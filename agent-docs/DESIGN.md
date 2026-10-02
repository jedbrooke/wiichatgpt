# WiiChatGPT Design Document

## Hardware Budget

| Resource | Spec | Notes |
|---|---|---|
| CPU | Broadway PowerPC 750CL, 729 MHz | 2x integer ALU, 1x FPU with paired singles |
| CPU FP peak | ~2.9 GFLOPS | via `ps_madd` (2 FMACs/cycle) |
| CPU INT peak | ~729M multiply/sec | single IU for `mullw` |
| GPU | Hollywood (ATI), 243 MHz | Fixed-function pipeline: XF + TEV + indirect tex |
| GPU TEV peak | ~15.6 GOPS (8-bit) | 243 MHz x 4 pixels/clock x 16 stages |
| GPU pixel fillrate | ~972 Mpixels/sec | 4 pixel pipes |
| MEM1 | 24 MB 1T-SRAM | CPU main memory, ~1.9 GB/s |
| MEM2 | 64 MB GDDR3 | GPU-accessible, ~3.9 GB/s |
| GPU embedded | 3 MB on-die | EFB + texture cache, internal bandwidth ~30 GB/s |
| SD card | ~25 MB/s sequential | Hardware controller bottleneck |
| USB 2.0 | ~15-20 MB/s practical | Via Starlet ARM IOS layer |
| DVD | ~12-13 MB/s | 3x DVD speed, high seek latency (~100ms) |

## Architecture Overview

See [MODEL.md](MODEL.md) for the full model specification. See [GX.md](GX.md) for the complete GX pipeline pseudocode.

The model is a Bi-Mamba-2 (binary SSM) with PLE. The architecture is hardware-driven: the hidden dimension (1920) and state dimension (528) are chosen to exactly fill the GPU's EFB.

```
 SD card
  |
  |  PLE table (~983MB INT8)
  |  ~7.7 KB per token lookup
  v
+-------------------------------+
|  MEM2 (64MB GDDR3)           |
|  Dense projections (~11MB)    |  binary, packed 8 per byte
|  Embedding (~31MB)            |  binary, packed 8 per byte
|  SSM params (~4MB INT8)       |
|  Weight textures (~4MB)       |  double-buffered
+-------------------------------+
  |              |
  v              v
+--------+   +----------------------------------+
| CPU    |   | GPU (Hollywood)                   |
| 750CL  |   | TEV: state update (full EFB)      |
| 729MHz |   | TEV: output head (43K pixels)     |
|        |   | Logic blend: binary XNOR          |
|        |   | CopyTex: state -> texture chain    |
+--------+   +----------------------------------+
  |              |
  | FFN (binary XNOR+popcount)
  | SSM recurrence (INT8)
  | norms, gating, PLE combine
  |              |
  |  state update rendered to EFB
  |  output head rendered to EFB
  v              v
+-------------------------------+
|  MEM1 (24MB 1T-SRAM)         |
|  Recurrent state (~4MB)       |  4 layers x 528 x 1920, INT8
|  Activations (~4MB)           |
|  Code + stack (~2MB)          |
+-------------------------------+
```

## GPU Compute Model

The GX pipeline (documented in [GX.md](GX.md)) is a fixed-function graphics pipeline. There are no programmable shaders. Compute is achieved by repurposing the graphics stages for arithmetic.

### The EFB as a Compute Surface

The Embedded Frame Buffer (EFB) is 640 x 528 pixels, each storing u8 RGBA color + u24 depth. For compute, the 3 color channels (R, G, B) are independent accumulators:

```
640 pixels x 528 rows x 3 channels = 1,013,760 accumulators
```

Each accumulator is s10 internally ([-1024..1023]) during TEV processing, clamped to u8 ([0..255]) on EFB write.

### TEV as a Multiply-Accumulate Unit

The TEV runs 16 sequential stages per pixel, 4 pixels in parallel. Each stage computes:

```
lerp = (a * (255 - c) + b * c + 128) >> 8
result = d +/- lerp
```

For a multiply-accumulate: `b = weight (from texture, per-pixel), c = activation (from konst, broadcast), d = CPREV (accumulator)`. This gives `acc += weight * activation / 255` per stage.

See [GX.md](GX.md) Stage 4+5 for the complete TEV pseudocode.

**Activation delivery via konst:** 4 konst registers x 4 RGBA channels = 16 unique scalar values per draw call. Each TEV stage selects one konst value via `kcsel`/`kasel`. This is the bottleneck for matmul inner dimension: 16 activation values per draw call, requiring `ceil(K/16)` draw calls for a K-element dot product.

**Weight delivery via texture:** Each pixel samples a different texel based on its texcoord, so weight data varies per pixel (per output row). Weight matrices are stored as GX textures in MEM2 (RGBA8 format, 4 values per texel).

### Logic Blend for Binary Ops

The Pixel Engine supports bitwise logic operations between the incoming fragment color and existing EFB contents:

```
BM_LOGIC with logic_op = EQUIV computes ~(src XOR dst) = XNOR per byte
```

XNOR is the binary dot product primitive: `dot(w, x) = popcount(XNOR(w_bits, x_bits)) * 2 - N`. The EFB holds packed binary weights, the fragment color carries packed binary activations, and the logic blend produces the XNOR result in a single draw call at full pixel throughput.

Popcount is not a native operation, but can be approximated via indirect texture LUT (a 256-entry I8 texture where `entry[i] = popcount(i)`).

### CopyTex: The GPU-to-GPU Data Path

`GX_CopyTex` DMAs a rectangular EFB region to a texture in MEM2. This texture can then be sampled by the next render pass, enabling multi-pass GPU computation without CPU readback:

```
Render pass 1 -> EFB -> CopyTex -> texture A
Render pass 2 (reads texture A) -> EFB -> CopyTex -> texture B
...
```

Each CopyTex costs ~0.5-1 ms regardless of pixel count (pipeline flush + DMA). This is the primary overhead in multi-pass GPU computation.

### Z Buffer for Conditional Write

The Z buffer (u24 per pixel) provides a hardware compare-and-discard mechanism. Setting `z_mode.func = GEQUAL` and pre-loading the Z buffer with a causal mask enables hardware causal masking: future positions fail the Z test and their fragments are discarded, preventing their color from being written.

### Why Mamba Fills the EFB (and Transformers Don't)

At batch=1, a standard transformer matmul produces M output elements where M is the layer width (a few hundred to a few thousand). This fills less than 1% of the EFB's 337K pixels. The output head (M = vocab_size) fills ~13%. The GPU surface is mostly wasted.

Mamba-2's recurrent state is a 2D matrix: [state_dim, head_dim] per head. With 1 head, state_dim=528, head_dim=1920, the state is 528 x 1920 = 1,013,760 elements, exactly matching the EFB's 1,013,760 accumulators. The state update (an outer product + decay) produces one output per accumulator, achieving 100% EFB utilization.

## Compute Strategy

### GPU: Recurrent State Update

The core GPU operation. Each layer's state update computes:

```
S_new[i][j] = decay * S_old[i][j] + v[i] * k[j]
```

This maps to the TEV as:

```
Texture 0: S_old (528 x 640, RGBA8, from CopyTex of previous state)
Texture 1: v vector (528 x 1, column vector, varies per EFB row)
Texture 2: k vector (1 x 640, row vector, varies per EFB column)

Stage 0: b = tex0 (S_old[i][j]),  c = KONST (decay),  d = ZERO
         -> TEVPREV = S_old * decay / 255

Stage 1: b = tex1 (v[i]),  c = ZERO,  d = ZERO
         -> TEVREG0 = v[i]  (load v into register)

Stage 2: b = tex2 (k[j]),  c = TEVREG0 (v[i]),  d = CPREV
         -> TEVPREV += v[i] * k[j] / 255

Write to EFB. CopyTex -> state texture for next layer.
```

The texcoord for texture 1 maps each pixel's row (0-527) to the v vector index. The texcoord for texture 2 maps each pixel's column (0-639, with 3 channels) to the k vector index. These are set up via XF texgen with appropriate texture matrices.

**Per-layer cost:** ~0.35 ms render (337K pixels at 972M pixels/sec) + ~0.5 ms CopyTex = ~0.85 ms.
**4 layers:** ~3.4 ms.

### GPU: Output Head

The output head [128K, 1920] is a standard matmul mapping to the TEV:

- 128K output rows / 3 channels = 43K pixels (12.7% EFB)
- Weights stored as textures (per-pixel via texcoord)
- Activations delivered via konst (16 per draw call)
- Inner dimension K=1920: `ceil(1920/16) = 120` draw calls
- Each draw call: ~0.15 ms setup + ~0.05 ms render = ~0.2 ms
- Total: 120 x 0.15 ms + 1 CopyTex = ~19 ms

### CPU: FFN (Binary)

FFN gate [1920, 5120], up [1920, 5120], down [5120, 1920] per layer. All binary weights.

Binary matmul on CPU: XOR + `popcntb` (byte popcount). Each 32-bit XOR+popcount computes 32 binary MACs. At ~250-350M groups/sec (2-3 cycles per XOR+popcount pair): ~8-11B binary MACs/sec.

Per-layer FFN: ~30M binary MACs. At ~8B MACs/sec: ~3.6 ms.
4 layers: ~14.4 ms.

### CPU: SSM Recurrence

The sequential state update `h_t = A * h_{t-1} + B * x_t` runs on CPU at INT8 precision (the GPU handles the parallel outer product portion of the state update, the CPU handles the sequential scalar recurrence parameters A, B, C, D).

4 layers: ~1 ms.

### CPU: Everything Else

| Operation | Engine | Time |
|---|---|---|
| RMSNorm (4 layers) | CPU | ~0.3 ms |
| Gating / SiLU (4 layers) | CPU | ~0.3 ms |
| PLE lookup + combine | CPU + SD I/O | ~0.3 ms |
| Residual adds | CPU | negligible |
| Sampling (argmax / top-p) | CPU | negligible |

### Operation Assignment

**GPU (TEV):**

| Operation | Shape | EFB pixels | Draw calls | Time |
|---|---|---|---|---|
| State update (4 layers) | [528, 1920] outer product | 337K (100%) | 4 | ~3.4 ms |
| Output head | [128K, 1920] matmul | 43K (12.7%) | 120 | ~19 ms |

**CPU:**

| Operation | Rationale |
|---|---|
| FFN (binary matmul) | Per-layer matmuls too small for EFB (FFN gate+up = 3.4K pixels = 1% EFB), draw call overhead dominates |
| SSM recurrence | Sequential, precision-sensitive |
| RMSNorm | Requires cross-element reduction |
| Gating / activation | Elementwise, trivial on CPU |
| PLE combine | Elementwise multiply + add, negligible compute |
| Sampling | Sequential |

### CPU/GPU Overlap

The GPU state update and output head can overlap with CPU work. The pipeline for one token:

```
CPU: prepare v, k vectors for layer 0
GPU: render state update layer 0 -------> CopyTex
CPU: FFN layer 0 (binary)                 prepare v, k for layer 1
GPU:                                       render state update layer 1 -> CopyTex
CPU: FFN layer 1                           ...
     (repeat for layers 2-3)
CPU: norms, gating, PLE combine
GPU: render output head (120 draw calls) ---------> CopyTex
CPU:                                                 sampling
```

The FFN (CPU) and state update (GPU) for the same layer can partially overlap since they operate on different data. The output head (GPU) runs after all layers complete, during which the CPU prepares the activation vector and handles sampling setup.

## Inference Stack

**Starting point:** [jserv/bitmamba.c](https://github.com/jserv/bitmamba.c) (portable C inference for ternary Mamba-2, weights packed 4 per byte, non-ternary components in float32).

**Why BitMamba.c:**
- Already C, no C++ STL dependencies
- Handles binary/ternary weight packing and XNOR+popcount inference
- Mamba-2 architecture matches our target
- Small codebase, easy to port to devkitPPC/newlib

**Additional references:**
- [Tangshengku/Bi-Mamba](https://github.com/Tangshengku/Bi-Mamba): 1-bit Mamba training code (PyTorch, for training on modern GPUs)
- [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai): PLE implementation reference (28.9M model on ESP32)
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c): minimal C inference reference, ported to [Dreamcast](https://github.com/thinkelastic/dc_llama2) and [ESP32](https://github.com/Circuit-Digest/ESP32-Tiny-LLM)
- [crtc-demos/tevsl](https://github.com/crtc-demos/tevsl): TEV shader compiler for GX TEV stage configuration
- [Tilka/llvm-ppc750cl](https://github.com/Tilka/llvm-ppc750cl): LLVM fork with paired singles support

**Evolution path:**
1. Port BitMamba.c to devkitPPC (validates toolchain, basic Mamba inference on Wii CPU)
2. Add PLE support (SD card lookup + elementwise combine, adapted from esp32-ai)
3. Add GX TEV backend for state update (the full-EFB outer product render)
4. Add GX TEV backend for output head (tiled matmul with konst activation delivery)
5. Train custom Bi-Mamba + PLE model (H=1920, state_dim=528, L=4, V=128K) using Bi-Mamba training code
6. End-to-end inference on Dolphin emulator, then real hardware

## Performance Estimates

See [MODEL.md](MODEL.md) for the full per-token breakdown.

**Summary:** ~1.27B total params (44M binary dense + 246M binary embed + 4M INT8 SSM in RAM, 983M INT8 PLE on disk). ~39 ms/token, ~26 tok/s. 2.6x above the 10 tok/s real-time target.

| Operation | Engine | Time | Share |
|---|---|---|---|
| State update (4 layers) | GPU | ~3.4 ms | 9% |
| FFN (4 layers) | CPU | ~14.4 ms | 37% |
| Output head | GPU | ~19 ms | 49% |
| SSM + norms + PLE + overhead | CPU | ~2.6 ms | 7% |
| **Total** | | **~39 ms** | |

The output head dominates at 49%. The FFN is second at 37%. Both are binary operations (XNOR+popcount on CPU, TEV multiply on GPU). The GPU state update is fast (9%) because the outer product is a simple 3-stage TEV operation at full EFB throughput.

### Bandwidth Check

Total weight reads per token from MEM2: ~42 MB (binary dense + embedding, unpacked by the access pattern). At 3.9 GB/s GPU bus: ~11 ms. Compute (39 ms) exceeds bandwidth (11 ms); the model is compute-bound.

### Reference Comparisons

**Bi-Mamba 780M (reference, on modern GPU):** runs at hundreds of tok/s on an H100. Our 1.27B model at 26 tok/s on a 2001-era console CPU + fixed-function GPU is ~10-50x slower per parameter, which is consistent with the ~1000x hardware gap.

**esp32-ai (embedded reference):** 28.9M model at 9.88 tok/s on ESP32-S3 (240MHz, no SIMD). Our 1.27B model is ~44x larger, running at 2.6x the speed, enabled by binary weights (8x compression), PLE (cheap disk params), and paired-singles SIMD.

## Open Questions

1. **Binary + PLE quality:** No prior work combines binary Mamba with PLE. The interaction between 1-bit dense weights and INT8 PLE conditioning is untested. PLE may compensate well for binary precision loss (it provides high-precision per-token signals at every layer), or the 4-layer depth may be too shallow regardless. The V=32K/L=8 fallback trades vocab for depth if needed.

2. **CopyTex latency:** Estimated at ~0.5-1 ms per call based on Dolphin profiling. Real hardware timing is not well documented. The state update path requires 4 CopyTex operations per token (one per layer). If CopyTex is slower than estimated, the state update cost rises proportionally.

3. **TEV state update precision:** The outer product `v[i] * k[j]` is computed at 8-bit via the TEV lerp formula, accumulated with the decay term. The decay multiply `S_old * decay / 255` introduces rounding error each layer. Over 4 layers this may accumulate. Needs validation in the GX simulator.

4. **Indirect texture popcount viability:** The proposed binary XNOR+popcount pipeline on the GPU (logic blend for XNOR, indirect texture LUT for popcount) has not been validated. If it works, it could move the FFN from CPU to GPU, reducing the FFN share from 37% to near zero and bringing total time to ~25 ms (~40 tok/s).