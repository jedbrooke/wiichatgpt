# WiiChatGPT Design Document

## Hardware Budget

| Resource | Spec | Notes |
|---|---|---|
| CPU | Broadway PowerPC 750CL, 729 MHz | 2x integer ALU, 1x FPU with paired singles |
| CPU FP peak | ~2.9 GFLOPS | via `ps_madd` (2 FMACs/cycle) |
| CPU INT peak | ~729M multiply/sec | single IU for `mullw` |
| GPU | Hollywood (ATI), 243 MHz | Fixed-function pipeline: XF + TEV + indirect tex |
| GPU TEV peak | ~15.6 GOPS (8-bit) | 243 MHz x 4 pixels/clock x 16 stages |
| GPU XF peak | ~6-18M vertices/sec | fp32 matrix-vector transforms |
| GPU pixel fillrate | ~972 Mpixels/sec | 4 pixel pipes |
| MEM1 | 24 MB 1T-SRAM | CPU main memory, ~1.9 GB/s |
| MEM2 | 64 MB GDDR3 | GPU-accessible, ~3.9 GB/s |
| GPU embedded | 3 MB on-die | EFB + texture cache, internal bandwidth ~30 GB/s |
| L1 cache | 32 KB I + 32 KB D | |
| L2 cache | 256 KB | |
| SD card | ~25 MB/s sequential | Hardware controller bottleneck |
| USB 2.0 | ~15-20 MB/s practical | Via Starlet ARM IOS layer |
| DVD | ~12-13 MB/s | 3x DVD speed, high seek latency (~100ms) |

## Architecture Overview

```
 SD/USB/DVD (slow storage)
  |
  |  PLE table (~430MB INT4)
  |  ~25 MB/s SD, ~4KB per token lookup
  v
+---------------------------+
|  MEM2 (64MB GDDR3)       |
|  Dense transformer core   |
|  (~70MB INT4)             |
|  Weight textures (tiled)  |
+---------------------------+
  |          |
  v          v
+------+  +---------------------------+
| CPU  |  | GPU (Hollywood)           |
| 750CL|  | XF: vertex transforms     |
| 729M |  | TEV: 16-stage pixel pipe  |
|      |  | Indirect tex: LUT lookups |
+------+  +---------------------------+
  |          |
  | activations, reductions,
  | attention, PLE, small matvecs
  |          |
  |   large matvecs (tiled)
  |   (output head, QKV, FFN)
  v          v
+---------------------------+
|  MEM1 (24MB 1T-SRAM)     |
|  Activations, KV cache,  |
|  working buffers          |
+---------------------------+
```

## Per-Layer Embeddings (PLE)

Borrowed from Google's Gemma 3n and validated by slvDev's [esp32-ai](https://github.com/slvDev/esp32-ai).

The core idea: most parameters in a transformer can be restructured as a **lookup table** rather than dense compute weights. Each token gets a per-layer embedding vector retrieved by index, not by matrix multiply.

**How it works:**
1. A large embedding table (the PLE table) maps `token_id -> packed vector` covering all layers
2. At each transformer layer, the relevant slice is extracted and gated into the residual stream
3. The lookup is a pure array index, no matmul, effectively free compute

**Memory split:**

| Component | Size (1B model, INT4) | Location |
|---|---|---|
| PLE table (~86% of params) | ~430 MB | SD card / USB |
| Dense core (~14% of params) | ~70 MB | MEM2 (64MB) |
| Standard embedding + lm_head | ~10-20 MB | MEM2 |
| Activations, KV cache | ~5-10 MB | MEM1 |

**Per-token I/O from disk:**
- ~24 layers x ~1-2 KB per layer = ~24-48 KB per token
- At 25 MB/s SD read: **~1-2 ms per token** for PLE I/O (negligible)

This is what makes a ~1B model feasible on 88MB of total RAM. The dense core that actually needs matrix multiplies is small enough to fit in memory. The bulk of the "knowledge" lives in the PLE table on disk, accessed by cheap random reads.

**Reference:** esp32-ai runs a 28.9M param model (25M in PLE, 559K dense core) at 9.88 tok/s on an ESP32-S3 with only 512KB SRAM + 8MB PSRAM + 16MB flash.

## GPU Hardware Inventory

The Hollywood GPU has three compute-capable subsystems. Previous versions of this document only considered the TEV. All three are relevant.

### TEV (Texture Environment Unit)

The per-pixel color combiner. 16 programmable stages, 4 pixel pipes running at 243 MHz.

**Per-stage formula:**
```
out = (d +/- ((1-c)*a + c*b) + bias) * scale
```

With `a=ZERO, b=KONST, c=RASC, d=ZERO`, this computes `RASC * KONST` per channel (a multiply). With `d=accumulator`, it becomes a fused multiply-add.

**Capabilities:**
- 4 RGBA channels per pixel (4 values processed in parallel)
- 8-bit unsigned / 10-bit signed per channel
- Operations: multiply, add, subtract, lerp, clamp, scale (x1, x2, x4, /2)
- 4 konst color registers, 4 TEV output registers (TEVPREV + TEVREG0-2)
- 8 texture map slots, 8 texture coordinate slots
- EFB: 640x528 = 337K pixels as compute output elements

**Cannot do:** exp, sqrt, division, cross-pixel reduction, anything requiring >8-bit precision per pass.

**Peak:** ~15.6 GOPS (8-bit MACs), ~972 Mpixels/sec fillrate.

### XF (Transform Unit)

The vertex transform engine. Runs at 243 MHz, does fp32 matrix-vector multiplies in hardware.

**Resources:**
- 10 position/normal matrix slots (4x3 fp32 each, 12 floats per matrix)
- 10 texture matrix slots (4x3 fp32)
- 20 dual/post-transform matrix slots (4x3 fp32)
- 8 hardware lights (each computes N.L dot product: fp32 internally, 8-bit color output)
- 8 texgen stages (each computes a 2x4 or 3x4 matrix * input vector, results as fp32 texture coordinates)

**Throughput:** ~6-18M vertices/sec depending on configuration.

**For compute:** Texgen stages are the most interesting: each produces 2-3 fp32 outputs per vertex as texture coordinates. With 8 texgen stages, one vertex produces up to 24 fp32 values. The lighting unit computes 8 independent N.L dot products per vertex, but outputs are clamped to 8-bit color, limiting usefulness for precision-sensitive work.

**Precision through the pipeline:** XF texgen outputs are fp32. The rasterizer interpolates them in fp32 (perspective-correct). The TEV receives texture coordinates as **s17.7 fixed-point** (signed, 24 usable bits), confirmed via the Dolphin software rasterizer (`pixel_float * 128`). This is distinct from the TEV *color* pipeline, which is 8-bit. Texcoords used as texture addresses retain much more precision than TEV color math.

**Current status:** Not yet used. Evaluated for general fp32 matmul acceleration and rejected: XF throughput is ~48-144M fp32 dot products/sec (8 texgens * 6-18M vertices/sec), which is 6-17x slower than the CPU paired-singles path (~850M MACs/sec sustained). There is no direct CPU readback of texcoord values (only EFB color and depth are readable). Packing work into many vertices to increase XF utilization starves the TEV (the XF becomes the pipeline bottleneck). The XF is viable for one specific role: using fp32 dot product results as high-precision (s17.7) LUT addresses for activation functions. See Stretch Goal 3.

### Indirect Texture Unit

Per-TEV-stage texture coordinate manipulation via a 2x3 fixed-point matrix multiply.

**Resources:**
- 3 programmable 2x3 matrices (S0.10 fixed-point, ~11-bit precision per element)
- Scale exponent: power-of-2 shift from 2^-32 to 2^+31
- Applied per-pixel as part of texture coordinate generation before the final texture fetch

**For compute:** The primary use is activation function LUTs. Bake `sigmoid(x)`, `exp(x)`, or `1/sqrt(x)` into a 256-entry texture. An indirect texture lookup converts a TEV intermediate value through the nonlinearity without leaving the GPU.

**Current status:** Not yet used. Planned as a stretch goal for keeping the FFN pipeline (gate matmul -> SiLU -> elementwise mul -> down matmul) entirely on the GPU.

### EFB Blending

Hardware alpha/color blending at the framebuffer write stage.

**For compute:** `GX_SetBlendMode(GX_BM_BLEND, GX_BL_ONE, GX_BL_ONE, ...)` configures additive blending: each new render pass adds its results directly to the existing EFB contents. This enables multi-pass dot product accumulation without CPU readback between passes. Instead of: render pass 1 -> EFB copy -> CPU accumulate -> render pass 2, we get: render pass 1 -> render pass 2 (additive) -> ... -> render pass N -> single EFB copy.

**Precision constraint:** Results are still clamped to 8-bit per channel after each blend. For this to work, per-pass partial sums must be small enough that accumulated totals stay within [0, 255]. For INT4 weights in [0,1] range with 64 MACs per pass, each pass contributes roughly ~8-16/255, so 8 passes total ~64-128/255. This is within range.

## Compute Strategy

### Design Principles

1. **CPU is the baseline for everything.** The paired-singles INT8 dequant path (`psq_l` + `ps_madd`) handles all operations at fp32 precision. Every GPU acceleration must beat the CPU path or it is not worth the complexity.

2. **GPU wins on parallelism, not precision.** The CPU does fp32 natively. The GPU's TEV is 8-bit. GPU paths are worth it only when we can process enough independent operations in parallel to overcome setup overhead and tolerate quantization noise.

3. **Minimize CPU<->GPU round-trips.** Each EFB readback costs ~1-2 ms. The best GPU workloads render many pixels across multiple passes (using additive blend), then read back once.

4. **The output head dominates compute.** Per esp32-ai's profile (the closest reference architecture), the output head consumes ~63% of per-token compute. It is the highest-priority GPU target.

### GPU: Tiled TEV Matmul with Additive Blend

The GPU handles large matrix-vector multiplies: operations where the output dimension is large enough to fill a significant portion of the EFB, and the inner dimension is large enough that the compute exceeds GPU setup costs.

**How it works (for y = W*x, W is [M, K]):**

1. Pack weight matrix tiles as GX textures (RGBA8 = 4 values per texel). Each texel holds 4 weight values.

2. Encode the input activation vector as vertex colors (4 values per vertex via RGBA) or as a second texture.

3. Configure TEV stages for multiply-accumulate:
   - `a=ZERO, b=KONST, c=RASC, d=ZERO` computes `RASC * KONST` per channel
   - Chain 16 stages for 16 x 4 = 64 MACs per pixel per pass
   - Each pixel in the EFB computes one output element's partial dot product

4. Render a full-screen quad. With EFB at 640x528 = 337K pixels, each draw call produces up to 337K output elements. For a matvec with M output rows, use M pixels.

5. For K > 64 (dot products longer than one pass), use **EFB additive blending** (`GX_BL_ONE + GX_BL_ONE`). Each subsequent pass uploads the next weight tile, re-renders, and the hardware adds the new partial sums directly to the EFB contents. No CPU readback between passes.

6. After all passes complete, one `GX_CopyTex` reads the final accumulated results from the EFB into a MEM2 buffer. The CPU dequantizes the 8-bit results back to fp32.

**Per-pass timing:**
- Full-screen render at 4 pixels/clock: 337K / (243M x 4) = ~0.35 ms
- With texture fetch + 16 TEV stages: ~1-2 ms per pass
- K=512 needs 8 passes: ~8-16 ms total render time
- One final EFB copy: ~1-2 ms
- Total per large matvec: ~10-18 ms

**Tile size considerations:**
- EFB: 640 x 528 pixels, each producing one output element
- With 64 MACs/pixel/pass, a K=512 dot product needs ~8 passes
- Output dimension up to ~337K per draw call (far more than any single matvec needs)

**Precision model:**
- Weights and activations quantized to [0, 255] (8-bit unsigned)
- TEV multiply: `a * b / 255` per channel (fixed-point with +128 rounding)
- Additive blend accumulates 8-bit partial sums, clamped to [0, 255]
- CPU reads back 8-bit results and scales to fp32
- For INT4 weights (values -7..7 mapped to ~0..255 via offset), the quantization noise is roughly comparable to running the full model at INT4, which is the storage precision anyway

### CPU: Paired Singles + GQR Dequant

The CPU is the primary compute path. All operations run here. Large matvecs are optionally offloaded to the GPU when the GPU path is validated.

**Peak:** ~2.9 GFLOPS (729 MHz x 2 lanes x 2 ops/FMAC).
**Sustained:** ~1.7 GFLOPS at ~60% utilization.

**The GQR hardware dequant path:**
- `psq_l frD, addr, W, GQR_index` loads 2x INT8 values from memory and **automatically converts them to fp32 pairs** in a single instruction
- GQR Type field `6` = signed 8-bit (s8), with a configurable power-of-2 scale factor
- Combined with `ps_madd` for the multiply-accumulate

Weights stay as INT8 in memory (half the bandwidth vs fp32), computation happens in fp32 via hardware dequant-on-load. This is the same pattern as llama.cpp's Q8_0 format.

### Operation Assignment

**GPU (TEV tiled matmul, large matvecs only):**

| Operation | Shape (per layer) | Priority | Rationale |
|---|---|---|---|
| Output head | [V, D] * x | Highest | ~63% of compute, V=32K independent dot products |
| FFN gate | [F, D] * x | High | Large, many output elements |
| FFN up | [F, D] * x | High | Same shape as gate, can batch |
| FFN down | [D, F] * g | High | Large inner dimension |
| QKV projection | [3D, D] * x | High | Large, 3D output elements |
| Attention output proj | [D, D] * att | High | Large |
| PLE model projection | [L*P, D] * x | Medium | Large (L*P can be ~3000+) |

**CPU (always, not offloaded):**

| Operation | Rationale |
|---|---|
| RMSNorm | Requires reduction (sum of squares) then elementwise. Reductions cannot parallelize across pixels. |
| Softmax | Requires reduction (max, sum) and exp(). Reduction-heavy. |
| SiLU / GELU activation | Elementwise nonlinear. Trivial on CPU, not worth GPU round-trip unless part of a larger GPU pipeline (see stretch goals). |
| SwiGLU elementwise multiply | Small elementwise op between two FFN results. |
| RoPE | Trig (cos/sin) + elementwise on small vectors. Computed once per position, reused across layers. |
| Residual adds | Trivial elementwise (size D), not worth GPU overhead. |
| Attention scores (Q.K) | Many small dot products (head_size ~32-96) + softmax + weighted V sum. Precision-sensitive (softmax requires fp32). Reduction-heavy. |
| PLE table lookups | SD card I/O, CPU-managed. |
| PLE combine | Small elementwise (size L*P). |
| Small matvecs: PLE gate [P, D], PLE proj [D, P] | P=128-256, too small to amortize GPU setup overhead. |
| KV cache management | Memory operations. |
| GX draw call setup, texture uploads | CPU-side GPU control. |
| Sampling (argmax / top-p) | Sequential, tiny. |

### CPU/GPU Overlap

While the GPU renders the current matvec, the CPU processes:
- The previous matvec's result (activation function, normalization)
- PLE lookups from SD card (I/O overlapped with GPU compute)
- KV cache writes
- Next matvec's weight texture preparation and upload

The pipeline within a single layer looks like:

```
CPU: rmsnorm -> [upload QKV weights]
GPU:            [render QKV matvec...............] -> copytex
CPU:                                                  RoPE, attn scores, softmax, weighted V
CPU: [upload attn_proj weights]
GPU: [render attn_proj matvec...] -> copytex
CPU:                                 residual add, rmsnorm -> [upload FFN weights]
GPU:                                                          [render gate+up matvecs...] -> copytex
CPU:                                                                                        SiLU, SwiGLU mul
...
```

### Stretch Goals

1. **Indirect texture activation LUTs:** Bake SiLU/GELU into a 256-entry texture. After the FFN gate matvec renders into the EFB, apply the activation via an indirect texture lookup in a follow-up TEV pass, keeping the data on-GPU between the gate and the SwiGLU multiply. This avoids a CPU round-trip for the activation.

2. **Full FFN pipeline on GPU:** gate matmul -> SiLU (via indirect LUT) -> elementwise mul with up matmul -> down matmul, all as consecutive GPU render passes without EFB readback until the final result. This is the maximum GPU utilization scenario.

3. **XF texgen-addressed activation LUTs:** Use XF texgen to compute a fp32 dot product per vertex, then use the result (delivered to the TEV as an s17.7 texcoord, 24 usable bits) to address a 1D LUT texture containing SiLU/GELU values. This applies the nonlinearity at 24-bit addressing precision. In the FFN pipeline, the gate matmul partial sums accumulate in the TEV at 8-bit, but the activation function lookup uses the higher-precision texcoord path, improving the error profile where nonlinearities are most sensitive (near zero). Requires routing the gate output through a texgen stage, which limits this to cases where the dot product can be expressed as a single 3x4 or 2x4 matrix-vector multiply per texgen channel.

### Throughput: GPU Path to Full Utilization

The gap between theoretical TEV peak (~15.6 GOPS) and realistic throughput:

1. **EFB additive blend precision.** Multi-pass accumulation in 8-bit space. Per-pass partial sums must stay small enough to avoid saturation. Validation needed via the GX simulator, then on real hardware.

2. **Draw call setup overhead.** Each pass requires configuring TEV stages, loading weight textures, issuing vertex data. The GX FIFO command stream adds latency. Mitigation: minimize passes by using all 16 TEV stages, pre-build command buffers where possible.

3. **Texture upload bandwidth.** Weight tiles must be uploaded as GX textures before each pass. At ~3.9 GB/s MEM2 bandwidth, a 256x256 RGBA8 tile (256KB) takes ~0.065ms, but `GX_LoadTexObj` has CPU overhead. Mitigation: double-buffer textures, upload the next tile while GPU renders the current one.

4. **EFB copy latency.** The final readback via `GX_CopyTex` after all passes. Estimated ~1-2 ms per copy. Only one copy per matvec (not per pass, thanks to additive blend).

5. **Quantization error.** 8-bit TEV math with 8-bit EFB accumulation introduces noise. For INT4 weights already stored at 4-bit precision, the additional TEV quantization error may be acceptable. Needs empirical validation.

**Validation milestones:**
1. Tiled matmul with additive blend in the GX simulator (correctness + precision)
2. Benchmark on Dolphin emulator with GPU timing
3. Benchmark on real Wii hardware
4. Compare GPU matmul throughput vs CPU paired-singles baseline at each matrix size
5. Profile EFB copy and draw call overhead to find the real utilization ceiling
6. End-to-end inference quality comparison: GPU-accelerated vs CPU-only

## Inference Stack

**Starting point:** [karpathy/llama2.c](https://github.com/karpathy/llama2.c) (~700 lines, pure C, no dependencies) and [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (PLE architecture, custom C runtime).

**Why these, not llama.cpp:**
- llama.cpp is STL-heavy C++, depends on mmap, pthreads, CMake. Porting to devkitPPC/newlib is a large surface.
- llama2.c is already ported to [Dreamcast](https://github.com/thinkelastic/dc_llama2) (close Wii analog: retro console, fixed-function GPU, 16MB RAM, bare-metal OS) and [ESP32](https://github.com/Circuit-Digest/ESP32-Tiny-LLM).
- esp32-ai has a working PLE implementation at the right scale.

**Evolution path:**
1. Get llama2.c running on devkitPPC with the 260K TinyStories model (validates toolchain, SD loading, basic inference)
2. Integrate PLE from esp32-ai for the embedding architecture
3. Replace the naive matmul with paired-singles CPU kernel (`psq_l` + `ps_madd`)
4. Add TEV-accelerated tiled matmul backend for large matvecs
5. Scale up model size, validate with increasingly large models
6. If needed, migrate toward ggml backend for better quantization formats (GGUF, Q4_K, etc.)

**Other references:**
- [Tilka/llvm-ppc750cl](https://github.com/Tilka/llvm-ppc750cl): LLVM fork with paired singles support, could help get compiler-emitted `ps_madd`/`psq_l`
- [crtc-demos/tevsl](https://github.com/crtc-demos/tevsl): TEV shader compiler, higher-level syntax for TEV stage configuration
- [Scottcjn/llama-cpp-tigerleopard](https://github.com/Scottcjn/llama-cpp-tigerleopard): llama.cpp on PowerPC Mac (G4/G5), reference for PPC-specific ggml changes
- [JustVugg/colibri](https://github.com/JustVugg/colibri): pure C inference with disk-streaming weight architecture

## Model

Custom trained/fine-tuned, ~500M-1B params, PLE architecture.

**Quantization:** INT4 for weights (dense core + PLE table), fp32 accumulation. The Wii has no FP16 hardware. INT8 via GQR dequant is the native fast path on the CPU. INT4 packs tighter for storage but requires unpacking to INT8/fp32 at load time.

**Training:** handled externally (Jake). Architecture should follow the Gemma 3n PLE pattern:
- Dense core: attention (Q/K/V/O projections), FFN (gate/up/down), layer norms
- PLE table: per-layer embedding vectors, indexed by token ID
- Standard embedding + tied lm_head for output projection

**Domain:** Wii trivia, popular Wii games, Nintendo lore. Not general-purpose.

### Per-token operation breakdown

From esp32-ai's profile (28.9M model, 559K dense core, ESP32-S3), the per-token compute breaks down as:

| Stage | ms/token | Share | Operations |
|---|---|---|---|
| Output head | 59.4 | 63% | Single large matvec [V, D] * x |
| Attention | 20.5 | 22% | QKV matvec + RoPE + Q.K scores + softmax + weighted V + attn_proj matvec |
| PLE path | 6.4 | 7% | PLE model proj matvec + PLE gate matvec + GELU + PLE proj matvec + RMSNorm |
| FFN | 6.5 | 7% | Gate matvec + up matvec + SiLU + SwiGLU mul + down matvec + RMSNorm |
| Input | 2.2 | 2% | Embedding lookup + PLE table lookup + PLE combine |

The output head alone is 63% of compute. At the Wii's larger model sizes the ratios shift (the head shrinks proportionally as the per-layer ops grow with more layers), but it remains a substantial fraction.

## Memory Map (target)

```
MEM1 (24 MB, 1T-SRAM, ~1.9 GB/s):
  Activations buffer       ~4 MB
  KV cache                 ~8 MB  (depends on context length)
  Working scratch          ~4 MB
  Code + stack             ~2 MB
  GX FIFO + command buffer ~1 MB
  Remaining                ~5 MB

MEM2 (64 MB, GDDR3, ~3.9 GB/s):
  Dense core weights       ~50-70 MB (INT4)
  Standard embeddings      ~10 MB
  Weight texture tiles     ~2-4 MB (double-buffered)
  Remaining                ~0-2 MB

SD/USB (external, ~25 MB/s):
  PLE table                ~430 MB (INT4)
  Tokenizer vocabulary     ~1 MB
```

## Performance Estimates (Napkin Math)

All numbers assume the PLE architecture (dense core in RAM, PLE table on SD card). PLE I/O overhead (~24-48 KB per token from SD at 25 MB/s = ~1-2 ms) is negligible at all model sizes and ignored below.

**Ground rules:**
- FLOPs per token for a transformer at batch=1: roughly `2 * P_dense` (one multiply + one add per weight)
- At batch=1, inference is memory-bandwidth-bound: every weight is read once per token
- CPU memory ceiling: ~1.9 GB/s (MEM2 via CPU bus)
- GPU memory ceiling: ~3.9 GB/s (MEM2 via GPU bus)

### Model size sweep (with ~86% PLE / 14% dense split)

| Total model | Dense core | Dense size (INT4) | Dense size (INT8) | Fits MEM2 (64MB)? |
|---|---|---|---|---|
| 100M | 14M | 7 MB | 14 MB | Yes, easily |
| 300M | 42M | 21 MB | 42 MB | Yes |
| 500M | 70M | 35 MB | 70 MB | INT4 yes, INT8 barely |
| 1B | 140M | 70 MB | 140 MB | INT4 tight (70/64MB), INT8 no |

### tok/s estimates: CPU-only (paired singles, ~1.7 GFLOPS sustained at ~60% utilization)

| Dense core | FLOPs/token | Time/token | tok/s |
|---|---|---|---|
| 14M (100M total) | 28M | 16 ms | ~62 |
| 42M (300M total) | 84M | 49 ms | ~20 |
| 70M (500M total) | 140M | 82 ms | ~12 |
| 140M (1B total) | 280M | 165 ms | ~6 |

### tok/s estimates: GPU TEV (memory-bandwidth-bound at 3.9 GB/s, INT8 weights)

| Dense core | BW ceiling tok/s | At 30% util | At 50% util |
|---|---|---|---|
| 14M | 278 | 83 | 139 |
| 42M | 93 | 28 | 46 |
| 70M | 56 | 17 | 28 |
| 140M | 28 | 8.4 | 14 |

### tok/s estimates: CPU + GPU combined (overlapped)

GPU handles the large matvecs (output head, QKV, FFN projections), CPU handles attention scores, activations, reductions, PLE, and small matvecs in parallel.

| Dense core | Total model | Conservative | Optimistic |
|---|---|---|---|
| 14M | 100M | ~60 tok/s | ~100+ tok/s |
| 42M | 300M | ~20 tok/s | ~35 tok/s |
| 70M | 500M | ~15 tok/s | ~25 tok/s |
| 140M | 1B | ~8 tok/s | ~12 tok/s |

### Targets

| Target | Required tok/s | Max dense core (conservative) | Max total model (with PLE) |
|---|---|---|---|
| Real-time (10 tok/s) | 10 | ~120M | ~850M |
| Minimum viable (1 tok/s) | 1 | ~1B+ | ~7B+ |

**The sweet spot is 500M-1B total model with PLE**, giving a 70-140M dense core. At the low end (500M, 70M dense), we comfortably clear 10 tok/s. At the high end (~850M-1B, ~120-140M dense), we're right on the edge of real-time and GPU utilization is the difference between hitting 10 tok/s and falling short. INT4 quantization for the dense core is required at the 1B end to fit MEM2.

### Sanity checks

**Memory bandwidth check (1B model):** 140M INT8 weights = 140MB read per token. At 1.9 GB/s CPU bus: 74ms/token = 13.5 tok/s ceiling. At 3.9 GB/s GPU bus: 36ms/token = 28 tok/s ceiling. We're compute-bound on CPU, bandwidth-bound on GPU.

**Dreamcast reference:** 15M dense model at ~1 tok/s on 200MHz SH-4 (naive code, no SIMD). Our 14M dense estimate is ~62 tok/s on a 729MHz PPC with paired singles. The ~60x gap: 3.6x clock, ~2x SIMD, plus optimized kernels. Aggressive but plausible.

**esp32-ai reference:** 559K dense core at 9.88 tok/s on ESP32-S3 (240MHz, no FP SIMD). Direction checks out for our scaling.

## Open Questions

1. **EFB additive blend precision:** Can multi-pass 8-bit additive accumulation produce usable results for a 512-element dot product? Per-pass partial sums must stay within [0, 255] range across all passes. Needs validation in the GX simulator and empirically on hardware.

2. **EFB copy latency:** Real hardware timing for `GX_CopyTex` is not well documented. Only one copy per matvec (thanks to additive blend), but this is still the GPU path's fixed overhead. This is the make-or-break number for GPU utilization.

3. **GPU vs CPU crossover point:** At what matrix size does the GPU TEV path actually beat CPU paired-singles? Small matvecs (PLE gate at [128, D]) almost certainly lose to CPU due to setup overhead. Large ones (output head at [32K, D]) should win. The crossover needs measurement.

4. **Model architecture details:** Exact layer count, hidden dim, head count, PLE dim, for a ~1B model that fits the memory budget. Jake needs these constraints for training.

5. **DVD random read performance:** PLE lookups are random-access by token ID. DVD seek latency (~100ms) may make DVD impractical for PLE; SD card (low-ms seek) is likely required. DVD may still work for sequential weight loading at boot.

6. **Context length:** KV cache size scales linearly with context. At 24 layers x 512 hidden x 2 (K+V) x fp32, each token position costs ~96KB. A 256-token context = ~24MB of KV cache, consuming most of MEM1. May need INT8 KV cache or shorter context.

7. **XF compute potential (resolved):** The XF is not viable as a general fp32 matmul accelerator: throughput is 6-17x slower than CPU paired-singles, there is no CPU readback of texcoord values (only EFB color/depth), and packing work into many vertices starves the TEV. However, texgen outputs enter the TEV as s17.7 fixed-point texcoords (24 usable bits), which makes the XF useful for high-precision LUT addressing in activation functions. See Stretch Goal 3.
