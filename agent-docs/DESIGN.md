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

See [MODEL.md](MODEL.md) for the full model specification (architecture, parameter counts, quantization, memory layout, PLE details).

```
 SD/USB/DVD (slow storage)
  |
  |  PLE table (~245MB INT8)
  |  ~25 MB/s SD, ~7.5KB per token lookup
  v
+---------------------------+
|  MEM2 (64MB GDDR3)       |
|  Dense core (~31MB INT8)  |
|  Embeddings (~12.3MB)     |
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
  | attention, PLE, norms
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

**Precision constraint:** Results are still clamped to 8-bit per channel after each blend. For this to work, per-pass partial sums must be small enough that accumulated totals stay within [0, 255]. With INT8 weights mapped to [0, 255] and 64 MACs per pass, the scale factor must be chosen so per-pass contributions fit within the accumulation range.

## Compute Strategy

### Design Principles

1. **CPU is the baseline for everything.** The paired-singles INT8 dequant path (`psq_l` + `ps_madd`) handles all operations at fp32 precision. Every GPU acceleration must beat the CPU path or it is not worth the complexity.

2. **GPU wins on parallelism, not precision.** The CPU does fp32 natively. The GPU's TEV is 8-bit. GPU paths are worth it only when we can process enough independent operations in parallel to overcome setup overhead and tolerate quantization noise.

3. **Minimize CPU<->GPU round-trips.** Each EFB readback costs ~1-2 ms. The best GPU workloads render many pixels across multiple passes (using additive blend), then read back once.

4. **The output head dominates compute.** The output head (384 x 32K) accounts for ~14 ms of ~54 ms per-token compute (~26%). It is the highest-priority GPU target due to the large number of independent dot products (32K).

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
- K=384 needs 6 passes: ~6-12 ms total render time
- One final EFB copy: ~1-2 ms
- Total per large matvec: ~8-14 ms

**Tile size considerations:**
- EFB: 640 x 528 pixels, each producing one output element
- With 64 MACs/pixel/pass, a K=384 dot product needs ~6 passes
- Output dimension up to ~337K per draw call (far more than any single matvec needs)

**Precision model:**
- Weights and activations quantized to [0, 255] (8-bit unsigned)
- TEV multiply: `a * b / 255` per channel (fixed-point with +128 rounding)
- Additive blend accumulates 8-bit partial sums, clamped to [0, 255]
- CPU reads back 8-bit results and scales to fp32
- The entire model uses INT8 quantization (see [MODEL.md](MODEL.md)), so the TEV's 8-bit pipeline is a natural match

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

Shapes use the actual model dimensions: D=384, F=1024, V=32K, H=6 heads, head_dim=64. See [MODEL.md](MODEL.md) for the full architecture.

**GPU (TEV tiled matmul, large matvecs only):**

| Operation | Shape (per layer) | Priority | Rationale |
|---|---|---|---|
| Output head | [32K, 384] * x | Highest | ~26% of compute, 32K independent dot products |
| FFN gate | [1024, 384] * x | High | Large, many output elements |
| FFN up | [1024, 384] * x | High | Same shape as gate, can batch |
| FFN down | [384, 1024] * g | High | Large inner dimension |
| QKV projection | [1152, 384] * x | High | 3x384 = 1152 output elements |
| Attention output proj | [384, 384] * att | High | Large |

**CPU (always, not offloaded):**

| Operation | Rationale |
|---|---|
| RMSNorm | Requires reduction (sum of squares) then elementwise. Reductions cannot parallelize across pixels. |
| Softmax | Requires reduction (max, sum) and exp(). Reduction-heavy. |
| SiLU activation | Elementwise nonlinear. Trivial on CPU, not worth GPU round-trip unless part of a larger GPU pipeline (see stretch goals). |
| SwiGLU elementwise multiply | Small elementwise op between two FFN results. |
| RoPE | Trig (cos/sin) + elementwise on small vectors. Computed once per position, reused across layers. |
| Residual adds | Trivial elementwise (384 values), not worth GPU overhead. |
| Attention scores (Q.K) | Small dot products (head_dim=64) + softmax + weighted V sum. Precision-sensitive (softmax requires fp32). Reduction-heavy. |
| PLE lookup + combine | SD card I/O (7.5 KB per token) + element-wise multiply and add (384 dims, 20 layers). Negligible compute (~0.3 ms total). |
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
CPU:                                 residual add, PLE combine, rmsnorm -> [upload FFN weights]
GPU:                                                                       [render gate+up matvecs...] -> copytex
CPU:                                                                                                      SiLU, SwiGLU mul
...
```

### Stretch Goals

1. **Indirect texture activation LUTs:** Bake SiLU into a 256-entry texture. After the FFN gate matvec renders into the EFB, apply the activation via an indirect texture lookup in a follow-up TEV pass, keeping the data on-GPU between the gate and the SwiGLU multiply. This avoids a CPU round-trip for the activation.

2. **Full FFN pipeline on GPU:** gate matmul -> SiLU (via indirect LUT) -> elementwise mul with up matmul -> down matmul, all as consecutive GPU render passes without EFB readback until the final result. This is the maximum GPU utilization scenario.

3. **XF texgen-addressed activation LUTs:** Use XF texgen to compute a fp32 dot product per vertex, then use the result (delivered to the TEV as an s17.7 texcoord, 24 usable bits) to address a 1D LUT texture containing SiLU values. This applies the nonlinearity at 24-bit addressing precision. In the FFN pipeline, the gate matmul partial sums accumulate in the TEV at 8-bit, but the activation function lookup uses the higher-precision texcoord path, improving the error profile where nonlinearities are most sensitive (near zero). Requires routing the gate output through a texgen stage, which limits this to cases where the dot product can be expressed as a single 3x4 or 2x4 matrix-vector multiply per texgen channel.

### Throughput: GPU Path to Full Utilization

The gap between theoretical TEV peak (~15.6 GOPS) and realistic throughput:

1. **EFB additive blend precision.** Multi-pass accumulation in 8-bit space. Per-pass partial sums must stay small enough to avoid saturation. Validation needed via the GX simulator, then on real hardware.

2. **Draw call setup overhead.** Each pass requires configuring TEV stages, loading weight textures, issuing vertex data. The GX FIFO command stream adds latency. Mitigation: minimize passes by using all 16 TEV stages, pre-build command buffers where possible.

3. **Texture upload bandwidth.** Weight tiles must be uploaded as GX textures before each pass. At ~3.9 GB/s MEM2 bandwidth, a 256x256 RGBA8 tile (256KB) takes ~0.065ms, but `GX_LoadTexObj` has CPU overhead. Mitigation: double-buffer textures, upload the next tile while GPU renders the current one.

4. **EFB copy latency.** The final readback via `GX_CopyTex` after all passes. Estimated ~1-2 ms per copy. Only one copy per matvec (not per pass, thanks to additive blend).

5. **Quantization error.** 8-bit TEV math with 8-bit EFB accumulation introduces noise. With INT8 weights, the TEV pipeline matches the storage precision, so additional quantization error should be minimal. Needs empirical validation.

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
5. Scale up to the 288M model, validate end-to-end inference

**Other references:**
- [Tilka/llvm-ppc750cl](https://github.com/Tilka/llvm-ppc750cl): LLVM fork with paired singles support, could help get compiler-emitted `ps_madd`/`psq_l`
- [crtc-demos/tevsl](https://github.com/crtc-demos/tevsl): TEV shader compiler, higher-level syntax for TEV stage configuration
- [Scottcjn/llama-cpp-tigerleopard](https://github.com/Scottcjn/llama-cpp-tigerleopard): llama.cpp on PowerPC Mac (G4/G5), reference for PPC-specific ggml changes
- [JustVugg/colibri](https://github.com/JustVugg/colibri): pure C inference with disk-streaming weight architecture

## Model

288M total params (31M dense core, 12.3M embeddings, 245M PLE table). INT8 quantization throughout. See [MODEL.md](MODEL.md) for the full specification.

The PLE architecture lets us have a 288M parameter model while keeping only ~43 MB of weights in RAM. The 245M PLE parameters live on disk (SD card primary, DVD future target) and are accessed via cheap O(1) lookups at 7.5 KB per token.

### esp32-ai Per-Token Profile (reference)

From esp32-ai's profile (28.9M model, 559K dense core, ESP32-S3):

| Stage | Share | Operations |
|---|---|---|
| Output head | 63% | Single large matvec [V, D] * x |
| Attention | 22% | QKV matvec + RoPE + Q.K scores + softmax + weighted V + attn_proj matvec |
| PLE path | 7% | PLE model proj matvec + PLE gate matvec + GELU + PLE proj matvec + RMSNorm |
| FFN | 7% | Gate matvec + up matvec + SiLU + SwiGLU mul + down matvec + RMSNorm |
| Input | 2% | Embedding lookup + PLE table lookup + PLE combine |

Note: esp32-ai uses PLE projection matrices (PLE dim != hidden_dim). Our model has PLE dim = hidden_dim, so PLE requires no projection matvecs, just element-wise multiply + add. The 7% PLE path cost from esp32-ai drops to near zero in our architecture, shifting compute share to the other stages. Our per-token breakdown from MODEL.md: ~36 ms dense core matmuls, ~14 ms output head, ~4 ms overhead, ~0.3 ms PLE = ~54 ms total (~18 tok/s).

## Performance Estimates

See [MODEL.md](MODEL.md) for the detailed per-token breakdown and KV cache scaling table.

**Summary:** 288M total model, 31M dense core in RAM (INT8), ~54 ms/token, ~18 tok/s CPU-only. With GPU acceleration on large matvecs: ~25+ tok/s estimated.

**Ground rules:**
- FLOPs per token for a transformer at batch=1: roughly `2 * P_dense` (one multiply + one add per weight)
- At batch=1, inference is memory-bandwidth-bound: every weight is read once per token
- CPU memory ceiling: ~1.9 GB/s (MEM2 via CPU bus)
- GPU memory ceiling: ~3.9 GB/s (MEM2 via GPU bus)

### CPU-only estimate

31M dense core at INT8 = ~62M FLOPs/token. At ~1.7 GFLOPS sustained: ~36 ms for dense core matmuls. Add ~14 ms for the output head (384 x 32K) and ~4 ms overhead (norms, softmax, activations, residuals, RoPE). Total: ~54 ms/token, ~18 tok/s.

### GPU TEV estimate

For the output head alone (384 x 32K): K=384 requires ~6 TEV passes. At ~1-2 ms/pass plus ~1-2 ms EFB copy: ~8-14 ms total. The CPU path takes ~14 ms for the same operation, so the GPU breaks even or wins here.

For per-layer matvecs (QKV [1152, 384], FFN gate/up [1024, 384], FFN down [384, 1024], attn proj [384, 384]): each needs ~6 passes at K=384 or ~16 passes at K=1024. GPU wins on the larger ones (FFN down), CPU may win on smaller ones (attn proj). The crossover point needs measurement (see Open Questions).

### Bandwidth sanity check

31M INT8 weights = 31 MB read per token. At 1.9 GB/s CPU bus: ~16 ms bandwidth ceiling = ~62 tok/s. At 3.9 GB/s GPU bus: ~8 ms bandwidth ceiling = ~125 tok/s. We are compute-bound, not bandwidth-bound: the ~54 ms compute time is well above the ~16 ms bandwidth floor.

### Reference comparisons

**Dreamcast:** 15M dense model at ~1 tok/s on 200MHz SH-4 (naive code, no SIMD). Our 31M dense model at ~18 tok/s on 729MHz PPC with paired singles. Accounting for 2x model size, the per-parameter throughput gap is ~36x: 3.6x clock, ~2x SIMD, plus optimized kernels. Aggressive but plausible.

**esp32-ai:** 559K dense core at 9.88 tok/s on ESP32-S3 (240MHz, no FP SIMD). Our 31M dense core is ~56x larger. At ~18 tok/s we are ~1.8x faster despite the larger model, thanks to 3x clock and paired-singles SIMD.

## Open Questions

1. **EFB additive blend precision:** Can multi-pass 8-bit additive accumulation produce usable results for a 384-element dot product? Per-pass partial sums must stay within [0, 255] range across all 6 passes. Needs validation in the GX simulator and empirically on hardware.

2. **EFB copy latency:** Real hardware timing for `GX_CopyTex` is not well documented. Only one copy per matvec (thanks to additive blend), but this is still the GPU path's fixed overhead. This is the make-or-break number for GPU utilization.

3. **GPU vs CPU crossover point:** At what matrix size does the GPU TEV path actually beat CPU paired-singles? Small matvecs (attn proj at [384, 384]) may lose to CPU due to setup overhead. Large ones (output head at [32K, 384]) should win. The crossover needs measurement.

4. **XF compute potential (resolved):** The XF is not viable as a general fp32 matmul accelerator: throughput is 6-17x slower than CPU paired-singles, there is no CPU readback of texcoord values (only EFB color/depth), and packing work into many vertices starves the TEV. However, texgen outputs enter the TEV as s17.7 fixed-point texcoords (24 usable bits), which makes the XF useful for high-precision LUT addressing in activation functions. See Stretch Goal 3.
