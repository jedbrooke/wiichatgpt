# WiiChatGPT Design Document

## Hardware Budget

| Resource | Spec | Notes |
|---|---|---|
| CPU | Broadway PowerPC 750CL, 729 MHz | 2x integer ALU, 1x FPU with paired singles |
| CPU FP peak | ~2.9 GFLOPS | via `ps_madd` (2 FMACs/cycle) |
| CPU INT peak | ~729M multiply/sec | single IU for `mullw` |
| GPU | Hollywood (ATI), 243 MHz | Fixed-function TEV pipeline, 4 pixel pipes |
| GPU TEV peak | ~15.6 GOPS (8-bit) | 243 MHz x 4 pixels/clock x 16 stages |
| MEM1 | 24 MB 1T-SRAM | CPU main memory, ~1.9 GB/s |
| MEM2 | 64 MB GDDR3 | GPU-accessible, ~3.9 GB/s |
| GPU embedded | 3 MB on-die | EFB, internal bandwidth ~30 GB/s |
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
+------+  +------------------+
| CPU  |  | GPU (TEV)        |
| 750CL|  | 16 stages        |
| 729M |  | 4 pixels/clock   |
+------+  +------------------+
  |          |
  | activations, reductions,
  | PLE lookup, setup
  |          |
  |   tiled matmul
  |   (attn proj, FFN)
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

## Compute Strategy

### GPU: Tiled TEV Matmul

**Theoretical peak:** 243 MHz x 4 pixels/clock x 16 TEV stages x 4 RGBA channels = **~62 billion 8-bit multiply-accumulate ops/sec** (counting 4 channels per pixel per stage).

More conservatively, counting useful MAC ops: **~15.6 GOPS** at 8-bit precision.

**TEV combiner formula per stage:**
```
out = (d +/- ((1-c)*a + c*b) + bias) * scale
```

With `a=ZERO, b=KONST, c=RASC, d=accumulator`, this computes `RASC * KONST + accumulator` per channel, a fused multiply-add.

**Tiled approach:**
1. Pack weight matrix tiles as GX textures (RGBA8 = 4 values per texel)
2. Activations encoded as vertex colors or a second texture
3. Render a large quad covering many output pixels (EFB is 640x528 = 337K pixels)
4. Each pixel computes one output element's partial dot product
5. Chain 16 TEV stages for 16 x 4 = 64 MACs per pixel per pass
6. Multi-pass with `GX_CopyTex` for dot products longer than 64 elements
7. Read results back via batch `GX_CopyTex` (not per-pixel `GX_PeekARGB`)

**Per-pass throughput at theoretical peak:**
- 337K pixels x 64 MACs/pixel = ~21.6M MACs per pass
- Full-screen render at 4 pixels/clock: 337K / (243M x 4) = ~0.35 ms
- With setup + EFB copy overhead, estimate ~2-3 ms per pass
- Effective throughput: **~7-21 GOPS**

**Tile size considerations:**
- EFB: 640 x 528 pixels, each producing one output element
- With 64 MACs/pixel/pass, a K=512 dot product needs ~8 passes
- Output matrix up to ~580 x 580 per draw call

### CPU: Paired Singles + GQR Dequant

**Peak:** ~2.9 GFLOPS (729 MHz x 2 lanes x 2 ops/FMAC)

**The GQR hardware dequant path:**
- `psq_l frD, addr, W, GQR_index` loads 2x INT8 values from memory and **automatically converts them to fp32 pairs** in a single instruction
- GQR Type field `6` = signed 8-bit (s8), with a configurable power-of-2 scale factor
- Combined with `ps_madd` for the multiply-accumulate

This means: weights stay as INT8 in memory (half the bandwidth vs fp32), but computation happens in fp32 via hardware dequant-on-load. This is the same pattern as llama.cpp's Q8_0 format.

**CPU handles:**
- Activation functions (SiLU, softmax, RMSNorm): elementwise, no TEV benefit
- Partial sum reduction from multi-pass TEV results
- PLE table lookups from SD card
- GX draw call setup, texture uploads
- KV cache management

### Combined Throughput

Assuming the GPU handles the big matmuls (attention projections, FFN) and the CPU handles everything else in parallel:

| Component | Theoretical | Notes |
|---|---|---|
| GPU TEV matmul | ~15.6 GOPS (8-bit) | Tiled, batched, multi-pass |
| CPU paired singles | ~2.9 GFLOPS (fp32) | Activations, reductions |
| PLE disk I/O | ~1-2 ms/token | Negligible |
| Combined | ~18+ GOPS effective | If CPU/GPU overlap well |

### Path to Full GPU Utilization

The gap between theoretical (~15.6 GOPS) and realistic throughput comes from:

1. **EFB copy latency.** Each multi-pass round trip requires a `GX_CopyTex` blit. Measured latency on real hardware is needed, but estimated at ~1-2ms per copy. Mitigation: maximize MACs per pass to amortize copy cost (use all 16 stages, full EFB coverage).

2. **Draw call setup overhead.** Each TEV pass requires configuring stages, loading textures, issuing vertex data. The GX FIFO command stream adds latency. Mitigation: minimize the number of passes by packing more work per pass; pre-build command lists where possible.

3. **8-bit precision accumulation.** TEV operates on 8-bit unsigned / signed 10-bit fixed-point. A 512-element dot product accumulated in 8-bit space will overflow. Mitigation: multi-pass with CPU-side fp32 accumulation of per-pass partial sums. The TEV computes 64-element partial dot products (fitting in 8-bit), CPU accumulates them.

4. **Texture upload bandwidth.** Weight tiles must be uploaded as GX textures before each pass. At ~3.9 GB/s MEM2 bandwidth, a 256x256 RGBA8 tile (256KB) takes ~0.065ms to upload, but `GX_LoadTexObj` has CPU overhead. Mitigation: double-buffer textures, upload while GPU is rendering the previous tile.

5. **Quantization error.** The TEV's 8-bit fixed-point introduce quantization noise. For INT4 weights dequantized to 8-bit TEV space, this may be acceptable. Needs empirical validation via the GX simulator.

**Validation milestones:**
1. Tiled matmul in the GX simulator (correctness)
2. Benchmark on Dolphin emulator with GPU timing
3. Benchmark on real Wii hardware
4. Compare GPU matmul throughput vs CPU paired-singles baseline
5. Profile EFB copy and draw call overhead to find the real utilization ceiling

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
4. Add TEV-accelerated tiled matmul backend for large matmuls
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

GPU handles the big matmuls (attention projections, FFN), CPU handles activations/reductions in parallel.

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

1. **TEV matmul precision:** Can 8-bit TEV math with multi-pass CPU accumulation match INT8 inference quality? Needs simulation and empirical testing.
2. **EFB copy latency:** Real hardware timing for `GX_CopyTex` is not well documented. This is the make-or-break number for GPU utilization.
3. **Model architecture details:** Exact layer count, hidden dim, head count, PLE dim, for a ~1B model that fits the memory budget. Jake needs these constraints for training.
4. **DVD random read performance:** PLE lookups are random-access by token ID. DVD seek latency (~100ms) may make DVD impractical for PLE; SD card (low-ms seek) is likely required. DVD may still work for sequential weight loading at boot.
5. **Context length:** KV cache size scales linearly with context. At 24 layers x 512 hidden x 2 (K+V) x fp32, each token position costs ~96KB. A 256-token context = ~24MB of KV cache, consuming most of MEM1. May need INT8 KV cache or shorter context.
