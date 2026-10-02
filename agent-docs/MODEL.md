# WiiChatGPT Model Specification

## Architecture

Bi-Mamba-2 (binary SSM) with Per-Layer Embeddings (PLE). The architecture is hardware-driven: the model's hidden dimension and state dimensions are chosen to exactly fill the Wii GPU's Embedded Frame Buffer (EFB), enabling 100% GPU utilization during the recurrent state update.

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | Mamba-2 (SSM) | O(1) per-token inference, no KV cache, recurrent state maps to EFB |
| hidden_dim (H) | 1920 | = 640 EFB pixels x 3 RGB channels (EFB row width) |
| state_dim | 528 | = 528 EFB rows (EFB column height) |
| num_heads | 1 | head_dim = H = 1920, single head fills entire EFB |
| num_layers | 4 | Fits RAM budget with V=128K |
| FFN intermediate | 5120 (SwiGLU, ~2.67x H) | |
| Vocab | 128K BPE (EN + JA + KO + ZH) | CJK support for the Wii community |
| Context length | Unlimited | Recurrent state, no KV cache growth |
| Embedding/lm_head | Weight-tied | |
| PLE dim | 1920 (= H, no projection) | |

### Why Mamba-2 (not Transformer)

Standard transformers at batch=1 produce small matmul output dimensions (M = hidden_dim or FFN_dim, a few thousand elements), filling less than 1% of the EFB's 337K pixels. The output head (M = vocab_size) fills ~13%, but per-layer matmuls waste >99% of the GPU surface.

Mamba-2's recurrent state update is fundamentally different: it produces state_dim x head_dim output elements per layer, which is a 2D grid (not a 1D vector). With state_dim=528 and head_dim=1920, the state is 1,013,760 elements, exactly matching the EFB's 640 x 528 x 3 = 1,013,760 accumulators.

### Why Binary Weights

At INT8 (1 byte per weight), H=1920 with 4 layers requires ~53 MB for dense weights alone, leaving no room for the embedding table. At binary (1 bit per weight), the same architecture uses ~11 MB for dense weights, freeing ~31 MB for a 128K-entry embedding.

Binary weights reduce per-weight information capacity, but the model compensates via:
- More total parameters (1.27B vs 288M at the same RAM budget)
- Large PLE table on disk (983M params, INT8)
- Wide hidden dimension (1920 vs 384)

References: [Bi-Mamba (arXiv:2411.11843)](https://arxiv.org/abs/2411.11843) demonstrates 1-bit Mamba trained from scratch at 780M-2.7B scale, matching full-precision Mamba on language modeling benchmarks.

### Why 1 Head

For SSMs, total EFB utilization = `num_heads * state_dim * head_dim / 3 pixels`. With a fixed H, fewer heads means larger head_dim, which means more state elements per head, which means more EFB utilization. At 1 head with head_dim=1920, the state fills 100% of the EFB. With 2 heads, it fills 51.7%.

Research (Ghodsi 2025, "How Many Heads Make an SSM?") shows that more heads increase expressivity via independent recurrent channels, but at head_dim=1920 the single head has enormous capacity (1.01M state elements). If quality is insufficient, 2 heads (head_dim=960, state_dim=528, 51.7% EFB) is a viable fallback.

## EFB-Native State Mapping

The recurrent state maps to the EFB via identity:

```
State[528][1920]  ->  EFB[528][640].rgb

State[i][j]       ->  EFB[row=i][col=j/3].channel[j%3]
                      where channel 0=R, 1=G, 2=B
```

- Each state row IS an EFB row (528 rows)
- Each state element IS one color channel of one pixel (1920 = 640 * 3)
- Zero waste, no packing tricks
- The GPU renders the state update at full EFB resolution: 337,920 pixels, all utilized

## Parameter Count

| Component | Params | Precision | Location |
|---|---|---|---|
| Dense core (4 layers: SSM projections + FFN) | ~44.4M | Binary (1-bit) | MEM2 (RAM) |
| Embedding (128K x 1920, tied with lm_head) | ~245.8M | Binary (1-bit) | MEM2 (RAM) |
| SSM recurrence params (A, B, C, D) | ~4.1M | INT8 | MEM2 (RAM) |
| PLE table (128K x 1920 x 4 layers) | ~983M | INT8 | SD card |
| **Total** | **~1.27B** | | |

### Dense Core Per-Layer Breakdown

Mamba-2 per-layer projections (all binary):

| Component | Params per layer |
|---|---|
| Input projection (H -> 2H) | 1920 x 3840 = 7.37M |
| Output projection (2H -> H) | 3840 x 1920 = 7.37M |
| Conv1d (kernel=4, channels=2H) | 4 x 3840 = 15K |
| SSM params A (diagonal, per state_dim) | 528 |
| SSM params B, C (state_dim x H each) | 2 x 528 x 1920 = 2.03M |
| SSM param D (diagonal) | 1920 |
| Layer norm | 2 x 1920 = 3840 |
| **Per layer total** | **~16.8M** |
| **4 layers** | **~67.3M** |

Note: the ~44.4M binary dense figure above uses the simplified `6 * H^2` estimate. The detailed breakdown gives ~67M, the difference being the expansion factor (Mamba-2 projects to 2H internally, not H). The precise sizing should be validated against the Bi-Mamba reference implementation during training.

### FFN Per-Layer Breakdown (binary)

| Component | Params per layer |
|---|---|
| FFN gate (1920 x 5120) | 9.83M |
| FFN up (1920 x 5120) | 9.83M |
| FFN down (5120 x 1920) | 9.83M |
| **FFN total per layer** | **~29.5M** |
| **4 layers** | **~118M** |

## Quantization

| Aspect | Format |
|---|---|
| Dense projections (linear layers) | Binary (1-bit, packed 8 per byte) |
| Embedding / lm_head | Binary (1-bit, packed 8 per byte) |
| SSM recurrence params (A, B, C, D) | INT8 (recurrence is precision-sensitive) |
| Norms | INT8 |
| PLE table (on disk) | INT8 |
| Activations | fp32 (CPU), u8 (GPU TEV path) |
| Recurrent state | INT8 |

Binary projections reduce matmuls to XNOR + popcount (no floating-point multiply). On the GPU, the TEV's logic blend mode (`BM_LOGIC`, `EQUIV` = XNOR) computes this natively. On the CPU, Broadway's bitwise ops + `popcntb` handle the equivalent.

The SSM recurrence (`h_t = A * h_{t-1} + B * x_t`) must remain in higher precision because errors accumulate through the sequential state updates. This matches the Bi-Mamba paper's approach (binary projections, float32 recurrence).

## Per-Layer Embeddings (PLE)

### What It Is

Each SSM layer gets a per-token conditioning signal looked up from a table on disk. PLE refreshes the token identity at every layer, compensating for the shallow depth (4 layers) and binary weight precision.

### How It Works (per layer)

1. Look up 1920 INT8 values from the PLE table on disk, indexed by token ID
2. Element-wise multiply with the hidden state (1920 values)
3. Add result to hidden state

No projection matrices are needed because PLE dim = H = 1920.

### Table Layout on Disk

`[token_id][layer_0_data (1920 bytes) | layer_1_data | layer_2_data | layer_3_data]`

All 4 layers for a given token are contiguous: one seek, one 7,680-byte sequential read per token.

### Per-Token I/O Cost

- Read size: 4 layers x 1920 bytes = 7,680 bytes per token
- SD card at 25 MB/s: ~0.3 ms per token (negligible)
- Compute: element-wise multiply + add at 1920 dimensions, 4 layers = ~15K ops (negligible)

### Why PLE Is Critical Here

With only 4 binary layers, the dense core has limited representational capacity per layer. PLE provides a large, high-precision (INT8) conditioning signal that compensates: 983M PLE params vs ~44M binary dense params. The PLE table is 22x larger than the dense core and carries 8x more information per parameter (INT8 vs binary).

### References

- Google Gemma 3n: originated PLE for transformers
- [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai): PLE on ESP32, 28.9M model at 9.88 tok/s
- [Bi-Mamba (arXiv:2411.11843)](https://arxiv.org/abs/2411.11843): binary Mamba training methodology

## Memory Layout

### MEM2 (64 MB, GPU-accessible GDDR3)

| Component | Size |
|---|---|
| Dense projections (binary, packed) | ~11.1 MB |
| Embedding (binary, packed) | ~30.7 MB |
| SSM recurrence params (INT8) | ~4.1 MB |
| Weight texture buffers | ~4 MB |
| GX overhead | ~2 MB |
| UI (framebuffer, assets) | ~2 MB |
| **Total** | **~54.0 MB** |
| **Free** | **~10.0 MB** |

### MEM1 (24 MB, CPU 1T-SRAM)

| Component | Size |
|---|---|
| Recurrent state (4 layers x 528 x 1920, INT8) | ~4.1 MB |
| Activations buffer | ~4 MB |
| Code + stack | ~2 MB |
| GX FIFO + command buffer | ~1 MB |
| **Total** | **~11.1 MB** |
| **Free** | **~12.9 MB** |

No KV cache. The recurrent state is fixed at 4.1 MB regardless of context length.

The tokenizer is loaded temporarily into MEM1 free space during prompt encoding, then freed before generation starts.

### SD Card

| Component | Size |
|---|---|
| PLE table | ~983 MB |
| Tokenizer model | ~4 MB |
| **Total** | **~987 MB** |

Fits on a 1 GB SD card. For a 2 GB card, the PLE table can be duplicated for redundancy.

## Performance Estimates

### Per-Token Breakdown

| Operation | Engine | Time |
|---|---|---|
| State update (4 layers) | GPU (TEV, full EFB) | ~3.4 ms |
| FFN (4 layers) | CPU (binary XNOR+popcount) | ~14.4 ms |
| Output head [128K, 1920] | GPU (TEV, 43K pixels) | ~19 ms |
| SSM recurrence (4 layers) | CPU (INT8) | ~1 ms |
| PLE I/O + combine | CPU + SD | ~0.3 ms |
| Norms, gating, overhead | CPU | ~1 ms |
| **Total** | | **~39 ms (~26 tok/s)** |

Target: 10 tok/s (real-time), 1 tok/s (bare minimum). At 26 tok/s, the architecture has 2.6x headroom above real-time.

### Bandwidth Check

Total weight data read per token: ~11.1 MB (dense) + ~30.7 MB (embedding rows accessed) + ~7.7 KB (PLE). Most of this is sequential texture reads from MEM2 at ~3.9 GB/s, taking ~11 ms. Compute time (39 ms) exceeds bandwidth time; the model is compute-bound.

## Fallback Configuration

If 4 layers proves too shallow for acceptable quality, a deeper variant trades vocab size for depth:

| | Primary (V=128K, L=4) | Fallback (V=32K, L=8) |
|---|---|---|
| Vocab | 128K (EN + CJK) | 32K (EN only) |
| Layers | 4 | 8 |
| Dense params (binary) | ~44M | ~89M |
| Embed params (binary) | ~246M | ~61M |
| PLE on disk | ~983M | ~491M |
| Total params | ~1.27B | ~641M |
| RAM used | ~54 MB | ~50 MB |
| tok/s | ~26 | ~17 |
| EFB state fill | 100% | 100% |

The fallback is deeper (better per-layer quality) but has fewer total params and no CJK support. Both variants are above the 10 tok/s real-time target.

## DVD Target (Future)

Same model, no architecture or weight changes. Disk layout optimizations only:

- Dense core weights and embedding loaded once at boot (~42 MB at 12 MB/s = ~3.5s)
- PLE table duplicated across disc zones (~4x copies, ~3.9 GB, within 4.7 GB DVD)
- Nearest copy selected at runtime to minimize seek distance
- Per-token DVD access: ~8-12 ms (overlapped with compute)
- Result: compute-bound at ~26 tok/s, same as SD card

## Training Notes

- Binary-aware training following Bi-Mamba methodology (straight-through estimator for binary gradients, autoregressive distillation loss)
- PLE injection: element-wise multiply with hidden state, then add (same as Gemma 3n pattern, adapted for SSM)
- All parameters (binary dense, binary embedding, INT8 SSM, INT8 PLE) trained jointly
- Domain: Wii trivia, popular Wii games, Nintendo lore, general conversational (EN, JA, KO, ZH)
- Export format: packed binary weights (8 weights per byte) + INT8 tensors with a simple header
- Weight tying: input embedding = lm_head (transposed, both binary)
- Tokenizer: 128K BPE trained with SentencePiece on multilingual corpus (EN + JA + KO + ZH)
- Training reference: [Bi-Mamba code (github.com/Tangshengku/Bi-Mamba)](https://github.com/Tangshengku/Bi-Mamba)
- Inference reference: [BitMamba.c (github.com/jserv/bitmamba.c)](https://github.com/jserv/bitmamba.c) (portable C Mamba inference)