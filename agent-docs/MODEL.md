# WiiChatGPT Model Specification

## Architecture

Transformer decoder with Per-Layer Embeddings (PLE), based on the Gemma 3n pattern.

| Parameter | Value |
|---|---|
| hidden_dim | 384 |
| num_layers | 20 |
| num_heads | 6 (head_dim = 64) |
| FFN intermediate | 1024 (SwiGLU, ~2.67x hidden) |
| PLE dim | 384 (matches hidden_dim, no projection matrices) |
| Vocab | 32K BPE (SentencePiece) |
| Context length | 512 (configurable up to ~768) |
| Embedding/lm_head | Weight-tied |

## Parameter Count

| Component | Params | Location |
|---|---|---|
| Dense core (20 layers: attn + FFN + norms) | ~31M | MEM2 (RAM) |
| Embedding (32K x 384, tied with lm_head) | ~12.3M | MEM2 (RAM) |
| PLE table (32K x 384 x 20) | ~245M | SD card (or DVD) |
| **Total** | **~288M** | |

~2.3x GPT-2 (124M).

### Dense Core Per-Layer Breakdown

| Component | Params per layer |
|---|---|
| Q projection (384 x 384) | 147K |
| K projection (384 x 384) | 147K |
| V projection (384 x 384) | 147K |
| Output projection (384 x 384) | 147K |
| FFN gate (384 x 1024) | 393K |
| FFN up (384 x 1024) | 393K |
| FFN down (1024 x 384) | 393K |
| Layer norms (2 x 384) | 768 |
| **Per layer total** | **~1.77M** |
| **20 layers** | **~35.4M** |

Note: the ~31M figure for the dense core excludes the PLE projection matrices (which are not needed since PLE dim = hidden_dim). The ~35.4M figure includes all attention and FFN weights. The difference is rounding and the norms.

## Quantization

| Aspect | Format |
|---|---|
| All weights (disk and RAM) | INT8 |
| Compute | INT8 via `psq_l` type 6 (s8) + `ps_madd` (native Broadway hardware dequant) |
| Accumulation | fp32 (paired singles) |
| KV cache | INT8 |
| Activations | fp32 |

No INT4, no fp16. The Broadway CPU's GQR registers natively support INT8 load-and-dequant via the `psq_l` instruction. This is the fastest compute path available on the hardware.

## Per-Layer Embeddings (PLE)

### What It Is

Each decoder layer gets a per-token conditioning signal looked up from a large table on disk. Instead of the token identity being embedded once at the input and diluted through 20 layers, PLE refreshes it at every layer via a direct element-wise modulation of the residual stream.

### How It Works (per layer)

1. Read 384 INT8 values from the PLE table on disk, indexed by token ID
2. Element-wise multiply with the residual stream (384 values)
3. Add result to residual stream

No projection matrices are needed because PLE dim = hidden_dim. The PLE vector directly modulates the residual.

### Table Layout on Disk

`[token_id][layer_0_data (384 bytes) | layer_1_data | ... | layer_19_data]`

All 20 layers for a given token are contiguous: one seek, one 7,680 byte sequential read per token.

### Per-Token I/O Cost

- Read size: 20 layers x 384 bytes = 7.5 KB per token
- SD card at 25 MB/s: ~0.3 ms per token (negligible)
- Compute: element-wise multiply + add at 384 dimensions, 20 layers = ~15K ops (negligible)

### Why PLE Works Here

PLE lookups are O(1) per token regardless of table size. The table can be arbitrarily large without increasing per-token compute. Only the row for the current token is read. This makes PLE parameters extremely cheap at inference time compared to dense parameters (which require matrix multiplications). On the Wii, where fast RAM is scarce (64 MB) but slow storage is plentiful (SD card, DVD), PLE lets us have a 288M parameter model while only keeping ~43 MB of weights in RAM.

### References

- Google Gemma 3n: originated PLE, deployed at 5B total / 1.9B active params
- [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai): 28.9M model (25M PLE, 560K dense) at 9.88 tok/s on ESP32-S3 with 512KB SRAM + 8MB PSRAM + 16MB flash

## Memory Layout

### MEM2 (64 MB, GPU-accessible GDDR3)

| Component | Size |
|---|---|
| Dense core weights | ~31 MB |
| Embedding (tied lm_head) | ~12.3 MB |
| Weight texture buffers (double-buffered) | ~4 MB |
| GX overhead | ~2 MB |
| UI (framebuffer, assets) | ~8 MB |
| **Total** | **~57.3 MB** |
| **Free** | **~6.7 MB** |

### MEM1 (24 MB, CPU 1T-SRAM)

| Component | Size |
|---|---|
| KV cache (512 context) | ~7.5 MB |
| Activations buffer | ~3 MB |
| Code + stack | ~2 MB |
| GX FIFO + command buffer | ~1 MB |
| **Total** | **~13.5 MB** |
| **Free** | **~10.5 MB** |

The tokenizer (~6 MB SentencePiece runtime) is loaded temporarily into MEM1 free space during prompt encoding, then freed before generation starts.

### SD Card (Primary Target)

| Component | Size |
|---|---|
| PLE table | ~245 MB |
| Tokenizer model | ~2 MB |
| **Total** | **~247 MB** |

## Performance Estimates

### Per-Token Breakdown (CPU-only, paired singles, ~1.7 GFLOPS sustained)

| Stage | Time |
|---|---|
| Dense core matmuls (attn + FFN, 20 layers) | ~36 ms |
| Output head (384 x 32K) | ~14 ms |
| PLE I/O + compute | ~0.3 ms |
| Other (norms, softmax, activations, residuals, RoPE) | ~4 ms |
| **Total** | **~54 ms (~18 tok/s)** |

With GPU acceleration on large matvecs (output head, QKV, FFN): ~25+ tok/s estimated.

### KV Cache Scaling

Per token position: 20 layers x 2 (K+V) x 384 x 1 byte = 15 KB.

| Context length | KV cache | MEM1 free after |
|---|---|---|
| 256 | 3.8 MB | 14.2 MB |
| 512 (default) | 7.5 MB | 10.5 MB |
| 768 | 11.3 MB | 6.7 MB |

## DVD Target (Future)

Same model, no architecture or weight changes. Disk layout optimizations only:

- Dense core weights placed at inner track (loaded once at boot, ~3s sequential read at 12 MB/s)
- PLE table duplicated ~10x across disc zones (~2.45 GB total, well within the 4.7 GB DVD capacity)
- At runtime, the nearest copy is selected to minimize seek distance
- Reduces worst-case per-token DVD seek from ~100 ms (full disc) to ~8-12 ms (1/20th of disc)
- Prefetching during dense compute hides the remaining seek latency
- Result: compute-bound at ~18 tok/s, same as SD card

### Why Duplication Works

The PLE table is 245 MB. A Wii DVD holds 4.7 GB. With 10 copies spread evenly across the disc, the maximum distance the laser needs to travel to reach the nearest copy is ~1/20th of the disc radius. DVD seek time is roughly proportional to distance, so seek drops from ~100 ms to ~5-10 ms. Add rotational latency (~4-8 ms floor) and the total per-token DVD access is ~8-12 ms, which is less than the ~54 ms dense compute time and can be fully overlapped via prefetching.

## Training Notes

- Architecture follows the Gemma 3n PLE pattern, but with PLE dim = hidden_dim (no projection matrices)
- PLE injection: element-wise multiply with residual stream, then add
- All parameters (dense core, embeddings, PLE table) trained jointly via gradient descent
- Domain: Wii trivia, popular Wii games, Nintendo lore
- Export format: raw INT8 tensors with a simple header
- Weight tying: input embedding = lm_head (transposed)
- Tokenizer: 32K BPE trained with SentencePiece
