#!/usr/bin/env python3
"""
Read stories42M.bin (llama2.c FP32 format), quantize to INT8, write stories42M_int8.bin.

Output format:
  - 7 x int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
  - Per tensor: 1 x float32 (scale) + N x int8 (quantized values)

Tensor order matches llama2.c memory_map_weights:
  token_embedding_table, rms_att_weight, wq, wk, wv, wo,
  rms_ffn_weight, w1, w2, w3, rms_final_weight, [wcls if unshared]
"""

import struct
import sys
import os
import numpy as np

def read_config(f):
    header = f.read(7 * 4)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', header)
    return {
        'dim': dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
    }

def quantize_tensor(tensor):
    """Per-tensor absmax INT8 quantization."""
    amax = np.max(np.abs(tensor))
    if amax == 0:
        scale = 1.0
    else:
        scale = amax / 127.0
    quantized = np.round(tensor / scale).astype(np.int8)
    return scale, quantized

def main():
    input_path = os.path.join('reference', 'stories42M.bin')
    output_path = os.path.join('reference', 'stories42M_int8.bin')

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'rb') as f:
        cfg = read_config(f)

    # Negative vocab_size signals unshared weights in llama2.c format
    shared_weights = cfg['vocab_size'] > 0
    vocab_size = abs(cfg['vocab_size'])
    cfg['vocab_size'] = vocab_size

    dim = cfg['dim']
    hidden_dim = cfg['hidden_dim']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_kv_heads = cfg['n_kv_heads']
    seq_len = cfg['seq_len']
    head_size = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads

    print(f"Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, "
          f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}, "
          f"seq_len={seq_len}")
    print(f"head_size={head_size}, kv_dim={kv_dim}, shared_weights={shared_weights}")

    # Define tensor sizes in order
    tensor_defs = [
        ('token_embedding_table', vocab_size * dim),
        ('rms_att_weight', n_layers * dim),
        ('wq', n_layers * dim * (n_heads * head_size)),
        ('wk', n_layers * dim * kv_dim),
        ('wv', n_layers * dim * kv_dim),
        ('wo', n_layers * (n_heads * head_size) * dim),
        ('rms_ffn_weight', n_layers * dim),
        ('w1', n_layers * dim * hidden_dim),
        ('w2', n_layers * hidden_dim * dim),
        ('w3', n_layers * dim * hidden_dim),
        ('rms_final_weight', dim),
    ]

    # Skip freq_cis_real and freq_cis_imag
    skip_size = seq_len * head_size // 2  # freq_cis_real
    skip_size2 = seq_len * head_size // 2  # freq_cis_imag

    if not shared_weights:
        tensor_defs.append(('wcls', vocab_size * dim))

    # Read all FP32 data after the config header
    with open(input_path, 'rb') as f:
        f.seek(7 * 4)  # skip config
        remaining = f.read()

    fp32_data = np.frombuffer(remaining, dtype=np.float32)
    print(f"Total FP32 values: {len(fp32_data)}")

    # Write output
    with open(output_path, 'wb') as out:
        # Write config header (use original vocab_size sign convention: positive = shared)
        if shared_weights:
            out_vocab = vocab_size
        else:
            out_vocab = -vocab_size
        out.write(struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                              out_vocab, seq_len))

        offset = 0
        total_fp32_bytes = 0
        total_int8_bytes = 7 * 4  # header

        for name, size in tensor_defs:
            if name == 'wcls' and shared_weights:
                continue

            # Handle the skip for freq_cis between rms_final_weight and wcls
            if name == 'rms_final_weight':
                tensor = fp32_data[offset:offset + size]
                offset += size
                # Now skip freq_cis_real and freq_cis_imag
                offset += skip_size + skip_size2
            else:
                tensor = fp32_data[offset:offset + size]
                offset += size

            if len(tensor) != size:
                print(f"Error: expected {size} values for {name}, got {len(tensor)}", file=sys.stderr)
                sys.exit(1)

            scale, quantized = quantize_tensor(tensor)

            # Write scale (float32) + quantized data (int8)
            out.write(struct.pack('f', scale))
            out.write(quantized.tobytes())

            fp32_bytes = size * 4
            int8_bytes = 4 + size  # scale + data
            total_fp32_bytes += fp32_bytes
            total_int8_bytes += int8_bytes

            # Compute quantization error
            reconstructed = quantized.astype(np.float32) * scale
            max_err = np.max(np.abs(tensor - reconstructed))
            mean_err = np.mean(np.abs(tensor - reconstructed))
            print(f"  {name:30s}: {size:>10d} params, scale={scale:.6f}, "
                  f"max_err={max_err:.6f}, mean_err={mean_err:.6f}")

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print(f"\nInput:  {input_path} ({input_size:,} bytes, {input_size/1024/1024:.1f} MB)")
    print(f"Output: {output_path} ({output_size:,} bytes, {output_size/1024/1024:.1f} MB)")
    print(f"Compression: {input_size/output_size:.2f}x")

if __name__ == '__main__':
    main()