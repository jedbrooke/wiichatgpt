# Two-Stage TEV Multiply-Accumulate Pipeline

Uses the GX TEV combiner to multiply and accumulate texture-sourced vectors within the s10 register file.

## TEV Lerp as Integer Multiply

The TEV combiner's lerp formula:

```
lerp = (a * (255 - c) + b * c + 128) >> 8
```

Setting `a = ZERO` reduces this to a multiply:

```
lerp = (b * c + 128) >> 8
```

Each input is a **fint8**: an unsigned 8-bit value representing the range 0.0..1.0 (0 = 0.0, 255 = 1.0). The `>> 8` is the standard fixed-point Q8 multiply: `fint8 * fint8` produces a 16-bit intermediate, and shifting right by 8 extracts the high byte, yielding a fint8 result. The `+128` rounds to nearest instead of truncating. This is not a scale factor, just how fixed-point fractional multiply works.

## Two-Stage Configuration

Each draw call uses 2 textures and 2 TEV stages:

```
Stage 0 (load texture A):
  tex_map  = 0
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=TEXC)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = texA

Stage 1 (multiply by texture B, halve for headroom):
  tex_map  = 1
  color_in = (a=ZERO, b=CPREV, c=TEXC, d=ZERO)
  color_op = (op=ADD, bias=ZERO, scale=CS_DIVIDE_2, clamp=true, out_reg=TEVPREV)
  Result:  TEVPREV = ((texA * texB + 128) >> 8) / 2
```

Two stages are needed because each TEV stage performs only one texture fetch. Stage 0 parks texture A's value in TEVPREV; stage 1 fetches texture B and multiplies it against TEVPREV. For accumulating multiple products, see the Batched 4-Product variant below, which uses the s10 register file instead of EFB blending.

## Throughput

The TEV color combine processes R, G, B channels independently, and the alpha combine handles A. Each RGBA texture packs 4 independent values per texel.

One draw call (2 textures, 2 TEV stages, fullscreen quad):

```
640 * 528 * 4 channels = 1,351,680 multiply-accumulates
```

## Batched 4-Product Single Draw Call (10 TEV Stages)

Uses all 8 texture units to compute 4 texture-pair products and accumulate them in a single draw call, reducing draw call overhead. Only 2 TEV registers are needed (TEVPREV + TEVREG0), leaving 6 stages free.

The pattern alternates load-into-TEVREG0 and multiply-accumulate-into-TEVPREV. The `d` input passes through the adder exactly (no lerp involved), so accumulation introduces zero precision loss beyond each multiply's inherent ±0.5 LSB rounding.

```
Stage 0 (load tex0):
  tex_map  = 0
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=TEXC)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = tex0

Stage 1 (multiply tex0 * tex1):
  tex_map  = 1
  color_in = (a=ZERO, b=CPREV, c=TEXC, d=ZERO)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = (tex0 * tex1 + 128) >> 8

Stage 2 (load tex2):
  tex_map  = 2
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=TEXC)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVREG0)
  Result:  TEVREG0 = tex2

Stage 3 (multiply tex2 * tex3, accumulate):
  tex_map  = 3
  color_in = (a=ZERO, b=C0, c=TEXC, d=CPREV)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = TEVPREV + (tex2 * tex3 + 128) >> 8

Stage 4 (load tex4):
  tex_map  = 4
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=TEXC)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVREG0)
  Result:  TEVREG0 = tex4

Stage 5 (multiply tex4 * tex5, accumulate):
  tex_map  = 5
  color_in = (a=ZERO, b=C0, c=TEXC, d=CPREV)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = TEVPREV + (tex4 * tex5 + 128) >> 8

Stage 6 (load tex6):
  tex_map  = 6
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=TEXC)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVREG0)
  Result:  TEVREG0 = tex6

Stage 7 (multiply tex6 * tex7, accumulate):
  tex_map  = 7
  color_in = (a=ZERO, b=C0, c=TEXC, d=CPREV)
  color_op = (op=ADD, bias=ZERO, scale=CS_SCALE_1, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = TEVPREV + (tex6 * tex7 + 128) >> 8

Stage 8 (rescale /2):
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=CPREV)
  color_op = (op=ADD, bias=ZERO, scale=CS_DIVIDE_2, clamp=false, out_reg=TEVPREV)
  Result:  TEVPREV = TEVPREV / 2

Stage 9 (rescale /2, final clamp):
  color_in = (a=ZERO, b=ZERO, c=ZERO, d=CPREV)
  color_op = (op=ADD, bias=ZERO, scale=CS_DIVIDE_2, clamp=true, out_reg=TEVPREV)
  Result:  TEVPREV = clamp(TEVPREV / 2, 0, 255)
```

### Accumulator Trace (worst case, all inputs = 255)

Each multiply produces `(255 * 255 + 128) >> 8 = 254`.

| After stage | Operation | Max TEVPREV | Max TEVREG0 |
|---|---|---|---|
| 0 | load tex0 | 255 | - |
| 1 | tex0 * tex1 | 254 | - |
| 2 | load tex2 | 254 | 255 |
| 3 | +tex2 * tex3 | 508 | - |
| 4 | load tex4 | 508 | 255 |
| 5 | +tex4 * tex5 | 762 | - |
| 6 | load tex6 | 762 | 255 |
| 7 | +tex6 * tex7 | 1016 | - |
| 8 | /2 | 508 | - |
| 9 | /2, clamp | 254 | - |

Peak value 1016 stays within s10 range [-1024..1023]. Final output fits in u8 [0..254].

### Scale Factor

The lerp `>> 8` is inherent to fint8 fixed-point multiply (extracts the high byte of a Q8*Q8 product) and introduces no systematic scale, only ±0.5 LSB rounding per multiply. The two rescale stages apply /2 each (/4 total). Per-product scale: **/4**.

Each R, G, B, A channel of each texel is an independent scalar lane. Per channel per texel, the CPU recovers the true 4-term partial sum:

```
partial_sum = output_channel * 4
```

One draw call produces `640 * 528 * 4 = 1,351,680` independent 4-term partial sums (one per channel per texel).

## Per-Block Shared Exponent (future)

For dynamic range beyond u8, a block floating point scheme can be layered on top:

- Each texture block stores 8-bit mantissas with a shared per-block scale factor (held on CPU)
- TEV handles mantissa-domain multiply and accumulate
- After EFB readback (CopyTex), the CPU multiplies the partial sum by `scale_weight * scale_activation`
- Cross-block sums are combined by the CPU with proper exponent alignment

This keeps the O(n*m) multiply work on GX and limits the CPU to cheap O(blocks) scaling.