# Konst-Weight TEV Multiply-Accumulate Pipeline

Uses CPU-set constant weight values (konst registers, TEV registers, vertex color) broadcast to all pixels, multiplied against per-pixel activations from textures. Requires only 1 TEV stage per MAC (no load stage needed), unlike the two-stage texture*texture path in TWO_STAGE_TEV_MAC.md.

## Weight Sources (constant across all pixels)

| Source | Registers | Channels | Values |
|---|---|---|---|
| Konst K0..K3 | 4 | RGBA each | 16 u8 |
| TEVREG0..2 | 3 | RGBA each | 12 s10 |
| Vertex color | 1 | RGBA | 4 u8 |
| **Total** | | | **32** |

## Activation Source (per-channel)

8 textures, RGBA each = 8 u8 values per channel.

## Pipeline (10 TEV stages, 8 MACs per draw call)

```
Stages 0-3: konst * texture
  Stage 0: d=CPREV, a=ZERO, b=KONST(K0), c=TEXC(tex0), op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 1: d=CPREV, a=ZERO, b=KONST(K1), c=TEXC(tex1), op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 2: d=CPREV, a=ZERO, b=KONST(K2), c=TEXC(tex2), op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 3: d=CPREV, a=ZERO, b=KONST(K3), c=TEXC(tex3), op=ADD, scale=CS_DIVIDE_2, clamp=false

Stages 4-6: TEVREG * texture
  Stage 4: d=CPREV, a=ZERO, b=C0, c=TEXC(tex4), op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 5: d=CPREV, a=ZERO, b=C1, c=TEXC(tex5), op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 6: d=CPREV, a=ZERO, b=C2, c=TEXC(tex6), op=ADD, scale=CS_DIVIDE_2, clamp=false

Stage 7: vertex color * texture
  Stage 7: d=CPREV, a=ZERO, b=RASC, c=TEXC(tex7), op=ADD, scale=CS_DIVIDE_2, clamp=false

Stages 8-9: rescale accumulator to u8 range
  Stage 8: d=CPREV, a=b=c=ZERO, op=ADD, scale=CS_DIVIDE_2, clamp=false
  Stage 9: d=CPREV, a=b=c=ZERO, op=ADD, scale=CS_DIVIDE_2, clamp=true
```

## Accumulator Trace (worst case, all inputs = 255)

| After stage | Operation | Max TEVPREV per channel |
|---|---|---|
| 0 | +127 | 127 |
| 1 | +127 | 254 |
| 2 | +127 | 381 |
| 3 | +127 | 508 |
| 4 | +127 | 635 |
| 5 | +127 | 762 |
| 6 | +127 | 889 |
| 7 | +127 | 1016 |
| 8 | /2 | 508 |
| 9 | /2, clamp | 254 |

Stays within s10 range (-1024..1023) at all times. Final output fits in u8 (0..254).

## Scale Factor

Each MAC stage applies CS_DIVIDE_2 to the product: /2. Two rescale stages apply /2 each: /4. Total per product: **/8**.

## CPU Readback

Read the EFB pixel(s). Multiply each channel by 8 to recover the true partial sum. Sum the 4 RGBA channels for the full 32-element dot product:

```
dot_product = 8 * (R + G + B + A)
```

## Per-Channel View

Each color channel independently computes an 8-element dot product:

```
R_out = (w0.r * a0.r + w1.r * a1.r + ... + w7.r * a7.r) / 8
```

where w0..w7 are the 8 weight sources (K0, K1, K2, K3, TEVREG0, TEVREG1, TEVREG2, ras_color) and a0..a7 are the 8 texture activations.

RGBA gives 4 parallel 8-element dot products per pixel.

## Precision

| Quantity | Value |
|---|---|
| Product range per term (after /2) | 0..127 |
| Effective bits per term | ~6 |
| Max accumulated result (8 terms, 1 channel) | 1016 (before rescale) |
| Final u8 output range | 0..254 |
| CPU-recovered range (per channel, *8) | 0..2032 |
| CPU-recovered range (4 channels summed, *8) | 0..8128 |

## Comparison with Two-Stage Texture*Texture Path

| Aspect | Konst-weight path | Two-texture path |
|---|---|---|
| TEV stages per MAC | 1 | 2 |
| MACs per draw call per channel | 8 | 1 |
| Weight source | CPU constants (same for all pixels) | Texture (per-pixel) |
| Use case | Small weight vector broadcast across pixels | Large weight matrices, per-pixel weights |