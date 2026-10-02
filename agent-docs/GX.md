Here's the full Wii GX pipeline as pseudocode, sourced from the Nintendo SDK GX reference, YAGCD, and Dolphin's hardware-accurate implementation. Data shapes and register/buffer names are annotated throughout.

```c
============================================================================
 WII GX GRAPHICS PIPELINE - COMPLETE PSEUDOCODE
 Sources: Nintendo GX SDK ref, YAGCD ch5/ch8, Dolphin PixelShaderGen.cpp
============================================================================

GLOBAL STATE (configured by CPU via GX FIFO commands before each draw call):
  # XF register bank (fp32)
  pos_nrm_matrices[10]    : f32[3][4]     # 4x3 modelview matrices (10 slots)
  current_pnmtx           : u8            # 0..9, set by GX_SetCurrentMtx, selects active pos/nrm matrix
  tex_matrices[10]        : f32[3][4]     # 4x3 texture matrices (10 slots)
  projection.values        : f32[6]       # 6 non-zero elements of a sparse 4x4 matrix (XF regs 0x1020-0x1025)
  projection.mode          : u8           # 0=PERSPECTIVE, 1=ORTHO (XF reg 0x1026)
  viewport                : f32[6]        # screen mapping (x0, y0, z_range, x1, y1, far)

  # XF texgen config (per texgen channel, up to 8)
  num_texgens             : u8            # 0..8, set by GX_SetNumTexGens
  texgen[8].src           : enum          # POS, NRM, TEX0..TEX7, BINRM, TANGENT
  texgen[8].type          : enum          # MTX2x4, MTX3x4, BUMP, SRTG
  texgen[8].mtx           : index         # which tex_matrix slot (or IDENTITY)

  # BP register bank (per-TEV-stage config)
  num_tev_stages          : u8            # 1..16
  tev_stage[16].color_in  : (a,b,c,d)    # 4 input selectors, each from ~16 sources
  tev_stage[16].alpha_in  : (a,b,c,d)    # 4 input selectors, each from ~8 sources
  tev_stage[16].color_op  : (op, bias, scale, clamp, out_reg)
  tev_stage[16].alpha_op  : (op, bias, scale, clamp, out_reg)
  tev_stage[16].tex_map   : u8            # which of 8 texture map slots
  tev_stage[16].tex_coord : u8            # which of 8 texcoord channels
  tev_stage[16].ras_chan  : u8            # which rasterized color channel
  tev_stage[16].kcsel     : u8            # konst color selector
  tev_stage[16].kasel     : u8            # konst alpha selector
  tev_stage[16].ras_swap  : u8            # swap table index for raster color
  tev_stage[16].tex_swap  : u8            # swap table index for texture color
  tev_stage[16].ind_*     : ...           # indirect texture config (stage, matrix, format, etc.)

  # TEV register file (persistent across stages within a fragment, reset per draw setup)
  tev_regs[4]             : s10[4]        # TEVPREV, TEVREG0, TEVREG1, TEVREG2
                                          # each is (R, G, B, A) in [-1024..1023]

  # Konst colors (loaded by CPU)
  konst_colors[4]         : u8[4]         # K0..K3, each RGBA

  # Swap tables
  swap_tables[4]          : u8[4]         # each is a permutation of {R=0, G=1, B=2, A=3}

  # Texture state
  textures[8].data        : u8*           # pointer to texture data in MEM2
  textures[8].width       : u16
  textures[8].height      : u16
  textures[8].format      : enum          # I8, IA8, RGB565, RGBA8, etc.
  textures[8].wrap_s/t    : enum          # CLAMP, REPEAT, MIRROR

  # Indirect texture state
  ind_stages[4].tex_coord : u8
  ind_stages[4].tex_map   : u8
  ind_stages[4].scale_s/t : u8
  ind_matrices[3]         : f32[2][3]     # 2x3 offset matrix
  ind_matrices[3].scale   : s8            # power-of-2 exponent

  # Pixel Engine state
  blend.mode              : enum          # NONE, BLEND, SUBTRACT, LOGIC
  blend.src_factor        : enum          # ZERO, ONE, SRCALPHA, INVSRCALPHA, ...
  blend.dst_factor        : enum          # ZERO, ONE, SRCALPHA, INVSRCALPHA, ...
  blend.logic_op          : enum          # AND, OR, XOR, COPY, ...
  alpha_compare           : (func0, ref0, op, func1, ref1)
  z_mode                  : (enable, func, write_enable)
  color_update_enable     : bool
  alpha_update_enable     : bool

  # EFB (on-die eDRAM, 2 MB)
  efb[528][640]           : { u8 r, g, b, a; u24 z; }


=====================================
 STAGE 1: COMMAND PROCESSOR (CP)
=====================================
# CPU has written vertex data + state commands into the GX FIFO.
# CP reads the FIFO and decodes vertex data according to the vertex descriptor.

input:
  fifo_stream             : byte stream from CPU
  vtx_desc                : which attributes are present (pos, nrm, clr0, tex0..7)
  vtx_attr_fmt            : data types and component counts per attribute

output (N = vertex count from GX_Begin, fields present depend on vtx_desc):
  positions                : f32[N][3]    # (x, y, z) per vertex
  normals                  : f32[N][3]    # (nx, ny, nz) per vertex
  colors                   : u8[N][4]     # (r, g, b, a) per vertex
  texcoords[i]             : f32[N][2]    # (s, t) per vertex, per active texcoord slot i


=====================================
 STAGE 2: TRANSFORM UNIT (XF)
=====================================
# All math here is fp32. Operations are batched over all N vertices.
# The XF hardware appends w=1.0 to position/normal vectors internally
# before matmul; vertex data in memory is always f32[3] (or f32[2] for texcoords).

input:
  positions                : f32[N][3]    # from CP
  normals                  : f32[N][3]    # from CP (if present)
  colors                   : u8[N][4]     # from CP (if present)
  texcoords[i]             : f32[N][2]    # from CP, per active texcoord slot
  pos_nrm_matrices[]       : f32[3][4]
  tex_matrices[]           : f32[3][4]
  texgen[] config
  projection               : f32[6]
  viewport                 : f32[6]

output:
  screen_positions         : f32[N][3]    # (px, py, z) in screen space
  lit_colors               : u8[N][4]     # post-lighting vertex colors
  gen_tc[i]                : f32[N][R]    # texgen outputs per active channel (R=2 or 3)

# --- Position transform (all N vertices) ---
mtx = pos_nrm_matrices[current_pnmtx]              # f32[3][4]
# XF appends w=1 to each 3-component position internally:
eye_positions = mtx @ [positions, 1].T              # f32[3][4] @ f32[4][N] -> f32[3][N]

# --- Projection (eye space -> clip space, all N vertices) ---
# Hardware stores 6 non-zero elements of a sparse 4x4 projection matrix.
# p = projection.values[0..5]. Reconstructed full matrix by mode:
#
#   PERSPECTIVE:                          ORTHO:
#   | p[0]  0     p[1]  0    |           | p[0]  0     0     p[1] |
#   | 0     p[2]  p[3]  0    |           | 0     p[2]  0     p[3] |
#   | 0     0     p[4]  p[5] |           | 0     0     p[4]  p[5] |
#   | 0     0     -1    0    |           | 0     0     0     1    |
#
P = reconstruct_proj_matrix(projection)             # f32[4][4]
clip = P @ [eye_positions; 1].T                     # f32[4][4] @ f32[4][N] -> f32[4][N]
# clip[0..2] = clip xyz, clip[3] = clip w
# (position transform and projection can be composed: P @ [mtx; 0 0 0 1])

# --- Perspective divide (clip -> NDC) ---
ndc = clip[0:3] / clip[3]                          # f32[3][N], elementwise divide by w

# --- Viewport transform (NDC -> screen pixels) ---
# vp registers: x0=half_width, y0=-half_height, x1=center_x+342,
#               y1=center_y+342, fp=far*16M, z=range*16M
screen_positions[:, 0] = ndc[0] * vp.x0 + vp.x1   # f32[N]
screen_positions[:, 1] = ndc[1] * vp.y0 + vp.y1   # f32[N]
screen_positions[:, 2] = ndc[2] * vp.z  + vp.fp   # f32[N], 24-bit Z range

# --- Lighting (per color channel, all N vertices) ---
# Two independent channels (COLOR0, COLOR1), each with separate config.
# Color and alpha computed separately. Per-vertex, not per-pixel.
# Per-channel config: mat_source, amb_source, diffuse_func, attn_func, light_mask.

if not lighting_enabled:
    lit_colors = colors                              # u8[N][4], pass through
else:
    mat = colors if mat_source == VERTEX else material_register  # u8[N][4]
    amb = colors if amb_source == VERTEX else ambient_register   # u8[N][4]
    lacc = amb                                       # s32[N][4], accumulator starts at ambient

    for each light j where light_mask bit j is set:
        L = lights[j]                                # {color: u8[4], cosatt: f32[3],
                                                     #  distatt: f32[3], pos: f32[3], dir: f32[3]}

        # direction from vertex to light
        ldir = L.pos - eye_positions.T               # f32[3][N] (broadcast subtract)
        # dot product of each ldir column with itself (per vertex)
        dist2 = dot(ldir, ldir)                      # f32[3][N] . f32[3][N] -> f32[N]
        dist = sqrt(dist2)                           # f32[N]
        ldir = ldir / dist                           # f32[3][N] (normalize)

        # Attenuation (mode-dependent)
        # dot(A, B) convention: sum over 3-component dim, broadcast scalar B across N vertices
        match attn_func:
            NONE or DIR:
                attn = 1.0                                       # f32[N]

            SPOT:
                cos_angle = clamp(
                    dot(ldir, L.dir),                            # f32[3][N] . f32[3] -> f32[N]
                    0, inf)
                attn_num = clamp(
                    L.cosatt[0]                                  # f32 (broadcast to N)
                    + L.cosatt[1] * cos_angle                    # f32 * f32[N] -> f32[N]
                    + L.cosatt[2] * cos_angle * cos_angle,       # f32 * f32[N] -> f32[N]
                    0, inf)                                      # -> f32[N]
                attn_den =
                    L.distatt[0]                                 # f32 (broadcast to N)
                    + L.distatt[1] * dist                        # f32 * f32[N] -> f32[N]
                    + L.distatt[2] * dist2                       # f32 * f32[N] -> f32[N]
                                                                 # -> f32[N]
                attn = attn_num / attn_den                       # f32[N] / f32[N] -> f32[N]

            SPEC:
                NdotL = dot(normals.T, ldir)                     # f32[3][N] . f32[3][N] -> f32[N]
                NdotH = where(NdotL >= 0,
                    clamp(dot(normals.T, L.dir), 0, inf),        # f32[3][N] . f32[3] -> f32[N]
                    0)                                           # -> f32[N]
                attn_num = clamp(
                    L.cosatt[0]
                    + L.cosatt[1] * NdotH
                    + L.cosatt[2] * NdotH * NdotH,
                    0, inf)                                      # -> f32[N]
                attn_den =
                    L.distatt[0]
                    + L.distatt[1] * NdotH
                    + L.distatt[2] * NdotH * NdotH              # -> f32[N]
                attn = attn_num / attn_den                       # f32[N] / f32[N] -> f32[N]

        # Diffuse factor (mode-dependent)
        match diffuse_func:
            NONE:  diffuse = 1.0                                 # f32[N]
            SIGN:  diffuse = dot(ldir, normals.T)                # f32[3][N] . f32[3][N] -> f32[N]
            CLAMP: diffuse = clamp(
                       dot(ldir, normals.T), 0, inf)             # f32[3][N] . f32[3][N] -> f32[N]

        # Accumulate: attn * diffuse * light_color
        lacc += round(attn * diffuse * L.color)      # broadcast: f32[N] * u8[4] -> s32[N][4]

    lacc = clamp(lacc, 0, 255)                       # u8[N][4]

    # Material * lighting multiply (fixed-point rounding):
    lit_colors = (mat * (lacc + (lacc >> 7))) >> 8   # u8[N][4]

# --- Texture coordinate generation (per active channel, all N vertices) ---
for i in 0..num_texgens-1:
    # Select input array based on texgen[i].src.
    # XF appends 1.0 to form homogeneous vectors internally (not stored in memory).
    match texgen[i].src:
        POS:  inputs = [positions, 1]                # conceptually f32[N][4], w=1 implicit
        NRM:  inputs = [normals, 1]                  # conceptually f32[N][4], w=1 implicit
        TEXj: inputs = [texcoords[j], 1, 1]          # conceptually f32[N][4], z=1, w=1 implicit

    if texgen[i].mtx == IDENTITY:
        gen_tc[i] = inputs[:, 0:2]                   # f32[N][2], pass through s,t
    else:
        T = tex_matrices[texgen[i].mtx]              # f32[R][4] (R=2 for MTX2x4, R=3 for MTX3x4)
        gen_tc[i] = (T @ inputs.T).T                 # f32[R][4] @ f32[4][N] -> f32[N][R]
        # R=2: (s, t) per vertex. R=3: (s, t, q), q for perspective-correct division later


=====================================
 STAGE 3: RASTERIZER
=====================================
# Scan-converts primitives into pixel fragments.
# Linearly interpolates all per-vertex attributes across the primitive surface.
# Interpolation is perspective-correct for texcoords (divide by w).

input:
  screen_positions         : f32[N][3]    # from XF
  lit_colors               : u8[N][4]     # from XF
  gen_tc[i]                : f32[N][R]    # from XF
  primitive_type           : TRIANGLES, QUADS, TRISTRIP, etc.

output (M = total fragments produced, depends on primitive screen coverage, max 640*528):
  # Streamed to TEV per-fragment via on-chip wiring (not CPU-accessible).
  fragment_positions       : u16[M][2]    # (px, py) pixel coordinates, 0..639 / 0..527
  fragment_depths          : u24[M]       # interpolated Z per fragment
  fragment_ras_colors      : u8[M][4]     # interpolated vertex color (RGBA), 8-bit per channel
  fragment_texcoords[i]    : s24[M][2]    # interpolated texcoords per active channel
                                          # s17.7 fixed-point (~24 usable bits, signed)
                                          # higher precision than TEV color path (8-bit)

helpers:
    # Barycentric weights for a triangle via edge functions (2D cross products).
    # Each weight is the ratio of the sub-triangle opposite that vertex to the full triangle.
    # w0=1 at V0, w0=0 on edge V1-V2, etc. All three sum to 1.0.
    # Real hardware uses fixed-point incremental evaluation, not fp32 recomputation per pixel.
    def barycentric_weights(px, py, V0, V1, V2) -> (f32, f32, f32):
        # 2x signed area of the full triangle (z-component of 2D cross product of two edges)
        area = (V1.x - V0.x) * (V2.y - V0.y) - (V2.x - V0.x) * (V1.y - V0.y)  : f32

        # sub-triangle areas (triangle formed by pixel and two vertices)
        w0 = (V1.x - px) * (V2.y - py) - (V2.x - px) * (V1.y - py)             : f32
        w1 = (V2.x - px) * (V0.y - py) - (V0.x - px) * (V2.y - py)             : f32
        w2 = (V0.x - px) * (V1.y - py) - (V1.x - px) * (V0.y - py)             : f32

        # normalize
        w0 /= area;  w1 /= area;  w2 /= area
        return (w0, w1, w2)

    # For quads (V0=TL, V1=TR, V2=BR, V3=BL), bilinear interpolation is equivalent:
    #   u = (px - min_x) / (max_x - min_x)                                       : f32 [0..1]
    #   v = (py - min_y) / (max_y - min_y)                                       : f32 [0..1]
    #   attr = (1-u)*(1-v)*V0 + u*(1-v)*V1 + u*v*V2 + (1-u)*v*V3

# Primitive assembly: partition vertex indices [0..N-1] into triangles.
# Each triangle is (i0, i1, i2), indexing into screen_positions, lit_colors, gen_tc.
match primitive_type:
    TRIANGLES:     tris = [(i, i+1, i+2)             for i in range(0, N, 3)]
    TRIANGLESTRIP: tris = [(i, i+1, i+2)             for i in range(0, N-2)]
    TRIANGLEFAN:   tris = [(0, i, i+1)               for i in range(1, N-1)]
    QUADS:         tris = [(i, i+1, i+2), (i, i+2, i+3) for i in range(0, N, 4)]

# Rasterize each triangle
for (i0, i1, i2) in tris:
    V0 = { pos: screen_positions[i0], color: lit_colors[i0], tc: gen_tc[:][i0] }
    V1 = { pos: screen_positions[i1], color: lit_colors[i1], tc: gen_tc[:][i1] }
    V2 = { pos: screen_positions[i2], color: lit_colors[i2], tc: gen_tc[:][i2] }

    # Bounding box of the triangle, clipped to EFB dimensions
    bbox_x0 = max(0,              floor(min(V0.pos.x, V1.pos.x, V2.pos.x)))   : int
    bbox_x1 = min(EFB_WIDTH - 1,  floor(max(V0.pos.x, V1.pos.x, V2.pos.x)))   : int
    bbox_y0 = max(0,              floor(min(V0.pos.y, V1.pos.y, V2.pos.y)))   : int
    bbox_y1 = min(EFB_HEIGHT - 1, floor(max(V0.pos.y, V1.pos.y, V2.pos.y)))   : int

    for py in bbox_y0..bbox_y1:
        for px in bbox_x0..bbox_x1:
            (w0, w1, w2) = barycentric_weights(px, py, V0.pos, V1.pos, V2.pos)
            if w0 < 0 or w1 < 0 or w2 < 0: continue  # pixel outside triangle

            fragment.z = w0*V0.pos.z + w1*V1.pos.z + w2*V2.pos.z

            for ch in {r, g, b, a}:
                fragment.ras_color[ch] = clamp(
                    w0*V0.color[ch] + w1*V1.color[ch] + w2*V2.color[ch],
                    0, 255)                                     # u8

            for i in 0..7:
                fragment.texcoord[i][0] = w0*V0.tc[i][0] + w1*V1.tc[i][0] + w2*V2.tc[i][0]
                fragment.texcoord[i][1] = w0*V0.tc[i][1] + w1*V1.tc[i][1] + w2*V2.tc[i][1]
                # (perspective correction: divide by interpolated 1/w; omitted for clarity)

            emit fragment to next stage


=====================================
 STAGE 4+5: TEXTURE FETCH + TEV
=====================================
# Per fragment. The TEV runs up to 16 stages sequentially.
# 4 pixel pipes execute this in parallel on real hardware.
# All TEV arithmetic is integer, in the range [-1024..1023] (s10) internally,
# clamped to [0..255] (u8) on output when clamp=true.

input (M fragments from rasterizer, streamed on-chip):
  fragment_positions       : u16[M][2]    # (px, py) per fragment
  fragment_depths          : u24[M]       # interpolated Z per fragment
  fragment_ras_colors      : u8[M][4]     # interpolated vertex color (RGBA) per fragment
  fragment_texcoords[i]    : s24[M][2]    # interpolated texcoords per active channel (s17.7)
  (all TEV config from global state)

output (M fragments, streamed on-chip to PE):
  fragment_final_colors    : u8[M][4]     # TEV output color per fragment
  fragment_positions       : u16[M][2]    # passed through from rasterizer
  fragment_depths          : u24[M]       # passed through from rasterizer

helpers:
    def wrap(coord : f32, dim : u16, mode : enum) -> u16:
        # Convert normalized coordinate to texel index with wrapping
        texel = floor(coord * dim)                   : s32
        match mode:
            CLAMP:  return clamp(texel, 0, dim - 1)  # u16
            REPEAT: return texel % dim                # u16 (add dim if negative)
            MIRROR: return dim - 1 - abs((texel % (2*dim)) - dim)  # u16, fold at edges

    def texture_sample(tex_map_idx : u8, s : s24, t : s24) -> u8[4]:
        # Fetch one texel from the texture bound at tex_map_idx.
        # s, t are texcoords (s17.7 fixed-point from rasterizer, or f32 after indirect offset).
        # Nearest-neighbor sampling (no bilinear interpolation in this pseudocode).
        tex = textures[tex_map_idx]
        ix = wrap(s, tex.width, tex.wrap_s)          # u16, texel x index
        iy = wrap(t, tex.height, tex.wrap_t)         # u16, texel y index
        raw = tex.data[iy * tex.width + ix]          # raw bytes from texture in MEM2
        return decode_format(raw, tex.format)        # -> u8[4] (RGBA after format decode)
        # decode_format converts raw texel bytes to u8[4] (R, G, B, A):
        #   GX_TF_I4      : 4-bit intensity.   (v*17, v*17, v*17, 255)        4 bits/texel
        #   GX_TF_I8      : 8-bit intensity.   (v, v, v, 255)                 1 byte/texel
        #   GX_TF_IA4     : 4-bit I + 4-bit A. (i*17, i*17, i*17, a*17)      1 byte/texel
        #   GX_TF_IA8     : 8-bit I + 8-bit A. (v, v, v, a)                  2 bytes/texel
        #   GX_TF_RGB565  : 5R/6G/5B.          (r<<3, g<<2, b<<3, 255)       2 bytes/texel
        #   GX_TF_RGB5A3  : MSB=1: 5R/5G/5B.   (r<<3, g<<3, b<<3, 255)      2 bytes/texel
        #                   MSB=0: 4R/4G/4B/3A. (r<<4, g<<4, b<<4, a<<5)
        #   GX_TF_RGBA8   : 8R/8G/8B/8A.       (r, g, b, a)                  4 bytes/texel
        #   GX_TF_C4      : 4-bit index -> tlut[index] (max 16 entries).     4 bits/texel + palette
        #   GX_TF_C8      : 8-bit index -> tlut[index] (max 256 entries).    1 byte/texel + palette
        #   GX_TF_C14X2   : 14-bit index -> tlut[index] (max 16384 entries). 2 bytes/texel + palette
        #                   Palette entry format (per TLUT): IA8, RGB565, or RGB5A3.
        #                   Palette stored in 16 KB on-chip TLUT memory, loaded via GX_LoadTlut.
        #   GX_TF_CMPR    : S3TC/DXT1 block compression (4x4 block, 4 bits/texel avg)

    def mux_color(selector : u4, ch : u2, regs, tex_color, ras_color, konst_rgb) -> s16:
        # Select one color channel value based on the GX_CC_* selector.
        # ch: 0=R, 1=G, 2=B
        match selector:
            GX_CC_CPREV  (0):  return regs[TEVPREV][ch]          # s10, previous stage color
            GX_CC_APREV  (1):  return regs[TEVPREV][3]           # s10, previous stage alpha
            GX_CC_C0     (2):  return regs[TEVREG0][ch]          # s10, register 0 color
            GX_CC_A0     (3):  return regs[TEVREG0][3]           # s10, register 0 alpha
            GX_CC_C1     (4):  return regs[TEVREG1][ch]          # s10
            GX_CC_A1     (5):  return regs[TEVREG1][3]           # s10
            GX_CC_C2     (6):  return regs[TEVREG2][ch]          # s10
            GX_CC_A2     (7):  return regs[TEVREG2][3]           # s10
            GX_CC_TEXC   (8):  return tex_color[ch]              # u8, texture color channel
            GX_CC_TEXA   (9):  return tex_color[3]               # u8, texture alpha
            GX_CC_RASC  (10):  return ras_color[ch]              # u8, rasterized color channel
            GX_CC_RASA  (11):  return ras_color[3]               # u8, rasterized alpha
            GX_CC_ONE   (12):  return 255                        # constant 1.0
            GX_CC_HALF  (13):  return 128                        # constant 0.5
            GX_CC_KONST (14):  return konst_rgb[ch]              # u8, konst color channel
            GX_CC_ZERO  (15):  return 0                          # constant 0.0

    def mux_alpha(selector : u3, regs, tex_color, ras_color, konst_a) -> s16:
        # Select one alpha value based on the GX_CA_* selector.
        match selector:
            GX_CA_APREV  (0):  return regs[TEVPREV][3]           # s10
            GX_CA_A0     (1):  return regs[TEVREG0][3]           # s10
            GX_CA_A1     (2):  return regs[TEVREG1][3]           # s10
            GX_CA_A2     (3):  return regs[TEVREG2][3]           # s10
            GX_CA_TEXA   (4):  return tex_color[3]               # u8
            GX_CA_RASA   (5):  return ras_color[3]               # u8
            GX_CA_KONST  (6):  return konst_a                    # u8
            GX_CA_ZERO   (7):  return 0                          # constant 0.0

# Process each fragment through the TEV pipeline (4 pixel pipes in parallel on real HW)
for f in 0..M-1:
    ras_color_f = fragment_ras_colors[f]               # u8[4], this fragment interpolated color
    texcoords_f = fragment_texcoords[:][f]             # s24[8][2], this fragment texcoords

    # Initialize TEV register file (reset per fragment from CPU-preloaded values)
    regs[0..3] = tev_regs[0..3]                        # s10[4] each, (R, G, B, A)

    for stage in 0..num_tev_stages-1:
        cfg = tev_stage[stage]

        # --- INDIRECT TEXTURE (optional, per stage) ---
        if cfg.ind_enable:
            ist = ind_stages[cfg.ind_stage]
            ind_texel = texture_sample(ist.tex_map, texcoords_f[ist.tex_coord])  : u8[4]

            offset = [(ind_texel[0] >> shift) - (128 >> shift),    : s32[3]
                      (ind_texel[1] >> shift) - (128 >> shift),
                      (ind_texel[2] >> shift) - (128 >> shift)]
            # shift depends on ind_format: ITF_8=0, ITF_5=3, ITF_4=4, ITF_3=5

            if not (cfg.ind_bias_sel & 1): offset[0] = 0    # S
            if not (cfg.ind_bias_sel & 2): offset[1] = 0    # T
            if not (cfg.ind_bias_sel & 4): offset[2] = 0    # U

            if cfg.ind_matrix_sel in {ITM_0, ITM_1, ITM_2}:
                ind_M = ind_matrices[cfg.ind_matrix_sel - 1]             : f32[2][3]
                scale = 2^(ind_M.scale)                                  : f32
                [ds, dt] = ind_M @ offset * scale    # f32[2][3] @ s32[3] * f32 -> f32[2]
            else:
                ds = dt = 0

            tex_s = texcoords_f[cfg.tex_coord][0] + ds             : f32
            tex_t = texcoords_f[cfg.tex_coord][1] + dt             : f32
        else:
            tex_s = texcoords_f[cfg.tex_coord][0]
            tex_t = texcoords_f[cfg.tex_coord][1]

        # --- TEXTURE SAMPLE (main fetch for this TEV stage) ---
        if cfg.tex_map != NULL and cfg.tex_coord != NULL:
            tex_color_raw = texture_sample(cfg.tex_map, tex_s, tex_t)  # u8[4]
        else:
            tex_color_raw = [0, 0, 0, 0]                              # u8[4]

        # --- SWAP TABLES ---
        # Each swap table is a 4-element channel permutation {R=0, G=1, B=2, A=3}.
        # Indexing by the table remaps channels, e.g. table={0,2,1,3} swaps G and B.
        perm = swap_tables[cfg.tex_swap]                         # u8[4], e.g. {0, 1, 2, 3}
        for i in 0..3:
            tex_color[i] = tex_color_raw[perm[i]]               # u8
        perm = swap_tables[cfg.ras_swap]                         # u8[4]
        for i in 0..3:
            ras_color[i] = ras_color_f[perm[i]]                 # u8

        # --- RESOLVE KONST ---
        konst_rgb = resolve_konst_color(cfg.kcsel)               : u8[3]
        konst_a   = resolve_konst_alpha(cfg.kasel)               : u8

        # --- COLOR COMBINE (R, G, B independently) ---
        for ch in {0, 1, 2}:
            a = mux_color(cfg.color_in.a, ch, regs, tex_color, ras_color, konst_rgb)  : s16
            b = mux_color(cfg.color_in.b, ch, regs, tex_color, ras_color, konst_rgb)  : s16
            c = mux_color(cfg.color_in.c, ch, regs, tex_color, ras_color, konst_rgb)  : s16
            d = mux_color(cfg.color_in.d, ch, regs, tex_color, ras_color, konst_rgb)  : s16

            lerp = (a * (255 - c) + b * c + 128) >> 8           : s32 -> s16

            if cfg.color_op.op == ADD:    result = d + lerp
            else if cfg.color_op.op == SUB:  result = d - lerp

            match cfg.color_op.bias:
                TB_ZERO:    pass
                TB_ADDHALF: result += 128
                TB_SUBHALF: result -= 128

            match cfg.color_op.scale:
                CS_SCALE_1:  pass
                CS_SCALE_2:  result *= 2
                CS_SCALE_4:  result *= 4
                CS_DIVIDE_2: result /= 2

            if cfg.color_op.clamp: result = clamp(result, 0, 255)
            else:                  result = clamp(result, -1024, 1023)

            regs[cfg.color_op.out_reg][ch] = result              : s10

        # --- ALPHA COMBINE (single channel) ---
        a = mux_alpha(cfg.alpha_in.a, regs, tex_color, ras_color, konst_a)  : s16
        b = mux_alpha(cfg.alpha_in.b, regs, tex_color, ras_color, konst_a)  : s16
        c = mux_alpha(cfg.alpha_in.c, regs, tex_color, ras_color, konst_a)  : s16
        d = mux_alpha(cfg.alpha_in.d, regs, tex_color, ras_color, konst_a)  : s16

        lerp = (a * (255 - c) + b * c + 128) >> 8
        if cfg.alpha_op.op == ADD:    result = d + lerp
        elif cfg.alpha_op.op == SUB:  result = d - lerp
        # (same bias, scale, clamp as color)
        regs[cfg.alpha_op.out_reg][3] = result                   : s10

    # After all stages for this fragment, TEVPREV holds the final color
    fragment_final_colors[f] = [clamp(regs[TEVPREV][0], 0, 255),  # u8[4]
                                clamp(regs[TEVPREV][1], 0, 255),
                                clamp(regs[TEVPREV][2], 0, 255),
                                clamp(regs[TEVPREV][3], 0, 255)]


=====================================
 STAGE 6: PIXEL ENGINE (PE)
=====================================
input (M fragments from TEV, streamed on-chip):
  fragment_final_colors    : u8[M][4]     # TEV output color per fragment
  fragment_positions       : u16[M][2]    # (px, py) pixel coordinates per fragment
  fragment_depths          : u24[M]       # interpolated Z per fragment
  efb[528][640]            : { u8 r, g, b, a; u24 z; }  # existing framebuffer (on-die eDRAM)

output:
  efb[528][640]            : { u8 r, g, b, a; u24 z; }  # modified in-place
  # M_p <= M fragments survive alpha compare + Z test and write to EFB

# --- Alpha Compare (conditional discard) ---
test0 = compare(fragment.final_color.a, alpha_compare.func0, alpha_compare.ref0)  : bool
test1 = compare(fragment.final_color.a, alpha_compare.func1, alpha_compare.ref1)  : bool
match alpha_compare.op:
    AND:  pass_alpha = test0 AND test1
    OR:   pass_alpha = test0 OR test1
    XOR:  pass_alpha = test0 XOR test1
    XNOR: pass_alpha = test0 XNOR test1
if not pass_alpha: discard fragment

# --- Z Test ---
if z_mode.enable:
    if not compare(fragment.z, z_mode.func, efb[py][px].z):
        discard fragment
    if z_mode.write_enable:
        efb[py][px].z = fragment.z

# --- EFB Blending ---
src = fragment.final_color                                        : u8[4]
dst = efb[py][px].color                                           : u8[4]

match blend.mode:
    BM_NONE:
        # overwrite
        out = src

    BM_BLEND:
        # per channel: out = (src * src_factor + dst * dst_factor) / 255
        # factor functions resolve to a u8 multiplier based on mode:
        #   ZERO=0, ONE=255, SRCALPHA=src.a, INVSRCALPHA=255-src.a,
        #   DSTALPHA=dst.a, INVDSTALPHA=255-dst.a,
        #   SRCCLR=src[ch], DSTCLR=dst[ch], etc.
        for ch in {r, g, b, a}:
            sf = resolve_blend_factor(blend.src_factor, src, dst, ch)  : u8
            df = resolve_blend_factor(blend.dst_factor, src, dst, ch)  : u8
            out[ch] = clamp((src[ch]*sf + dst[ch]*df + 128) / 256, 0, 255)  : u8

    BM_SUBTRACT:
        for ch in {r, g, b, a}:
            out[ch] = clamp(dst[ch] - src[ch], 0, 255)

    BM_LOGIC:
        # bitwise operation per byte
        for ch in {r, g, b, a}:
            out[ch] = logic_op(blend.logic_op, src[ch], dst[ch])
            # CLEAR=0, AND=s&d, REVAND=s&~d, COPY=s, INVAND=~s&d,
            # NOOP=d, XOR=s^d, OR=s|d, NOR=~(s|d), EQUIV=~(s^d),
            # INV=~d, REVOR=s|~d, INVCOPY=~s, INVOR=~s|d, NAND=~(s&d), SET=0xFF

# --- Write to EFB ---
if color_update_enable:
    efb[py][px].r = out.r
    efb[py][px].g = out.g
    efb[py][px].b = out.b
if alpha_update_enable:
    efb[py][px].a = out.a


=====================================
 STAGE 7: EFB READBACK
=====================================
# Not part of the per-fragment pipeline. Initiated by separate GX commands.

# CopyTex: DMA a rectangular EFB region to a texture buffer in MEM2
GX_CopyTex(dest_ptr, clear):
    for y in src_top..src_top+src_height-1:
        for x in src_left..src_left+src_width-1:
            dest_ptr[(y-src_top)*src_width + (x-src_left)] = efb[y][x].rgba  : u8[4]
    if clear:
        fill copied region with clear_color, clear_z

# CopyDisp: DMA EFB to the external framebuffer for TV display
# (involves Y-scaling and optional RGBA->YCbCr conversion)

# Poke/Peek: single-pixel direct access (slow, no pipeline involvement)
GX_PokeARGB(x, y, color): efb[y][x].rgba = color
GX_PeekARGB(x, y) -> color: return efb[y][x].rgba
```

### Data shape summary

| Location | Shape | Precision | Notes |
|---|---|---|---|
| Vertex position (model space) | f32[3] | fp32 | From CPU via FIFO |
| Vertex color | u8[4] | 8-bit | RGBA |
| Vertex texcoord (raw) | f32[2] per slot | fp32 | Up to 8 slots |
| Position matrix | f32[3][4] | fp32 | 10 slots in XF |
| Texture matrix | f32[3][4] | fp32 | 10 slots in XF |
| Texgen output (per vertex) | f32[2] or f32[3] | fp32 | Fed to rasterizer |
| Rasterized color (per pixel) | u8[4] | 8-bit | Interpolated, quantized |
| Rasterized texcoord (per pixel) | s17.7 fixed point | ~24 bits | Interpolated in rasterizer |
| Texture texel | u8[4] | 8-bit | After format decode |
| TEV register file | s10[4] x 4 regs | 10-bit signed | [-1024..1023] per channel |
| TEV lerp intermediate | s32 | 32-bit | Before >>8 rounding |
| TEV output (clamped) | u8[4] | 8-bit | [0..255] |
| Konst colors | u8[4] x 4 | 8-bit | |
| Indirect texture matrix | f32[2][3] x 3 | fp32 | Plus s8 scale exponent |
| EFB pixel | u8[4] + u24 | 8-bit color, 24-bit depth | 640x528 pixels |
| Blend factors | u8 | 8-bit | Per channel |