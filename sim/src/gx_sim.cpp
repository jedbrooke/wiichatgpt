// GX Software Simulator
// Implements the subset of libogc's GX API needed for compute-via-TEV.
// Focuses on: TEV combiner pipeline, EFB read/write, texture sampling,
// and full-screen quad rasterization.

#include "gx.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Internal state

static const int MAX_TEV_STAGES = 16;
static const int MAX_TEXMAPS = 8;
static const int MAX_KONST_COLORS = 4;
static const int MAX_TEV_REGS = 4;   // TEVPREV + TEVREG0..2
static const int MAX_SWAP_TABLES = 4;

struct TEVStageConfig {
    // Color inputs (a, b, c, d selectors)
    u8 color_a, color_b, color_c, color_d;
    // Alpha inputs
    u8 alpha_a, alpha_b, alpha_c, alpha_d;
    // Color operation
    u8 color_op, color_bias, color_scale, color_clamp, color_regid;
    // Alpha operation
    u8 alpha_op, alpha_bias, alpha_scale, alpha_clamp, alpha_regid;
    // Texture/color channel binding
    u8 texcoord;
    u32 texmap;
    u8 colorchan;
    // Konst selectors
    u8 kcsel, kasel;
    // Swap mode selectors
    u8 ras_swap, tex_swap;
};

struct SwapTable {
    u8 r, g, b, a;  // channel remapping indices (0=R, 1=G, 2=B, 3=A)
};

struct TextureState {
    void* img_ptr;
    u16 width;
    u16 height;
    u8 fmt;
    u8 wrap_s;
    u8 wrap_t;
    bool loaded;
};

struct EFBPixel {
    // Stored as signed 10-bit range (-1024..1023) to match TEV internal precision.
    // On real hardware the EFB is 8-bit per channel after clamping, but we keep
    // higher precision internally for chained TEV computations.
    s16 r, g, b, a;
    u32 z;
};

struct TexCopyRegion {
    u16 left, top, width, height;
};

struct VertexData {
    f32 pos[3];
    u8 color[4];
    f32 texcoord[8][2];  // up to 8 texcoord slots
    bool has_pos;
    bool has_color;
    bool has_texcoord[8];
};

// Global simulator state
static struct {
    // TEV
    u8 num_tev_stages;
    TEVStageConfig tev_stages[MAX_TEV_STAGES];
    GXColorS10 tev_regs[MAX_TEV_REGS];    // TEVPREV, TEVREG0, TEVREG1, TEVREG2
    GXColor konst_colors[MAX_KONST_COLORS];
    SwapTable swap_tables[MAX_SWAP_TABLES];

    // Textures
    TextureState textures[MAX_TEXMAPS];

    // EFB
    EFBPixel efb[GX_EFB_HEIGHT][GX_EFB_WIDTH];
    GXColor clear_color;
    u32 clear_z;

    // EFB copy region
    TexCopyRegion tex_copy_src;
    u16 tex_copy_dst_wd;
    u16 tex_copy_dst_ht;
    u32 tex_copy_dst_fmt;

    // Drawing state
    u8 current_primitive;
    u8 current_vtxfmt;
    u16 expected_vtx_count;
    std::vector<VertexData> vertex_buffer;
    VertexData current_vertex;

    // Vertex attribute state
    bool vtx_has_pos;
    bool vtx_has_color;
    bool vtx_has_texcoord[8];

    // Color update
    bool color_update_enabled;
    bool alpha_update_enabled;

    // Number of texcoord gens
    u32 num_texgens;
    u8 num_color_chans;

    // Initialized flag
    bool initialized;
} gx;

// Clamp to TEV internal range (-1024..1023 for s10, 0..255 for output)
static s16 clamp_s10(s16 val) {
    if (val < -1024) return -1024;
    if (val > 1023) return 1023;
    return val;
}

static u8 clamp_u8(s16 val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (u8)val;
}

// Texture sampling (nearest neighbor, unnormalized coords mapped to texels)
static void sample_texture(u8 texmap, f32 s, f32 t, u8 out[4]) {
    TextureState& tex = gx.textures[texmap];
    if (!tex.loaded || !tex.img_ptr) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }

    // s, t are in [0..1] range
    int ix = (int)(s * tex.width);
    int iy = (int)(t * tex.height);

    // Wrap
    if (tex.wrap_s == GX_CLAMP) {
        ix = std::max(0, std::min((int)tex.width - 1, ix));
    } else if (tex.wrap_s == GX_REPEAT) {
        ix = ix % tex.width;
        if (ix < 0) ix += tex.width;
    }
    if (tex.wrap_t == GX_CLAMP) {
        iy = std::max(0, std::min((int)tex.height - 1, iy));
    } else if (tex.wrap_t == GX_REPEAT) {
        iy = iy % tex.height;
        if (iy < 0) iy += tex.height;
    }

    // Sample based on format
    switch (tex.fmt) {
        case GX_TF_RGBA8: {
            const u8* p = (const u8*)tex.img_ptr + (iy * tex.width + ix) * 4;
            out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = p[3];
            break;
        }
        case GX_TF_I8: {
            u8 v = ((const u8*)tex.img_ptr)[iy * tex.width + ix];
            out[0] = out[1] = out[2] = v;
            out[3] = 255;
            break;
        }
        case GX_TF_IA8: {
            const u8* p = (const u8*)tex.img_ptr + (iy * tex.width + ix) * 2;
            out[0] = out[1] = out[2] = p[0];
            out[3] = p[1];
            break;
        }
        default:
            out[0] = out[1] = out[2] = out[3] = 0;
            break;
    }
}

// Apply swap table to a 4-channel value
static void apply_swap(const u8 in[4], const SwapTable& swap, u8 out[4]) {
    u8 channels[4] = { in[0], in[1], in[2], in[3] };
    out[0] = channels[swap.r];
    out[1] = channels[swap.g];
    out[2] = channels[swap.b];
    out[3] = channels[swap.a];
}

// Resolve a TEV color input selector to a value (0..255 range, returned as s16)
static s16 resolve_color_input(u8 sel, int channel,
                               const s16 tev_regs[MAX_TEV_REGS][4],
                               const u8 tex_color[4],
                               const u8 ras_color[4],
                               const u8 konst_color[4]) {
    // channel: 0=R, 1=G, 2=B
    switch (sel) {
        case GX_CC_CPREV:  return tev_regs[GX_TEVPREV][channel];
        case GX_CC_APREV:  return tev_regs[GX_TEVPREV][3];
        case GX_CC_C0:     return tev_regs[GX_TEVREG0][channel];
        case GX_CC_A0:     return tev_regs[GX_TEVREG0][3];
        case GX_CC_C1:     return tev_regs[GX_TEVREG1][channel];
        case GX_CC_A1:     return tev_regs[GX_TEVREG1][3];
        case GX_CC_C2:     return tev_regs[GX_TEVREG2][channel];
        case GX_CC_A2:     return tev_regs[GX_TEVREG2][3];
        case GX_CC_TEXC:   return tex_color[channel];
        case GX_CC_TEXA:   return tex_color[3];
        case GX_CC_RASC:   return ras_color[channel];
        case GX_CC_RASA:   return ras_color[3];
        case GX_CC_ONE:    return 255;
        case GX_CC_HALF:   return 128;
        case GX_CC_KONST:  return konst_color[channel];
        case GX_CC_ZERO:   return 0;
        default:           return 0;
    }
}

// Resolve a TEV alpha input selector
static s16 resolve_alpha_input(u8 sel,
                               const s16 tev_regs[MAX_TEV_REGS][4],
                               const u8 tex_color[4],
                               const u8 ras_color[4],
                               const u8 konst_alpha) {
    switch (sel) {
        case GX_CA_APREV:  return tev_regs[GX_TEVPREV][3];
        case GX_CA_A0:     return tev_regs[GX_TEVREG0][3];
        case GX_CA_A1:     return tev_regs[GX_TEVREG1][3];
        case GX_CA_A2:     return tev_regs[GX_TEVREG2][3];
        case GX_CA_TEXA:   return tex_color[3];
        case GX_CA_RASA:   return ras_color[3];
        case GX_CA_KONST:  return konst_alpha;
        case GX_CA_ZERO:   return 0;
        default:           return 0;
    }
}

// Resolve konst color for a TEV stage
static void resolve_konst_color(u8 kcsel, u8 out[4]) {
    switch (kcsel) {
        case GX_TEV_KCSEL_1:   out[0]=out[1]=out[2]=255; break;
        case GX_TEV_KCSEL_7_8: out[0]=out[1]=out[2]=224; break;
        case GX_TEV_KCSEL_3_4: out[0]=out[1]=out[2]=192; break;
        case GX_TEV_KCSEL_5_8: out[0]=out[1]=out[2]=160; break;
        case GX_TEV_KCSEL_1_2: out[0]=out[1]=out[2]=128; break;
        case GX_TEV_KCSEL_3_8: out[0]=out[1]=out[2]=96;  break;
        case GX_TEV_KCSEL_1_4: out[0]=out[1]=out[2]=64;  break;
        case GX_TEV_KCSEL_1_8: out[0]=out[1]=out[2]=32;  break;
        case GX_TEV_KCSEL_K0:
            out[0]=gx.konst_colors[0].r; out[1]=gx.konst_colors[0].g;
            out[2]=gx.konst_colors[0].b; break;
        case GX_TEV_KCSEL_K1:
            out[0]=gx.konst_colors[1].r; out[1]=gx.konst_colors[1].g;
            out[2]=gx.konst_colors[1].b; break;
        case GX_TEV_KCSEL_K2:
            out[0]=gx.konst_colors[2].r; out[1]=gx.konst_colors[2].g;
            out[2]=gx.konst_colors[2].b; break;
        case GX_TEV_KCSEL_K3:
            out[0]=gx.konst_colors[3].r; out[1]=gx.konst_colors[3].g;
            out[2]=gx.konst_colors[3].b; break;
        case GX_TEV_KCSEL_K0_R: out[0]=out[1]=out[2]=gx.konst_colors[0].r; break;
        case GX_TEV_KCSEL_K1_R: out[0]=out[1]=out[2]=gx.konst_colors[1].r; break;
        case GX_TEV_KCSEL_K2_R: out[0]=out[1]=out[2]=gx.konst_colors[2].r; break;
        case GX_TEV_KCSEL_K3_R: out[0]=out[1]=out[2]=gx.konst_colors[3].r; break;
        case GX_TEV_KCSEL_K0_G: out[0]=out[1]=out[2]=gx.konst_colors[0].g; break;
        case GX_TEV_KCSEL_K1_G: out[0]=out[1]=out[2]=gx.konst_colors[1].g; break;
        case GX_TEV_KCSEL_K2_G: out[0]=out[1]=out[2]=gx.konst_colors[2].g; break;
        case GX_TEV_KCSEL_K3_G: out[0]=out[1]=out[2]=gx.konst_colors[3].g; break;
        case GX_TEV_KCSEL_K0_B: out[0]=out[1]=out[2]=gx.konst_colors[0].b; break;
        case GX_TEV_KCSEL_K1_B: out[0]=out[1]=out[2]=gx.konst_colors[1].b; break;
        case GX_TEV_KCSEL_K2_B: out[0]=out[1]=out[2]=gx.konst_colors[2].b; break;
        case GX_TEV_KCSEL_K3_B: out[0]=out[1]=out[2]=gx.konst_colors[3].b; break;
        case GX_TEV_KCSEL_K0_A: out[0]=out[1]=out[2]=gx.konst_colors[0].a; break;
        case GX_TEV_KCSEL_K1_A: out[0]=out[1]=out[2]=gx.konst_colors[1].a; break;
        case GX_TEV_KCSEL_K2_A: out[0]=out[1]=out[2]=gx.konst_colors[2].a; break;
        case GX_TEV_KCSEL_K3_A: out[0]=out[1]=out[2]=gx.konst_colors[3].a; break;
        default: out[0]=out[1]=out[2]=0; break;
    }
    out[3] = 255;
}

// Resolve konst alpha for a TEV stage
static u8 resolve_konst_alpha(u8 kasel) {
    switch (kasel) {
        case GX_TEV_KASEL_1:   return 255;
        case GX_TEV_KASEL_7_8: return 224;
        case GX_TEV_KASEL_3_4: return 192;
        case GX_TEV_KASEL_5_8: return 160;
        case GX_TEV_KASEL_1_2: return 128;
        case GX_TEV_KASEL_3_8: return 96;
        case GX_TEV_KASEL_1_4: return 64;
        case GX_TEV_KASEL_1_8: return 32;
        case GX_TEV_KASEL_K0_R: return gx.konst_colors[0].r;
        case GX_TEV_KASEL_K1_R: return gx.konst_colors[1].r;
        case GX_TEV_KASEL_K2_R: return gx.konst_colors[2].r;
        case GX_TEV_KASEL_K3_R: return gx.konst_colors[3].r;
        case GX_TEV_KASEL_K0_G: return gx.konst_colors[0].g;
        case GX_TEV_KASEL_K1_G: return gx.konst_colors[1].g;
        case GX_TEV_KASEL_K2_G: return gx.konst_colors[2].g;
        case GX_TEV_KASEL_K3_G: return gx.konst_colors[3].g;
        case GX_TEV_KASEL_K0_B: return gx.konst_colors[0].b;
        case GX_TEV_KASEL_K1_B: return gx.konst_colors[1].b;
        case GX_TEV_KASEL_K2_B: return gx.konst_colors[2].b;
        case GX_TEV_KASEL_K3_B: return gx.konst_colors[3].b;
        case GX_TEV_KASEL_K0_A: return gx.konst_colors[0].a;
        case GX_TEV_KASEL_K1_A: return gx.konst_colors[1].a;
        case GX_TEV_KASEL_K2_A: return gx.konst_colors[2].a;
        case GX_TEV_KASEL_K3_A: return gx.konst_colors[3].a;
        default: return 0;
    }
}

// TEV combiner formula: out = (d OP ((1-c)*a + c*b) + bias) * scale
// All values are in 0..255 range (treated as fixed point), math done in s16
// to handle intermediate overflow.
static s16 tev_combine(s16 a, s16 b, s16 c, s16 d,
                       u8 op, u8 bias, u8 scale, u8 clamp) {
    // The formula:  d OP ((1-c)*a + c*b + bias) * scale
    // where c is in 0..255 representing 0.0..1.0
    // Multiply-add: (255-c)*a + c*b  (all in 0..255 fixed point)
    s32 lerp = ((255 - c) * a + c * b);
    // The hardware does a +128 and >>8 to round the lerp
    lerp = (lerp + 128) >> 8;

    // Apply operation
    s32 result;
    if (op == GX_TEV_ADD) {
        result = d + lerp;
    } else {
        result = d - lerp;
    }

    // Apply bias
    switch (bias) {
        case GX_TB_ADDHALF: result += 128; break;
        case GX_TB_SUBHALF: result -= 128; break;
        default: break;
    }

    // Apply scale
    switch (scale) {
        case GX_CS_SCALE_2:  result *= 2; break;
        case GX_CS_SCALE_4:  result *= 4; break;
        case GX_CS_DIVIDE_2: result /= 2; break;
        default: break;
    }

    // Clamp
    if (clamp) {
        if (result < 0) result = 0;
        if (result > 255) result = 255;
    } else {
        if (result < -1024) result = -1024;
        if (result > 1023) result = 1023;
    }

    return (s16)result;
}

// Process a single fragment through the TEV pipeline
static void process_fragment(int px, int py,
                             const u8 ras_color_raw[4],
                             const f32 texcoords[8][2]) {
    // TEV registers: [reg][channel], channel 0=R, 1=G, 2=B, 3=A
    s16 tev_regs[MAX_TEV_REGS][4];
    for (int r = 0; r < MAX_TEV_REGS; r++) {
        tev_regs[r][0] = gx.tev_regs[r].r;
        tev_regs[r][1] = gx.tev_regs[r].g;
        tev_regs[r][2] = gx.tev_regs[r].b;
        tev_regs[r][3] = gx.tev_regs[r].a;
    }

    for (int stage = 0; stage < gx.num_tev_stages; stage++) {
        const TEVStageConfig& cfg = gx.tev_stages[stage];

        // Sample texture for this stage
        u8 tex_color_raw[4] = {0, 0, 0, 0};
        if (cfg.texmap != GX_TEXMAP_NULL && cfg.texcoord != GX_TEXCOORDNULL) {
            f32 s = texcoords[cfg.texcoord][0];
            f32 t = texcoords[cfg.texcoord][1];
            sample_texture((u8)cfg.texmap, s, t, tex_color_raw);
        }

        // Apply swap tables
        u8 tex_color[4], ras_color[4];
        apply_swap(tex_color_raw, gx.swap_tables[cfg.tex_swap], tex_color);
        apply_swap(ras_color_raw, gx.swap_tables[cfg.ras_swap], ras_color);

        // Resolve konst values for this stage
        u8 konst_color[4];
        resolve_konst_color(cfg.kcsel, konst_color);
        u8 konst_alpha = resolve_konst_alpha(cfg.kasel);

        // Process color channels (R, G, B)
        for (int ch = 0; ch < 3; ch++) {
            s16 a = resolve_color_input(cfg.color_a, ch, tev_regs, tex_color, ras_color, konst_color);
            s16 b = resolve_color_input(cfg.color_b, ch, tev_regs, tex_color, ras_color, konst_color);
            s16 c = resolve_color_input(cfg.color_c, ch, tev_regs, tex_color, ras_color, konst_color);
            s16 d = resolve_color_input(cfg.color_d, ch, tev_regs, tex_color, ras_color, konst_color);

            s16 result = tev_combine(a, b, c, d,
                                     cfg.color_op, cfg.color_bias,
                                     cfg.color_scale, cfg.color_clamp);
            tev_regs[cfg.color_regid][ch] = result;
        }

        // Process alpha channel
        {
            s16 a = resolve_alpha_input(cfg.alpha_a, tev_regs, tex_color, ras_color, konst_alpha);
            s16 b = resolve_alpha_input(cfg.alpha_b, tev_regs, tex_color, ras_color, konst_alpha);
            s16 c = resolve_alpha_input(cfg.alpha_c, tev_regs, tex_color, ras_color, konst_alpha);
            s16 d = resolve_alpha_input(cfg.alpha_d, tev_regs, tex_color, ras_color, konst_alpha);

            s16 result = tev_combine(a, b, c, d,
                                     cfg.alpha_op, cfg.alpha_bias,
                                     cfg.alpha_scale, cfg.alpha_clamp);
            tev_regs[cfg.alpha_regid][3] = result;
        }
    }

    // Write to EFB (TEVPREV is the final output)
    if (px >= 0 && px < GX_EFB_WIDTH && py >= 0 && py < GX_EFB_HEIGHT) {
        EFBPixel& pixel = gx.efb[py][px];
        if (gx.color_update_enabled) {
            pixel.r = tev_regs[GX_TEVPREV][0];
            pixel.g = tev_regs[GX_TEVPREV][1];
            pixel.b = tev_regs[GX_TEVPREV][2];
        }
        if (gx.alpha_update_enabled) {
            pixel.a = tev_regs[GX_TEVPREV][3];
        }
    }
}

// Rasterize a quad (4 vertices) as a screen-aligned rectangle
static void rasterize_quad(const VertexData verts[4]) {
    // Find bounding box of the quad
    f32 min_x = verts[0].pos[0], max_x = verts[0].pos[0];
    f32 min_y = verts[0].pos[1], max_y = verts[0].pos[1];
    for (int i = 1; i < 4; i++) {
        min_x = std::min(min_x, verts[i].pos[0]);
        max_x = std::max(max_x, verts[i].pos[0]);
        min_y = std::min(min_y, verts[i].pos[1]);
        max_y = std::max(max_y, verts[i].pos[1]);
    }

    int ix0 = std::max(0, (int)min_x);
    int iy0 = std::max(0, (int)min_y);
    int ix1 = std::min(GX_EFB_WIDTH - 1, (int)max_x);
    int iy1 = std::min(GX_EFB_HEIGHT - 1, (int)max_y);

    f32 width = max_x - min_x;
    f32 height = max_y - min_y;
    if (width < 0.001f) width = 0.001f;
    if (height < 0.001f) height = 0.001f;

    for (int py = iy0; py <= iy1; py++) {
        for (int px = ix0; px <= ix1; px++) {
            // Compute normalized position within the quad for texcoord interpolation
            f32 u = (px - min_x) / width;
            f32 v = (py - min_y) / height;
            u = std::max(0.0f, std::min(1.0f, u));
            v = std::max(0.0f, std::min(1.0f, v));

            // Interpolate texcoords (bilinear between the 4 corners)
            // Assuming vertex order: 0=TL, 1=TR, 2=BR, 3=BL (or similar quad ordering)
            f32 texcoords[8][2];
            for (int t = 0; t < 8; t++) {
                if (verts[0].has_texcoord[t]) {
                    f32 top = verts[0].texcoord[t][0] * (1 - u) + verts[1].texcoord[t][0] * u;
                    f32 bot = verts[3].texcoord[t][0] * (1 - u) + verts[2].texcoord[t][0] * u;
                    texcoords[t][0] = top * (1 - v) + bot * v;

                    top = verts[0].texcoord[t][1] * (1 - u) + verts[1].texcoord[t][1] * u;
                    bot = verts[3].texcoord[t][1] * (1 - u) + verts[2].texcoord[t][1] * u;
                    texcoords[t][1] = top * (1 - v) + bot * v;
                } else {
                    texcoords[t][0] = u;
                    texcoords[t][1] = v;
                }
            }

            // Interpolate vertex color
            u8 ras_color[4];
            for (int c = 0; c < 4; c++) {
                f32 top = verts[0].color[c] * (1 - u) + verts[1].color[c] * u;
                f32 bot = verts[3].color[c] * (1 - u) + verts[2].color[c] * u;
                f32 val = top * (1 - v) + bot * v;
                ras_color[c] = clamp_u8((s16)(val + 0.5f));
            }

            process_fragment(px, py, ras_color, texcoords);
        }
    }
}

// Forward declarations
static void maybe_push_vertex();
static void flush_vertices();

// Flush accumulated vertices
static void flush_vertices() {
    if (gx.current_primitive == GX_QUADS) {
        // Process quads: every 4 vertices is one quad
        for (size_t i = 0; i + 3 < gx.vertex_buffer.size(); i += 4) {
            VertexData quad[4] = {
                gx.vertex_buffer[i],
                gx.vertex_buffer[i + 1],
                gx.vertex_buffer[i + 2],
                gx.vertex_buffer[i + 3]
            };
            rasterize_quad(quad);
        }
    }
    // Other primitive types can be added as needed
    gx.vertex_buffer.clear();
}

// Public API implementation

GXFifoObj* GX_Init(void* base, u32 size) {
    memset(&gx, 0, sizeof(gx));

    // Default TEV stage config (pass through rasterized color)
    for (int i = 0; i < MAX_TEV_STAGES; i++) {
        TEVStageConfig& s = gx.tev_stages[i];
        s.color_a = GX_CC_ZERO;
        s.color_b = GX_CC_ZERO;
        s.color_c = GX_CC_ZERO;
        s.color_d = GX_CC_RASC;
        s.alpha_a = GX_CA_ZERO;
        s.alpha_b = GX_CA_ZERO;
        s.alpha_c = GX_CA_ZERO;
        s.alpha_d = GX_CA_RASA;
        s.color_op = GX_TEV_ADD;
        s.alpha_op = GX_TEV_ADD;
        s.color_bias = GX_TB_ZERO;
        s.alpha_bias = GX_TB_ZERO;
        s.color_scale = GX_CS_SCALE_1;
        s.alpha_scale = GX_CS_SCALE_1;
        s.color_clamp = 1;
        s.alpha_clamp = 1;
        s.color_regid = GX_TEVPREV;
        s.alpha_regid = GX_TEVPREV;
        s.texcoord = GX_TEXCOORDNULL;
        s.texmap = GX_TEXMAP_NULL;
        s.colorchan = GX_COLOR0A0;
        s.kcsel = GX_TEV_KCSEL_1;
        s.kasel = GX_TEV_KASEL_1;
        s.ras_swap = 0;
        s.tex_swap = 0;
    }
    gx.num_tev_stages = 1;

    // Default swap tables (identity)
    for (int i = 0; i < MAX_SWAP_TABLES; i++) {
        gx.swap_tables[i] = {0, 1, 2, 3};
    }

    gx.color_update_enabled = true;
    gx.alpha_update_enabled = true;
    gx.num_texgens = 0;
    gx.num_color_chans = 1;

    // Clear EFB
    for (int y = 0; y < GX_EFB_HEIGHT; y++) {
        for (int x = 0; x < GX_EFB_WIDTH; x++) {
            gx.efb[y][x] = {0, 0, 0, 255, 0x00FFFFFF};
        }
    }

    gx.initialized = true;

    static GXFifoObj fifo;
    fifo.base = base;
    fifo.size = size;
    return &fifo;
}

void GX_SetCPUFifo(GXFifoObj*) {}
void GX_SetGPFifo(GXFifoObj*) {}

void GX_Flush(void) {}

void GX_DrawDone(void) {
    flush_vertices();
}

void GX_ClearVtxDesc(void) {
    gx.vtx_has_pos = false;
    gx.vtx_has_color = false;
    for (int i = 0; i < 8; i++) gx.vtx_has_texcoord[i] = false;
}

void GX_SetVtxDesc(u8 attr, u8 type) {
    bool enabled = (type != GX_NONE);
    if (attr == GX_VA_POS) gx.vtx_has_pos = enabled;
    else if (attr == GX_VA_CLR0) gx.vtx_has_color = enabled;
    else if (attr >= GX_VA_TEX0 && attr <= GX_VA_TEX7)
        gx.vtx_has_texcoord[attr - GX_VA_TEX0] = enabled;
}

void GX_SetVtxAttrFmt(u8, u32, u32, u32, u32) {}

void GX_SetNumTevStages(u8 num) {
    gx.num_tev_stages = num;
}

void GX_SetTevOp(u8 tevstage, u8 mode) {
    TEVStageConfig& s = gx.tev_stages[tevstage];
    switch (mode) {
        case GX_REPLACE:
            s.color_a = GX_CC_ZERO; s.color_b = GX_CC_ZERO;
            s.color_c = GX_CC_ZERO; s.color_d = GX_CC_TEXC;
            s.alpha_a = GX_CA_ZERO; s.alpha_b = GX_CA_ZERO;
            s.alpha_c = GX_CA_ZERO; s.alpha_d = GX_CA_TEXA;
            break;
        case GX_MODULATE:
            s.color_a = GX_CC_ZERO; s.color_b = GX_CC_TEXC;
            s.color_c = GX_CC_RASC; s.color_d = GX_CC_ZERO;
            s.alpha_a = GX_CA_ZERO; s.alpha_b = GX_CA_TEXA;
            s.alpha_c = GX_CA_RASA; s.alpha_d = GX_CA_ZERO;
            break;
        case GX_DECAL:
            s.color_a = GX_CC_RASC; s.color_b = GX_CC_TEXC;
            s.color_c = GX_CC_TEXA; s.color_d = GX_CC_ZERO;
            s.alpha_a = GX_CA_ZERO; s.alpha_b = GX_CA_ZERO;
            s.alpha_c = GX_CA_ZERO; s.alpha_d = GX_CA_RASA;
            break;
        case GX_BLEND:
            s.color_a = GX_CC_RASC; s.color_b = GX_CC_TEXC;
            s.color_c = GX_CC_TEXA; s.color_d = GX_CC_ZERO;
            s.alpha_a = GX_CA_ZERO; s.alpha_b = GX_CA_TEXA;
            s.alpha_c = GX_CA_RASA; s.alpha_d = GX_CA_ZERO;
            break;
        case GX_PASSCLR:
            s.color_a = GX_CC_ZERO; s.color_b = GX_CC_ZERO;
            s.color_c = GX_CC_ZERO; s.color_d = GX_CC_RASC;
            s.alpha_a = GX_CA_ZERO; s.alpha_b = GX_CA_ZERO;
            s.alpha_c = GX_CA_ZERO; s.alpha_d = GX_CA_RASA;
            break;
    }
    s.color_op = GX_TEV_ADD;
    s.color_bias = GX_TB_ZERO;
    s.color_scale = GX_CS_SCALE_1;
    s.color_clamp = 1;
    s.color_regid = GX_TEVPREV;
    s.alpha_op = GX_TEV_ADD;
    s.alpha_bias = GX_TB_ZERO;
    s.alpha_scale = GX_CS_SCALE_1;
    s.alpha_clamp = 1;
    s.alpha_regid = GX_TEVPREV;
}

void GX_SetTevOrder(u8 tevstage, u8 texcoord, u32 texmap, u8 color) {
    gx.tev_stages[tevstage].texcoord = texcoord;
    gx.tev_stages[tevstage].texmap = texmap;
    gx.tev_stages[tevstage].colorchan = color;
}

void GX_SetTevColorIn(u8 tevstage, u8 a, u8 b, u8 c, u8 d) {
    gx.tev_stages[tevstage].color_a = a;
    gx.tev_stages[tevstage].color_b = b;
    gx.tev_stages[tevstage].color_c = c;
    gx.tev_stages[tevstage].color_d = d;
}

void GX_SetTevAlphaIn(u8 tevstage, u8 a, u8 b, u8 c, u8 d) {
    gx.tev_stages[tevstage].alpha_a = a;
    gx.tev_stages[tevstage].alpha_b = b;
    gx.tev_stages[tevstage].alpha_c = c;
    gx.tev_stages[tevstage].alpha_d = d;
}

void GX_SetTevColorOp(u8 tevstage, u8 tevop, u8 tevbias, u8 tevscale, u8 clamp, u8 tevregid) {
    TEVStageConfig& s = gx.tev_stages[tevstage];
    s.color_op = tevop;
    s.color_bias = tevbias;
    s.color_scale = tevscale;
    s.color_clamp = clamp;
    s.color_regid = tevregid;
}

void GX_SetTevAlphaOp(u8 tevstage, u8 tevop, u8 tevbias, u8 tevscale, u8 clamp, u8 tevregid) {
    TEVStageConfig& s = gx.tev_stages[tevstage];
    s.alpha_op = tevop;
    s.alpha_bias = tevbias;
    s.alpha_scale = tevscale;
    s.alpha_clamp = clamp;
    s.alpha_regid = tevregid;
}

void GX_SetTevColor(u8 tev_regid, GXColor color) {
    gx.tev_regs[tev_regid].r = color.r;
    gx.tev_regs[tev_regid].g = color.g;
    gx.tev_regs[tev_regid].b = color.b;
    gx.tev_regs[tev_regid].a = color.a;
}

void GX_SetTevColorS10(u8 tev_regid, GXColorS10 color) {
    gx.tev_regs[tev_regid] = color;
}

void GX_SetTevKColor(u8 sel, GXColor col) {
    if (sel < MAX_KONST_COLORS) {
        gx.konst_colors[sel] = col;
    }
}

void GX_SetTevKColorSel(u8 tevstage, u8 sel) {
    gx.tev_stages[tevstage].kcsel = sel;
}

void GX_SetTevKAlphaSel(u8 tevstage, u8 sel) {
    gx.tev_stages[tevstage].kasel = sel;
}

void GX_SetTevSwapMode(u8 tevstage, u8 ras_sel, u8 tex_sel) {
    gx.tev_stages[tevstage].ras_swap = ras_sel;
    gx.tev_stages[tevstage].tex_swap = tex_sel;
}

void GX_SetTevSwapModeTable(u8 swapid, u8 r, u8 g, u8 b, u8 a) {
    if (swapid < MAX_SWAP_TABLES) {
        gx.swap_tables[swapid] = {r, g, b, a};
    }
}

void GX_SetTevDirect(u8) {}

void GX_InitTexObj(GXTexObj* obj, void* img_ptr, u16 wd, u16 ht, u8 fmt,
                   u8 wrap_s, u8 wrap_t, u8 mipmap) {
    obj->img_ptr = img_ptr;
    obj->width = wd;
    obj->height = ht;
    obj->fmt = fmt;
    obj->wrap_s = wrap_s;
    obj->wrap_t = wrap_t;
    obj->mipmap = mipmap;
}

void GX_LoadTexObj(GXTexObj* obj, u8 mapid) {
    if (mapid < MAX_TEXMAPS) {
        TextureState& ts = gx.textures[mapid];
        ts.img_ptr = obj->img_ptr;
        ts.width = obj->width;
        ts.height = obj->height;
        ts.fmt = obj->fmt;
        ts.wrap_s = obj->wrap_s;
        ts.wrap_t = obj->wrap_t;
        ts.loaded = true;
    }
}

void GX_InvalidateTexAll(void) {
    for (int i = 0; i < MAX_TEXMAPS; i++) {
        gx.textures[i].loaded = false;
    }
}

u32 GX_GetTexBufferSize(u16 wd, u16 ht, u32 fmt, u8, u8) {
    u32 bpp = 4;  // bytes per pixel
    switch (fmt) {
        case GX_TF_I8:  bpp = 1; break;
        case GX_TF_IA8: bpp = 2; break;
        case GX_TF_RGB565: bpp = 2; break;
        case GX_TF_RGB5A3: bpp = 2; break;
        case GX_TF_RGBA8: bpp = 4; break;
        default: bpp = 4; break;
    }
    return wd * ht * bpp;
}

void GX_SetNumTexGens(u32 nr) { gx.num_texgens = nr; }
void GX_SetTexCoordGen(u16, u32, u32, u32) {}
void GX_SetNumChans(u8 num) { gx.num_color_chans = num; }

void GX_Begin(u8 primitive, u8 vtxfmt, u16 vtxcnt) {
    gx.current_primitive = primitive;
    gx.current_vtxfmt = vtxfmt;
    gx.expected_vtx_count = vtxcnt;
    gx.vertex_buffer.clear();
    gx.vertex_buffer.reserve(vtxcnt);
    memset(&gx.current_vertex, 0, sizeof(gx.current_vertex));
}

void GX_End(void) {
    // Push the final in-progress vertex before flushing
    maybe_push_vertex();
    flush_vertices();
}

// Vertex submission: accumulates per-vertex data, pushes when complete.
// The "completion" heuristic: position always comes first, so when we get
// a new position call and we already have one, push the previous vertex.

static void maybe_push_vertex() {
    // We require at least a position to form a valid vertex
    if (gx.current_vertex.has_pos) {
        // Set defaults for missing attributes
        if (!gx.current_vertex.has_color) {
            gx.current_vertex.color[0] = 255;
            gx.current_vertex.color[1] = 255;
            gx.current_vertex.color[2] = 255;
            gx.current_vertex.color[3] = 255;
        }
        gx.vertex_buffer.push_back(gx.current_vertex);
    }
}

void GX_Position3f32(f32 x, f32 y, f32 z) {
    maybe_push_vertex();
    memset(&gx.current_vertex, 0, sizeof(gx.current_vertex));
    gx.current_vertex.pos[0] = x;
    gx.current_vertex.pos[1] = y;
    gx.current_vertex.pos[2] = z;
    gx.current_vertex.has_pos = true;
}

void GX_Position2f32(f32 x, f32 y) {
    GX_Position3f32(x, y, 0.0f);
}

void GX_Color4u8(u8 r, u8 g, u8 b, u8 a) {
    gx.current_vertex.color[0] = r;
    gx.current_vertex.color[1] = g;
    gx.current_vertex.color[2] = b;
    gx.current_vertex.color[3] = a;
    gx.current_vertex.has_color = true;
}

void GX_Color1u32(u32 clr) {
    GX_Color4u8((clr >> 24) & 0xFF, (clr >> 16) & 0xFF,
                (clr >> 8) & 0xFF, clr & 0xFF);
}

void GX_TexCoord2f32(f32 s, f32 t) {
    // Find the first unused texcoord slot for this vertex
    for (int i = 0; i < 8; i++) {
        if (!gx.current_vertex.has_texcoord[i]) {
            gx.current_vertex.texcoord[i][0] = s;
            gx.current_vertex.texcoord[i][1] = t;
            gx.current_vertex.has_texcoord[i] = true;
            break;
        }
    }
}

void GX_Normal3f32(f32, f32, f32) {}

// EFB operations

void GX_SetPixelFmt(u8, u8) {}

void GX_SetCopyClear(GXColor color, u32 zvalue) {
    gx.clear_color = color;
    gx.clear_z = zvalue;
}

void GX_SetTexCopySrc(u16 left, u16 top, u16 wd, u16 ht) {
    gx.tex_copy_src = {left, top, wd, ht};
}

void GX_SetTexCopyDst(u16 wd, u16 ht, u32 fmt, u8) {
    gx.tex_copy_dst_wd = wd;
    gx.tex_copy_dst_ht = ht;
    gx.tex_copy_dst_fmt = fmt;
}

void GX_CopyTex(void* dest, u8 clear) {
    // Copy EFB region to a texture buffer in memory
    u16 sx = gx.tex_copy_src.left;
    u16 sy = gx.tex_copy_src.top;
    u16 sw = gx.tex_copy_src.width;
    u16 sh = gx.tex_copy_src.height;

    // For RGBA8 format, copy as 4 bytes per pixel
    u8* dst = (u8*)dest;
    for (u16 y = 0; y < sh && (sy + y) < GX_EFB_HEIGHT; y++) {
        for (u16 x = 0; x < sw && (sx + x) < GX_EFB_WIDTH; x++) {
            const EFBPixel& p = gx.efb[sy + y][sx + x];
            u32 idx = (y * sw + x) * 4;
            dst[idx + 0] = clamp_u8(p.r);
            dst[idx + 1] = clamp_u8(p.g);
            dst[idx + 2] = clamp_u8(p.b);
            dst[idx + 3] = clamp_u8(p.a);
        }
    }

    if (clear) {
        for (u16 y = 0; y < sh && (sy + y) < GX_EFB_HEIGHT; y++) {
            for (u16 x = 0; x < sw && (sx + x) < GX_EFB_WIDTH; x++) {
                gx.efb[sy + y][sx + x] = {
                    (s16)gx.clear_color.r, (s16)gx.clear_color.g,
                    (s16)gx.clear_color.b, (s16)gx.clear_color.a,
                    gx.clear_z
                };
            }
        }
    }
}

void GX_SetDispCopySrc(u16, u16, u16, u16) {}
void GX_SetDispCopyDst(u16, u16) {}
void GX_SetDispCopyYScale(f32) {}
void GX_CopyDisp(void*, u8) {}
void GX_PixModeSync(void) {}

void GX_PokeARGB(u16 x, u16 y, GXColor color) {
    if (x < GX_EFB_WIDTH && y < GX_EFB_HEIGHT) {
        gx.efb[y][x].r = color.r;
        gx.efb[y][x].g = color.g;
        gx.efb[y][x].b = color.b;
        gx.efb[y][x].a = color.a;
    }
}

void GX_PeekARGB(u16 x, u16 y, GXColor* color) {
    if (x < GX_EFB_WIDTH && y < GX_EFB_HEIGHT) {
        color->r = clamp_u8(gx.efb[y][x].r);
        color->g = clamp_u8(gx.efb[y][x].g);
        color->b = clamp_u8(gx.efb[y][x].b);
        color->a = clamp_u8(gx.efb[y][x].a);
    } else {
        color->r = color->g = color->b = color->a = 0;
    }
}

void GX_PokeZ(u16 x, u16 y, u32 z) {
    if (x < GX_EFB_WIDTH && y < GX_EFB_HEIGHT) {
        gx.efb[y][x].z = z;
    }
}

void GX_PeekZ(u16 x, u16 y, u32* z) {
    if (x < GX_EFB_WIDTH && y < GX_EFB_HEIGHT) {
        *z = gx.efb[y][x].z;
    } else {
        *z = 0;
    }
}

// Stubs for render state (not critical for compute)
void GX_SetZMode(u8, u8, u8) {}
void GX_SetZCompLoc(u8) {}
void GX_SetBlendMode(u8, u8, u8, u8) {}
void GX_SetColorUpdate(u8 enable) { gx.color_update_enabled = (enable != 0); }
void GX_SetAlphaUpdate(u8 enable) { gx.alpha_update_enabled = (enable != 0); }
void GX_SetAlphaCompare(u8, u8, u8, u8, u8) {}
void GX_SetDither(u8) {}
