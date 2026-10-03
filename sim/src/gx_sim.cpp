// GX Software Simulator
// Implements the subset of libogc's GX API needed for compute-via-TEV.
// Covers: TEV combiner pipeline, EFB read/write with blending, texture sampling,
// indirect texture coordinate manipulation, XF texgen matrix transforms,
// and full-screen quad rasterization.

#include "gx.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef GX_SIM_TIMING
#include <chrono>
#include <thread>
static void sim_sleep_us(int us) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}
static const int COPYTEX_LATENCY_US  = 500;  // ~0.5 ms pipeline flush + DMA
static const int DRAWDONE_LATENCY_US = 100;  // ~0.1 ms pipeline drain
#else
static void sim_sleep_us(int) {}
static const int COPYTEX_LATENCY_US  = 0;
static const int DRAWDONE_LATENCY_US = 0;
#endif

// Internal state

static const int MAX_TEV_STAGES = 16;
static const int MAX_TEXMAPS = 8;
static const int MAX_KONST_COLORS = 4;
static const int MAX_TEV_REGS = 4;   // TEVPREV + TEVREG0..2
static const int MAX_SWAP_TABLES = 4;
static const int MAX_TEXGENS = 8;
static const int MAX_TEXMTX = 10;
static const int MAX_PNMTX = 10;
static const int MAX_IND_STAGES = 4;
static const int MAX_IND_MTX = 3;

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
    // Indirect texture config
    u8 ind_stage;
    u8 ind_format;
    u8 ind_bias_sel;
    u8 ind_matrix_sel;
    u8 ind_wrap_s, ind_wrap_t;
    u8 ind_add_prev;
    u8 ind_utc_lod;
    u8 ind_alpha_sel;
    bool ind_enable;
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
    // EFB stores 8-bit per channel after TEV output clamping, matching real hardware.
    // TEV internal precision is s10, but once written to EFB it's u8.
    u8 r, g, b, a;
    u32 z;
};

struct TexCopyRegion {
    u16 left, top, width, height;
};

struct VertexData {
    f32 pos[3];
    f32 normal[3];
    u8 color[4];
    f32 texcoord[8][2];  // up to 8 texcoord slots
    bool has_pos;
    bool has_normal;
    bool has_color;
    bool has_texcoord[8];
};

struct TexGenConfig {
    u32 tgen_type;  // GX_TG_MTX2x4 or GX_TG_MTX3x4
    u32 tgen_src;   // GX_TG_POS, GX_TG_NRM, GX_TG_TEX0..7
    u32 mtx_src;    // GX_TEXMTX0..9 or GX_IDENTITY
    bool active;
};

// 4x3 matrix stored as float[3][4] (3 rows, 4 columns) matching libogc Mtx format
// For a 2x4 texgen, only rows 0-1 are used, producing 2 output components (s, t).
// For a 3x4 texgen, rows 0-2 produce 3 components (s, t, q).
struct TexMatrix {
    f32 m[3][4];
    bool loaded;
};

struct PosNrmMatrix {
    f32 m[3][4];
    bool loaded;
};

struct BlendState {
    u8 mode;       // GX_BM_NONE, GX_BM_BLEND, GX_BM_LOGIC, GX_BM_SUBTRACT
    u8 src_factor; // GX_BL_*
    u8 dst_factor; // GX_BL_*
    u8 logic_op;   // GX_LO_*
};

struct IndTexStageConfig {
    u8 texcoord;
    u8 texmap;
    u8 scale_s;
    u8 scale_t;
};

struct IndTexMatrix {
    f32 m[2][3];   // 2x3 offset matrix
    s8 scale_exp;  // power-of-2 scale exponent
    bool loaded;
};

// A draw command captured at GX_End() time. On real hardware the CPU writes
// state changes and vertex data into the GX FIFO; the GPU processes them
// asynchronously. We model this by snapshotting all pipeline state into a
// DrawCommand, then processing the queue when GX_DrawDone() is called.
struct DrawCommand {
    u8 num_tev_stages;
    TEVStageConfig tev_stages[MAX_TEV_STAGES];
    GXColorS10 tev_regs[MAX_TEV_REGS];
    GXColor konst_colors[MAX_KONST_COLORS];
    SwapTable swap_tables[MAX_SWAP_TABLES];
    TextureState textures[MAX_TEXMAPS];
    u32 num_texgens;
    TexGenConfig texgens[MAX_TEXGENS];
    TexMatrix tex_matrices[MAX_TEXMTX];
    u8 num_ind_stages;
    IndTexStageConfig ind_stages[MAX_IND_STAGES];
    IndTexMatrix ind_matrices[MAX_IND_MTX];
    BlendState blend;
    bool color_update_enabled;
    bool alpha_update_enabled;
    u8 primitive_type;
    std::vector<VertexData> vertices;
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
    bool vtx_has_normal;
    bool vtx_has_color;
    bool vtx_has_texcoord[8];

    // Color update
    bool color_update_enabled;
    bool alpha_update_enabled;

    // Texgen
    u32 num_texgens;
    TexGenConfig texgens[MAX_TEXGENS];

    // XF matrices
    TexMatrix tex_matrices[MAX_TEXMTX];
    PosNrmMatrix pn_matrices[MAX_PNMTX];
    u32 current_pnmtx;

    // Color channels
    u8 num_color_chans;

    // Blend state
    BlendState blend;

    // Indirect texture
    u8 num_ind_stages;
    IndTexStageConfig ind_stages[MAX_IND_STAGES];
    IndTexMatrix ind_matrices[MAX_IND_MTX];

    // FIFO draw queue: commands queued by GX_End(), flushed by GX_DrawDone()
    std::vector<DrawCommand> draw_queue;

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

// Resolve a blend factor to a multiplier in [0..255] range for one channel.
// src_val is the fragment (source) value, dst_val is the existing EFB value.
static u8 resolve_blend_factor(u8 factor, u8 src_r, u8 src_g, u8 src_b, u8 src_a,
                               u8 dst_r, u8 dst_g, u8 dst_b, u8 dst_a,
                               int channel) {
    // channel: 0=R, 1=G, 2=B, 3=A
    u8 src_ch[4] = {src_r, src_g, src_b, src_a};
    u8 dst_ch[4] = {dst_r, dst_g, dst_b, dst_a};
    switch (factor) {
        case GX_BL_ZERO:        return 0;
        case GX_BL_ONE:         return 255;
        // GX_BL_SRCCLR / GX_BL_DSTCLR share the same encoding (2)
        // Context determines meaning: as dst_factor it's SRCCLR, as src_factor it's DSTCLR
        // We handle both the same way since the caller passes the right value.
        case GX_BL_SRCCLR:      return src_ch[channel];  // also GX_BL_DSTCLR
        case GX_BL_INVSRCCLR:   return 255 - src_ch[channel]; // also GX_BL_INVDSTCLR
        case GX_BL_SRCALPHA:    return src_a;
        case GX_BL_INVSRCALPHA: return 255 - src_a;
        case GX_BL_DSTALPHA:    return dst_a;
        case GX_BL_INVDSTALPHA: return 255 - dst_a;
        default:                return 255;
    }
}

// Apply EFB blending. fragment is the TEV output (clamped to 0..255).
// Reads the existing EFB pixel, applies blend, writes back.
static void blend_and_write(int px, int py, u8 frag_r, u8 frag_g, u8 frag_b, u8 frag_a) {
    if (px < 0 || px >= GX_EFB_WIDTH || py < 0 || py >= GX_EFB_HEIGHT) return;

    EFBPixel& pixel = gx.efb[py][px];

    if (gx.blend.mode == GX_BM_NONE) {
        // No blending: overwrite
        if (gx.color_update_enabled) {
            pixel.r = frag_r;
            pixel.g = frag_g;
            pixel.b = frag_b;
        }
        if (gx.alpha_update_enabled) {
            pixel.a = frag_a;
        }
        return;
    }

    u8 dst_r = clamp_u8(pixel.r);
    u8 dst_g = clamp_u8(pixel.g);
    u8 dst_b = clamp_u8(pixel.b);
    u8 dst_a = clamp_u8(pixel.a);

    if (gx.blend.mode == GX_BM_BLEND) {
        // out = src * src_factor + dst * dst_factor, per channel, clamped to [0, 255]
        for (int ch = 0; ch < 4; ch++) {
            bool update = (ch < 3) ? gx.color_update_enabled : gx.alpha_update_enabled;
            if (!update) continue;

            u8 src_val = (ch == 0) ? frag_r : (ch == 1) ? frag_g : (ch == 2) ? frag_b : frag_a;
            u8 dst_val = (ch == 0) ? dst_r : (ch == 1) ? dst_g : (ch == 2) ? dst_b : dst_a;

            u8 sf = resolve_blend_factor(gx.blend.src_factor,
                                         frag_r, frag_g, frag_b, frag_a,
                                         dst_r, dst_g, dst_b, dst_a, ch);
            u8 df = resolve_blend_factor(gx.blend.dst_factor,
                                         frag_r, frag_g, frag_b, frag_a,
                                         dst_r, dst_g, dst_b, dst_a, ch);

            // Fixed-point multiply: (val * factor + 128) >> 8
            s32 blended = ((s32)src_val * sf + 128) / 256 + ((s32)dst_val * df + 128) / 256;
            if (blended < 0) blended = 0;
            if (blended > 255) blended = 255;

            u8* dest = (ch == 0) ? &pixel.r : (ch == 1) ? &pixel.g : (ch == 2) ? &pixel.b : &pixel.a;
            *dest = (u8)blended;
        }
        return;
    }

    if (gx.blend.mode == GX_BM_SUBTRACT) {
        // out = dst - src, clamped to [0, 255]
        if (gx.color_update_enabled) {
            pixel.r = (u8)std::max(0, (int)dst_r - (int)frag_r);
            pixel.g = (u8)std::max(0, (int)dst_g - (int)frag_g);
            pixel.b = (u8)std::max(0, (int)dst_b - (int)frag_b);
        }
        if (gx.alpha_update_enabled) {
            pixel.a = (u8)std::max(0, (int)dst_a - (int)frag_a);
        }
        return;
    }

    if (gx.blend.mode == GX_BM_LOGIC) {
        // Logic ops applied per byte
        auto apply_logic = [](u8 src, u8 dst, u8 op) -> u8 {
            switch (op) {
                case GX_LO_CLEAR:  return 0;
                case GX_LO_AND:    return src & dst;
                case GX_LO_REVAND: return src & ~dst;
                case GX_LO_COPY:   return src;
                case GX_LO_INVAND: return ~src & dst;
                case GX_LO_NOOP:   return dst;
                case GX_LO_XOR:    return src ^ dst;
                case GX_LO_OR:     return src | dst;
                case GX_LO_NOR:    return ~(src | dst);
                case GX_LO_EQUIV:  return ~(src ^ dst);
                case GX_LO_INV:    return ~dst;
                case GX_LO_REVOR:  return src | ~dst;
                case GX_LO_INVCOPY: return ~src;
                case GX_LO_INVOR:  return ~src | dst;
                case GX_LO_NAND:   return ~(src & dst);
                case GX_LO_SET:    return 0xFF;
                default:           return src;
            }
        };
        if (gx.color_update_enabled) {
            pixel.r = apply_logic(frag_r, dst_r, gx.blend.logic_op);
            pixel.g = apply_logic(frag_g, dst_g, gx.blend.logic_op);
            pixel.b = apply_logic(frag_b, dst_b, gx.blend.logic_op);
        }
        if (gx.alpha_update_enabled) {
            pixel.a = apply_logic(frag_a, dst_a, gx.blend.logic_op);
        }
        return;
    }

    // Unknown mode: overwrite
    if (gx.color_update_enabled) {
        pixel.r = frag_r;
        pixel.g = frag_g;
        pixel.b = frag_b;
    }
    if (gx.alpha_update_enabled) {
        pixel.a = frag_a;
    }
}

// Apply indirect texture coordinate offset for a TEV stage.
// Modifies the texture coordinates in-place before the main texture fetch.
static void apply_indirect_texcoord(const TEVStageConfig& cfg,
                                    const f32 texcoords[8][2],
                                    f32* out_s, f32* out_t) {
    if (!cfg.ind_enable) {
        *out_s = texcoords[cfg.texcoord][0];
        *out_t = texcoords[cfg.texcoord][1];
        return;
    }

    const IndTexStageConfig& ist = gx.ind_stages[cfg.ind_stage];

    // Sample the indirect texture
    u8 ind_texel[4] = {0, 0, 0, 0};
    if (ist.texmap < MAX_TEXMAPS) {
        f32 ind_s = texcoords[ist.texcoord][0];
        f32 ind_t = texcoords[ist.texcoord][1];
        sample_texture(ist.texmap, ind_s, ind_t, ind_texel);
    }

    // Extract offset components based on format (number of bits to use)
    // GX_ITF_8: full 8-bit, GX_ITF_5: top 5 bits, etc.
    s32 ind_offset[3]; // S, T, U components
    int shift = 0;
    switch (cfg.ind_format) {
        case GX_ITF_8: shift = 0; break;
        case GX_ITF_5: shift = 3; break;
        case GX_ITF_4: shift = 4; break;
        case GX_ITF_3: shift = 5; break;
    }
    // Convert unsigned texel values to signed offsets (centered at 128)
    ind_offset[0] = ((s32)ind_texel[0] >> shift) - (128 >> shift);
    ind_offset[1] = ((s32)ind_texel[1] >> shift) - (128 >> shift);
    ind_offset[2] = ((s32)ind_texel[2] >> shift) - (128 >> shift);

    // Apply bias
    if (!(cfg.ind_bias_sel & 1)) ind_offset[0] = 0; // S bias disabled
    if (!(cfg.ind_bias_sel & 2)) ind_offset[1] = 0; // T bias disabled
    if (!(cfg.ind_bias_sel & 4)) ind_offset[2] = 0; // U bias disabled

    f32 coord_s = texcoords[cfg.texcoord][0];
    f32 coord_t = texcoords[cfg.texcoord][1];

    // Apply indirect matrix transform if selected
    if (cfg.ind_matrix_sel >= GX_ITM_0 && cfg.ind_matrix_sel <= GX_ITM_2) {
        int mtx_idx = cfg.ind_matrix_sel - GX_ITM_0;
        if (mtx_idx < MAX_IND_MTX && gx.ind_matrices[mtx_idx].loaded) {
            const IndTexMatrix& mtx = gx.ind_matrices[mtx_idx];
            f32 scale = ldexpf(1.0f, mtx.scale_exp);
            f32 offs_s = (mtx.m[0][0] * ind_offset[0] +
                          mtx.m[0][1] * ind_offset[1] +
                          mtx.m[0][2] * ind_offset[2]) * scale;
            f32 offs_t = (mtx.m[1][0] * ind_offset[0] +
                          mtx.m[1][1] * ind_offset[1] +
                          mtx.m[1][2] * ind_offset[2]) * scale;
            coord_s += offs_s;
            coord_t += offs_t;
        }
    }

    *out_s = coord_s;
    *out_t = coord_t;
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

        // Sample texture for this stage (with optional indirect texcoord offset)
        u8 tex_color_raw[4] = {0, 0, 0, 0};
        if (cfg.texmap != GX_TEXMAP_NULL && cfg.texcoord != GX_TEXCOORDNULL) {
            f32 s, t;
            apply_indirect_texcoord(cfg, texcoords, &s, &t);
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

    // Write to EFB through the blend unit (TEVPREV is the final output)
    u8 frag_r = clamp_u8(tev_regs[GX_TEVPREV][0]);
    u8 frag_g = clamp_u8(tev_regs[GX_TEVPREV][1]);
    u8 frag_b = clamp_u8(tev_regs[GX_TEVPREV][2]);
    u8 frag_a = clamp_u8(tev_regs[GX_TEVPREV][3]);

    blend_and_write(px, py, frag_r, frag_g, frag_b, frag_a);
}

// Convert a texgen matrix slot ID (GX_TEXMTX0..GX_TEXMTX9, GX_IDENTITY) to
// an index into our tex_matrices array. Returns -1 for GX_IDENTITY.
static int texmtx_to_index(u32 mtxsrc) {
    if (mtxsrc == GX_IDENTITY) return -1;
    if (mtxsrc >= GX_TEXMTX0 && mtxsrc <= GX_TEXMTX9) {
        return (mtxsrc - GX_TEXMTX0) / 3;
    }
    return -1;
}

// Apply XF texgen transforms to a vertex's attributes, producing the
// per-vertex texture coordinates that will be interpolated by the rasterizer.
static void apply_texgens(const VertexData& vtx, f32 out_texcoords[8][2]) {
    for (u32 tg = 0; tg < gx.num_texgens && tg < MAX_TEXGENS; tg++) {
        const TexGenConfig& gen = gx.texgens[tg];
        if (!gen.active) {
            // Use vertex-submitted texcoords as-is
            out_texcoords[tg][0] = vtx.has_texcoord[tg] ? vtx.texcoord[tg][0] : 0.0f;
            out_texcoords[tg][1] = vtx.has_texcoord[tg] ? vtx.texcoord[tg][1] : 0.0f;
            continue;
        }

        // Determine input vector based on tgen_src
        f32 input[4] = {0, 0, 0, 1}; // x, y, z, w=1
        switch (gen.tgen_src) {
            case GX_TG_POS:
                input[0] = vtx.pos[0];
                input[1] = vtx.pos[1];
                input[2] = vtx.pos[2];
                break;
            case GX_TG_NRM:
                input[0] = vtx.normal[0];
                input[1] = vtx.normal[1];
                input[2] = vtx.normal[2];
                break;
            default:
                // GX_TG_TEX0..GX_TG_TEX7
                if (gen.tgen_src >= GX_TG_TEX0 && gen.tgen_src <= GX_TG_TEX3) {
                    int ti = gen.tgen_src - GX_TG_TEX0;
                    if (vtx.has_texcoord[ti]) {
                        input[0] = vtx.texcoord[ti][0];
                        input[1] = vtx.texcoord[ti][1];
                    }
                    input[2] = 1.0f;
                }
                break;
        }

        int mtx_idx = texmtx_to_index(gen.mtx_src);
        if (mtx_idx < 0 || !gx.tex_matrices[mtx_idx].loaded) {
            // Identity: pass through input as texcoords
            out_texcoords[tg][0] = input[0];
            out_texcoords[tg][1] = input[1];
        } else {
            // Matrix * input vector
            const f32 (*m)[4] = gx.tex_matrices[mtx_idx].m;
            if (gen.tgen_type == GX_TG_MTX2x4) {
                // 2x4 matrix: 2 output components (s, t)
                out_texcoords[tg][0] = m[0][0]*input[0] + m[0][1]*input[1] + m[0][2]*input[2] + m[0][3]*input[3];
                out_texcoords[tg][1] = m[1][0]*input[0] + m[1][1]*input[1] + m[1][2]*input[2] + m[1][3]*input[3];
            } else {
                // 3x4 matrix: 3 output components (s, t, q). We store s, t.
                // q could be used for perspective correction, not modeled here.
                out_texcoords[tg][0] = m[0][0]*input[0] + m[0][1]*input[1] + m[0][2]*input[2] + m[0][3]*input[3];
                out_texcoords[tg][1] = m[1][0]*input[0] + m[1][1]*input[1] + m[1][2]*input[2] + m[1][3]*input[3];
            }
        }
    }

    // For any remaining texcoord slots beyond num_texgens, pass through vertex data
    for (u32 tg = gx.num_texgens; tg < 8; tg++) {
        out_texcoords[tg][0] = vtx.has_texcoord[tg] ? vtx.texcoord[tg][0] : 0.0f;
        out_texcoords[tg][1] = vtx.has_texcoord[tg] ? vtx.texcoord[tg][1] : 0.0f;
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

    // Pre-compute texgen outputs for each corner vertex
    f32 corner_tc[4][8][2];
    for (int v = 0; v < 4; v++) {
        apply_texgens(verts[v], corner_tc[v]);
    }

    for (int py = iy0; py <= iy1; py++) {
        for (int px = ix0; px <= ix1; px++) {
            // Compute normalized position within the quad for interpolation
            f32 u = (px - min_x) / width;
            f32 v = (py - min_y) / height;
            u = std::max(0.0f, std::min(1.0f, u));
            v = std::max(0.0f, std::min(1.0f, v));

            // Interpolate texcoords (bilinear between the 4 corners)
            // Vertex order: 0=TL, 1=TR, 2=BR, 3=BL
            f32 texcoords[8][2];
            for (int t = 0; t < 8; t++) {
                f32 top_s = corner_tc[0][t][0] * (1 - u) + corner_tc[1][t][0] * u;
                f32 bot_s = corner_tc[3][t][0] * (1 - u) + corner_tc[2][t][0] * u;
                texcoords[t][0] = top_s * (1 - v) + bot_s * v;

                f32 top_t = corner_tc[0][t][1] * (1 - u) + corner_tc[1][t][1] * u;
                f32 bot_t = corner_tc[3][t][1] * (1 - u) + corner_tc[2][t][1] * u;
                texcoords[t][1] = top_t * (1 - v) + bot_t * v;
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

// Default-initialize a TEV stage config
static void init_tev_stage(TEVStageConfig& s) {
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
    s.ind_stage = 0;
    s.ind_format = GX_ITF_8;
    s.ind_bias_sel = GX_ITB_NONE;
    s.ind_matrix_sel = GX_ITM_OFF;
    s.ind_wrap_s = GX_ITW_OFF;
    s.ind_wrap_t = GX_ITW_OFF;
    s.ind_add_prev = 0;
    s.ind_utc_lod = 0;
    s.ind_alpha_sel = GX_ITBA_OFF;
    s.ind_enable = false;
}

// Public API implementation

GXFifoObj* GX_Init(void* base, u32 size) {
    memset(&gx, 0, sizeof(gx));

    // Default TEV stage config (pass through rasterized color)
    for (int i = 0; i < MAX_TEV_STAGES; i++) {
        init_tev_stage(gx.tev_stages[i]);
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

    // Default blend: none (overwrite)
    gx.blend.mode = GX_BM_NONE;
    gx.blend.src_factor = GX_BL_ONE;
    gx.blend.dst_factor = GX_BL_ZERO;
    gx.blend.logic_op = GX_LO_COPY;

    // Clear XF state
    for (int i = 0; i < MAX_TEXGENS; i++) {
        gx.texgens[i].active = false;
    }
    for (int i = 0; i < MAX_TEXMTX; i++) {
        gx.tex_matrices[i].loaded = false;
    }
    for (int i = 0; i < MAX_PNMTX; i++) {
        gx.pn_matrices[i].loaded = false;
    }
    gx.current_pnmtx = 0;

    // Clear indirect texture state
    gx.num_ind_stages = 0;
    for (int i = 0; i < MAX_IND_STAGES; i++) {
        gx.ind_stages[i] = {GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_ITS_1, GX_ITS_1};
    }
    for (int i = 0; i < MAX_IND_MTX; i++) {
        memset(&gx.ind_matrices[i], 0, sizeof(IndTexMatrix));
    }

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
    sim_sleep_us(DRAWDONE_LATENCY_US);
}

void GX_ClearVtxDesc(void) {
    gx.vtx_has_pos = false;
    gx.vtx_has_normal = false;
    gx.vtx_has_color = false;
    for (int i = 0; i < 8; i++) gx.vtx_has_texcoord[i] = false;
}

void GX_SetVtxDesc(u8 attr, u8 type) {
    bool enabled = (type != GX_NONE);
    if (attr == GX_VA_POS) gx.vtx_has_pos = enabled;
    else if (attr == GX_VA_NRM) gx.vtx_has_normal = enabled;
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

void GX_SetTevDirect(u8 tevstage) {
    gx.tev_stages[tevstage].ind_enable = false;
    gx.tev_stages[tevstage].ind_matrix_sel = GX_ITM_OFF;
}

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

void GX_SetTexCoordGen(u16 texcoord, u32 tgen_typ, u32 tgen_src, u32 mtxsrc) {
    if (texcoord < MAX_TEXGENS) {
        gx.texgens[texcoord].tgen_type = tgen_typ;
        gx.texgens[texcoord].tgen_src = tgen_src;
        gx.texgens[texcoord].mtx_src = mtxsrc;
        gx.texgens[texcoord].active = true;
    }
}

void GX_LoadTexMtxImm(const f32 mtx[][4], u32 texmtx, u8 type) {
    int idx = texmtx_to_index(texmtx);
    if (idx >= 0 && idx < MAX_TEXMTX) {
        int rows = (type == GX_TG_MTX2x4) ? 2 : 3;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 4; c++) {
                gx.tex_matrices[idx].m[r][c] = mtx[r][c];
            }
        }
        // Zero remaining rows if 2x4
        for (int r = rows; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                gx.tex_matrices[idx].m[r][c] = 0.0f;
            }
        }
        gx.tex_matrices[idx].loaded = true;
    }
}

void GX_LoadPosMtxImm(const f32 mtx[][4], u32 pnmtx) {
    int idx = pnmtx / 3;
    if (idx < MAX_PNMTX) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                gx.pn_matrices[idx].m[r][c] = mtx[r][c];
            }
        }
        gx.pn_matrices[idx].loaded = true;
    }
}

void GX_LoadNrmMtxImm(const f32 mtx[][4], u32 pnmtx) {
    // For the sim, normal matrices share the same slots as position matrices.
    // On real hardware these have separate storage, but we don't transform
    // normals in the sim currently.
    GX_LoadPosMtxImm(mtx, pnmtx);
}

void GX_SetCurrentMtx(u32 pnmtx) {
    gx.current_pnmtx = pnmtx;
}

// Indirect texture API

void GX_SetNumIndStages(u8 num) {
    gx.num_ind_stages = num;
}

void GX_SetIndTexOrder(u8 ind_stage, u8 texcoord, u8 texmap) {
    if (ind_stage < MAX_IND_STAGES) {
        gx.ind_stages[ind_stage].texcoord = texcoord;
        gx.ind_stages[ind_stage].texmap = texmap;
    }
}

void GX_SetIndTexCoordScale(u8 ind_stage, u8 scale_s, u8 scale_t) {
    if (ind_stage < MAX_IND_STAGES) {
        gx.ind_stages[ind_stage].scale_s = scale_s;
        gx.ind_stages[ind_stage].scale_t = scale_t;
    }
}

void GX_SetIndTexMtx(u8 ind_mtx, const f32 offset_mtx[2][3], s8 scale_exp) {
    int idx = ind_mtx - GX_ITM_0;
    if (idx >= 0 && idx < MAX_IND_MTX) {
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                gx.ind_matrices[idx].m[r][c] = offset_mtx[r][c];
            }
        }
        gx.ind_matrices[idx].scale_exp = scale_exp;
        gx.ind_matrices[idx].loaded = true;
    }
}

void GX_SetTevIndirect(u8 tevstage, u8 ind_stage, u8 format,
                        u8 bias_sel, u8 matrix_sel,
                        u8 wrap_s, u8 wrap_t,
                        u8 add_prev, u8 utc_lod, u8 alpha_sel) {
    if (tevstage < MAX_TEV_STAGES) {
        TEVStageConfig& s = gx.tev_stages[tevstage];
        s.ind_enable = true;
        s.ind_stage = ind_stage;
        s.ind_format = format;
        s.ind_bias_sel = bias_sel;
        s.ind_matrix_sel = matrix_sel;
        s.ind_wrap_s = wrap_s;
        s.ind_wrap_t = wrap_t;
        s.ind_add_prev = add_prev;
        s.ind_utc_lod = utc_lod;
        s.ind_alpha_sel = alpha_sel;
    }
}

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

void GX_Normal3f32(f32 nx, f32 ny, f32 nz) {
    gx.current_vertex.normal[0] = nx;
    gx.current_vertex.normal[1] = ny;
    gx.current_vertex.normal[2] = nz;
    gx.current_vertex.has_normal = true;
}

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
    sim_sleep_us(COPYTEX_LATENCY_US);
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
                    gx.clear_color.r, gx.clear_color.g,
                    gx.clear_color.b, gx.clear_color.a,
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

// Render state
void GX_SetZMode(u8, u8, u8) {}
void GX_SetZCompLoc(u8) {}

void GX_SetBlendMode(u8 type, u8 src_fact, u8 dst_fact, u8 op) {
    gx.blend.mode = type;
    gx.blend.src_factor = src_fact;
    gx.blend.dst_factor = dst_fact;
    gx.blend.logic_op = op;
}

void GX_SetColorUpdate(u8 enable) { gx.color_update_enabled = (enable != 0); }
void GX_SetAlphaUpdate(u8 enable) { gx.alpha_update_enabled = (enable != 0); }
void GX_SetAlphaCompare(u8, u8, u8, u8, u8) {}
void GX_SetDither(u8) {}
