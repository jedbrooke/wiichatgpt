// GX TEV backend for wii_math.h
// Implements math operations by encoding them as GX TEV pipeline operations.
// On desktop this runs against gx_sim.cpp, on Wii it runs against real hardware.
//
// Matmul strategy:
//   Weight values delivered via TEXTURE (per-pixel, varies per output column).
//   Activation values delivered via KONST (broadcast to all pixels in a draw call).
//
//   Using KCSEL_K0 mode, each RGBA channel of the konst gets a different value
//   (K0.R, K0.G, K0.B, K0.A), and the texture RGBA holds 4 corresponding weights.
//   One TEV stage computes 4 independent multiplies per pixel:
//     R: tex.R * K0.R,  G: tex.G * K0.G,  B: tex.B * K0.B,  A: tex.A * K0.A
//
//   Each pixel represents one output element. The 4 RGBA products cover 4
//   consecutive k-indices of the dot product. CPU sums the 4 channels on readback.
//   ceil(K/4) draw calls per row of A, processing all N output columns in parallel.
//
//   Results are read back via GX_CopyTex (the proper pipeline path).

#include "wii_math.h"
#include "gx.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Quantization helpers: map float to 0..255 and back.
// Affine mapping over [0, 1] -> [0, 255].

static u8 float_to_u8(float v) {
    int iv = (int)(v * 255.0f + 0.5f);
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    return (u8)iv;
}

static float u8_to_float(u8 v) {
    return v / 255.0f;
}

static u8 fifo_buf[256 * 1024] __attribute__((aligned(32)));

void wii_math_init(void) {
    GX_Init(fifo_buf, sizeof(fifo_buf));

    GX_SetNumChans(1);
    GX_SetNumTexGens(0);
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GX_SetTevOp(GX_TEVSTAGE0, GX_PASSCLR);

    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_COPY);
}

void wii_math_shutdown(void) {}

// ---------------------------------------------------------------------------
// Matmul: C = A * B
// A is M x K, B is K x N, C is M x N, all row-major, values in [0,1].
//
// Per draw call: 1 TEV stage, 4 k-values packed into RGBA channels.
//   Konst K0: R=A[i][k+0], G=A[i][k+1], B=A[i][k+2], A=A[i][k+3]
//   Texture (width=N, height=1): texel j has R=B[k+0][j], G=B[k+1][j], ...
//   TEV produces per-pixel: R=tex.R*K0.R, G=tex.G*K0.G, B=tex.B*K0.B, A=tex.A*K0.A
//   CPU reads back RGBA, sums 4 channels to get partial dot product for that k-chunk.
//
// ceil(K/4) draw calls per A-row. All N output columns processed in parallel.
// ---------------------------------------------------------------------------
void wii_matmul(const float* A, const float* B, float* C,
                int M, int K, int N) {
    // Weight texture buffer: width=N, height=1, RGBA8
    int tex_bytes = N * 1 * 4;
    u8* tex_data = (u8*)calloc(tex_bytes > 0 ? tex_bytes : 1, 1);

    // Readback buffer: N pixels, RGBA8
    int rb_bytes = N * 4;
    u8* readback = (u8*)calloc(rb_bytes > 0 ? rb_bytes : 1, 1);

    // Configure TEV for multiply: out = TEXC * KONST (per channel)
    // KCSEL_K0: R->K0.R, G->K0.G, B->K0.B (each channel gets different konst)
    // KASEL_K0_A: alpha gets K0.A
    GX_SetNumChans(1);
    GX_SetNumTexGens(1);
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

    // Color: lerp(ZERO, TEXC, KONST) = TEXC * KONST / 255, d=ZERO
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_TEXC, GX_CC_KONST, GX_CC_ZERO);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);

    // Alpha: lerp(ZERO, TEXA, KONST) = TEXA * KONST_A / 255, d=ZERO
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_TEXA, GX_CA_KONST, GX_CA_ZERO);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);

    // No blending (overwrite EFB each draw)
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_COPY);

    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);

    for (int i = 0; i < M; i++) {
        // Zero the output row
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }

        for (int k_base = 0; k_base < K; k_base += 4) {
            int chunk = K - k_base;
            if (chunk > 4) chunk = 4;

            // Pack 4 activation values into konst K0 (RGBA)
            u8 act_r = (k_base + 0 < K) ? float_to_u8(A[i * K + k_base + 0]) : 0;
            u8 act_g = (k_base + 1 < K) ? float_to_u8(A[i * K + k_base + 1]) : 0;
            u8 act_b = (k_base + 2 < K) ? float_to_u8(A[i * K + k_base + 2]) : 0;
            u8 act_a = (k_base + 3 < K) ? float_to_u8(A[i * K + k_base + 3]) : 0;
            GX_SetTevKColor(0, (GXColor){act_r, act_g, act_b, act_a});

            // Pack weight texture: width=N, height=1, RGBA8
            // Texel j: R=B[k+0][j], G=B[k+1][j], B=B[k+2][j], A=B[k+3][j]
            for (int j = 0; j < N; j++) {
                int off = j * 4;
                tex_data[off + 0] = (k_base + 0 < K) ? float_to_u8(B[(k_base + 0) * N + j]) : 0;
                tex_data[off + 1] = (k_base + 1 < K) ? float_to_u8(B[(k_base + 1) * N + j]) : 0;
                tex_data[off + 2] = (k_base + 2 < K) ? float_to_u8(B[(k_base + 2) * N + j]) : 0;
                tex_data[off + 3] = (k_base + 3 < K) ? float_to_u8(B[(k_base + 3) * N + j]) : 0;
            }

            // Load weight texture
            GXTexObj tex_obj;
            GX_InitTexObj(&tex_obj, tex_data, (u16)N, 1, GX_TF_RGBA8,
                          GX_CLAMP, GX_CLAMP, GX_FALSE);
            GX_LoadTexObj(&tex_obj, GX_TEXMAP0);

            // Draw N-pixel quad (one pixel per output column)
            // Texcoord s maps pixel position to texture column: [0, 1] across N texels
            // Texcoord t = 0.5 (center of the single row)
            GX_ClearVtxDesc();
            GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
            GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);
            GX_SetVtxDesc(GX_VA_TEX0, GX_DIRECT);

            GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
                GX_Position2f32(0.0f, 0.0f);
                GX_Color4u8(255, 255, 255, 255);
                GX_TexCoord2f32(0.0f, 0.5f);

                GX_Position2f32((float)N, 0.0f);
                GX_Color4u8(255, 255, 255, 255);
                GX_TexCoord2f32(1.0f, 0.5f);

                GX_Position2f32((float)N, 1.0f);
                GX_Color4u8(255, 255, 255, 255);
                GX_TexCoord2f32(1.0f, 0.5f);

                GX_Position2f32(0.0f, 1.0f);
                GX_Color4u8(255, 255, 255, 255);
                GX_TexCoord2f32(0.0f, 0.5f);
            GX_End();
            GX_DrawDone();

            // Read back products via CopyTex
            GX_SetTexCopySrc(0, 0, (u16)N, 1);
            GX_SetTexCopyDst((u16)N, 1, GX_TF_RGBA8, GX_FALSE);
            GX_CopyTex(readback, GX_FALSE);

            // CPU accumulation: sum RGBA channels per pixel
            for (int j = 0; j < N; j++) {
                int off = j * 4;
                float sum = 0.0f;
                if (k_base + 0 < K) sum += u8_to_float(readback[off + 0]);
                if (k_base + 1 < K) sum += u8_to_float(readback[off + 1]);
                if (k_base + 2 < K) sum += u8_to_float(readback[off + 2]);
                if (k_base + 3 < K) sum += u8_to_float(readback[off + 3]);
                C[i * N + j] += sum;
            }
        }
    }

    free(tex_data);
    free(readback);
}

void wii_matvec(const float* A, const float* x, float* y,
                int M, int K) {
    wii_matmul(A, x, y, M, K, 1);
}

// ---------------------------------------------------------------------------
// Elementwise add: y[i] = a[i] + b[i]
// Uses TEV addition: a=KONST, b=ZERO, c=ZERO, d=RASC -> out = RASC + KONST
// 4 elements per pixel via RGBA channels.
// ---------------------------------------------------------------------------
void wii_add(const float* a, const float* b, float* y, int n) {
    GX_SetNumChans(1);
    GX_SetNumTexGens(0);
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_COPY);
    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);

    for (int i = 0; i < n; i += 4) {
        u8 ar = (i + 0 < n) ? float_to_u8(a[i + 0]) : 0;
        u8 ag = (i + 1 < n) ? float_to_u8(a[i + 1]) : 0;
        u8 ab = (i + 2 < n) ? float_to_u8(a[i + 2]) : 0;
        u8 aa = (i + 3 < n) ? float_to_u8(a[i + 3]) : 0;

        u8 br = (i + 0 < n) ? float_to_u8(b[i + 0]) : 0;
        u8 bg = (i + 1 < n) ? float_to_u8(b[i + 1]) : 0;
        u8 bb = (i + 2 < n) ? float_to_u8(b[i + 2]) : 0;
        u8 ba = (i + 3 < n) ? float_to_u8(b[i + 3]) : 0;

        GX_SetTevKColor(0, (GXColor){br, bg, bb, ba});

        GX_ClearVtxDesc();
        GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
        GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

        GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
            GX_Position2f32(0.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(0.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
        GX_End();
        GX_DrawDone();

        u8 result[4];
        GX_SetTexCopySrc(0, 0, 1, 1);
        GX_SetTexCopyDst(1, 1, GX_TF_RGBA8, GX_FALSE);
        GX_CopyTex(result, GX_FALSE);

        if (i + 0 < n) y[i + 0] = u8_to_float(result[0]);
        if (i + 1 < n) y[i + 1] = u8_to_float(result[1]);
        if (i + 2 < n) y[i + 2] = u8_to_float(result[2]);
        if (i + 3 < n) y[i + 3] = u8_to_float(result[3]);
    }
}

// ---------------------------------------------------------------------------
// Elementwise mul: y[i] = a[i] * b[i]
// Uses TEV multiply: a=ZERO, b=KONST, c=RASC, d=ZERO -> out = RASC * KONST
// 4 elements per pixel via RGBA channels.
// ---------------------------------------------------------------------------
void wii_mul(const float* a, const float* b, float* y, int n) {
    GX_SetNumChans(1);
    GX_SetNumTexGens(0);
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_KONST, GX_CC_RASC, GX_CC_ZERO);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_KONST, GX_CA_RASA, GX_CA_ZERO);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_COPY);
    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);

    for (int i = 0; i < n; i += 4) {
        u8 ar = (i + 0 < n) ? float_to_u8(a[i + 0]) : 0;
        u8 ag = (i + 1 < n) ? float_to_u8(a[i + 1]) : 0;
        u8 ab = (i + 2 < n) ? float_to_u8(a[i + 2]) : 0;
        u8 aa = (i + 3 < n) ? float_to_u8(a[i + 3]) : 0;

        u8 br = (i + 0 < n) ? float_to_u8(b[i + 0]) : 0;
        u8 bg = (i + 1 < n) ? float_to_u8(b[i + 1]) : 0;
        u8 bb = (i + 2 < n) ? float_to_u8(b[i + 2]) : 0;
        u8 ba = (i + 3 < n) ? float_to_u8(b[i + 3]) : 0;

        GX_SetTevKColor(0, (GXColor){br, bg, bb, ba});

        GX_ClearVtxDesc();
        GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
        GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

        GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
            GX_Position2f32(0.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(0.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
        GX_End();
        GX_DrawDone();

        u8 result[4];
        GX_SetTexCopySrc(0, 0, 1, 1);
        GX_SetTexCopyDst(1, 1, GX_TF_RGBA8, GX_FALSE);
        GX_CopyTex(result, GX_FALSE);

        if (i + 0 < n) y[i + 0] = u8_to_float(result[0]);
        if (i + 1 < n) y[i + 1] = u8_to_float(result[1]);
        if (i + 2 < n) y[i + 2] = u8_to_float(result[2]);
        if (i + 3 < n) y[i + 3] = u8_to_float(result[3]);
    }
}

// ---------------------------------------------------------------------------
// CPU fallbacks for operations that don't map to the TEV pipeline.
// These run on the Broadway PPC750, not the GPU.
// ---------------------------------------------------------------------------

void wii_relu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void wii_silu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sig;
    }
}

void wii_softmax(const float* x, float* y, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }
    for (int i = 0; i < n; i++) {
        y[i] /= sum;
    }
}

void wii_rmsnorm(const float* x, const float* weight, float* y, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * ss * weight[i];
    }
}

void wii_matvec_q8(const signed char* W, float scale, const float* x,
                   float* y, int M, int K) {
    (void)W; (void)scale; (void)x; (void)y; (void)M; (void)K;
    fprintf(stderr, "wii_matvec_q8: signed INT8 not supported on GX TEV backend\n");
    exit(1);
}