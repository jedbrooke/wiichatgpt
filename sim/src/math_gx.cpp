// GX TEV backend for wii_math.h
// Implements math operations by encoding them as GX TEV pipeline operations.
// On desktop this runs against gx_sim.cpp, on Wii it runs against real hardware.
//
// Strategy for matmul via TEV:
//   The TEV combiner computes: out = (d + ((1-c)*a + c*b)) per channel.
//   With c=255 (1.0), this becomes: out = d + b (per channel, in 0..255 space).
//   With the MODULATE preset, we get: out = tex * ras (a multiply).
//
//   For a dot product of two vectors encoded in texture pixels and vertex colors,
//   we can use the TEV to compute a[i]*b[i] per pixel, then accumulate with
//   multi-pass rendering (read back EFB, add to accumulator).
//
//   For now: a placeholder that uses GX_PokeARGB / GX_PeekARGB to encode
//   values in the EFB and process them through TEV stages. The full tiled
//   matmul kernel will be developed iteratively as we validate the TEV math.

#include "wii_math.h"
#include "gx.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Quantization helpers: map float to 0..255 and back.
// For initial validation we use a simple affine mapping over a fixed range.
// This limits us to values in [0, 1] mapped to [0, 255].

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

    // Set up for compute: 1 TEV stage, no texgens, 1 color channel
    GX_SetNumChans(1);
    GX_SetNumTexGens(0);
    GX_SetNumTevStages(1);

    // Default TEV: pass through rasterized color
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GX_SetTevOp(GX_TEVSTAGE0, GX_PASSCLR);

    // Enable color and alpha writes
    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);
}

void wii_math_shutdown(void) {}

// Matmul using TEV multiply-accumulate.
// Strategy: for each output element C[i][j], compute the dot product of
// row i of A with column j of B.
//
// We encode A values as vertex colors and B values as texture colors,
// use TEV MODULATE to multiply them, then accumulate in the EFB by
// reading back and adding on the CPU side.
//
// This is deliberately naive (one element at a time, per-channel) to
// validate the TEV math before optimizing with tiled/batched approaches.
void wii_matmul(const float* A, const float* B, float* C,
                int M, int K, int N) {
    // For each output element
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;

            // Accumulate dot product in chunks of 4 (RGBA channels)
            int k = 0;
            while (k < K) {
                int chunk = K - k;
                if (chunk > 4) chunk = 4;

                // Encode A values as vertex color (RGBA)
                u8 a_r = (k + 0 < K) ? float_to_u8(A[i * K + k + 0]) : 0;
                u8 a_g = (k + 1 < K) ? float_to_u8(A[i * K + k + 1]) : 0;
                u8 a_b = (k + 2 < K) ? float_to_u8(A[i * K + k + 2]) : 0;
                u8 a_a = (k + 3 < K) ? float_to_u8(A[i * K + k + 3]) : 0;

                // Encode B values as konst color
                u8 b_r = (k + 0 < K) ? float_to_u8(B[(k + 0) * N + j]) : 0;
                u8 b_g = (k + 1 < K) ? float_to_u8(B[(k + 1) * N + j]) : 0;
                u8 b_b = (k + 2 < K) ? float_to_u8(B[(k + 2) * N + j]) : 0;
                u8 b_a = (k + 3 < K) ? float_to_u8(B[(k + 3) * N + j]) : 0;

                // Set up TEV to compute: out = ras_color * konst_color
                // Using manual TEV config: a=ZERO, b=KONST, c=RASC, d=ZERO
                // Formula: (1-c)*0 + c*KONST = RASC * KONST
                GX_SetTevKColor(0, (GXColor){b_r, b_g, b_b, b_a});
                GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
                GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

                GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_KONST, GX_CC_RASC, GX_CC_ZERO);
                GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_KONST, GX_CA_RASA, GX_CA_ZERO);
                GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
                GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
                GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

                // Draw a single pixel at position (0, 0) using a 1x1 quad
                GX_ClearVtxDesc();
                GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
                GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

                GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
                    GX_Position2f32(0.0f, 0.0f);
                    GX_Color4u8(a_r, a_g, a_b, a_a);
                    GX_Position2f32(1.0f, 0.0f);
                    GX_Color4u8(a_r, a_g, a_b, a_a);
                    GX_Position2f32(1.0f, 1.0f);
                    GX_Color4u8(a_r, a_g, a_b, a_a);
                    GX_Position2f32(0.0f, 1.0f);
                    GX_Color4u8(a_r, a_g, a_b, a_a);
                GX_End();

                // Read back the result
                GXColor result;
                GX_PeekARGB(0, 0, &result);

                // Accumulate: each channel holds one product a[k+ch] * b[k+ch]
                // The TEV computed (ras * konst) / 255 due to fixed-point scaling,
                // so the actual product in float is (result / 255) which represents
                // (a_val * b_val) where both were in [0,1].
                if (k + 0 < K) acc += u8_to_float(result.r);
                if (k + 1 < K) acc += u8_to_float(result.g);
                if (k + 2 < K) acc += u8_to_float(result.b);
                if (k + 3 < K) acc += u8_to_float(result.a);

                k += 4;
            }

            C[i * N + j] = acc;
        }
    }
}

void wii_matvec(const float* A, const float* x, float* y,
                int M, int K) {
    // Reuse matmul with N=1 (x treated as K x 1 matrix)
    wii_matmul(A, x, y, M, K, 1);
}

// Elementwise operations: use TEV to process 4 elements per pixel (RGBA)

void wii_add(const float* a, const float* b, float* y, int n) {
    // TEV add: configure d=RASC, a=KONST, b=ZERO, c=ZERO -> out = d + (1-0)*a = RASC + KONST
    // But TEV formula is d + lerp(a,b,c), with c=ZERO: out = d + a
    for (int i = 0; i < n; i += 4) {
        int chunk = n - i;
        if (chunk > 4) chunk = 4;

        u8 ar = (i + 0 < n) ? float_to_u8(a[i + 0]) : 0;
        u8 ag = (i + 1 < n) ? float_to_u8(a[i + 1]) : 0;
        u8 ab = (i + 2 < n) ? float_to_u8(a[i + 2]) : 0;
        u8 aa = (i + 3 < n) ? float_to_u8(a[i + 3]) : 0;

        u8 br = (i + 0 < n) ? float_to_u8(b[i + 0]) : 0;
        u8 bg = (i + 1 < n) ? float_to_u8(b[i + 1]) : 0;
        u8 bb = (i + 2 < n) ? float_to_u8(b[i + 2]) : 0;
        u8 ba = (i + 3 < n) ? float_to_u8(b[i + 3]) : 0;

        GX_SetTevKColor(0, (GXColor){br, bg, bb, ba});
        GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
        GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

        // out = RASC + KONST: a=KONST, b=ZERO, c=ZERO, d=RASC
        GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
        GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
        GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
        GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
        GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

        GX_ClearVtxDesc();
        GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
        GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

        GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
            GX_Position2f32(0.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(0.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
        GX_End();

        GXColor result;
        GX_PeekARGB(0, 0, &result);

        if (i + 0 < n) y[i + 0] = u8_to_float(result.r);
        if (i + 1 < n) y[i + 1] = u8_to_float(result.g);
        if (i + 2 < n) y[i + 2] = u8_to_float(result.b);
        if (i + 3 < n) y[i + 3] = u8_to_float(result.a);
    }
}

void wii_mul(const float* a, const float* b, float* y, int n) {
    // Same as matmul's per-element multiply: out = RASC * KONST
    for (int i = 0; i < n; i += 4) {
        int chunk = n - i;
        if (chunk > 4) chunk = 4;

        u8 ar = (i + 0 < n) ? float_to_u8(a[i + 0]) : 0;
        u8 ag = (i + 1 < n) ? float_to_u8(a[i + 1]) : 0;
        u8 ab = (i + 2 < n) ? float_to_u8(a[i + 2]) : 0;
        u8 aa = (i + 3 < n) ? float_to_u8(a[i + 3]) : 0;

        u8 br = (i + 0 < n) ? float_to_u8(b[i + 0]) : 0;
        u8 bg = (i + 1 < n) ? float_to_u8(b[i + 1]) : 0;
        u8 bb = (i + 2 < n) ? float_to_u8(b[i + 2]) : 0;
        u8 ba = (i + 3 < n) ? float_to_u8(b[i + 3]) : 0;

        GX_SetTevKColor(0, (GXColor){br, bg, bb, ba});
        GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
        GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

        GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_KONST, GX_CC_RASC, GX_CC_ZERO);
        GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_KONST, GX_CA_RASA, GX_CA_ZERO);
        GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
        GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
        GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

        GX_ClearVtxDesc();
        GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
        GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

        GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
            GX_Position2f32(0.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 0.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(1.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
            GX_Position2f32(0.0f, 1.0f); GX_Color4u8(ar, ag, ab, aa);
        GX_End();

        GXColor result;
        GX_PeekARGB(0, 0, &result);

        if (i + 0 < n) y[i + 0] = u8_to_float(result.r);
        if (i + 1 < n) y[i + 1] = u8_to_float(result.g);
        if (i + 2 < n) y[i + 2] = u8_to_float(result.b);
        if (i + 3 < n) y[i + 3] = u8_to_float(result.a);
    }
}

// Activation functions: fall back to CPU for now.
// These could be approximated with multi-stage TEV (piecewise linear)
// but that's a later optimization.

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
