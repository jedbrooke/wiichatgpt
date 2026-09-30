// Tests for the GX TEV simulator.
// Validates the TEV combiner formula, EFB read/write, texture sampling,
// and various TEV stage configurations.

#include "gx.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name()
#define RUN(name) do { \
    printf("  %-50s ", #name); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL\n    %s:%d: %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabs((double)(a) - (double)(b)) > (tol)) { \
        printf("FAIL\n    %s:%d: %.4f != %.4f (tol=%.4f)\n", \
               __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        tests_failed++; \
        return; \
    } \
} while(0)

static unsigned char fifo_buf[64 * 1024] __attribute__((aligned(32)));

static void init_gx() {
    GX_Init(fifo_buf, sizeof(fifo_buf));
    GX_SetNumChans(1);
    GX_SetNumTexGens(0);
    GX_SetNumTevStages(1);
    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_TRUE);
}

// EFB poke/peek round-trip
TEST(test_efb_poke_peek) {
    init_gx();
    GXColor in = {100, 150, 200, 250};
    GX_PokeARGB(10, 20, in);

    GXColor out;
    GX_PeekARGB(10, 20, &out);
    ASSERT_EQ(out.r, 100);
    ASSERT_EQ(out.g, 150);
    ASSERT_EQ(out.b, 200);
    ASSERT_EQ(out.a, 250);
}

// TEV PASSCLR: output = rasterized color
TEST(test_tev_passclr) {
    init_gx();
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GX_SetTevOp(GX_TEVSTAGE0, GX_PASSCLR);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(5.0f, 5.0f);  GX_Color4u8(42, 84, 126, 168);
        GX_Position2f32(6.0f, 5.0f);  GX_Color4u8(42, 84, 126, 168);
        GX_Position2f32(6.0f, 6.0f);  GX_Color4u8(42, 84, 126, 168);
        GX_Position2f32(5.0f, 6.0f);  GX_Color4u8(42, 84, 126, 168);
    GX_End();

    GXColor out;
    GX_PeekARGB(5, 5, &out);
    ASSERT_EQ(out.r, 42);
    ASSERT_EQ(out.g, 84);
    ASSERT_EQ(out.b, 126);
    ASSERT_EQ(out.a, 168);
}

// TEV REPLACE with konst color: output = konst
TEST(test_tev_konst_replace) {
    init_gx();

    GX_SetTevKColor(0, (GXColor){77, 88, 99, 110});
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

    // Configure: out = KONST (d=KONST, a=b=c=ZERO)
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(255, 255, 255, 255);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(255, 255, 255, 255);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(255, 255, 255, 255);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(255, 255, 255, 255);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);
    ASSERT_EQ(out.r, 77);
    ASSERT_EQ(out.g, 88);
    ASSERT_EQ(out.b, 99);
    ASSERT_EQ(out.a, 110);
}

// TEV multiply: out = ras * konst (via lerp trick)
// a=ZERO, b=KONST, c=RASC, d=ZERO -> lerp(0, KONST, RASC) = RASC * KONST / 255
TEST(test_tev_multiply) {
    init_gx();

    // ras = 200, konst = 128 -> product = 200 * 128 / 255 ~ 100
    GX_SetTevKColor(0, (GXColor){128, 128, 128, 128});
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
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(200, 200, 200, 200);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);

    // Expected: lerp = ((255-200)*0 + 200*128 + 128) >> 8 = (25600 + 128) >> 8 = 100
    ASSERT_EQ(out.r, 100);
    ASSERT_EQ(out.g, 100);
    ASSERT_EQ(out.b, 100);
    ASSERT_EQ(out.a, 100);
}

// TEV addition: out = ras + konst (clamped to 255)
TEST(test_tev_add) {
    init_gx();

    // ras = 100, konst = 50 -> sum = 150
    GX_SetTevKColor(0, (GXColor){50, 50, 50, 50});
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

    // a=KONST, b=ZERO, c=ZERO, d=RASC -> lerp(KONST,0,0) = KONST, then result = RASC + KONST
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(100, 100, 100, 100);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(100, 100, 100, 100);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(100, 100, 100, 100);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(100, 100, 100, 100);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);

    // lerp = ((255-0)*50 + 0*0 + 128) >> 8 = (12750 + 128) >> 8 = 50
    // result = d + lerp = 100 + 50 = 150
    ASSERT_EQ(out.r, 150);
    ASSERT_EQ(out.g, 150);
    ASSERT_EQ(out.b, 150);
    ASSERT_EQ(out.a, 150);
}

// TEV subtraction: out = d - lerp(a,b,c)
TEST(test_tev_subtract) {
    init_gx();

    // ras = 200, konst = 50. We want out = RASC - KONST
    GX_SetTevKColor(0, (GXColor){50, 50, 50, 50});
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);

    // a=KONST, b=ZERO, c=ZERO, d=RASC, op=SUB
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_SUB, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_SUB, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(200, 200, 200, 200);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(200, 200, 200, 200);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);

    // lerp = ((255-0)*50 + 128) >> 8 = 50
    // result = 200 - 50 = 150
    ASSERT_EQ(out.r, 150);
    ASSERT_EQ(out.g, 150);
    ASSERT_EQ(out.b, 150);
    ASSERT_EQ(out.a, 150);
}

// TEV scale x2
TEST(test_tev_scale_x2) {
    init_gx();

    // d=RASC=60, a=ZERO, b=ZERO, c=ZERO, scale=x2 -> result = 60 * 2 = 120
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_2, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_2, GX_TRUE, GX_TEVPREV);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(60, 60, 60, 60);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(60, 60, 60, 60);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(60, 60, 60, 60);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(60, 60, 60, 60);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);
    ASSERT_EQ(out.r, 120);
    ASSERT_EQ(out.g, 120);
}

// Multi-stage TEV: stage0 writes TEVREG0, stage1 reads it and adds to RASC
TEST(test_tev_two_stages) {
    init_gx();
    GX_SetNumTevStages(2);

    // Stage 0: write konst (30,30,30,30) to TEVREG0
    GX_SetTevKColor(0, (GXColor){30, 30, 30, 30});
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0);
    GX_SetTevKAlphaSel(GX_TEVSTAGE0, GX_TEV_KASEL_K0_A);
    GX_SetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST);
    GX_SetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
    GX_SetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVREG0);
    GX_SetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVREG0);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    // Stage 1: out = RASC + TEVREG0
    // a=C0 (TEVREG0 color), b=ZERO, c=ZERO, d=RASC
    GX_SetTevColorIn(GX_TEVSTAGE1, GX_CC_C0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC);
    GX_SetTevAlphaIn(GX_TEVSTAGE1, GX_CA_A0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA);
    GX_SetTevColorOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevAlphaOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevOrder(GX_TEVSTAGE1, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLOR0A0);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);

    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(70, 70, 70, 70);
        GX_Position2f32(1.0f, 0.0f); GX_Color4u8(70, 70, 70, 70);
        GX_Position2f32(1.0f, 1.0f); GX_Color4u8(70, 70, 70, 70);
        GX_Position2f32(0.0f, 1.0f); GX_Color4u8(70, 70, 70, 70);
    GX_End();

    GXColor out;
    GX_PeekARGB(0, 0, &out);
    // stage0: TEVREG0 = 30
    // stage1: lerp(30, 0, 0) = ((255)*30 + 128)>>8 = 30, result = 70 + 30 = 100
    ASSERT_EQ(out.r, 100);
    ASSERT_EQ(out.g, 100);
}

// Texture sampling through TEV
TEST(test_tev_texture_replace) {
    init_gx();

    // Create a 2x2 RGBA8 texture
    u8 tex_data[2 * 2 * 4] = {
        255, 0,   0,   255,    // (0,0) = red
        0,   255, 0,   255,    // (1,0) = green
        0,   0,   255, 255,    // (0,1) = blue
        255, 255, 0,   255,    // (1,1) = yellow
    };

    GXTexObj tex;
    GX_InitTexObj(&tex, tex_data, 2, 2, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
    GX_LoadTexObj(&tex, GX_TEXMAP0);

    GX_SetNumTexGens(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    GX_SetTevOp(GX_TEVSTAGE0, GX_REPLACE);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_CLR0, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_TEX0, GX_DIRECT);

    // Draw a 2x2 quad mapping the texture
    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
        GX_Position2f32(0.0f, 0.0f); GX_Color4u8(255,255,255,255); GX_TexCoord2f32(0.0f, 0.0f);
        GX_Position2f32(2.0f, 0.0f); GX_Color4u8(255,255,255,255); GX_TexCoord2f32(1.0f, 0.0f);
        GX_Position2f32(2.0f, 2.0f); GX_Color4u8(255,255,255,255); GX_TexCoord2f32(1.0f, 1.0f);
        GX_Position2f32(0.0f, 2.0f); GX_Color4u8(255,255,255,255); GX_TexCoord2f32(0.0f, 1.0f);
    GX_End();

    // Check pixel (0,0): should be red (texcoord ~(0,0) -> texel (0,0))
    GXColor out;
    GX_PeekARGB(0, 0, &out);
    ASSERT_EQ(out.r, 255);
    ASSERT_EQ(out.g, 0);
    ASSERT_EQ(out.b, 0);
    ASSERT_EQ(out.a, 255);
}

// EFB CopyTex
TEST(test_efb_copytex) {
    init_gx();

    // Poke some known values
    GX_PokeARGB(0, 0, (GXColor){10, 20, 30, 40});
    GX_PokeARGB(1, 0, (GXColor){50, 60, 70, 80});
    GX_PokeARGB(0, 1, (GXColor){90, 100, 110, 120});
    GX_PokeARGB(1, 1, (GXColor){130, 140, 150, 160});

    GX_SetTexCopySrc(0, 0, 2, 2);
    GX_SetTexCopyDst(2, 2, GX_TF_RGBA8, GX_FALSE);

    u8 dest[2 * 2 * 4];
    memset(dest, 0, sizeof(dest));
    GX_CopyTex(dest, GX_FALSE);

    // Verify copied data
    ASSERT_EQ(dest[0], 10);  ASSERT_EQ(dest[1], 20);
    ASSERT_EQ(dest[2], 30);  ASSERT_EQ(dest[3], 40);
    ASSERT_EQ(dest[4], 50);  ASSERT_EQ(dest[5], 60);
    ASSERT_EQ(dest[6], 70);  ASSERT_EQ(dest[7], 80);
    ASSERT_EQ(dest[8], 90);  ASSERT_EQ(dest[9], 100);
    ASSERT_EQ(dest[10], 110); ASSERT_EQ(dest[11], 120);
    ASSERT_EQ(dest[12], 130); ASSERT_EQ(dest[13], 140);
    ASSERT_EQ(dest[14], 150); ASSERT_EQ(dest[15], 160);
}

int main() {
    printf("test_tev:\n");
    RUN(test_efb_poke_peek);
    RUN(test_tev_passclr);
    RUN(test_tev_konst_replace);
    RUN(test_tev_multiply);
    RUN(test_tev_add);
    RUN(test_tev_subtract);
    RUN(test_tev_scale_x2);
    RUN(test_tev_two_stages);
    RUN(test_tev_texture_replace);
    RUN(test_efb_copytex);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
