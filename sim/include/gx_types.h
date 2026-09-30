#pragma once

// GX type definitions and constants matching libogc's gx.h
// This is a subset focused on the TEV pipeline and EFB operations
// needed for compute-via-GPU.

#include <cstdint>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef float f32;

// GXColor: 8-bit per channel RGBA
typedef struct {
    u8 r, g, b, a;
} GXColor;

// GXColorS10: signed 10-bit per channel (used for TEV constant colors)
typedef struct {
    s16 r, g, b, a;
} GXColorS10;

// Opaque objects (internally defined in gx_sim.cpp)
typedef struct _GXTexObj GXTexObj;
typedef struct _GXFifoObj GXFifoObj;

// Primitive types
#define GX_POINTS         0xB8
#define GX_LINES          0xA8
#define GX_LINESTRIP      0xB0
#define GX_TRIANGLES      0x90
#define GX_TRIANGLESTRIP  0x98
#define GX_TRIANGLEFAN    0xA0
#define GX_QUADS          0x80

// Vertex format slots
#define GX_VTXFMT0  0
#define GX_VTXFMT1  1
#define GX_VTXFMT2  2

// Vertex attributes
#define GX_VA_POS     9
#define GX_VA_CLR0   11
#define GX_VA_CLR1   12
#define GX_VA_TEX0   13
#define GX_VA_TEX1   14
#define GX_VA_TEX2   15
#define GX_VA_TEX3   16
#define GX_VA_TEX4   17
#define GX_VA_TEX5   18
#define GX_VA_TEX6   19
#define GX_VA_TEX7   20

// Vertex attribute types
#define GX_NONE    0
#define GX_DIRECT  1
#define GX_INDEX8  2
#define GX_INDEX16 3

// Component types for vertex attributes
#define GX_POS_XYZ  1
#define GX_CLR_RGBA 5
#define GX_TEX_ST   1

// Component sizes
#define GX_F32  4
#define GX_RGBA8 5

// TEV stages
#define GX_TEVSTAGE0   0
#define GX_TEVSTAGE1   1
#define GX_TEVSTAGE2   2
#define GX_TEVSTAGE3   3
#define GX_TEVSTAGE4   4
#define GX_TEVSTAGE5   5
#define GX_TEVSTAGE6   6
#define GX_TEVSTAGE7   7
#define GX_TEVSTAGE8   8
#define GX_TEVSTAGE9   9
#define GX_TEVSTAGE10 10
#define GX_TEVSTAGE11 11
#define GX_TEVSTAGE12 12
#define GX_TEVSTAGE13 13
#define GX_TEVSTAGE14 14
#define GX_TEVSTAGE15 15

// TEV combiner mode (simplified presets)
#define GX_MODULATE  0
#define GX_DECAL     1
#define GX_BLEND     2
#define GX_REPLACE   3
#define GX_PASSCLR   4

// TEV color input selectors
#define GX_CC_CPREV   0   // previous stage color
#define GX_CC_APREV   1   // previous stage alpha
#define GX_CC_C0      2   // TEV register 0 color
#define GX_CC_A0      3   // TEV register 0 alpha
#define GX_CC_C1      4   // TEV register 1 color
#define GX_CC_A1      5   // TEV register 1 alpha
#define GX_CC_C2      6   // TEV register 2 color
#define GX_CC_A2      7   // TEV register 2 alpha
#define GX_CC_TEXC    8   // texture color
#define GX_CC_TEXA    9   // texture alpha
#define GX_CC_RASC   10   // rasterized color
#define GX_CC_RASA   11   // rasterized alpha
#define GX_CC_ONE    12   // constant 1.0
#define GX_CC_HALF   13   // constant 0.5
#define GX_CC_KONST  14   // konst color
#define GX_CC_ZERO   15   // constant 0.0

// TEV alpha input selectors
#define GX_CA_APREV   0
#define GX_CA_A0      1
#define GX_CA_A1      2
#define GX_CA_A2      3
#define GX_CA_TEXA    4
#define GX_CA_RASA    5
#define GX_CA_KONST   6
#define GX_CA_ZERO    7

// TEV operation
#define GX_TEV_ADD  0
#define GX_TEV_SUB  1

// TEV bias
#define GX_TB_ZERO     0
#define GX_TB_ADDHALF  1
#define GX_TB_SUBHALF  2

// TEV scale
#define GX_CS_SCALE_1   0
#define GX_CS_SCALE_2   1
#define GX_CS_SCALE_4   2
#define GX_CS_DIVIDE_2  3

// TEV output register IDs
#define GX_TEVPREV  0
#define GX_TEVREG0  1
#define GX_TEVREG1  2
#define GX_TEVREG2  3

// TEV konst color selectors
#define GX_TEV_KCSEL_1     0x00
#define GX_TEV_KCSEL_7_8   0x01
#define GX_TEV_KCSEL_3_4   0x02
#define GX_TEV_KCSEL_5_8   0x03
#define GX_TEV_KCSEL_1_2   0x04
#define GX_TEV_KCSEL_3_8   0x05
#define GX_TEV_KCSEL_1_4   0x06
#define GX_TEV_KCSEL_1_8   0x07
#define GX_TEV_KCSEL_K0    0x0C
#define GX_TEV_KCSEL_K1    0x0D
#define GX_TEV_KCSEL_K2    0x0E
#define GX_TEV_KCSEL_K3    0x0F
#define GX_TEV_KCSEL_K0_R  0x10
#define GX_TEV_KCSEL_K1_R  0x11
#define GX_TEV_KCSEL_K2_R  0x12
#define GX_TEV_KCSEL_K3_R  0x13
#define GX_TEV_KCSEL_K0_G  0x14
#define GX_TEV_KCSEL_K1_G  0x15
#define GX_TEV_KCSEL_K2_G  0x16
#define GX_TEV_KCSEL_K3_G  0x17
#define GX_TEV_KCSEL_K0_B  0x18
#define GX_TEV_KCSEL_K1_B  0x19
#define GX_TEV_KCSEL_K2_B  0x1A
#define GX_TEV_KCSEL_K3_B  0x1B
#define GX_TEV_KCSEL_K0_A  0x1C
#define GX_TEV_KCSEL_K1_A  0x1D
#define GX_TEV_KCSEL_K2_A  0x1E
#define GX_TEV_KCSEL_K3_A  0x1F

// TEV konst alpha selectors
#define GX_TEV_KASEL_1     0x00
#define GX_TEV_KASEL_7_8   0x01
#define GX_TEV_KASEL_3_4   0x02
#define GX_TEV_KASEL_5_8   0x03
#define GX_TEV_KASEL_1_2   0x04
#define GX_TEV_KASEL_3_8   0x05
#define GX_TEV_KASEL_1_4   0x06
#define GX_TEV_KASEL_1_8   0x07
#define GX_TEV_KASEL_K0_R  0x10
#define GX_TEV_KASEL_K1_R  0x11
#define GX_TEV_KASEL_K2_R  0x12
#define GX_TEV_KASEL_K3_R  0x13
#define GX_TEV_KASEL_K0_G  0x14
#define GX_TEV_KASEL_K1_G  0x15
#define GX_TEV_KASEL_K2_G  0x16
#define GX_TEV_KASEL_K3_G  0x17
#define GX_TEV_KASEL_K0_B  0x18
#define GX_TEV_KASEL_K1_B  0x19
#define GX_TEV_KASEL_K2_B  0x1A
#define GX_TEV_KASEL_K3_B  0x1B
#define GX_TEV_KASEL_K0_A  0x1C
#define GX_TEV_KASEL_K1_A  0x1D
#define GX_TEV_KASEL_K2_A  0x1E
#define GX_TEV_KASEL_K3_A  0x1F

// Texture map slots
#define GX_TEXMAP0  0
#define GX_TEXMAP1  1
#define GX_TEXMAP2  2
#define GX_TEXMAP3  3
#define GX_TEXMAP4  4
#define GX_TEXMAP5  5
#define GX_TEXMAP6  6
#define GX_TEXMAP7  7
#define GX_TEXMAP_NULL 0xFF

// Texture coordinate slots
#define GX_TEXCOORD0  0
#define GX_TEXCOORD1  1
#define GX_TEXCOORD2  2
#define GX_TEXCOORD3  3
#define GX_TEXCOORD4  4
#define GX_TEXCOORD5  5
#define GX_TEXCOORD6  6
#define GX_TEXCOORD7  7
#define GX_TEXCOORDNULL 0xFF

// Texture formats
#define GX_TF_I8     1   // 8-bit intensity
#define GX_TF_IA8    3   // 8-bit intensity + alpha
#define GX_TF_RGB565 4
#define GX_TF_RGB5A3 5
#define GX_TF_RGBA8  6

// Texture wrap modes
#define GX_CLAMP  0
#define GX_REPEAT 1
#define GX_MIRROR 2

// Texture filter modes
#define GX_NEAR   0
#define GX_LINEAR 1

// Color channel IDs
#define GX_COLOR0    0
#define GX_COLOR1    1
#define GX_ALPHA0    2
#define GX_ALPHA1    3
#define GX_COLOR0A0  4
#define GX_COLOR1A1  5
#define GX_COLORZERO 6
#define GX_COLORNULL 0xFF

// Pixel format
#define GX_PF_RGB8_Z24   0
#define GX_PF_RGBA6_Z24  1
#define GX_PF_RGB565_Z16 2

// Z format
#define GX_ZC_LINEAR  0
#define GX_ZC_NEAR    1
#define GX_ZC_MID     2
#define GX_ZC_FAR     3

// Blend modes
#define GX_BM_NONE     0
#define GX_BM_BLEND    1
#define GX_BM_LOGIC    2
#define GX_BM_SUBTRACT 3

// Compare functions
#define GX_NEVER   0
#define GX_LESS    1
#define GX_EQUAL   2
#define GX_LEQUAL  3
#define GX_GREATER 4
#define GX_NEQUAL  5
#define GX_GEQUAL  6
#define GX_ALWAYS  7

// Alpha op
#define GX_AOP_AND  0
#define GX_AOP_OR   1
#define GX_AOP_XOR  2
#define GX_AOP_XNOR 3

// Texcoord generation type
#define GX_TG_MTX2x4  0
#define GX_TG_MTX3x4  1

// Texcoord generation source
#define GX_TG_POS  0
#define GX_TG_NRM  1
#define GX_TG_TEX0 4
#define GX_TG_TEX1 5
#define GX_TG_TEX2 6
#define GX_TG_TEX3 7

// Texture matrix slots
#define GX_TEXMTX0    30
#define GX_IDENTITY   60

// Boolean
#define GX_TRUE  1
#define GX_FALSE 0

// EFB dimensions (maximum)
#define GX_EFB_WIDTH   640
#define GX_EFB_HEIGHT  528

// Size of GXTexObj in bytes (matches libogc)
#define GX_TEXOBJ_SIZE 32

// Actual struct for GXTexObj (internal, opaque to user code on real hardware)
struct _GXTexObj {
    void* img_ptr;
    u16 width;
    u16 height;
    u8 fmt;
    u8 wrap_s;
    u8 wrap_t;
    u8 mipmap;
    u8 pad[GX_TEXOBJ_SIZE - sizeof(void*) - 2*sizeof(u16) - 4*sizeof(u8)];
};

// GXFifoObj (stub, not needed for simulator)
struct _GXFifoObj {
    void* base;
    u32 size;
};
