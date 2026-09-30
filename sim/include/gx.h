#pragma once

// GX function declarations matching libogc signatures.
// On desktop, these are implemented by gx_sim.cpp (software simulator).
// On Wii, these come from libogc and this header is not used.

#include "gx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
GXFifoObj* GX_Init(void* base, u32 size);
void GX_SetCPUFifo(GXFifoObj* fifo);
void GX_SetGPFifo(GXFifoObj* fifo);

// FIFO / sync
void GX_Flush(void);
void GX_DrawDone(void);

// Vertex descriptor
void GX_ClearVtxDesc(void);
void GX_SetVtxDesc(u8 attr, u8 type);
void GX_SetVtxAttrFmt(u8 vtxfmt, u32 vtxattr, u32 comptype, u32 compsize, u32 frac);

// TEV configuration
void GX_SetNumTevStages(u8 num);
void GX_SetTevOp(u8 tevstage, u8 mode);
void GX_SetTevOrder(u8 tevstage, u8 texcoord, u32 texmap, u8 color);
void GX_SetTevColorIn(u8 tevstage, u8 a, u8 b, u8 c, u8 d);
void GX_SetTevAlphaIn(u8 tevstage, u8 a, u8 b, u8 c, u8 d);
void GX_SetTevColorOp(u8 tevstage, u8 tevop, u8 tevbias, u8 tevscale, u8 clamp, u8 tevregid);
void GX_SetTevAlphaOp(u8 tevstage, u8 tevop, u8 tevbias, u8 tevscale, u8 clamp, u8 tevregid);
void GX_SetTevColor(u8 tev_regid, GXColor color);
void GX_SetTevColorS10(u8 tev_regid, GXColorS10 color);
void GX_SetTevKColor(u8 sel, GXColor col);
void GX_SetTevKColorSel(u8 tevstage, u8 sel);
void GX_SetTevKAlphaSel(u8 tevstage, u8 sel);
void GX_SetTevSwapMode(u8 tevstage, u8 ras_sel, u8 tex_sel);
void GX_SetTevSwapModeTable(u8 swapid, u8 r, u8 g, u8 b, u8 a);
void GX_SetTevDirect(u8 tevstage);

// Texture
void GX_InitTexObj(GXTexObj* obj, void* img_ptr, u16 wd, u16 ht, u8 fmt,
                   u8 wrap_s, u8 wrap_t, u8 mipmap);
void GX_LoadTexObj(GXTexObj* obj, u8 mapid);
void GX_InvalidateTexAll(void);
u32 GX_GetTexBufferSize(u16 wd, u16 ht, u32 fmt, u8 mipmap, u8 maxlod);

// Texture coordinate generation
void GX_SetNumTexGens(u32 nr);
void GX_SetTexCoordGen(u16 texcoord, u32 tgen_typ, u32 tgen_src, u32 mtxsrc);

// XF matrix loading
void GX_LoadTexMtxImm(const f32 mtx[][4], u32 texmtx, u8 type);
void GX_LoadPosMtxImm(const f32 mtx[][4], u32 pnmtx);
void GX_LoadNrmMtxImm(const f32 mtx[][4], u32 pnmtx);
void GX_SetCurrentMtx(u32 pnmtx);

// Indirect texture configuration
void GX_SetNumIndStages(u8 num);
void GX_SetIndTexOrder(u8 ind_stage, u8 texcoord, u8 texmap);
void GX_SetIndTexCoordScale(u8 ind_stage, u8 scale_s, u8 scale_t);
void GX_SetIndTexMtx(u8 ind_mtx, const f32 offset_mtx[2][3], s8 scale_exp);
void GX_SetTevIndirect(u8 tevstage, u8 ind_stage, u8 format,
                        u8 bias_sel, u8 matrix_sel,
                        u8 wrap_s, u8 wrap_t,
                        u8 add_prev, u8 utc_lod, u8 alpha_sel);

// Color channels
void GX_SetNumChans(u8 num);

// Drawing
void GX_Begin(u8 primitive, u8 vtxfmt, u16 vtxcnt);
void GX_End(void);

// Vertex submission (immediate mode)
void GX_Position3f32(f32 x, f32 y, f32 z);
void GX_Position2f32(f32 x, f32 y);
void GX_Color4u8(u8 r, u8 g, u8 b, u8 a);
void GX_Color1u32(u32 clr);
void GX_TexCoord2f32(f32 s, f32 t);
void GX_Normal3f32(f32 nx, f32 ny, f32 nz);

// EFB operations
void GX_SetPixelFmt(u8 pix_fmt, u8 z_fmt);
void GX_SetCopyClear(GXColor color, u32 zvalue);
void GX_SetTexCopySrc(u16 left, u16 top, u16 wd, u16 ht);
void GX_SetTexCopyDst(u16 wd, u16 ht, u32 fmt, u8 mipmap);
void GX_CopyTex(void* dest, u8 clear);
void GX_SetDispCopySrc(u16 left, u16 top, u16 wd, u16 ht);
void GX_SetDispCopyDst(u16 wd, u16 ht);
void GX_SetDispCopyYScale(f32 yscale);
void GX_CopyDisp(void* dest, u8 clear);
void GX_PixModeSync(void);

// Direct EFB pixel access
void GX_PokeARGB(u16 x, u16 y, GXColor color);
void GX_PeekARGB(u16 x, u16 y, GXColor* color);
void GX_PokeZ(u16 x, u16 y, u32 z);
void GX_PeekZ(u16 x, u16 y, u32* z);

// Render state (stubs for compatibility, not critical for compute)
void GX_SetZMode(u8 enable, u8 func, u8 update_enable);
void GX_SetZCompLoc(u8 before_tex);
void GX_SetBlendMode(u8 type, u8 src_fact, u8 dst_fact, u8 op);
void GX_SetColorUpdate(u8 enable);
void GX_SetAlphaUpdate(u8 enable);
void GX_SetAlphaCompare(u8 comp0, u8 ref0, u8 aop, u8 comp1, u8 ref1);
void GX_SetDither(u8 dither);

#ifdef __cplusplus
}
#endif
