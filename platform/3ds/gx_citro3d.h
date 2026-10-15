// GX -> citro3d/PICA200 backend (STUB: design contract, not yet implemented).
//
// Why this exists: sms-pc-port gets its graphics from Aurora, which reimplements
// GX on top of WebGPU (Vulkan/Metal/D3D12). None of those exist on the 3DS, so the
// 3DS port needs its own GX layer targeting citro3d. This header lists what the
// decomp's GX calls need to turn into. Implement in gx_citro3d.c as you go,
// driven by what the game actually calls (log unimplemented calls at runtime).
//
// Known hardware mismatches to design around (see docs/PORTING.md):
//   * GX has up to 16 TEV stages; PICA200 has 6 texenv stages.
//   * GX has 8 texture coordinate generators; PICA200 has 3 texture units.
//   * GX texture formats (CMPR, I4, IA8, RGB5A3, ...) must be converted to
//     PICA formats (ETC1, RGBA8, RGB565, ...) and swizzled/tiled at load time.
//   * GX vertex data is big-endian and indexed via vertex descriptors; PICA wants
//     little-endian attribute buffers in linear memory.
//   * EFB copies (used for reflections, water, shadows) map to render-to-texture.
#pragma once
#include <3ds.h>
#include <citro3d.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	C3D_RenderTarget* top;
	C3D_RenderTarget* bottom; // optional (HUD/debug)
	bool initialized;
} GX3DS;

bool gx3ds_init(GX3DS* gx);
void gx3ds_shutdown(GX3DS* gx);

void gx3ds_begin_frame(GX3DS* gx);
void gx3ds_end_frame(GX3DS* gx);

// Bring-up test (the original triangle): proves shader + projection + TexEnv work.
void gx3ds_draw_test_triangle(GX3DS* gx);

// ---- To be filled in, in this order (cheapest wins first) -------------------
// 1. GXSetViewport / GXSetScissor / GXSetCopyClear     -> C3D viewport/scissor/clear
// 2. GXLoadPosMtxImm / GXLoadProjectionMtx / GXSetProjection -> vertex shader uniforms
// 3. GXBegin..GXEnd + GXPosition3f32/GXColor4u8/GXTexCoord2f32 -> C3D_Imm* (immediate mode)
// 4. GXInitTexObj / GXLoadTexObj + texture decode      -> C3D_Tex (convert CMPR->ETC1)
// 5. GXSetTevOp / GXSetTevOrder / GXSetTevColorIn...   -> C3D_TexEnv (collapse to <= 6 stages)
// 6. GXSetChanCtrl / GXLoadLightObj (hardware lighting)-> PICA fragment lighting or vertex shader
// 7. GXCopyTex / GXSetTexCopySrc                       -> render-to-texture targets
// 8. Fog, Z-compare, blend modes                       -> C3D_FogGas / C3D_DepthTest / C3D_AlphaBlend

#ifdef __cplusplus
}
#endif
