#include "gx_citro3d.h"
#include <string.h>
#include "vshader_shbin.h"

#define CLEAR_COLOR 0x68B0D8FF

#define DISPLAY_TRANSFER_FLAGS \
	(GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) | \
	GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | \
	GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

static DVLB_s* s_dvlb;
static shaderProgram_s s_program;
static int s_uLocProjection;
static C3D_Mtx s_projection;

bool gx3ds_init(GX3DS* gx)
{
	memset(gx, 0, sizeof(*gx));
	gfxInitDefault();
	C3D_Init(C3D_DEFAULT_CMDBUF_SIZE);

	gx->top = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
	C3D_RenderTargetSetOutput(gx->top, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);

	s_dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
	shaderProgramInit(&s_program);
	shaderProgramSetVsh(&s_program, &s_dvlb->DVLE[0]);
	C3D_BindProgram(&s_program);

	s_uLocProjection = shaderInstanceGetUniformLocation(s_program.vertexShader, "projection");

	C3D_AttrInfo* attrInfo = C3D_GetAttrInfo();
	AttrInfo_Init(attrInfo);
	AttrInfo_AddLoader(attrInfo, 0, GPU_FLOAT, 3); // v0 = position
	AttrInfo_AddLoader(attrInfo, 1, GPU_FLOAT, 3); // v1 = color

	Mtx_OrthoTilt(&s_projection, 0.0, 400.0, 0.0, 240.0, 0.0, 1.0, true);

	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
	C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);

	gx->initialized = true;
	return true;
}

void gx3ds_shutdown(GX3DS* gx)
{
	if (!gx->initialized) return;
	shaderProgramFree(&s_program);
	DVLB_Free(s_dvlb);
	C3D_Fini();
	gfxExit();
	gx->initialized = false;
}

void gx3ds_begin_frame(GX3DS* gx)
{
	C3D_FrameBegin(C3D_FRAME_SYNCDRAW);
	C3D_RenderTargetClear(gx->top, C3D_CLEAR_ALL, CLEAR_COLOR, 0);
	C3D_FrameDrawOn(gx->top);
	C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, s_uLocProjection, &s_projection);
}

void gx3ds_end_frame(GX3DS* gx)
{
	(void)gx;
	C3D_FrameEnd(0);
}

void gx3ds_draw_test_triangle(GX3DS* gx)
{
	(void)gx;
	C3D_ImmDrawBegin(GPU_TRIANGLES);
		C3D_ImmSendAttrib(200.0f, 200.0f, 0.5f, 0.0f);
		C3D_ImmSendAttrib(1.0f, 0.0f, 0.0f, 1.0f);

		C3D_ImmSendAttrib(100.0f, 40.0f, 0.5f, 0.0f);
		C3D_ImmSendAttrib(0.0f, 1.0f, 0.0f, 1.0f);

		C3D_ImmSendAttrib(300.0f, 40.0f, 0.5f, 0.0f);
		C3D_ImmSendAttrib(0.0f, 0.0f, 1.0f, 1.0f);
	C3D_ImmDrawEnd();
}
