/* PICA200 draw path for the GX back end: see pica_draw.h. */
#include "pica_draw.h"

#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <string.h>

#include "gx_pass_shbin.h"

#define DISPLAY_TRANSFER_FLAGS                                                                     \
	(GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |                 \
	 GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | \
	 GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

/* A frame's vertices and indices live in one half of each ring; the other half belongs to the
 * frame the GPU may still be drawing. */
#define VTX_RING_BYTES (3u << 20)
#define IDX_RING_BYTES (1u << 20)
#define CMDBUF_BYTES   (1u << 20)
#define CMDBUF_LIMIT   0.92f /* leave room for the frame's end */

static int s_ready, s_failed, s_in_frame;
static C3D_RenderTarget* s_top;
static DVLB_s* s_dvlb;
static shaderProgram_s s_prog;
static int s_uProj;
static C3D_Mtx s_proj;

static uint8_t* s_vring[2];
static uint8_t* s_iring[2];
static uint32_t s_half, s_vused, s_iused;
static uint32_t s_dropped;

static uint32_t s_clear_rgba = 0x000000FF, s_clear_depth = 0xFFFFFF;
static int s_clear_pending = 1;

/* Last state sent, so unchanged state is not re-sent for every batch. */
static PicaState s_last;
static int s_last_valid;

int pica_init(void)
{
	if (s_ready) return 1;
	if (s_failed) return 0;
	if (!C3D_Init(CMDBUF_BYTES)) goto fail;
	s_top = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
	if (!s_top) goto fail;
	C3D_RenderTargetSetOutput(s_top, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);

	s_dvlb = DVLB_ParseFile((u32*)gx_pass_shbin, gx_pass_shbin_size);
	if (!s_dvlb) goto fail;
	shaderProgramInit(&s_prog);
	shaderProgramSetVsh(&s_prog, &s_dvlb->DVLE[0]);
	C3D_BindProgram(&s_prog);
	s_uProj = shaderInstanceGetUniformLocation(s_prog.vertexShader, "projection");

	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	AttrInfo_AddLoader(ai, 0, GPU_FLOAT, 4);          /* v0: position (EFB pixels, homogeneous) */
	AttrInfo_AddLoader(ai, 1, GPU_UNSIGNED_BYTE, 4);  /* v1: colour 0..255 */

	/* EFB pixels (y down) onto the rotated top screen. Depth: z = -(GameCube depth), so the
	 * default depth map (-z/w) stores the GameCube's own depth and its compare modes carry over. */
	Mtx_OrthoTilt(&s_proj, 0.0f, (float)PICA_EFB_WIDTH, (float)PICA_EFB_HEIGHT, 0.0f, 0.0f, 1.0f, true);
	s_proj.r[2].x = 0.0f;
	s_proj.r[2].y = 0.0f;
	s_proj.r[2].z = -1.0f;
	s_proj.r[2].w = 0.0f;
	C3D_DepthMap(true, -1.0f, 0.0f);

	/* Untextured: the fragment is the vertex colour. */
	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
	C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
	C3D_CullFace(GPU_CULL_NONE);

	for (int i = 0; i < 2; i++) {
		s_vring[i] = (uint8_t*)linearAlloc(VTX_RING_BYTES);
		s_iring[i] = (uint8_t*)linearAlloc(IDX_RING_BYTES);
		if (!s_vring[i] || !s_iring[i]) goto fail;
	}
	s_ready = 1;
	fprintf(stderr, "[pica] citro3d ready: EFB %dx%d -> top screen 400x240, rings %u+%u KB x2\n",
	        PICA_EFB_WIDTH, PICA_EFB_HEIGHT, VTX_RING_BYTES >> 10, IDX_RING_BYTES >> 10);
	return 1;
fail:
	s_failed = 1;
	fprintf(stderr, "[pica] citro3d init failed (linear free %lu KB)\n", (unsigned long)(linearSpaceFree() >> 10));
	return 0;
}

static void frame_begin(void)
{
	C3D_FrameBegin(C3D_FRAME_SYNCDRAW);
	s_half ^= 1;
	s_vused = s_iused = 0;
	s_last_valid = 0;
	if (s_clear_pending) {
		C3D_RenderTargetClear(s_top, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
		s_clear_pending = 0;
	}
	C3D_FrameDrawOn(s_top);
	C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, s_uProj, &s_proj);
	s_in_frame = 1;
}

int pica_alloc(uint32_t nverts, uint32_t nidx, PicaVtx** verts, uint16_t** idx)
{
	if (!pica_init()) return 0;
	if (!s_in_frame) frame_begin();
	uint32_t vbytes = nverts * (uint32_t)sizeof(PicaVtx);
	uint32_t ibytes = (nidx * 2u + 3u) & ~3u;
	if (nverts > 0xFFFF || s_vused + vbytes > VTX_RING_BYTES || s_iused + ibytes > IDX_RING_BYTES ||
	    C3D_GetCmdBufUsage() > CMDBUF_LIMIT) {
		s_dropped++;
		return 0;
	}
	*verts = (PicaVtx*)(s_vring[s_half] + s_vused);
	*idx = (uint16_t*)(s_iring[s_half] + s_iused);
	s_vused += (vbytes + 15u) & ~15u;
	s_iused += ibytes;
	return 1;
}

static void apply_state(const PicaState* st)
{
	if (s_last_valid && memcmp(st, &s_last, sizeof *st) == 0) return;
	C3D_DepthTest(st->depth_test, (GPU_TESTFUNC)st->depth_func,
	              (GPU_WRITEMASK)((st->color_write ? GPU_WRITE_COLOR : 0) | (st->depth_write ? GPU_WRITE_DEPTH : 0)));
	if (st->blend)
		C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, (GPU_BLENDFACTOR)st->blend_src, (GPU_BLENDFACTOR)st->blend_dst,
		               (GPU_BLENDFACTOR)st->blend_src, (GPU_BLENDFACTOR)st->blend_dst);
	else
		C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	s_last = *st;
	s_last_valid = 1;
}

void pica_draw_triangles(const PicaVtx* verts, const uint16_t* idx, uint32_t nidx, const PicaState* st)
{
	if (!s_in_frame || nidx == 0) return;
	apply_state(st);
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, verts, sizeof(PicaVtx), 2, 0x10);
	C3D_DrawElements(GPU_TRIANGLES, (int)nidx, C3D_UNSIGNED_SHORT, idx);
}

void pica_clear(uint32_t rgba, uint32_t depth24)
{
	s_clear_rgba = rgba;
	s_clear_depth = depth24 & 0xFFFFFF;
	if (s_in_frame) {
		C3D_RenderTargetClear(s_top, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
		C3D_FrameDrawOn(s_top);
		s_last_valid = 0;
	} else {
		s_clear_pending = 1;
	}
}

void pica_present(void)
{
	if (!pica_init()) return;
	if (!s_in_frame) frame_begin();
	/* The GPU reads the rings from memory: write back what the CPU wrote this frame. */
	if (s_vused) GSPGPU_FlushDataCache(s_vring[s_half], s_vused);
	if (s_iused) GSPGPU_FlushDataCache(s_iring[s_half], s_iused);
	C3D_FrameEnd(0);
	s_in_frame = 0;
	s_clear_pending = 1; /* the GameCube clears the EFB as it copies it out */
}

uint32_t pica_dropped(void) { return s_dropped; }
