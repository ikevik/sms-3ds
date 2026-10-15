/* PICA200 draw path for the GX back end: see pica_draw.h. */
#include "pica_draw.h"

#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
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

/* The frame is drawn into an offscreen "EFB" texture (unrotated, covering the GameCube's
 * 640x448 EFB), so EFB copies can be GPU texture copies of it; a display copy draws it onto
 * the rotated top screen as one quad. */
#define EFB_TEX_W 512
#define EFB_TEX_H 256

static int s_ready, s_failed, s_in_frame;
static C3D_RenderTarget* s_top;
static C3D_RenderTarget* s_efb;
static C3D_Tex s_efbTex;
static DVLB_s* s_dvlb;
static shaderProgram_s s_prog;
static int s_uProj;
static C3D_Mtx s_projTop; /* EFB pixels onto the rotated top screen (the present quad) */
static C3D_Mtx s_projEfb; /* EFB pixels onto the EFB texture (game draws) */

static uint8_t* s_vring[2];
static uint8_t* s_iring[2];
static uint32_t s_half, s_vused, s_iused;
static uint32_t s_vflushed, s_iflushed; /* ring bytes already written back for the GPU */
static uint32_t s_dropped, s_copies;

/* Copy viewer (pica_set_view): 0 shows the EFB as usual, n shows the n-th EFB copy made in the
 * previous frame instead, to inspect render-to-texture passes. */
#define MAX_VIEW_COPIES 4
static PicaTex* s_frameCopies[MAX_VIEW_COPIES];
static PicaTex* s_lastFrameCopies[MAX_VIEW_COPIES];
static int s_nFrameCopies, s_nLastFrameCopies, s_view;

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
	AttrInfo_AddLoader(ai, 1, GPU_FLOAT, 2);          /* v1: texture coordinate */
	AttrInfo_AddLoader(ai, 2, GPU_UNSIGNED_BYTE, 4);  /* v2: colour 0..255 */

	if (!C3D_TexInitVRAM(&s_efbTex, EFB_TEX_W, EFB_TEX_H, GPU_RGBA8)) goto fail;
	s_efb = C3D_RenderTargetCreateFromTex(&s_efbTex, GPU_TEXFACE_2D, 0, GPU_RB_DEPTH24_STENCIL8);
	if (!s_efb) goto fail;
	C3D_TexSetFilter(&s_efbTex, GPU_LINEAR, GPU_LINEAR);
	C3D_TexSetWrap(&s_efbTex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);

	/* EFB pixels (y down). Depth: z = -(GameCube depth), so the default depth map (-z/w) stores
	 * the GameCube's own depth and its compare modes carry over. */
	Mtx_OrthoTilt(&s_projTop, 0.0f, (float)PICA_EFB_WIDTH, (float)PICA_EFB_HEIGHT, 0.0f, 0.0f, 1.0f, true);
	Mtx_Ortho(&s_projEfb, 0.0f, (float)PICA_EFB_WIDTH, (float)PICA_EFB_HEIGHT, 0.0f, 0.0f, 1.0f, true);
	C3D_Mtx* m[2] = {&s_projTop, &s_projEfb};
	for (int i = 0; i < 2; i++) {
		m[i]->r[2].x = 0.0f;
		m[i]->r[2].y = 0.0f;
		m[i]->r[2].z = -1.0f;
		m[i]->r[2].w = 0.0f;
	}
	C3D_DepthMap(true, -1.0f, 0.0f);

	C3D_CullFace(GPU_CULL_NONE);

	for (int i = 0; i < 2; i++) {
		s_vring[i] = (uint8_t*)linearAlloc(VTX_RING_BYTES);
		s_iring[i] = (uint8_t*)linearAlloc(IDX_RING_BYTES);
		if (!s_vring[i] || !s_iring[i]) goto fail;
	}
	s_ready = 1;
	fprintf(stderr, "[pica] citro3d ready: EFB %dx%d drawn into a %dx%d texture -> top screen 400x240, "
	        "rings %u+%u KB x2\n", PICA_EFB_WIDTH, PICA_EFB_HEIGHT, EFB_TEX_W, EFB_TEX_H, VTX_RING_BYTES >> 10,
	        IDX_RING_BYTES >> 10);
	return 1;
fail:
	s_failed = 1;
	fprintf(stderr, "[pica] citro3d init failed (linear free %lu KB)\n", (unsigned long)(linearSpaceFree() >> 10));
	return 0;
}

/* ---- textures ------------------------------------------------------------------------- */
struct PicaTex {
	C3D_Tex tex;
	uint32_t bytes;
	C3D_RenderTarget* rt; /* EFB copy textures are drawn into */
};

/* A texture freed while a frame may still be drawing with it waits for two frame starts. */
#define MAX_DEFERRED 256
static PicaTex* s_dead[2][MAX_DEFERRED];
static int s_ndead[2];

static void release_dead(void)
{
	/* After C3D_FrameBegin the GPU has finished the previous frame (citro3d waits for it): the
	 * textures freed before that frame was submitted can go. */
	for (int i = 0; i < s_ndead[1]; i++) {
		if (s_dead[1][i]->rt) C3D_RenderTargetDelete(s_dead[1][i]->rt);
		C3D_TexDelete(&s_dead[1][i]->tex);
		free(s_dead[1][i]);
	}
	memcpy(s_dead[1], s_dead[0], sizeof(PicaTex*) * (size_t)s_ndead[0]);
	s_ndead[1] = s_ndead[0];
	s_ndead[0] = 0;
}

PicaTex* pica_tex_create(uint32_t w, uint32_t h)
{
	if (!pica_init()) return NULL;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	if (!C3D_TexInit(&t->tex, (u16)w, (u16)h, GPU_RGBA8)) {
		free(t);
		return NULL;
	}
	t->bytes = w * h * 4;
	return t;
}

/* The PICA stores textures in 8x8 tiles, texels in Morton (Z) order inside a tile, tile rows
 * in memory order from t = 0, and samples t = 0 at the image's bottom: the image is stored
 * bottom row first, so the back end's v = 0 (top row) lands where it samples. */
static inline uint32_t morton8(uint32_t x, uint32_t y)
{
	return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

void pica_tex_upload(PicaTex* t, const uint8_t* rgba)
{
	if (!t) return;
	uint32_t w = t->tex.width, h = t->tex.height;
	uint32_t* dst = (uint32_t*)t->tex.data;
	for (uint32_t y = 0; y < h; y++) {
		const uint8_t* row = rgba + (size_t)(h - 1 - y) * w * 4;
		uint32_t tileRow = (y >> 3) * (w >> 3) * 64;
		for (uint32_t x = 0; x < w; x++) {
			const uint8_t* p = row + x * 4;
			/* GPU_RGBA8 texel: 0xRRGGBBAA as a little-endian word */
			dst[tileRow + (x >> 3) * 64 + morton8(x & 7, y & 7)] =
			    ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
		}
	}
	GSPGPU_FlushDataCache(t->tex.data, t->bytes);
}

void pica_tex_free(PicaTex* t)
{
	if (!t) return;
	if (s_ndead[0] < MAX_DEFERRED) {
		s_dead[0][s_ndead[0]++] = t;
	} else {
		/* Should not happen; leak rather than free a texture the GPU may read. */
	}
}

uint32_t pica_tex_bytes(const PicaTex* t) { return t ? t->bytes : 0; }

static void frame_begin(void)
{
	/* No C3D_FRAME_SYNCDRAW: that waits for the 3DS's vblank, and this runs on the game thread
	 * while it holds the emulated CPU (sms-pc-port's OS scheduler). The game paces itself on
	 * its emulated VI retrace; blocking here as well left the retrace already due whenever the
	 * game waited for it, so the main thread never slept and lower-priority threads (stage
	 * setup, priority 17) were starved. citro3d still waits for the previous frame's GPU work. */
	C3D_FrameBegin(0);
	release_dead();
	s_half ^= 1;
	s_vused = s_iused = s_vflushed = s_iflushed = 0;
	s_last_valid = 0;
	if (s_clear_pending) {
		C3D_RenderTargetClear(s_efb, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
		s_clear_pending = 0;
	}
	C3D_FrameDrawOn(s_efb);
	C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, s_uProj, &s_projEfb);
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
	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	if (st->tex) {
		C3D_Tex* tex = &st->tex->tex;
		C3D_TexSetWrap(tex, (GPU_TEXTURE_WRAP_PARAM)st->wrap_s, (GPU_TEXTURE_WRAP_PARAM)st->wrap_t);
		C3D_TexSetFilter(tex, st->linear ? GPU_LINEAR : GPU_NEAREST, st->linear ? GPU_LINEAR : GPU_NEAREST);
		C3D_TexBind(0, tex);
		C3D_TexEnvSrc(env, C3D_Both, GPU_TEXTURE0, GPU_PRIMARY_COLOR, 0);
		C3D_TexEnvFunc(env, C3D_Both, GPU_MODULATE);
	} else {
		C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
		C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
	}
	C3D_AlphaTest(st->alpha_test, (GPU_TESTFUNC)st->alpha_func, st->alpha_ref);
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
	BufInfo_Add(bi, verts, sizeof(PicaVtx), 3, 0x210);
	C3D_DrawElements(GPU_TRIANGLES, (int)nidx, C3D_UNSIGNED_SHORT, idx);
}

void pica_clear(uint32_t rgba, uint32_t depth24)
{
	s_clear_rgba = rgba;
	s_clear_depth = depth24 & 0xFFFFFF;
	if (s_in_frame) {
		C3D_RenderTargetClear(s_efb, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
		C3D_FrameDrawOn(s_efb);
		s_last_valid = 0;
	} else {
		s_clear_pending = 1;
	}
}

/* The GPU reads the rings from memory: write back what the CPU wrote since the last time. */
static void flush_rings(void)
{
	if (s_vused > s_vflushed) GSPGPU_FlushDataCache(s_vring[s_half] + s_vflushed, s_vused - s_vflushed);
	if (s_iused > s_iflushed) GSPGPU_FlushDataCache(s_iring[s_half] + s_iflushed, s_iused - s_iflushed);
	s_vflushed = s_vused;
	s_iflushed = s_iused;
}

void pica_efb_uv(float x, float y, float* s, float* t)
{
	*s = x * (1.0f / PICA_EFB_WIDTH);
	*t = 1.0f - y * (1.0f / PICA_EFB_HEIGHT);
}

PicaTex* pica_copy_tex_create(void)
{
	if (!pica_init()) return NULL;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	if (!C3D_TexInitVRAM(&t->tex, EFB_TEX_W, EFB_TEX_H, GPU_RGBA8)) {
		free(t);
		return NULL;
	}
	t->rt = C3D_RenderTargetCreateFromTex(&t->tex, GPU_TEXFACE_2D, 0, -1);
	if (!t->rt) {
		C3D_TexDelete(&t->tex);
		free(t);
		return NULL;
	}
	t->bytes = EFB_TEX_W * EFB_TEX_H * 4;
	return t;
}

/* Draws the EFB texture over the whole of the bound target (EFB pixel space), as the present
 * pass and EFB copies do. alpha_one: alpha 1 instead of the EFB's (formats without alpha). */
static void draw_efb_quad(int alpha_one, C3D_Tex* src)
{
	PicaVtx* v;
	uint16_t* idx;
	uint32_t keep = s_dropped;
	int ok = pica_alloc(4, 6, &v, &idx);
	s_dropped = keep; /* not a game draw */
	if (!ok) return;
	static const float corners[4][2] = {{0, 0}, {PICA_EFB_WIDTH, 0}, {PICA_EFB_WIDTH, PICA_EFB_HEIGHT},
	                                    {0, PICA_EFB_HEIGHT}};
	for (int i = 0; i < 4; i++) {
		v[i].x = corners[i][0];
		v[i].y = corners[i][1];
		v[i].z = 0.0f;
		v[i].w = 1.0f;
		pica_efb_uv(corners[i][0], corners[i][1], &v[i].u, &v[i].v);
		v[i].rgba[0] = v[i].rgba[1] = v[i].rgba[2] = v[i].rgba[3] = 255;
	}
	static const uint16_t q[6] = {0, 1, 2, 0, 2, 3};
	memcpy(idx, q, sizeof q);

	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	C3D_TexEnvSrc(env, C3D_RGB, GPU_TEXTURE0, 0, 0);
	C3D_TexEnvFunc(env, C3D_RGB, GPU_REPLACE);
	if (alpha_one) {
		C3D_TexEnvColor(env, 0xFFFFFFFF);
		C3D_TexEnvSrc(env, C3D_Alpha, GPU_CONSTANT, 0, 0);
	} else {
		C3D_TexEnvSrc(env, C3D_Alpha, GPU_TEXTURE0, 0, 0);
	}
	C3D_TexEnvFunc(env, C3D_Alpha, GPU_REPLACE);
	C3D_TexBind(0, src);
	C3D_AlphaTest(false, GPU_ALWAYS, 0);
	C3D_DepthTest(false, GPU_ALWAYS, GPU_WRITE_ALL & ~GPU_WRITE_DEPTH);
	C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, v, sizeof(PicaVtx), 3, 0x210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, idx);
	s_last_valid = 0;
}

void pica_efb_copy_to(PicaTex* dst, int alpha_one)
{
	if (!dst || !dst->rt || !pica_init()) return;
	if (!s_in_frame) frame_begin();
	/* A draw pass, not a texture copy: it stays in the frame's command list (no CPU wait) and
	 * can fix alpha, as the GameCube's copy does for EFB formats without alpha. */
	C3D_FrameDrawOn(dst->rt);
	draw_efb_quad(alpha_one, &s_efbTex);
	C3D_FrameDrawOn(s_efb);
	s_copies++;
	if (s_nFrameCopies < MAX_VIEW_COPIES) s_frameCopies[s_nFrameCopies++] = dst;
}

void pica_present(void)
{
	if (!pica_init()) return;
	if (!s_in_frame) frame_begin();

	/* The EFB texture as one quad over the whole top screen. */
	C3D_FrameDrawOn(s_top);
	C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, s_uProj, &s_projTop);
	C3D_Tex* show = &s_efbTex;
	if (s_view > 0 && s_view <= s_nLastFrameCopies) show = &s_lastFrameCopies[s_view - 1]->tex;
	draw_efb_quad(1, show);
	flush_rings();
	memcpy(s_lastFrameCopies, s_frameCopies, sizeof s_frameCopies);
	s_nLastFrameCopies = s_nFrameCopies;
	s_nFrameCopies = 0;
	C3D_FrameEnd(0);
	s_in_frame = 0;
	s_last_valid = 0;
	s_clear_pending = 1; /* the GameCube clears the EFB as it copies it out */
}

uint32_t pica_copies(void) { return s_copies; }

int pica_cycle_view(void)
{
	s_view = s_view >= s_nLastFrameCopies ? 0 : s_view + 1;
	return s_view;
}

int pica_view(void) { return s_view; }
int pica_view_count(void) { return s_nLastFrameCopies; }

uint32_t pica_dropped(void) { return s_dropped; }
