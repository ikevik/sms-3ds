/* PICA200 draw path for the GX back end (platform/3ds/gx/gx3ds_backend.cpp).
 *
 * The back end transforms GX vertices on the CPU into homogeneous positions in GameCube EFB
 * pixels (x right 0..640, y down 0..PICA_EFB_HEIGHT, z the GameCube depth 0..1, all times w)
 * and hands them here. This file owns citro3d and is built with the toolchain's defaults (it
 * includes libctru/citro3d headers, whose structs use variable-size enums), so its interface
 * uses plain integers only: the back end is built with the GameCube ABI (4-byte enums).
 *
 * All calls must come from one thread (the game thread, which runs GX). */
#ifndef SMS3DS_PICA_DRAW_H
#define SMS3DS_PICA_DRAW_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PICA_EFB_WIDTH  640
#define PICA_EFB_HEIGHT 448

#define PICA_MAX_TEX    3 /* texture units with their own coordinates (TEXTURE0-2) */
#define PICA_MAX_STAGES 6 /* texture combiner stages */

typedef struct {
	float x, y, z, w;            /* EFB pixels and GameCube depth, homogeneous */
	float tc[PICA_MAX_TEX][2];   /* texture coordinates per unit (normalised; v = 0 is the image's top row) */
	uint8_t rgba[4];
} PicaVtx;

/* A texture owned by this file (an opaque C3D_Tex). */
typedef struct PicaTex PicaTex;

/* One texture combiner stage, in PICA values: sources (GPU_TEVSOURCES of GPU_TEVSRC),
 * operands (GPU_TEVOPERANDS of GPU_TEVOP_RGB / GPU_TEVOP_A), GPU_COMBINEFUNC, GPU_TEVSCALE,
 * and the stage's constant colour (0xAABBGGRR). */
typedef struct {
	uint16_t src_rgb, src_alpha;
	uint16_t op_rgb, op_alpha;
	uint8_t func_rgb, func_alpha;
	uint8_t scale_rgb, scale_alpha;
	uint32_t color;
} PicaTevStage;

/* PICA values (GPU_TESTFUNC, GPU_BLENDFACTOR), filled by the back end from BP registers. The
 * PICA side compares states bytewise: zero the whole struct before filling it. */
typedef struct {
	uint8_t depth_test, depth_func, depth_write;
	uint8_t blend, blend_eq, blend_src, blend_dst;     /* colour: GPU_BLENDEQUATION, GPU_BLENDFACTOR */
	uint8_t blend_aeq, blend_asrc, blend_adst;        /* alpha */
	uint32_t blend_color;                             /* constant (0xAABBGGRR) */
	uint8_t write_mask;                               /* GPU_WRITE_RED..ALPHA (depth: depth_write) */
	uint8_t cull;                                     /* GPU_CULLMODE */
	uint8_t scissor;                                  /* scissor box in EFB pixels: x0, y0 (top) to x1, y1 */
	uint16_t sc_x0, sc_y0, sc_x1, sc_y1;
	uint8_t alpha_test, alpha_func, alpha_ref; /* GPU_TESTFUNC */
	PicaTex* tex[PICA_MAX_TEX];                /* texture units; null: unused */
	uint8_t wrap_s[PICA_MAX_TEX], wrap_t[PICA_MAX_TEX]; /* GPU_TEXTURE_WRAP_PARAM */
	uint8_t linear[PICA_MAX_TEX];              /* bilinear filtering */
	uint8_t nstages;                           /* combiner stages used; 0: vertex colour only */
	uint8_t buf_rgb, buf_alpha;                /* combiner buffer update masks (bit i: stage i) */
	uint32_t buf_color;                        /* combiner buffer's initial colour (0xAABBGGRR) */
	PicaTevStage stage[PICA_MAX_STAGES];
} PicaState;

/* Lazily initialises citro3d and the top-screen render target. Returns 0 on failure. */
int pica_init(void);

/* Room for `nverts` vertices and `nidx` indices in this frame's GPU-visible ring. Begins a
 * frame first if none is open. Returns 0 (and draws nothing) when the ring or the command
 * buffer is full. */
int pica_alloc(uint32_t nverts, uint32_t nidx, PicaVtx** verts, uint16_t** idx);

/* EFB pixel to EFB texture pixel: x right, rows from the texture's first (bottom) row. */
#define PICA_EFB_TEX_W 512
#define PICA_EFB_TEX_H 256

/* Textures: power-of-two sizes from 8 to 1024. pica_tex_upload takes RGBA8 rows, top row
 * first, and tiles them for the PICA. Returns null when out of linear memory. */
PicaTex* pica_tex_create(uint32_t w, uint32_t h);
void pica_tex_upload(PicaTex* t, const uint8_t* rgba);
void pica_tex_free(PicaTex* t);
uint32_t pica_tex_bytes(const PicaTex* t);

/* EFB copies: a copy texture is a whole-EFB snapshot (texture copy of the EFB render target).
 * pica_efb_uv maps an EFB pixel position to texture coordinates in such a texture (and in the
 * EFB texture). */
PicaTex* pica_copy_tex_create(void);
void pica_efb_copy_to(PicaTex* dst, int alpha_one); /* alpha_one: EFB format without alpha */
void pica_efb_uv(float x, float y, float* s, float* t);
uint32_t pica_copies(void);
/* Debug viewer: cycles what the top screen shows between the EFB (0) and each EFB copy of the
 * previous frame (1..n); returns the new view. */
int pica_cycle_view(void);
int pica_view(void);
int pica_view_count(void);

/* Draws triangles from the last pica_alloc. */
void pica_draw_triangles(const PicaVtx* verts, const uint16_t* idx, uint32_t nidx, const PicaState* st);

/* Clears colour (0xRRGGBBAA) and depth (24-bit) of the frame in progress, or of the next one. */
void pica_clear(uint32_t rgba, uint32_t depth24);
/* Clears an EFB rectangle (EFB pixels) of the frame in progress, in order with its draws, as a
 * texture copy's clear does. */
void pica_clear_rect(float x, float y, float w, float h, uint32_t rgba, uint32_t depth24);

/* Ends and presents the frame in progress (a GX display copy): the EFB onto the top screen. */
void pica_present(void);

/* Draws and drops made because the ring or command buffer was full (for the status screen). */
uint32_t pica_dropped(void);

#ifdef __cplusplus
}
#endif
#endif
