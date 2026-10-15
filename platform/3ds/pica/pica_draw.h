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

typedef struct {
	float x, y, z, w;  /* EFB pixels and GameCube depth, homogeneous */
	float u, v;        /* texture coordinate (normalised; v = 0 is the image's top row) */
	uint8_t rgba[4];
} PicaVtx;

/* A texture owned by this file (an opaque C3D_Tex). */
typedef struct PicaTex PicaTex;

/* PICA values (GPU_TESTFUNC, GPU_BLENDFACTOR), filled by the back end from BP registers. */
typedef struct {
	uint8_t depth_test, depth_func, depth_write;
	uint8_t blend, blend_src, blend_dst;
	uint8_t color_write;
	uint8_t alpha_test, alpha_func, alpha_ref; /* GPU_TESTFUNC */
	PicaTex* tex;                              /* null: vertex colour only */
	uint8_t wrap_s, wrap_t;                    /* GPU_TEXTURE_WRAP_PARAM */
	uint8_t linear;                            /* bilinear filtering */
} PicaState;

/* Lazily initialises citro3d and the top-screen render target. Returns 0 on failure. */
int pica_init(void);

/* Room for `nverts` vertices and `nidx` indices in this frame's GPU-visible ring. Begins a
 * frame first if none is open. Returns 0 (and draws nothing) when the ring or the command
 * buffer is full. */
int pica_alloc(uint32_t nverts, uint32_t nidx, PicaVtx** verts, uint16_t** idx);

/* Textures: power-of-two sizes from 8 to 1024. pica_tex_upload takes RGBA8 rows, top row
 * first, and tiles them for the PICA. Returns null when out of linear memory. */
PicaTex* pica_tex_create(uint32_t w, uint32_t h);
void pica_tex_upload(PicaTex* t, const uint8_t* rgba);
void pica_tex_free(PicaTex* t);
uint32_t pica_tex_bytes(const PicaTex* t);

/* Draws triangles from the last pica_alloc. */
void pica_draw_triangles(const PicaVtx* verts, const uint16_t* idx, uint32_t nidx, const PicaState* st);

/* Clears colour (0xRRGGBBAA) and depth (24-bit) of the frame in progress, or of the next one. */
void pica_clear(uint32_t rgba, uint32_t depth24);

/* Ends and presents the frame in progress (a GX display copy). */
void pica_present(void);

/* Draws and drops made because the ring or command buffer was full (for the status screen). */
uint32_t pica_dropped(void);

#ifdef __cplusplus
}
#endif
#endif
