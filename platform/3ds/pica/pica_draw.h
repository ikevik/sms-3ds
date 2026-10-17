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
/* LA8 textures (16 bits per texel), written directly: pica_tex_data is the texture's memory in the
 * PICA's tiled layout (8x8 tiles, Morton order, rows from the image's bottom); pica_tex_written
 * makes the writes visible to the GPU. */
PicaTex* pica_tex_create_la8(uint32_t w, uint32_t h);
void* pica_tex_data(PicaTex* t);
void pica_tex_written(PicaTex* t);
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

/* ---- GX mode: GameCube vertices transformed by the vertex shader -------------------------
 * The vertices are the GX front half's packed format (position at 0; see gx_internal.h
 * VtxFmtLayout); fields the draw does not use are skipped. Offsets are -1 for absent fields:
 * those read fixed values (normal (0, 0, 1), matrix index fixed_idx). */
#define PICA_GX_MTX_ROWS 30 /* XF position/normal matrix rows the shader can index (PNMTX0-9) */

typedef struct {
	uint16_t stride;
	int16_t off_nrm, off_clr, off_tex[3], off_mtx;
	int16_t fixed_idx;
} PicaGxLayout;

typedef struct {
	float F[4][4];  /* view space -> clip: pica_gx_projection */
	float T[6][4];  /* texture unit u: rows 2u, 2u+1 over its source vector */
	float W[6][4];  /* unit u's source: W[2u] = (pos, normal, tex a, tex b), W[2u+1].x = tex c */
	float CS[4];    /* colour = vertex colour * CS + CM (0..1) */
	float CM[4];
	float LD[4];    /* light direction, and w: how much of the shade applies (0 or 1) */
} PicaGxUniforms;

/* The free tail of this frame's vertex ring (begins a frame if none is open): a batch can be
 * decoded there and drawn in place. pica_ring_claim then reserves the bytes it used; it fails
 * (returns 0) when anything else took ring space meanwhile. */
void* pica_ring_tail(uint32_t* avail);
int pica_ring_claim(const void* p, uint32_t bytes);

/* Room for raw vertex bytes and indices in this frame's ring (as pica_alloc). */
int pica_alloc_raw(uint32_t vbytes, uint32_t nidx, void** verts, uint16_t** idx);
/* F for GX-mode draws: the PICA projection applied after o (view space -> EFB pixels and
 * GameCube depth, homogeneous; rows). */
void pica_gx_projection(const float o[4][4], PicaGxUniforms* u);
/* Draws GX-mode triangles. pos_rows/nrm_rows are XF matrix rows 0-29 (normal rows padded),
 * uploaded when matrices_dirty. Returns 0 when the layout cannot be expressed (draw it in CPU
 * mode instead). */
int pica_draw_gx(const void* verts, uint32_t nverts, const PicaGxLayout* L, const uint16_t* idx, uint32_t nidx,
                 const PicaGxUniforms* u, int matrices_dirty, const float (*pos_rows)[4], const float (*nrm_rows)[4],
                 const PicaState* st);

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
/* The most vertex-ring bytes a frame used since the last call, and one frame's capacity. */
uint32_t pica_ring_peak(void);
uint32_t pica_ring_capacity(void);
/* Kernel threads this process holds now, or -1 if unknown (the limit is 32: [perf] watches for leaks). */
int pica_kernel_threads(void);
/* Draws dropped so far because a vertex/index ring or the GPU command buffer was full. */
void pica_drop_reasons(uint32_t* ring, uint32_t* cmdbuf);
/* Most GPU command bytes a frame used since the last call (reset by it), and the buffer's size. */
uint32_t pica_cmd_peak(void);
uint32_t pica_cmd_capacity(void);
/* Linear (GPU-visible) memory still free, in bytes. */
uint32_t pica_linear_free(void);
/* Mid-frame submissions of the command list so far (the GPU starts on a frame's first parts). */
uint32_t pica_splits(void);
/* The GX queue's entry counters, for stall reports. */
const char* pica_queue_state(void);
/* Mid-frame submission on (default) or off. */
void pica_set_split(int on);
/* Overlapped frames: the CPU builds the next frame while the GPU draws the last (pica_draw.c). */
void pica_set_overlap(int on);
int pica_overlap(void);
/* GPU completions the frame-start watchdog had to supply (the hardware never reported them). */
uint32_t pica_lost_completions(void);
/* System ticks spent waiting in C3D_FrameBegin for the GPU to finish the previous frame. */
uint64_t pica_wait_ticks(void);
/* Sum of citro3d's GPU drawing time (ms) over the frames begun so far. */
double pica_gpu_ms_sum(void);
/* Cycles the GPU test mode (0 normal, 1 game draws scissored to 8x8 pixels); returns the new mode. */
int pica_cycle_gpu_test(void);
int pica_gpu_test(void);
/* For a hang report: what the renderer is doing on the game thread, and draws in this frame. */
const char* pica_where(void);
uint32_t pica_frame_draws(void);
/* Runs a frame's pieces one at a time and logs which, if any, the GPU never finishes ([gputest]
 * lines); returns 0 at the first that hangs. Call before the game thread starts. */
int pica_selftest(void);
/* On: every draw is submitted and waited for on its own, and one the GPU never finishes is logged
 * ([gpucheck] lines) with its layout, textures and combiner stages. Very slow. */
void pica_set_gpu_check(int on);
/* Quitting: no more GPU work is submitted from any thread after this. */
void pica_request_stop(void);

#ifdef __cplusplus
}
#endif
#endif
