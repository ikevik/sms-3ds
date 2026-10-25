/* PICA200 draw path for the GX back end: see pica_draw.h. */
#include "pica_draw.h"
#include "pad_3ds.h" /* sms3ds_park */

#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gx_pass_shbin.h"
#include "gx_proj_shbin.h" /* gx_pass with projective texgens on the GPU path (gpu_projective = on) */
#include "vshader_shbin.h" /* the bring-up shader: [vbench] I1 (two outputs) */

#define DISPLAY_TRANSFER_FLAGS                                                                     \
	(GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) |                 \
	 GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | \
	 GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

/* A frame's vertices and indices live in one half of each ring; the other half belongs to the
 * frame the GPU may still be drawing. */
/* Per frame. Delfino Plaza peaks near 2.4 MB with resident vertices (the rest of linear memory goes
 * to them); without them, or on the CPU vertex path, it needs 5 MB (pica_set_big_ring). */
static uint32_t s_vringBytes = 3584u << 10;
#define VTX_RING_BYTES s_vringBytes
#define IDX_RING_BYTES (1u << 20)
#define CMDBUF_BYTES   (1u << 20)
/* Command-buffer words a draw may not eat into: the rest of the frame's state and its end. */
#define CMDBUF_RESERVE_WORDS (32u << 10 >> 2)

/* The frame is drawn into an offscreen "EFB" texture (unrotated, covering the GameCube's
 * 640x448 EFB), so EFB copies can be GPU texture copies of it; a display copy draws it onto
 * the rotated top screen as one quad. */
#define EFB_TEX_W 512
#define EFB_TEX_H 256

static int s_ready, s_failed, s_in_frame;
void pica_set_big_ring(int on) { if (!s_ready) s_vringBytes = (on ? 5120u : 3584u) << 10; }
static C3D_RenderTarget* s_top;
static C3D_RenderTarget* s_efb;
static C3D_Tex s_efbTex;
static DVLB_s* s_dvlb;
static shaderProgram_s s_prog;
/* Vertex shader uniforms (shaders/gx_pass.v.pica) and the vertex mode they are set up for. */
static int s_uF, s_uP, s_uN, s_uT, s_uTQ, s_uTQ1, s_uW, s_uCS, s_uCM, s_uAMB, s_uLP, s_uLC, s_uLF, s_uLA, s_uLK, s_uLD, s_bCpu;
static int s_bLit, s_bL[2], s_bA[2], s_bS[2], s_bTu[3], s_bTd[3], s_bPj; /* the GX shader's per-draw shortcuts */
static int s_projShader; /* gx_proj.v.pica is bound (pica_set_projective_shader) */
static C3D_Tex s_projTex; /* unit 0's texture in projection mode (PicaState.proj0) */
enum { MODE_NONE = -1, MODE_CPU = 0, MODE_GX = 1 };
static int s_mode = MODE_NONE;
static const C3D_Mtx* s_cpuProj; /* what F holds in CPU mode */
static int s_fIsCpu;              /* F currently holds s_cpuProj */
static PicaGxLayout s_gxLayout;   /* attribute setup of the last GX-mode draw */
static int s_gxLayoutValid;
static int s_gxMatricesLoaded;
static C3D_Mtx s_projTop; /* EFB pixels onto the rotated top screen (the present quad) */
static C3D_Mtx s_projEfb; /* EFB pixels onto the EFB texture (game draws) */
static C3D_Tex s_efbDepthTex; /* the EFB's depth buffer as an RGBA8 texture (Z copies) */

static uint8_t* s_vring[2];
static uint8_t* s_iring[2];
static uint32_t s_half, s_vused, s_iused;
static uint32_t s_vflushed, s_iflushed; /* ring bytes already written back for the GPU */
static uint32_t s_dropped, s_copies;
static uint32_t s_dropRing, s_dropCmd; /* why draws were dropped (pica_drop_reasons) */
static u32* s_cmdFrameStart;           /* where this frame's commands begin */
static uint32_t s_cmdPeakWords;        /* most command words a frame used (pica_cmd_peak) */
/* Mid-frame GPU submission (maybe_split). */
extern u8 __C3D_Context[];
void gxCmdQueueInterrupt(GSPGPU_Event irq); /* libctru gxqueue.c: GSP interrupt handler, not in its headers */
static gxCmdQueue_s* s_queue;    /* citro3d's GX queue, or NULL: no mid-frame submission */
static uint32_t s_splitAtWords;  /* cmd_words_used() at this frame's last split */
static uint32_t s_splits;        /* mid-frame submissions so far (pica_splits) */
static u16 s_partEntries; /* queue entries up to and including this frame's last part */
static uint32_t s_lostCompletions; /* GPU completions the watchdog supplied (pica_lost_completions) */
/* Off by default: with parts in flight, GSP stopped reporting GPU completions altogether after
 * a while (levels loading in Delfino Plaza; every later command "sent, 0 done"). R at launch. */
static int s_splitOff = 1; /* pica_set_split(0): whole frames only, as citro3d does */
void pica_set_split(int on) { s_splitOff = !on; }

static uint32_t s_ringPeak, s_ringPeakNow; /* vertex ring bytes used: most in a frame (pica_ring_peak) */

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
/* Texture-content generation: bumped when texels may have changed (apply_state binds again). */
static uint32_t s_texGen;
/* pica_set_state_skip(0): unchanged settings are set again anyway (to compare). */
static int s_skipOff;

/* pica_request_stop: quitting. Every entry point goes through pica_init, so nothing new reaches
 * the GPU after this, and what is in flight finishes before the screens are shut down. */
static volatile int s_stopping;
static u32 s_stopThread; /* the thread that asked (the main thread, quitting): it never parks */
void pica_request_stop(void)
{
	svcGetThreadId(&s_stopThread, CUR_THREAD_HANDLE);
	__dmb();
	s_stopping = 1;
}
/* In the GPU waits' loops: once the app is quitting, the thread waiting (the render thread) parks
 * rather than poll on, the GPU's completions gone with gfxExit (pad_3ds.h, sms3ds_park). */
static void park_if_stopping(void)
{
	if (!s_stopping) return;
	u32 id = 0;
	svcGetThreadId(&id, CUR_THREAD_HANDLE);
	if (id != s_stopThread) sms3ds_park();
}

/* libctru's linear heap has no lock of its own: with the GX back end's render thread
 * (gx3ds_backend.cpp) the game thread (resident vertices) and the render thread (textures, frames)
 * both allocate from it. Every linear allocation and free here takes this. */
static LightLock s_linLock = 1; /* 1: unlocked */
#define LIN_LOCK() LightLock_Lock(&s_linLock)
#define LIN_UNLOCK() LightLock_Unlock(&s_linLock)

/* VRAM for the EFB-copy textures, taken whole at start-up and handed out in power-of-two runs
 * of 4 KB units at multiples of their own size (copy_arena_alloc), so freed runs join up again.
 * Made from VRAM at large, a copy texture could find it free but split into smaller pieces by
 * texture copies (vram_try), and one that gets no VRAM leaves the game sampling raw memory at
 * its address (noise on screen). */
/* 1.25 MB: Delfino Plaza's copies peak at 772 KB; a copy that finds no room in the arena gets
 * VRAM of its own, made by evicting texture copies if needed (pica_copy_tex_create). */
#define COPY_ARENA_BYTES (1280u << 10)
#define COPY_UNIT 4096u
#define COPY_UNITS (COPY_ARENA_BYTES / COPY_UNIT)
static uint8_t* s_copyArena;
static uint8_t s_copyUnitUsed[COPY_UNITS];
static uint32_t s_copyUnits, s_copyUnitsPeak;

/* The GX shader's uniform registers (gx_pass.v.pica or gx_proj.v.pica, whichever is bound). */
static void gx_shader_uniforms(void)
{
	shaderInstance_s* vs = s_prog.vertexShader;
	s_uF = shaderInstanceGetUniformLocation(vs, "F");
	s_uP = shaderInstanceGetUniformLocation(vs, "P");
	s_uN = shaderInstanceGetUniformLocation(vs, "N");
	s_uT = shaderInstanceGetUniformLocation(vs, "T");
	s_uTQ = shaderInstanceGetUniformLocation(vs, "TQ");
	s_uTQ1 = shaderInstanceGetUniformLocation(vs, "TQ1");
	s_uW = shaderInstanceGetUniformLocation(vs, "W");
	s_uCS = shaderInstanceGetUniformLocation(vs, "CS");
	s_uCM = shaderInstanceGetUniformLocation(vs, "CM");
	s_uAMB = shaderInstanceGetUniformLocation(vs, "AMB");
	s_uLP = shaderInstanceGetUniformLocation(vs, "LP");
	s_uLC = shaderInstanceGetUniformLocation(vs, "LC");
	s_uLF = shaderInstanceGetUniformLocation(vs, "LF");
	s_uLA = shaderInstanceGetUniformLocation(vs, "LA");
	s_uLK = shaderInstanceGetUniformLocation(vs, "LK");
	s_uLD = shaderInstanceGetUniformLocation(vs, "LD");
	s_bCpu = shaderInstanceGetUniformLocation(vs, "cpu");
	s_bLit = shaderInstanceGetUniformLocation(vs, "lit");
	s_bL[0] = shaderInstanceGetUniformLocation(vs, "l0");
	s_bL[1] = shaderInstanceGetUniformLocation(vs, "l1");
	s_bA[0] = shaderInstanceGetUniformLocation(vs, "a0");
	s_bA[1] = shaderInstanceGetUniformLocation(vs, "a1");
	s_bS[0] = shaderInstanceGetUniformLocation(vs, "s0");
	s_bS[1] = shaderInstanceGetUniformLocation(vs, "s1");
	s_bPj = shaderInstanceGetUniformLocation(vs, "pj");
	for (int k = 0; k < 3; k++) {
		char n[4] = {'t', 'u', (char)('0' + k), 0};
		s_bTu[k] = shaderInstanceGetUniformLocation(vs, n);
		n[1] = 'd';
		s_bTd[k] = shaderInstanceGetUniformLocation(vs, n);
	}
}

/* gx_proj.v.pica in place of gx_pass (settings.txt gpu_projective = on): draws with projective
 * texgens on the GPU path (pica_gx_projective). Called once, before the game draws. */
int pica_set_projective_shader(int on)
{
	if (!on || s_projShader) return s_projShader;
	if (!pica_init()) return 0;
	DVLB_s* d = DVLB_ParseFile((u32*)gx_proj_shbin, gx_proj_shbin_size);
	if (!d) return 0;
	shaderProgramFree(&s_prog);
	DVLB_Free(s_dvlb);
	s_dvlb = d;
	shaderProgramInit(&s_prog);
	shaderProgramSetVsh(&s_prog, &s_dvlb->DVLE[0]);
	C3D_BindProgram(&s_prog);
	gx_shader_uniforms();
	s_mode = MODE_NONE;
	s_projShader = s_uTQ >= 0 && s_uTQ1 >= 0;
	return s_projShader;
}

int pica_gx_projective(void) { return s_projShader; }

int pica_init(void)
{
	if (s_stopping) return 0;
	if (s_ready) return 1;
	if (s_failed) return 0;
	if (!C3D_Init(CMDBUF_BYTES)) goto fail;
	s_top = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, -1); /* no depth: the present pass needs none */
	if (!s_top) goto fail;
	C3D_RenderTargetSetOutput(s_top, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);

	s_dvlb = DVLB_ParseFile((u32*)gx_pass_shbin, gx_pass_shbin_size);
	if (!s_dvlb) goto fail;
	shaderProgramInit(&s_prog);
	shaderProgramSetVsh(&s_prog, &s_dvlb->DVLE[0]);
	C3D_BindProgram(&s_prog);
	gx_shader_uniforms();

	if (!C3D_TexInitVRAM(&s_efbTex, EFB_TEX_W, EFB_TEX_H, GPU_RGBA8)) goto fail;
	s_efb = C3D_RenderTargetCreateFromTex(&s_efbTex, GPU_TEXFACE_2D, 0, GPU_RB_DEPTH24_STENCIL8);
	if (!s_efb) goto fail;
	/* Its depth buffer read as a texture (Z copies): D24S8 texels are laid out as RGBA8 ones. */
	memset(&s_efbDepthTex, 0, sizeof s_efbDepthTex);
	s_efbDepthTex.data = s_efb->frameBuf.depthBuf;
	s_efbDepthTex.fmt = GPU_RGBA8;
	s_efbDepthTex.size = EFB_TEX_W * EFB_TEX_H * 4;
	s_efbDepthTex.width = EFB_TEX_W;
	s_efbDepthTex.height = EFB_TEX_H;
	s_efbDepthTex.param = GPU_TEXTURE_MAG_FILTER(GPU_NEAREST) | GPU_TEXTURE_MIN_FILTER(GPU_NEAREST) |
	                      GPU_TEXTURE_WRAP_S(GPU_CLAMP_TO_EDGE) | GPU_TEXTURE_WRAP_T(GPU_CLAMP_TO_EDGE);
	C3D_TexSetFilter(&s_efbTex, GPU_LINEAR, GPU_LINEAR);
	C3D_TexSetWrap(&s_efbTex, GPU_CLAMP_TO_EDGE, GPU_CLAMP_TO_EDGE);
	s_copyArena = (uint8_t*)vramAlloc(COPY_ARENA_BYTES);
	if (!s_copyArena)
		fprintf(stderr, "[pica] no %u KB of VRAM for the EFB-copy textures: each takes its own\n", COPY_ARENA_BYTES >> 10);

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
		LIN_LOCK();
		s_vring[i] = (uint8_t*)linearAlloc(VTX_RING_BYTES);
		s_iring[i] = (uint8_t*)linearAlloc(IDX_RING_BYTES);
		LIN_UNLOCK();
		if (!s_vring[i] || !s_iring[i]) goto fail;
	}
	s_ready = 1;
	{
		/* Only with the queue layout this was written for (32 entries, set by C3D_Init). */
		gxCmdQueue_s* q = (gxCmdQueue_s*)__C3D_Context;
		s_queue = q->entries && q->maxEntries == 32 && q->numEntries <= q->maxEntries ? q : NULL;
		if (!s_queue) fprintf(stderr, "[pica] citro3d queue not as expected: no mid-frame GPU submission\n");
	}
	fprintf(stderr, "[pica] citro3d ready: EFB %dx%d drawn into a %dx%d texture -> top screen 400x240, "
	        "rings %u+%u KB x2\n", PICA_EFB_WIDTH, PICA_EFB_HEIGHT, EFB_TEX_W, EFB_TEX_H, (unsigned)(VTX_RING_BYTES >> 10),
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
	int fmt; /* PICA_TF_* (pica_tex_upload's packing) */
	C3D_RenderTarget* rt; /* EFB copy textures are drawn into */
	void* vram;           /* non-null: a copy of tex.data in VRAM (vram_copy) */
	C3D_Tex vtex;         /* tex, pointing at the VRAM copy (bound once vready) */
	uint8_t vready, vpending, dead, noVram;
	uint8_t copyLinear;             /* EFB copy textures: in linear FCRAM (no VRAM was left) */
	uint16_t arenaUnit, arenaUnits; /* EFB copy textures: their run in s_copyArena (0 units: own VRAM) */
	float wx, wy, ww, wh;           /* EFB copy textures: the EFB window (EFB pixels) of their latest copy */
	uint32_t lastUse, vramUse, vcopyFrame, vtryFrame; /* frame numbers (s_frameNo) */
};

/* A texture freed while a frame may still be drawing with it waits for two frame starts. */
#define MAX_DEFERRED 256
static PicaTex* s_dead[2][MAX_DEFERRED];
static int s_ndead[2];

static void tex_count(const PicaTex* t, int sign);
static void etc_count(const PicaTex* t, int sign);
static void vram_unpend(PicaTex* t);
static void vram_drop(PicaTex* t);
static void copy_tex_mem_free(PicaTex* t);
static int s_copyTexCount; /* EFB-copy textures in VRAM (copy_reserve, below) */
static int s_dead_due = 1; /* a frame has started since release_dead last ran */

static void release_dead(void)
{
	/* Runs just before C3D_FrameBegin, outside a frame (citro3d panics on C3D_RenderTargetDelete
	 * inside one: the EFB-copy textures freed at FLUDD's prompt). s_dead[1] holds the textures
	 * freed two frames ago; the C3D_FrameBegin of the last frame waited for the GPU to finish
	 * the frame that could still draw with them, so they can go. */
	for (int i = 0; i < s_ndead[1]; i++) {
		PicaTex* t = s_dead[1][i];
		if (t->vram) vram_drop(t);
		if (t->vpending) vram_unpend(t);
		if (t->rt) {
			C3D_RenderTargetDelete(t->rt);
			s_copyTexCount--;
		} else {
			tex_count(t, -1);
		}
		if (t->rt) {
			copy_tex_mem_free(t);
		} else {
			etc_count(t, -1);
			LIN_LOCK();
			C3D_TexDelete(&t->tex);
			LIN_UNLOCK();
		}
		free(s_dead[1][i]);
	}
	memcpy(s_dead[1], s_dead[0], sizeof(PicaTex*) * (size_t)s_ndead[0]);
	s_ndead[1] = s_ndead[0];
	s_ndead[0] = 0;
}

/* VRAM copies of textures. The GPU reads VRAM faster than FCRAM, where its reads also compete
 * with the CPU's; but the CPU may not write VRAM (a data abort), so each texture stays in linear
 * FCRAM, where it is written, and the GPU copies it into VRAM (GX TextureCopy) at the head of
 * the frame's GX queue, before the frame's draws, which then bind the VRAM copy. Only with
 * overlapped frames, whose queue is filled while stopped (ovl_frame_begin); copies that do not
 * fit in this frame's queue wait for the next frame start, drawn from FCRAM meanwhile.
 * A texture gets its VRAM copy when a draw first binds it (vram_try), so VRAM holds what the
 * current scene draws with; copies of textures no draw has bound for a while make room
 * (vram_evict), for other textures and for the EFB-copy textures (pica_copy_tex_create). */
/* VRAM kept free for EFB-copy textures that do not fit in the arena (s_copyArena), or for all of
 * them when it could not be made: 8 of them (the back end keeps up to 16, kMaxEfbCopies, most of
 * them small goop maps), one more while a replaced one waits out its two frames, each at most a
 * whole-EFB texture. */
#define COPY_TEX_BYTES (EFB_TEX_W * EFB_TEX_H * 4u)
static uint32_t copy_reserve(void)
{
	const int left = 9 - s_copyTexCount;
	return (!s_copyArena && left > 0 ? (uint32_t)left * COPY_TEX_BYTES : 0) + (64u << 10);
}
#define VRAM_IDLE 30 /* frames a texture goes unbound before its VRAM copy may go */
#define VRAM_MAX 1024
static uint32_t s_frameNo;
static PicaTex* s_vlist[VRAM_MAX]; /* textures with a VRAM copy */
static int s_nvlist;
static uint32_t s_vramEvicted, s_vramAnimated;
#define VRAM_PENDING 1024
static int s_vramTex = 1;
static uint32_t s_vramTexBytes, s_vramTexCount, s_linTexBytes, s_linTexCount, s_vramCopies;
static PicaTex* s_vpend[VRAM_PENDING];
static int s_nvpend;
static int ovl_queue_room(void);
static int s_ovl; /* overlapped frames (defined with them, below) */
void pica_set_vram_textures(int on) { s_vramTex = on; }
void pica_tex_stats(uint32_t* vramKB, uint32_t* vramCount, uint32_t* linKB, uint32_t* linCount, uint32_t* vramFreeKB)
{
	*vramKB = s_vramTexBytes >> 10;
	*vramCount = s_vramTexCount;
	*linKB = s_linTexBytes >> 10;
	*linCount = s_linTexCount;
	*vramFreeKB = (uint32_t)(vramSpaceFree() >> 10);
}
uint32_t pica_vram_copies(void) { return s_vramCopies; }
uint32_t pica_vram_evicted(void) { return s_vramEvicted; }
uint32_t pica_vram_animated(void) { return s_vramAnimated; }
static void tex_count(const PicaTex* t, int sign)
{
	if (t->vram) {
		s_vramTexBytes += sign * (int)t->bytes;
		s_vramTexCount += sign;
	} else {
		s_linTexBytes += sign * (int)t->bytes;
		s_linTexCount += sign;
	}
}
static bool tex_init(C3D_Tex* tex, uint32_t w, uint32_t h, int maxLevel, GPU_TEXCOLOR fmt)
{
	C3D_TexInitParams p = {(u16)w, (u16)h, (u8)maxLevel, fmt, GPU_TEX_2D, false};
	LIN_LOCK();
	const bool ok = C3D_TexInitWithParams(tex, NULL, p);
	LIN_UNLOCK();
	return ok;
}
/* Frees VRAM copies of textures no draw has bound in VRAM_IDLE frames (and with no copy into
 * them still possibly queued), least recently used first, until `want` bytes are free. */
static bool vram_evict(uint32_t want, uint32_t minAge)
{
	while (vramSpaceFree() < want) {
		int best = -1;
		uint32_t bestAge = 0;
		for (int i = 0; i < s_nvlist; i++) {
			const PicaTex* t = s_vlist[i];
			const uint32_t age = s_frameNo - t->vramUse; /* frames since a draw last bound the VRAM copy */
			if (age >= minAge && s_frameNo - t->vcopyFrame > 2 && !t->vpending && age > bestAge) {
				best = i;
				bestAge = age;
			}
		}
		if (best < 0) return false;
		vram_drop(s_vlist[best]);
		s_vramEvicted++;
	}
	return true;
}
static void vram_copy(PicaTex* t);
/* A draw binds t: give it a VRAM copy if there is (or can be made) room. */
static void vram_try(PicaTex* t)
{
	if (t->noVram) return;
	if (t->vtryFrame && s_frameNo - t->vtryFrame < VRAM_IDLE) return; /* no room lately */
	t->vtryFrame = s_frameNo;
	if (s_nvlist >= VRAM_MAX) return;
	const uint32_t need = t->bytes + copy_reserve() + 4096;
	if (vramSpaceFree() < need && !vram_evict(need, VRAM_IDLE)) return;
	void* p = vramAlloc(t->bytes);
	if (!p) return;
	tex_count(t, -1);
	t->vram = p;
	t->vramUse = s_frameNo;
	tex_count(t, 1);
	s_vlist[s_nvlist++] = t;
	t->vtryFrame = 0;
	vram_copy(t);
}
static void vram_drop(PicaTex* t)
{
	for (int i = 0; i < s_nvlist; i++)
		if (s_vlist[i] == t) {
			s_vlist[i] = s_vlist[--s_nvlist];
			break;
		}
	if (t->vpending) vram_unpend(t);
	tex_count(t, -1);
	vramFree(t->vram);
	t->vram = NULL;
	t->vready = 0;
	tex_count(t, 1);
}
/* For the EFB-copy textures: room for `bytes` more in VRAM, made by evicting the VRAM copies of
 * textures no frame the GPU may still be drawing has bound (3 frames). */
static void vram_make_room(uint32_t bytes) { vram_evict(bytes + 4096, 3); }
static void vram_unpend(PicaTex* t)
{
	for (int i = 0; i < s_nvpend; i++)
		if (s_vpend[i] == t) {
			s_vpend[i] = s_vpend[--s_nvpend];
			break;
		}
	t->vpending = 0;
}
/* Queues the copy into VRAM if this frame's queue has room, else for the next frame start. */
static void vram_copy(PicaTex* t)
{
	/* Rewritten again within a few frames (animated: water, movies): such a texture stays in
	 * FCRAM. Its copies cost a DMA every frame, and a copy could read the FCRAM original while
	 * the CPU rewrites it for the next frame. Its VRAM copy goes unbound and is evicted later. */
	if (t->vcopyFrame && s_frameNo - t->vcopyFrame < 8) {
		t->noVram = 1;
		t->vready = 0;
		if (t->vpending) vram_unpend(t);
		s_vramAnimated++;
		return;
	}
	if (s_in_frame && s_ovl && ovl_queue_room()) {
		GX_TextureCopy((u32*)t->tex.data, 0, (u32*)t->vram, 0, t->bytes, 8);
		s_texGen++;
		t->vready = 1;
		t->vcopyFrame = s_frameNo;
		s_vramCopies++;
		if (t->vpending) vram_unpend(t);
	} else {
		t->vready = 0;
		if (!t->vpending && s_nvpend < VRAM_PENDING) {
			t->vpending = 1;
			s_vpend[s_nvpend++] = t;
		}
	}
}
/* At a frame start, with the frame's queue bound and empty. */
static void vram_drain(void)
{
	for (int i = s_nvpend - 1; i >= 0 && ovl_queue_room(); i--) {
		PicaTex* t = s_vpend[i];
		if (t->dead || t->noVram) continue;
		vram_copy(t);
	}
}
/* Self-check of the VRAM copies (each run, one texture at a time): the GPU copies a texture's
 * VRAM copy back into FCRAM, and two frame starts later (its queue has run by then) it is
 * compared with the FCRAM original it was copied from. Mismatches are logged with the texture's
 * size and format, as are the totals now and then. */
#define VFY_BYTES (1024u << 10)
static uint8_t* s_vfyBuf;
static PicaTex* s_vfyTex;
static uint32_t s_vfyFrame, s_vfyBytes, s_vfyNext, s_vfyOk, s_vfyBad, s_vfyLog;
static int s_vfyOn; /* off unless asked for (pica_set_vram_check) */
void pica_set_vram_check(int on) { s_vfyOn = on; }
static void vram_verify(void)
{
	if (!s_vfyBuf) {
		LIN_LOCK();
		s_vfyBuf = (uint8_t*)linearAlloc(VFY_BYTES);
		LIN_UNLOCK();
		if (!s_vfyBuf) return;
	}
	if (s_vfyTex) {
		if (s_frameNo - s_vfyFrame < 2) return;
		PicaTex* t = s_vfyTex;
		s_vfyTex = NULL;
		bool live = false;
		for (int i = 0; i < s_nvlist && !live; i++) live = s_vlist[i] == t;
		/* Gone, or copied again since (its original may have changed): no verdict. */
		if (!live || t->dead || t->vpending || (int32_t)(t->vcopyFrame - s_vfyFrame) >= 0) return;
		GSPGPU_InvalidateDataCache(s_vfyBuf, s_vfyBytes);
		const uint8_t* a = (const uint8_t*)t->tex.data;
		uint32_t first = s_vfyBytes, diff = 0;
		for (uint32_t i = 0; i < s_vfyBytes; i++)
			if (a[i] != s_vfyBuf[i]) {
				if (first == s_vfyBytes) first = i;
				diff++;
			}
		if (!diff) {
			s_vfyOk++;
		} else {
			s_vfyBad++;
			if (s_vfyLog++ < 40)
				fprintf(stderr,
				        "[vram] copy differs: %ux%u fmt %d levels %d, %lu bytes, %lu differ from offset %lu | vram %p "
				        "fcram %p, copied frame %lu, now %lu\n",
				        t->tex.width, t->tex.height, t->fmt, t->tex.maxLevel + 1, (unsigned long)s_vfyBytes,
				        (unsigned long)diff, (unsigned long)first, t->vram, t->tex.data, (unsigned long)t->vcopyFrame,
				        (unsigned long)s_frameNo);
		}
		if ((s_vfyOk + s_vfyBad) % 100 == 0)
			fprintf(stderr, "[vram] self-check: %lu copies match, %lu differ\n", (unsigned long)s_vfyOk,
			        (unsigned long)s_vfyBad);
		return;
	}
	if (!s_nvlist || !ovl_queue_room()) return;
	for (int n = 0; n < s_nvlist; n++) {
		PicaTex* t = s_vlist[(s_vfyNext + n) % s_nvlist];
		if (!t->vready || t->bytes > VFY_BYTES || s_frameNo - t->vcopyFrame <= 2) continue;
		s_vfyNext = (s_vfyNext + n + 1) % s_nvlist;
		GX_TextureCopy((u32*)t->vram, 0, (u32*)s_vfyBuf, 0, t->bytes, 8);
		s_vfyTex = t;
		s_vfyFrame = s_frameNo;
		s_vfyBytes = t->bytes;
		return;
	}
}

/* The CPU's caches and what the GPU reads or writes (pica_set_cache_way): 0, the kernel calls GSP
 * makes for GSPGPU_FlushDataCache and GSPGPU_InvalidateDataCache, made on this thread instead; 1,
 * through GSP, as citro3d does; 2, as 0 with the whole heap only written back at each frame's end
 * (svcStoreProcessDataCache), not also dropped from the caches. On the New 3DS GSP did the frame's
 * whole-heap flush (2-3 ms) and the goop map's invalidate (2 ms) on core 1, and the GPU took ~2 ms
 * longer over a frame while either ran (log-reverts, 2026-10-23). A failed kernel call goes
 * through GSP. */
static int s_cacheWay;
static uint32_t s_cacheFails;
void pica_set_cache_way(int way) { s_cacheWay = way; }
int pica_cache_way(void) { return s_cacheWay; }
static int cache_kernel(Result r)
{
	if (R_SUCCEEDED(r)) return 1;
	if (!s_cacheFails++) fprintf(stderr, "[pica] a cache call failed (%08lx): through GSP\n", (unsigned long)r);
	return 0;
}
static void cache_flush(const void* p, uint32_t n)
{
	if (s_cacheWay == 1 || !cache_kernel(svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)(uintptr_t)p, n)))
		GSPGPU_FlushDataCache(p, n);
}
static void cache_invalidate(const void* p, uint32_t n)
{
	if (s_cacheWay == 1 || !cache_kernel(svcInvalidateProcessDataCache(CUR_PROCESS_HANDLE, (u32)(uintptr_t)p, n)))
		GSPGPU_InvalidateDataCache(p, n);
}
static void heap_writeback(void)
{
	extern u32 __ctru_linear_heap, __ctru_linear_heap_size;
	if (s_cacheWay == 2 && cache_kernel(svcStoreProcessDataCache(CUR_PROCESS_HANDLE, __ctru_linear_heap, __ctru_linear_heap_size)))
		return;
	cache_flush((void*)__ctru_linear_heap, __ctru_linear_heap_size);
}

static void tex_flush(PicaTex* t)
{
	s_texGen++; /* bound textures are bound again (apply_state): the bind empties the texture cache */
	cache_flush(t->tex.data, t->bytes);
	if (t->vram && !t->noVram) vram_copy(t);
}

/* ETC1 textures (settings.txt etc1_textures): encoded on upload (etc1_enc.c), counted for [perf]. */
uint64_t etc1_encode_tex(void* dst, uint32_t w, uint32_t h, const uint8_t* rgba, int alpha);
static uint32_t s_etcCount, s_etcBytes, s_etcEncodes;
static uint64_t s_etcTicks, s_etcErr, s_etcTexels;
static void etc_count(const PicaTex* t, int sign)
{
	if (t->fmt < PICA_TF_ETC1) return;
	s_etcCount += (uint32_t)sign;
	s_etcBytes += (uint32_t)sign * t->bytes;
}
static void etc_encode(void* dst, uint32_t w, uint32_t h, const uint8_t* rgba, int fmt)
{
	const uint64_t t0 = svcGetSystemTick();
	const uint64_t err = etc1_encode_tex(dst, w, h, rgba, fmt == PICA_TF_ETC1A4);
	s_etcTicks += svcGetSystemTick() - t0;
	s_etcEncodes++;
	s_etcErr += err;
	s_etcTexels += (uint64_t)w * h;
	/* the worst ones, for the log: under 32 dB (mean squared error over 3 channels above ~41) */
	const double mse = (double)err / (3.0 * w * h);
	if (mse > 41.0) {
		static int s_said;
		if (s_said++ < 40)
			fprintf(stderr, "[pica] ETC1%s %lux%lu texture at %.1f dB\n", fmt == PICA_TF_ETC1A4 ? "A4" : "",
			        (unsigned long)w, (unsigned long)h, 10.0 * log10(255.0 * 255.0 / mse));
	}
}
void pica_etc1_stats(uint32_t* count, uint32_t* kb, uint32_t* encodes, float* ms, float* db)
{
	*db = s_etcTexels && s_etcErr ? (float)(10.0 * log10(255.0 * 255.0 * 3.0 * (double)s_etcTexels / (double)s_etcErr)) : 0.f;
	*count = s_etcCount;
	*kb = s_etcBytes >> 10;
	*encodes = s_etcEncodes;
	*ms = (float)((double)s_etcTicks * 1000.0 / SYSCLOCK_ARM11);
}

static PicaTex* tex_create(uint32_t w, uint32_t h, GPU_TEXCOLOR fmt, uint32_t bpp)
{
	if (!pica_init()) return NULL;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	const bool ok = tex_init(&t->tex, w, h, 0, fmt);
	if (!ok) {
		free(t);
		return NULL;
	}
	(void)bpp;
	t->bytes = t->tex.size; /* (ETC1: half a byte a texel) */
	tex_count(t, 1);
	return t;
}

PicaTex* pica_tex_create(uint32_t w, uint32_t h) { return tex_create(w, h, GPU_RGBA8, 4); }
PicaTex* pica_tex_create_fmt(uint32_t w, uint32_t h, int fmt)
{
	static const GPU_TEXCOLOR kGpu[] = {GPU_RGBA8, GPU_RGB565, GPU_RGBA5551, GPU_RGBA4, GPU_LA8, GPU_L8, GPU_ETC1, GPU_ETC1A4};
	static const uint8_t kBpp[] = {4, 2, 2, 2, 2, 1, 0, 1};
	if (fmt < 0 || fmt > PICA_TF_ETC1A4) fmt = PICA_TF_RGBA8;
	PicaTex* t = tex_create(w, h, kGpu[fmt], kBpp[fmt]);
	if (t) t->fmt = fmt;
	if (t) etc_count(t, 1);
	return t;
}
int pica_tex_format(const PicaTex* t) { return t ? t->fmt : PICA_TF_RGBA8; }
PicaTex* pica_tex_create_la8(uint32_t w, uint32_t h)
{
	PicaTex* t = tex_create(w, h, GPU_LA8, 2);
	if (t) t->fmt = PICA_TF_LA8;
	return t;
}
void* pica_tex_data(PicaTex* t) { return t ? t->tex.data : NULL; }
void pica_tex_written(PicaTex* t)
{
	if (t) tex_flush(t);
}

/* The PICA stores textures in 8x8 tiles, texels in Morton (Z) order inside a tile, tile rows
 * in memory order from t = 0, and samples t = 0 at the image's bottom: the image is stored
 * bottom row first, so the back end's v = 0 (top row) lands where it samples. */
static inline uint32_t morton8(uint32_t x, uint32_t y)
{
	return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

static void tex_swizzle(uint32_t* dst, uint32_t w, uint32_t h, const uint8_t* rgba)
{
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
}

/* tex_swizzle for the 16- and 8-bit formats: the same tiling, each texel packed from RGBA8. */
static void tex_swizzle_small(void* dst, uint32_t w, uint32_t h, const uint8_t* rgba, int fmt)
{
	uint16_t* d16 = (uint16_t*)dst;
	uint8_t* d8 = (uint8_t*)dst;
	for (uint32_t y = 0; y < h; y++) {
		const uint8_t* row = rgba + (size_t)(h - 1 - y) * w * 4;
		const uint32_t tileRow = (y >> 3) * (w >> 3) * 64;
		for (uint32_t x = 0; x < w; x++) {
			const uint8_t* p = row + x * 4;
			const uint32_t i = tileRow + (x >> 3) * 64 + morton8(x & 7, y & 7);
			switch (fmt) {
			case PICA_TF_RGB565: d16[i] = (uint16_t)((p[0] >> 3) << 11 | (p[1] >> 2) << 5 | p[2] >> 3); break;
			case PICA_TF_RGBA5551:
				d16[i] = (uint16_t)((p[0] >> 3) << 11 | (p[1] >> 3) << 6 | (p[2] >> 3) << 1 | p[3] >> 7);
				break;
			case PICA_TF_RGBA4: d16[i] = (uint16_t)((p[0] >> 4) << 12 | (p[1] >> 4) << 8 | (p[2] >> 4) << 4 | p[3] >> 4); break;
			case PICA_TF_LA8: d16[i] = (uint16_t)(p[0] << 8 | p[3]); break;
			default: d8[i] = p[0]; break; /* L8 */
			}
		}
	}
}

void pica_tex_upload(PicaTex* t, const uint8_t* rgba)
{
	if (!t) return;
	if (t->fmt >= PICA_TF_ETC1)
		etc_encode(t->tex.data, t->tex.width, t->tex.height, rgba, t->fmt);
	else if (t->fmt == PICA_TF_RGBA8)
		tex_swizzle((uint32_t*)t->tex.data, t->tex.width, t->tex.height, rgba);
	else
		tex_swizzle_small(t->tex.data, t->tex.width, t->tex.height, rgba, t->fmt);
	tex_flush(t);
}

PicaTex* pica_tex_create_mip(uint32_t w, uint32_t h, int levels, float lodBias, int fmt)
{
	static const GPU_TEXCOLOR kGpu[] = {GPU_RGBA8, GPU_RGB565, GPU_RGBA5551, GPU_RGBA4, GPU_LA8, GPU_L8, GPU_ETC1, GPU_ETC1A4};
	if (!pica_init() || levels < 1) return NULL;
	if (fmt < 0 || fmt > PICA_TF_ETC1A4) fmt = PICA_TF_RGBA8;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	const bool ok = tex_init(&t->tex, w, h, levels - 1, kGpu[fmt]);
	if (!ok) {
		free(t);
		return NULL;
	}
	t->fmt = fmt;
	C3D_TexSetFilterMipmap(&t->tex, GPU_LINEAR);
	C3D_TexSetLodBias(&t->tex, lodBias);
	t->bytes = C3D_TexCalcTotalSize(t->tex.size, levels - 1);
	tex_count(t, 1);
	etc_count(t, 1);
	return t;
}

void pica_tex_upload_level(PicaTex* t, int level, const uint8_t* rgba)
{
	if (!t || level > t->tex.maxLevel) return;
	void* dst = C3D_Tex2DGetImagePtr(&t->tex, level, NULL);
	if (t->fmt >= PICA_TF_ETC1)
		etc_encode(dst, t->tex.width >> level, t->tex.height >> level, rgba, t->fmt);
	else if (t->fmt == PICA_TF_RGBA8)
		tex_swizzle((uint32_t*)dst, t->tex.width >> level, t->tex.height >> level, rgba);
	else
		tex_swizzle_small(dst, t->tex.width >> level, t->tex.height >> level, rgba, t->fmt);
	if (level == t->tex.maxLevel) tex_flush(t);
}

void pica_tex_free(PicaTex* t)
{
	if (!t) return;
	t->dead = 1;
	if (s_ndead[0] < MAX_DEFERRED) {
		s_dead[0][s_ndead[0]++] = t;
	} else {
		/* Should not happen; leak rather than free a texture the GPU may read. */
	}
}

uint32_t pica_tex_bytes(const PicaTex* t) { return t ? t->bytes : 0; }

/* Draw trace check: level 0 of t against what an upload of rgba (RGBA8, the texture's size), or
 * of raw (already in t's layout), would have written, as the GPU reads it (the CPU's cache lines
 * of it dropped first; uploads write them back). Returns how many bytes differ (*first: the
 * first's offset); *bytes: level 0's size. */
uint32_t pica_tex_check(PicaTex* t, const uint8_t* rgba, const void* raw, uint32_t* bytes, uint32_t* first)
{
	*bytes = 0;
	*first = 0;
	if (!t || !t->tex.data || t->rt) return 0;
	const uint32_t w = t->tex.width, h = t->tex.height;
	const uint32_t n = t->tex.size; /* level 0 */
	static uint8_t* s_exp;
	static uint32_t s_expBytes;
	if (s_expBytes < n) {
		free(s_exp);
		s_exp = (uint8_t*)malloc(n);
		s_expBytes = s_exp ? n : 0;
		if (!s_exp) return 0;
	}
	if (raw)
		memcpy(s_exp, raw, n);
	else if (t->fmt >= PICA_TF_ETC1)
		(void)etc1_encode_tex(s_exp, w, h, rgba, t->fmt == PICA_TF_ETC1A4);
	else if (t->fmt == PICA_TF_RGBA8)
		tex_swizzle((uint32_t*)s_exp, w, h, rgba);
	else
		tex_swizzle_small(s_exp, w, h, rgba, t->fmt);
	cache_invalidate(t->tex.data, n);
	const uint8_t* a = (const uint8_t*)t->tex.data;
	uint32_t diff = 0;
	for (uint32_t i = 0; i < n; i++)
		if (a[i] != s_exp[i]) {
			if (!diff) *first = i;
			diff++;
		}
	*bytes = n;
	return diff;
}
uint32_t pica_tex_phys(const PicaTex* t, int vram)
{
	if (!t) return 0;
	const void* p = vram ? t->vram : t->tex.data;
	return p ? osConvertVirtToPhys(p) : 0;
}
/* 1: draws bind its VRAM copy; 2: it is kept in FCRAM for good (rewritten often); 0: in FCRAM. */
int pica_tex_where(const PicaTex* t)
{
	if (!t) return 0;
	if (t->noVram) return 2;
	return t->vram && t->vready ? 1 : 0;
}

/* ---- vertex modes ---------------------------------------------------------------------- */
static void set_cpu_proj(const C3D_Mtx* m)
{
	s_cpuProj = m;
	s_fIsCpu = 0;
}

/* CPU mode: PicaVtx vertices (v0 position, v1-v3 texture coordinates, v4 colour). */
static void cpu_mode(void)
{
	if (s_mode != MODE_CPU) {
		C3D_AttrInfo* ai = C3D_GetAttrInfo();
		AttrInfo_Init(ai);
		AttrInfo_AddLoader(ai, 0, GPU_FLOAT, 4);          /* v0: position (EFB pixels, homogeneous) */
		AttrInfo_AddLoader(ai, 1, GPU_FLOAT, 2);          /* v1-v3: texture coordinates 0-2 */
		AttrInfo_AddLoader(ai, 2, GPU_FLOAT, 2);
		AttrInfo_AddLoader(ai, 3, GPU_FLOAT, 2);
		AttrInfo_AddLoader(ai, 4, GPU_UNSIGNED_BYTE, 4);  /* v4: colour 0..255 */
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bCpu, true);
		s_mode = MODE_CPU;
		s_gxLayoutValid = 0;
	}
	if (!s_fIsCpu && s_cpuProj) {
		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, s_uF, s_cpuProj);
		s_fIsCpu = 1;
	}
}

extern uint64_t GX3DS_OursTicks; /* gx3ds_backend.cpp: all timed renderer work */
static uint64_t s_waitTicks;      /* C3D_FrameBegin waits (pica_wait_ticks) */
uint64_t pica_wait_ticks(void) { return s_waitTicks; }
static double s_gpuMsSum;          /* citro3d's GPU time of each finished frame (pica_gpu_ms_sum) */
double pica_gpu_ms_sum(void) { return s_gpuMsSum; }
/* The frame's end split up (pica_end_ticks): each mark adds the ticks since the last one to its
 * part. Only frames ended by pica_present are timed (s_endTimed). */
static uint64_t s_endTicks[PICA_END_COUNT], s_endT;
static int s_endTimed;
static void end_mark(int part)
{
	if (!s_endTimed) return;
	const u64 t = svcGetSystemTick();
	s_endTicks[part] += t - s_endT;
	s_endT = t;
}
void pica_end_ticks(uint64_t out[PICA_END_COUNT]) { memcpy(out, s_endTicks, sizeof s_endTicks); }

/* GPU test (pica_cycle_gpu_test, the profiler): 1 scissors every game draw to 8x8 pixels, so the
 * GPU still transforms every vertex but shades almost no pixels; 2 (GX-mode draws) puts every
 * vertex at one point, so they are still fetched and shaded but nothing is clipped, set up or
 * drawn; 3 sends every GX-mode draw's setup but not the draw. The steps between their GPU times
 * are pixels, clipping and triangle setup, vertex work, and what is left (state, copies).
 * 5 is 1 with the GX-mode shader's work beyond the position left out (no lighting, no texture
 * coordinates), 6 is 5 with only the position (and matrix index) fetched: what lighter shaders and
 * fewer attributes would save. 7 binds one small texture in place of every texture without a VRAM
 * copy, 8 in place of every texture: what textures in main memory, and textures at all, cost.
 * 9 changes nothing drawn: like every test it only leaves out the copies' readbacks (the
 * profiler's game-thread step, while the GX writes are muted and the copies hold nothing real).
 * 10-15 are 3 with more of what the GPU still does left out, for the parts of that: 10 the
 * CPU-path draws, 11 the EFB copies' passes, 12 the copies' clears of their rectangles, 13 the
 * EFB's fill at each frame's start, 14 the GX-mode draws' setup too, 15 all of these (presenting
 * and the screen transfer left). */
static int s_gpuTest;
enum { SKIP_CPU = 1, SKIP_COPY = 2, SKIP_RECT = 4, SKIP_FILL = 8, SKIP_SETUP = 16 };
/* GX-mode draws left out (tests 3, 10-15) */
static inline int test_no_draws(void) { return s_gpuTest == 3 || (s_gpuTest >= 10 && s_gpuTest <= 15); }
/* Whether tests 10-15 leave out this part (SKIP_*) */
static inline int test_skips(int part)
{
	static const uint8_t kSkips[6] = {SKIP_CPU, SKIP_COPY, SKIP_RECT, SKIP_FILL, SKIP_SETUP, 31};
	return s_gpuTest >= 10 && s_gpuTest <= 15 && (kSkips[s_gpuTest - 10] & part);
}
/* What the GPU does besides the GX-mode draws, for [perf] (pica_fixed_stats): CPU-path draws, EFB
 * copies and the pixels drawn into them, copies' rectangle clears and their pixels (in the EFB
 * texture's), EFB fills, render-target switches. */
enum { FX_CPU, FX_COPY, FX_COPY_PX, FX_RECT, FX_RECT_PX, FX_FILL, FX_TARGET, FX_COUNT };
static uint32_t s_fx[FX_COUNT];
void pica_fixed_stats(uint32_t out[FX_COUNT]) { memcpy(out, s_fx, sizeof s_fx); }
static C3D_Tex s_smallTex; /* tests 7 and 8: 8x8 white */
static int s_smallTexReady;
int pica_cycle_gpu_test(void)
{
	s_gpuTest = (s_gpuTest + 1) % 2;
	s_last_valid = 0;
	return s_gpuTest;
}
int pica_gpu_test(void) { return s_gpuTest; }
/* Set from the moment a test starts until the render thread ends a frame with none on (frame_end):
 * the frames that may hold test draws. Their copies of maps the game draws back (the goop) are
 * left out (gx3ds_backend.cpp), so the map in VRAM is as before the test: the goop map is drawn
 * from it each frame, and a copy of a frame whose goop draws were cut would empty it. */
static volatile int s_testSeen;
int pica_gpu_test_seen(void) { return s_testSeen; }
void pica_set_gpu_test(int mode)
{
	if (mode) s_testSeen = 1;
	s_gpuTest = mode;
	s_last_valid = 0;    /* the scissor, the textures: set again */
	s_gxLayoutValid = 0; /* test 6 changes the attributes */
	if ((mode == 7 || mode == 8) && !s_smallTexReady && C3D_TexInit(&s_smallTex, 8, 8, GPU_RGBA8)) {
		memset(s_smallTex.data, 0xFF, s_smallTex.size);
		GSPGPU_FlushDataCache(s_smallTex.data, s_smallTex.size);
		C3D_TexSetFilter(&s_smallTex, GPU_LINEAR, GPU_LINEAR);
		C3D_TexSetWrap(&s_smallTex, GPU_REPEAT, GPU_REPEAT);
		s_smallTexReady = 1;
	}
}
/* One object's draws cut to 8x8 pixels (the profiler's per-object part). */
static int s_drawCut;
void pica_set_draw_cut(int cut)
{
	if (cut != s_drawCut) s_last_valid = 0;
	s_drawCut = cut;
}

/* Where the game thread is in the renderer, for the main thread's hang report (pica_where). */
static const char* volatile s_where = "not started";
static volatile uint32_t s_frameDraws;
const char* pica_where(void) { return s_where; }
static int s_check;                         /* pica_set_gpu_check */
static uint32_t s_checkDraws, s_checkFrames;
static void s_checkFramesInc(void)
{
	if (!s_check) return;
	if (++s_checkFrames % 30 == 0)
		fprintf(stderr, "[gpucheck] %lu frames, %lu draws checked, none hung\n", (unsigned long)s_checkFrames,
		        (unsigned long)s_checkDraws);
}
uint32_t pica_frame_draws(void) { return s_frameDraws; }

/* Overlapped frames (pica_set_overlap). citro3d's C3D_FrameBegin waits for the GPU to finish the
 * previous frame before the CPU may build the next, so CPU and GPU took turns (each about 50-60
 * ms a frame in Delfino Plaza). Instead, two command buffers and two GX queues alternate: frame
 * N is built into one while the GPU runs frame N-1 from the other, and the wait for N-1 moves to
 * the end of frame N, just before N is queued. Each queue is filled while stopped and run whole,
 * once the other has finished, as citro3d does; nothing is added to a running queue (the
 * mid-frame parts above did that, and GSP then lost completions). libctru's queue code (2.x):
 * one queue runs at a time, gxCmdQueueWait waits for whichever runs, gxCmdQueueAdd on a queue
 * that is not running only stores the command. citro3d's frame calls are replaced here: its
 * C3D_FrameBegin would wait, and C3D_FrameDrawOn/C3D_FrameSplit need its own in-frame flag. */
bool C3Di_SplitFrame(u32** pBuf, u32* pSize); /* citro3d internal (base.c), exported */
#define OVL_ENTRIES 96
#define OVL_COPY_LIMIT 64 /* entries texture copies may fill; the rest for the frame's own */
static int s_ovlWanted;           /* pica_set_overlap */
static int s_ovl;                 /* frames are overlapped from the next frame start on */
static gxCmdQueue_s s_ovq[2];
static gxCmdEntry_s s_ovEntries[2][OVL_ENTRIES];
static u32* s_ovCmd[2];
static u32 s_ovCmdWords;
static int s_ovb;                         /* queue and command buffer of the frame being built */
static gxCmdQueue_s* s_ovRunning;         /* queued last; the GPU may still be running it */
static volatile int s_ovSwap[2];          /* the queue ends with a transfer to the top screen */
static volatile u64 s_ovRunT0, s_ovDoneT; /* when the last queue started and finished */
static u32* s_c3dCmdBuf;                  /* citro3d's command buffer (seen at its frame starts) */
static u32 s_c3dCmdWords;
void pica_set_overlap(int on) { s_ovlWanted = on; }
/* The whole linear heap written back at each frame's end, as C3D_FrameEnd does (pica_set_heap_flush;
 * settings.txt heap_flush). The vertex and index rings, resident vertices, textures and command
 * lists are also flushed by their own ranges, but something the GPU reads still needs the whole
 * heap: with it left out from launch, the GPU stopped completing work at the first scene change
 * (the Nintendo logo to the intro movie), every frame then waiting 500 ms. Left out only while
 * standing in Delfino Plaza it saved the game thread 0.6 ms and the frame 1.1 (the profiler's
 * game-thread steps). On by default until what else needs it is found. */
static int s_heapFlush = 1;
void pica_set_heap_flush(int on) { s_heapFlush = on; }
int pica_heap_flush(void) { return s_heapFlush; }
int pica_overlap(void) { return s_ovl; }

static void ovl_done(gxCmdQueue_s* q)
{
	/* GSP interrupt (libctru's queue callback): the frame is drawn and on its way to the screen. */
	s_ovDoneT = svcGetSystemTick();
	const int i = q == &s_ovq[1];
	if (s_ovSwap[i]) {
		s_ovSwap[i] = 0;
		gfxScreenSwapBuffers(GFX_TOP, false);
	}
}

/* Switches over from citro3d's own frames, between two frames (the GPU self-test runs on those). */
static int ovl_start(void)
{
	if (!s_queue || !s_c3dCmdBuf) return 0;
	s_ovCmd[0] = s_c3dCmdBuf;
	s_ovCmdWords = s_c3dCmdWords;
	LIN_LOCK();
	s_ovCmd[1] = (u32*)linearAlloc(s_ovCmdWords * 4u);
	LIN_UNLOCK();
	if (!s_ovCmd[1]) {
		fprintf(stderr, "[pica] no linear memory for a second command buffer: frames not overlapped\n");
		s_ovlWanted = 0;
		return 0;
	}
	gxCmdQueueWait(s_queue, -1); /* citro3d's last frame */
	gxCmdQueueStop(s_queue);
	gxCmdQueueClear(s_queue);
	for (int i = 0; i < 2; i++) {
		memset(&s_ovq[i], 0, sizeof s_ovq[i]);
		s_ovq[i].entries = s_ovEntries[i];
		s_ovq[i].maxEntries = OVL_ENTRIES;
		gxCmdQueueSetCallback(&s_ovq[i], ovl_done, NULL);
	}
	s_ovb = 0;
	s_ovRunning = NULL;
	s_ovl = 1;
	fprintf(stderr, "[pica] frames overlapped: the CPU builds the next frame while the GPU draws (%lu KB command "
	        "buffers x2)\n", (unsigned long)(s_ovCmdWords >> 8));
	return 1;
}

/* Waits for the queue the GPU runs, with the watchdog of frame_begin_ex. */
static void ovl_wait_running(void)
{
	if (!s_ovRunning) return;
	const u64 w0 = svcGetSystemTick();
	u64 since = w0;
	while (!gxCmdQueueWait(s_ovRunning, 0)) {
		park_if_stopping();
		svcSleepThread(200000); /* 0.2 ms */
		if (svcGetSystemTick() - since > SYSCLOCK_ARM11 / 2) {
			fprintf(stderr, "[pica] GPU completion missing for 500 ms (%s): supplied it\n", pica_queue_state());
			gxCmdQueueInterrupt(GSPGPU_EVENT_P3D);
			s_lostCompletions++;
			since = svcGetSystemTick();
		}
	}
	const u64 w = svcGetSystemTick() - w0;
	s_waitTicks += w;
	GX3DS_OursTicks += w;
	gxCmdQueueStop(s_ovRunning);
	if (s_ovDoneT > s_ovRunT0) s_gpuMsSum += (double)(s_ovDoneT - s_ovRunT0) / (SYSCLOCK_ARM11 / 1000.0);
	s_ovRunning = NULL;
}

static void ovl_frame_begin(void)
{
	/* This frame's queue last ran two frames ago, and the last frame's end waited for it. */
	gxCmdQueue_s* q = &s_ovq[s_ovb];
	gxCmdQueueClear(q);
	GX_BindQueue(q);
	GPUCMD_SetBuffer(s_ovCmd[s_ovb], s_ovCmdWords, 0);
}
static int ovl_queue_room(void) { return s_ovq[s_ovb].numEntries < OVL_COPY_LIMIT; }

static void readback_arrived(void);
static void readback_finish(void);
/* A screenshot (main.cpp: autorun.txt shots): the EFB as the GPU finished the last frame, saved as
 * sdmc:/sms/shot-<id>.ppm, its 512x256 texels (the texture's first row is the picture's top; 8x8
 * tiles, Morton order inside each). Taken at the next frame's end,
 * once that frame's queue has finished and before the next one (with its clear) runs. */
static volatile int s_shotQ[64];  /* asked for (main thread), saved one a frame (render thread) */
static volatile unsigned s_shotIn, s_shotOut;
void pica_request_shot(int id)
{
	if (s_shotIn - s_shotOut < 64) {
		s_shotQ[s_shotIn % 64] = id;
		__dmb();
		s_shotIn++;
	}
}
static void shot_save(int id)
{
	const C3D_Tex* t = &s_efbTex;
	const int w = t->width, h = t->height;
	char path[48];
	snprintf(path, sizeof path, "sdmc:/sms/shot-%d.ppm", id);
	FILE* f = fopen(path, "wb");
	if (!f) return;
	fprintf(f, "P6 %d %d 255\n", w, h);
	const uint8_t* src = (const uint8_t*)t->data;
	uint8_t* row = (uint8_t*)malloc((size_t)w * 3);
	for (int y = 0; row && y < h; y++) {
		const int r = y;
		for (int x = 0; x < w; x++) {
			const int ix = x & 7, iy = r & 7;
			const int m = (ix & 1) | (iy & 1) << 1 | (ix & 2) << 1 | (iy & 2) << 2 | (ix & 4) << 2 | (iy & 4) << 3;
			const uint8_t* p = src + (((size_t)(r >> 3) * (w >> 3) + (x >> 3)) * 64 + m) * 4; /* A B G R */
			row[x * 3] = p[3];
			row[x * 3 + 1] = p[2];
			row[x * 3 + 2] = p[1];
		}
		fwrite(row, 3, (size_t)w, f);
	}
	free(row);
	fclose(f);
	fprintf(stderr, "[pica] screenshot saved: %s\n", path);
	/* The presented top screen too (240 x 400, the screen's portrait layout): what is shown. */
	snprintf(path, sizeof path, "sdmc:/sms/shot-%d-top.ppm", id);
	const uint8_t* top = (const uint8_t*)s_top->frameBuf.colorBuf;
	if (!top || !(f = fopen(path, "wb"))) return;
	fprintf(f, "P6 240 400 255\n");
	uint8_t trow[240 * 3];
	for (int y = 0; y < 400; y++) {
		for (int x = 0; x < 240; x++) {
			const int ix = x & 7, iy = y & 7;
			const int m = (ix & 1) | (iy & 1) << 1 | (ix & 2) << 1 | (iy & 2) << 2 | (ix & 4) << 2 | (iy & 4) << 3;
			const uint8_t* p = top + (((size_t)(y >> 3) * 30 + (x >> 3)) * 64 + m) * 4;
			trow[x * 3] = p[3];
			trow[x * 3 + 1] = p[2];
			trow[x * 3 + 2] = p[1];
		}
		fwrite(trow, 3, 240, f);
	}
	fclose(f);
}
static void readback_queue(void);
/* Readbacks converted after the frame's queue starts (pica_set_readback_late; the profiler's revert
 * steps): first, the GPU waited for them, some 3 ms a frame on the New 3DS (invalidating the
 * 256 KB goop map's buffer 2.1, converting it 1.0) whenever it was the frame's limit. */
static int s_rbLate = 1;
void pica_set_readback_late(int on) { s_rbLate = on; }

static void ovl_frame_end(void)
{
	gxCmdQueue_s* q = &s_ovq[s_ovb];
	u32* buf;
	u32 size;
	if (C3Di_SplitFrame(&buf, &size)) GX_ProcessCommandList(buf, size * 4, GX_CMDLIST_FLUSH);
	GPUCMD_SetBuffer(NULL, 0, 0);
	end_mark(PICA_END_SUBMIT);
	if (s_heapFlush) heap_writeback(); /* as C3D_FrameEnd(0): the whole linear heap written back */
	end_mark(PICA_END_HEAP_FLUSH);
	s_where = "waiting for the GPU to finish the previous frame";
	ovl_wait_running();
	end_mark(PICA_END_WAIT);
	if (s_shotOut != s_shotIn) {
		__dmb();
		shot_save(s_shotQ[s_shotOut % 64]);
		s_shotOut++;
	}
	/* The previous frame's copies have arrived: converted into the game's memory once this frame's
	 * queue runs (s_rbLate), else first. */
	readback_arrived();
	if (!s_rbLate) readback_finish();
	end_mark(PICA_END_RB_CONVERT);
	readback_queue();  /* this frame's, fetched from VRAM after its draws */
	end_mark(PICA_END_RB_FETCH);
	/* The previous frame has swapped the screens: this transfer targets the new back buffer. */
	if (s_top->used) {
		s_top->used = false;
		C3D_FrameBufTransfer(&s_top->frameBuf, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);
		s_ovSwap[s_ovb] = 1;
	}
	s_ovRunT0 = svcGetSystemTick();
	gxCmdQueueRun(q);
	s_ovRunning = q;
	s_ovb ^= 1;
	end_mark(PICA_END_RUN);
	if (s_rbLate) {
		readback_finish();
		end_mark(PICA_END_RB_CONVERT);
	}
}

const char* pica_queue_state(void);

/* Linear memory for the GX back end's resident vertices (gx3ds_backend.cpp, display-list arena). */
void* pica_linear_alloc(uint32_t bytes)
{
	LIN_LOCK();
	void* p = linearAlloc(bytes);
	LIN_UNLOCK();
	return p;
}
void pica_linear_free_block(void* p)
{
	LIN_LOCK();
	linearFree(p);
	LIN_UNLOCK();
}
void pica_flush(const void* p, uint32_t bytes) { cache_flush(p, bytes); }
/* Waits until the GPU has finished every frame handed to it. */
void pica_gpu_idle(void)
{
	if (s_ovl) ovl_wait_running();
	else if (s_queue) gxCmdQueueWait(s_queue, -1);
}

/* C3D_FrameDrawOn, without citro3d's in-frame check in overlapped frames. */
static void draw_on(C3D_RenderTarget* t)
{
	s_texGen++; /* a texture drawn into is sampled with the texture cache emptied */
	s_fx[FX_TARGET]++;
	if (!s_ovl) {
		C3D_FrameDrawOn(t);
		return;
	}
	t->used = true;
	C3D_SetFrameBuf(&t->frameBuf);
	C3D_SetViewport(0, 0, t->frameBuf.width, t->frameBuf.height);
}

static void frame_end(void)
{
	if (s_ovl) ovl_frame_end();
	else C3D_FrameEnd(0);
	s_testSeen = s_gpuTest != 0;
}

/* flags: C3D_FRAME_NONBLOCK returns 0 instead of waiting while the GPU is busy. */
static int frame_begin_ex(u8 flags)
{
	/* No C3D_FRAME_SYNCDRAW: that waits for the 3DS's vblank, and this runs on the game thread
	 * while it holds the emulated CPU (sms-pc-port's OS scheduler). The game paces itself on
	 * its emulated VI retrace; blocking here as well left the retrace already due whenever the
	 * game waited for it, so the main thread never slept and lower-priority threads (stage
	 * setup, priority 17) were starved. citro3d still waits for the previous frame's GPU work. */
	if (s_dead_due) { /* once per started frame, even when a non-blocking begin fails */
		release_dead();
		s_dead_due = 0;
	}
	if (!s_ovl && s_ovlWanted && s_splitOff && !s_check) ovl_start();
	if (s_ovl) {
		ovl_frame_begin(); /* no wait: the GPU may still be drawing the last frame */
	} else {
	s_where = "C3D_FrameBegin (waiting for the GPU to finish the last frame)";
	{
		/* Time spent waiting here is the GPU (or the screen) holding the game back: counted as
		 * its own [perf] figure and left out of the front half's timers (GX3DS_OursTicks). */
		const u64 w0 = svcGetSystemTick();
		int ok;
		if (flags & C3D_FRAME_NONBLOCK) {
			ok = C3D_FrameBegin(flags);
		} else {
			/* Waiting for the GPU, with a watchdog: now and then the hardware never reports a GX
			 * command done (libctru then waits forever; seen with mid-frame parts and with the EFB
			 * clear run on its own). No frame takes the GPU 500 ms (80 ms at most so far), so
			 * after that the missing completion is supplied: a hitch instead of a freeze. */
			u64 since = w0;
			while (!(ok = C3D_FrameBegin(flags | C3D_FRAME_NONBLOCK))) {
				park_if_stopping();
				svcSleepThread(200000); /* 0.2 ms */
				if (s_queue && svcGetSystemTick() - since > SYSCLOCK_ARM11 / 2) {
					fprintf(stderr, "[pica] GPU completion missing for 500 ms (%s): supplied it\n",
					        pica_queue_state());
					gxCmdQueueInterrupt(GSPGPU_EVENT_P3D);
					s_lostCompletions++;
					since = svcGetSystemTick();
				}
			}
		}
		const u64 w = svcGetSystemTick() - w0;
		s_waitTicks += w;
		GX3DS_OursTicks += w;
		if (!ok) return 0;
		s_gpuMsSum += C3D_GetDrawingTime(); /* the frame just waited for */
	}
	s_c3dCmdBuf = gpuCmdBuf; /* citro3d's buffer, reused by overlapped frames */
	s_c3dCmdWords = gpuCmdBufSize;
	}
	s_where = "in a frame (game code or drawing)";
	s_dead_due = 1;
	s_frameNo++;
	s_cmdFrameStart = gpuCmdBuf;
	s_splitAtWords = 0;
	s_partEntries = 0;
	if (s_vused > s_ringPeakNow) s_ringPeakNow = s_vused;
	s_half ^= 1;
	s_vused = s_iused = s_vflushed = s_iflushed = 0;
	s_last_valid = 0;
	if (s_clear_pending) {
		if (!test_skips(SKIP_FILL)) {
			C3D_RenderTargetClear(s_efb, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
			s_fx[FX_FILL]++;
		}
		s_clear_pending = 0;
	}
	draw_on(s_efb);
	set_cpu_proj(&s_projEfb);
	s_in_frame = 1;
	if (s_ovl) vram_drain(); /* VRAM copies that did not fit in the last frame's queue */
	if (s_ovl && s_vfyOn) vram_verify();
	return 1;
}

static void frame_begin(void) { frame_begin_ex(0); }

/* Command-buffer room left in this frame. Not C3D_GetCmdBufUsage(): citro3d updates that only
 * when it splits the command list and never resets it at a frame's start, so after one frame ran
 * the buffer past the limit it kept reporting it full and every later draw was dropped (the
 * screen froze at the goop after FLUDD). libctru's own offset and size are always current. */
static int cmd_room(void) { return gpuCmdBufSize - gpuCmdBufOffset > CMDBUF_RESERVE_WORDS; }

static void flush_rings(void);

/* Handing the GPU the frame in parts. citro3d runs a frame's command list only at C3D_FrameEnd,
 * and SMS issues its draws right after the previous frame's display copy, so the next
 * C3D_FrameBegin waited for the GPU's whole frame (its drawing time and the wait matched to
 * 0.5 ms in Delfino Plaza, about 60 ms): CPU and GPU never worked at once. Every SPLIT_WORDS of
 * commands the list so far is queued and the queue started, and the GPU draws while the CPU
 * makes the rest. citro3d's queue (its context's first member, citro3d 1.7.1) holds 32 entries
 * a frame; the frame's end needs two (its last part and the screen transfer). */
#define SPLIT_WORDS (48u << 10 >> 2)
#define SPLIT_SPARE_ENTRIES 6

static uint32_t cmd_words_used(void);


/* The last part queued has completed (lastEntry counts completions). Only our own parts count:
 * the EFB clear at a frame's start queues a memory fill that waits until the queue first runs.
 * Running that fill early instead left the frame's command list never reported done (17:15
 * build, frame 2); see frame_begin_ex's watchdog for completions the hardware never reports. */
static int queue_idle(void)
{
	return *(volatile u16*)&s_queue->lastEntry >= s_partEntries;
}

/* Before C3D_FrameEnd queues the frame's last part: wait for the parts already running, so the
 * last list is again the only one in flight (see maybe_split). */
static void drain_parts(void)
{
	if (!s_queue || s_splitAtWords == 0) return;
	const u64 t0 = svcGetSystemTick();
	while (!queue_idle()) {
		park_if_stopping();
		svcSleepThread(100000); /* 0.1 ms */
		if (svcGetSystemTick() - t0 > SYSCLOCK_ARM11 / 4) break; /* 250 ms: the stall report says why */
	}
	const u64 w = svcGetSystemTick() - t0;
	s_waitTicks += w;
	GX3DS_OursTicks += w;
}

const char* pica_queue_state(void)
{
	static char buf[96];
	const gxCmdQueue_s* q = s_ovl ? (s_ovRunning ? s_ovRunning : &s_ovq[s_ovb]) : s_queue;
	if (!q) return "queue unknown";
	snprintf(buf, sizeof buf, "GX queue%s %u entries, %u sent, %u done", s_ovl ? " (overlapped)" : "",
	         (unsigned)q->numEntries, (unsigned)q->curEntry, (unsigned)q->lastEntry);
	return buf;
}

static void maybe_split(void)
{
	if (!s_queue || !s_in_frame || s_splitOff) return; /* the GPU check mode ends a frame after each draw */
	const uint32_t used = cmd_words_used();
	if (used - s_splitAtWords < SPLIT_WORDS || s_queue->numEntries + SPLIT_SPARE_ENTRIES > s_queue->maxEntries)
		return;
	/* Only once the GPU has finished every part queued so far. libctru keeps up to three GX
	 * commands in flight and counts their completions by interrupt; two command lists finishing
	 * back to back left it a completion short (GSP merges a repeated interrupt, it seems), and
	 * the next C3D_FrameBegin waited forever (Delfino Plaza, frame 791). One list in flight at a
	 * time cannot lose one. */
	if (!queue_idle()) return;
	flush_rings(); /* the vertices and indices the queued part draws */
	C3D_FrameSplit(GX_CMDLIST_FLUSH);
	s_partEntries = s_queue->numEntries;
	gxCmdQueueRun(s_queue); /* no-op once running; later parts start as they are added */
	s_splitAtWords = cmd_words_used();
	s_splits++;
}

uint32_t pica_splits(void) { return s_splits; }
uint32_t pica_lost_completions(void) { return s_lostCompletions; }

static uint32_t cmd_words_used(void)
{
	return s_cmdFrameStart ? (uint32_t)(gpuCmdBuf - s_cmdFrameStart) + gpuCmdBufOffset : 0;
}

static void count_drop(int ring_full)
{
	s_dropped++;
	if (ring_full) s_dropRing++;
	else s_dropCmd++;
}

int pica_alloc(uint32_t nverts, uint32_t nidx, PicaVtx** verts, uint16_t** idx)
{
	if (!pica_init()) return 0;
	if (!s_in_frame) frame_begin();
	uint32_t vbytes = nverts * (uint32_t)sizeof(PicaVtx);
	uint32_t ibytes = (nidx * 2u + 3u) & ~3u;
	if (nverts > 0xFFFF || s_vused + vbytes > VTX_RING_BYTES || s_iused + ibytes > IDX_RING_BYTES || !cmd_room()) {
		count_drop(cmd_room());
		return 0;
	}
	*verts = (PicaVtx*)(s_vring[s_half] + s_vused);
	*idx = (uint16_t*)(s_iring[s_half] + s_iused);
	s_vused += (vbytes + 15u) & ~15u;
	s_iused += ibytes;
	return 1;
}

int pica_alloc_raw(uint32_t vbytes, uint32_t nidx, void** verts, uint16_t** idx)
{
	if (!pica_init()) return 0;
	if (!s_in_frame) frame_begin();
	uint32_t ibytes = (nidx * 2u + 3u) & ~3u;
	if (s_vused + vbytes > VTX_RING_BYTES || s_iused + ibytes > IDX_RING_BYTES || !cmd_room()) {
		count_drop(cmd_room());
		return 0;
	}
	*verts = s_vring[s_half] + s_vused;
	*idx = (uint16_t*)(s_iring[s_half] + s_iused);
	s_vused += (vbytes + 15u) & ~15u;
	s_iused += ibytes;
	return 1;
}

static void apply_state(const PicaState* st, uint32_t same);
static void second_pass(GPU_Primitive_t prim, int count, const void* idx, const PicaState* st);
static void gpu_check(const char* kind, const PicaGxLayout* L, uint32_t nidx, const PicaState* st);

void* pica_ring_tail(uint32_t* avail)
{
	*avail = 0;
	if (!pica_init()) return NULL;
	if (!s_in_frame) frame_begin();
	*avail = VTX_RING_BYTES - s_vused;
	return s_vring[s_half] + s_vused;
}

int pica_ring_claim(const void* p, uint32_t bytes)
{
	if (!s_in_frame || p != s_vring[s_half] + s_vused || s_vused + bytes > VTX_RING_BYTES) return 0;
	s_vused += (bytes + 15u) & ~15u;
	return 1;
}

/* GX mode: the vertex shader's registers for each GX vertex field. */
enum { REG_POS = 0, REG_NRM, REG_CLR, REG_TA, REG_TB, REG_TC, REG_IDX, NUM_GX_REGS };

/* Sets up the attribute loaders for a GX vertex layout: the fields the draw uses are read from the
 * buffer in memory order, gaps skipped with padding entries (4-16 bytes each), absent fields given
 * fixed values. Returns the buffer's attribute count and permutation, or 0 when the layout needs
 * more entries than the PICA's 12. */
static int gx_attribs(const PicaGxLayout* L, u64* perm)
{
	struct Field { int off, size, reg, fmt, count; } f[NUM_GX_REGS];
	int nf = 0;
	const int offs[NUM_GX_REGS] = {0, L->off_nrm, L->off_clr, L->off_tex[0], L->off_tex[1], L->off_tex[2], L->off_mtx};
	/* The matrix index is read as 4 bytes (the shader uses the first): the GameCube block has 12. */
	static const int sizes[NUM_GX_REGS] = {12, 12, 4, 8, 8, 8, 4};
	static const int fmts[NUM_GX_REGS] = {GPU_FLOAT, GPU_FLOAT, GPU_UNSIGNED_BYTE, GPU_FLOAT, GPU_FLOAT, GPU_FLOAT,
	                                      GPU_UNSIGNED_BYTE};
	static const int counts[NUM_GX_REGS] = {3, 3, 4, 2, 2, 2, 4};
	for (int r = 0; r < NUM_GX_REGS; r++)
		if (offs[r] >= 0) f[nf++] = (struct Field){offs[r], sizes[r], r, fmts[r], counts[r]};
	/* memory order (insertion sort: at most 7 fields) */
	for (int i = 1; i < nf; i++)
		for (int j = i; j > 0 && f[j].off < f[j - 1].off; j--) {
			struct Field t = f[j];
			f[j] = f[j - 1];
			f[j - 1] = t;
		}
	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	u64 p = 0;
	int entries = 0, at = 0;
	for (int i = 0; i < nf; i++) {
		int gap = f[i].off - at;
		while (gap > 0) {
			int chunk = gap > 16 ? 16 : gap;
			if (chunk & 3 || entries >= 12) return 0;
			p |= (u64)(0xB + chunk / 4) << (4 * entries++); /* 0xC..0xF: 4..16 bytes of padding */
			gap -= chunk;
		}
		int attr = AttrInfo_AddLoader(ai, f[i].reg, (GPU_FORMATS)f[i].fmt, f[i].count);
		if (attr < 0 || entries >= 12) return 0;
		p |= (u64)attr << (4 * entries++);
		at = f[i].off + f[i].size;
	}
	/* The bytes after the last field are padding entries too: on a New 3DS the GPU locked up on a
	 * draw whose loaders covered 49 of its 60-byte vertices (Azahar stepped by the stride anyway). */
	for (int gap = (int)L->stride - at; gap > 0;) {
		int chunk = gap > 16 ? 16 : gap;
		if (chunk & 3 || entries >= 12) return 0;
		p |= (u64)(0xB + chunk / 4) << (4 * entries++);
		gap -= chunk;
	}
	for (int r = 0; r < NUM_GX_REGS; r++)
		if (offs[r] < 0) {
			int attr = AttrInfo_AddFixed(ai, r);
			if (attr < 0) return 0;
			if (r == REG_IDX)
				C3D_FixedAttribSet(attr, (float)L->fixed_idx, 0.0f, 0.0f, 1.0f);
			else if (r == REG_NRM)
				C3D_FixedAttribSet(attr, 0.0f, 0.0f, 1.0f, 0.0f);
			else
				C3D_FixedAttribSet(attr, 0.0f, 0.0f, 0.0f, 1.0f);
		}
	*perm = p;
	return entries;
}

static int s_gxEntries;
static u64 s_gxPerm;

/* Sets a vertex shader uniform only if it changed: citro3d uploads every vector marked dirty, and
 * a draw used to rewrite all 60 matrix rows and 19 other vectors (about 1.3 KB of GPU commands)
 * though usually one matrix had moved. C3D_FVUnif holds the values last set. */
/* GPU state sent, for [perf] (pica_state_stats): texture binds, combiner stages, matrix-row
 * uniforms, other uniform vectors, bool uniforms, draws (second passes too). */
enum { ST_TEXBIND, ST_STAGE, ST_ROWS, ST_UNIF, ST_BOOL, ST_DRAW, ST_COUNT };
static uint32_t s_stat[ST_COUNT];
void pica_state_stats(uint32_t out[6]) { memcpy(out, s_stat, sizeof s_stat); }

static inline void vunif(int id, float x, float y, float z, float w)
{
	const C3D_FVec* v = &C3D_FVUnif[GPU_VERTEX_SHADER][id];
	if (v->x != x || v->y != y || v->z != z || v->w != w) {
		C3D_FVUnifSet(GPU_VERTEX_SHADER, id, x, y, z, w);
		s_stat[(id >= s_uP && id < s_uP + PICA_GX_MTX_ROWS) || (id >= s_uN && id < s_uN + PICA_GX_MTX_ROWS) ? ST_ROWS
		                                                                                                    : ST_UNIF]++;
	}
}

static inline int vzero(const float* v) { return v[0] == 0.f && v[1] == 0.f && v[2] == 0.f && v[3] == 0.f; }

/* A vertex shader bool, set only if it changed: each set re-sends the bool register. */
static inline void bunif(int id, bool v)
{
	if (s_skipOff || (((C3D_BoolUnifs[GPU_VERTEX_SHADER] >> (id - 0x68)) & 1) != 0) != v)
	{
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, id, v);
		s_stat[ST_BOOL]++;
	}
}

/* The vertex buffer for the next draws, set only if it changed: touching the setup re-sends the
 * buffer registers. Every draw here sets its buffer through this. */
static const void* s_bufVerts;
static u64 s_bufPerm;
static int s_bufStride, s_bufEntries, s_bufValid;
static void set_buffer(const void* verts, int stride, int entries, u64 perm)
{
	if (!s_skipOff && s_bufValid && verts == s_bufVerts && stride == s_bufStride && entries == s_bufEntries &&
	    perm == s_bufPerm)
		return;
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, verts, stride, entries, perm);
	s_bufVerts = verts;
	s_bufStride = stride;
	s_bufEntries = entries;
	s_bufPerm = perm;
	s_bufValid = 1;
}

/* The shader's shortcuts for this draw (gx_pass.v.pica): each only skips work whose result the
 * uniforms already fix. Unshaded draws skip the normal and the shade; a unit whose rows T are zero
 * has the coordinate 0 either way; a unit whose source is exactly its own attribute (unit N from
 * v(3+N), the usual case) skips building the source. */
static void gx_shortcuts(const PicaGxUniforms* u)
{
	if (s_gpuTest == 5 || s_gpuTest == 6) { /* only the position and the colour */
		bunif(s_bLit, false);
		for (int k = 0; k < 3; k++) {
			bunif(s_bTu[k], false);
			bunif(s_bTd[k], false);
		}
		return;
	}
	bunif(s_bLit, u->lit != 0);
	bunif(s_bL[0], u->lit && u->lights >= 1);
	bunif(s_bL[1], u->lit && u->lights >= 2);
	bunif(s_bA[0], u->lit && u->lights >= 1 && (u->attn & 0x11) == 0x01);
	bunif(s_bA[1], u->lit && u->lights >= 2 && (u->attn & 0x22) == 0x02);
	bunif(s_bS[0], u->lit && u->lights >= 1 && (u->attn & 0x10));
	bunif(s_bS[1], u->lit && u->lights >= 2 && (u->attn & 0x20));
	for (int k = 0; k < 3; k++) {
		const float* w0 = u->W[2 * k];
		const float* w1 = u->W[2 * k + 1];
		int direct;
		if (k < 2)
			direct = w0[0] == 0.f && w0[1] == 0.f && w0[2] == (k == 0 ? 1.f : 0.f) &&
			         w0[3] == (k == 1 ? 1.f : 0.f) && vzero(w1);
		else
			direct = vzero(w0) && w1[0] == 1.f && w1[1] == 0.f && w1[2] == 0.f && w1[3] == 0.f;
		bunif(s_bTu[k], !(vzero(u->T[2 * k]) && vzero(u->T[2 * k + 1])));
		bunif(s_bTd[k], direct);
	}
}

/* The shader's normal matrices (gx_pass.v.pica): matrix k's rows in N[3k..3k+2], uploaded for lit
 * draws; a matrix is sent again only when its rows changed. */
static float s_lnRows[PICA_GX_MTX_ROWS / 3][9];
static uint8_t s_lnValid[PICA_GX_MTX_ROWS / 3];
static void gx_normal_rows(const float (*nrm_rows)[4])
{
	for (int k = 0; k < PICA_GX_MTX_ROWS / 3; k++) {
		float m[9];
		for (int i = 0; i < 3; i++) memcpy(&m[3 * i], nrm_rows[3 * k + i], 12);
		if (s_lnValid[k] && memcmp(m, s_lnRows[k], sizeof m) == 0) continue;
		memcpy(s_lnRows[k], m, sizeof m);
		s_lnValid[k] = 1;
		for (int i = 0; i < 3; i++) vunif(s_uN + 3 * k + i, m[3 * i], m[3 * i + 1], m[3 * i + 2], 0.0f);
	}
}

/* The setup a GX-mode draw needs (attribute loaders, state, uniforms, shortcuts, buffer): 0 when the
 * layout cannot be expressed. */
/* pica_draw_same (pica_draw.h), taken once by each draw: the number the caller gave with the state
 * object s_last was set from (apply_state: s_sameSrc, s_sameState) and with the layout gx_setup
 * last took (s_sameLayout). */
uint32_t pica_draw_same;
static int s_sameOff;  /* pica_draw_same ignored (pica_set_draw_same_off; the profiler's revert steps) */
void pica_set_draw_same_off(int on) { s_sameOff = on; }
static const PicaState* s_sameSrc;
static uint32_t s_sameState, s_sameLayout;

/* Field by field (no padding: the bytes memcmp compared), without a call. */
static inline int gx_layout_same(const PicaGxLayout* a, const PicaGxLayout* b)
{
	return a->stride == b->stride && a->off_nrm == b->off_nrm && a->off_clr == b->off_clr && a->off_tex[0] == b->off_tex[0] &&
	       a->off_tex[1] == b->off_tex[1] && a->off_tex[2] == b->off_tex[2] && a->off_mtx == b->off_mtx &&
	       a->fixed_idx == b->fixed_idx;
}

static int gx_setup(const void* verts, const PicaGxLayout* L, const PicaGxUniforms* u, uint32_t matrices_dirty,
                    const float (*pos_rows)[4], const float (*nrm_rows)[4], const PicaState* st, uint32_t same);

int pica_draw_gx(const void* verts, uint32_t nverts, const PicaGxLayout* L, const uint16_t* idx, uint32_t nidx,
                 const PicaGxUniforms* u, uint32_t matrices_dirty, const float (*pos_rows)[4], const float (*nrm_rows)[4],
                 const PicaState* st)
{
	(void)nverts;
	const uint32_t same = s_sameOff ? 0 : pica_draw_same;
	pica_draw_same = 0;
	if (!s_in_frame || nidx == 0) return 1;
	if (test_skips(SKIP_SETUP)) return 1;
	if (!gx_setup(verts, L, u, matrices_dirty, pos_rows, nrm_rows, st, same)) return 0;
	if (test_no_draws()) return 1;  /* the setup without the draw */
	s_frameDraws++;
	s_stat[ST_DRAW]++;
	C3D_DrawElements(GPU_TRIANGLES, (int)nidx, C3D_UNSIGNED_SHORT, idx);
	second_pass(GPU_TRIANGLES, (int)nidx, idx, st);
	gpu_check("GPU-transform draw", L, nidx, st);
	maybe_split();
	return 1;
}

static int gx_setup(const void* verts, const PicaGxLayout* L, const PicaGxUniforms* u, uint32_t matrices_dirty,
                    const float (*pos_rows)[4], const float (*nrm_rows)[4], const PicaState* st, uint32_t same)
{
	static PicaGxLayout s_posOnly;
	if (s_gpuTest == 6) { /* only the position and the matrix index fetched */
		s_posOnly = *L;
		s_posOnly.off_nrm = s_posOnly.off_clr = -1;
		for (int k = 0; k < 3; k++) s_posOnly.off_tex[k] = -1;
		L = &s_posOnly;
	}
	/* (same: the layout is as when gx_setup last took it, with the same number) */
	const int layoutSame = same && same == s_sameLayout;
	s_sameLayout = same;
	if (s_mode != MODE_GX || !s_gxLayoutValid || (!layoutSame && !gx_layout_same(L, &s_gxLayout))) {
		u64 perm;
		int n = gx_attribs(L, &perm);
		if (!n) {
			s_gxLayoutValid = 0;
			s_mode = MODE_NONE; /* the attribute setup was overwritten */
			return 0;
		}
		s_gxEntries = n;
		s_gxPerm = perm;
		s_gxLayout = *L;
		s_gxLayoutValid = 1;
		if (s_mode != MODE_GX) {
			C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bCpu, false);
			s_mode = MODE_GX;
		}
		if (!s_gxMatricesLoaded) matrices_dirty = (1u << PICA_GX_MTX_ROWS) - 1;
	}
	apply_state(st, same);
	if (!s_gxMatricesLoaded) memset(s_lnValid, 0, sizeof s_lnValid);  /* uniforms set elsewhere (self-test, [vbench]) */
	if (matrices_dirty || !s_gxMatricesLoaded) {
		/* the rows whose XF words changed (bit r), or all of them when the uniforms were set elsewhere:
		 * the others hold the values set last */
		uint32_t m = s_gxMatricesLoaded ? matrices_dirty : (1u << PICA_GX_MTX_ROWS) - 1;
		for (; m; m &= m - 1) {
			const int r = __builtin_ctz(m);
			vunif(s_uP + r, pos_rows[r][0], pos_rows[r][1], pos_rows[r][2], pos_rows[r][3]);
		}
		s_gxMatricesLoaded = 1;
	}
	if (u->lit) {
		gx_normal_rows(nrm_rows);
		vunif(s_uAMB, u->AMB[0], u->AMB[1], u->AMB[2], u->AMB[3]);
		for (int i = 0; i < u->lights; i++) {
			vunif(s_uLP + i, u->LP[i][0], u->LP[i][1], u->LP[i][2], 1.0f);
			vunif(s_uLC + i, u->LC[i][0], u->LC[i][1], u->LC[i][2], u->LC[i][3]);
			vunif(s_uLF + i, u->LF[i][0], u->LF[i][1], u->LF[i][2], 0.0f);
			if (u->attn & (0x11u << i)) {
				vunif(s_uLA + i, u->LA[i][0], u->LA[i][1], u->LA[i][2], 0.0f);
				vunif(s_uLK + i, u->LK[i][0], u->LK[i][1], u->LK[i][2], 0.0f);
				vunif(s_uLD + i, u->LD[i][0], u->LD[i][1], u->LD[i][2], 0.0f);
			}
		}
	}
	if (s_gpuTest == 2)
		for (int r = 0; r < 4; r++) vunif(s_uF + r, 0.0f, 0.0f, 0.0f, r == 3 ? 1.0f : 0.0f);
	else if (s_gpuTest == 4) /* the picture shrunk 8x about the centre: nothing crosses the sides */
		for (int r = 0; r < 4; r++) {
			const float k = r < 2 ? 0.125f : 1.0f;
			vunif(s_uF + r, k * u->F[r][0], k * u->F[r][1], k * u->F[r][2], k * u->F[r][3]);
		}
	else
		for (int r = 0; r < 4; r++) vunif(s_uF + r, u->F[r][0], u->F[r][1], u->F[r][2], u->F[r][3]);
	s_fIsCpu = 0;
	for (int r = 0; r < 6; r++) {
		vunif(s_uT + r, u->T[r][0], u->T[r][1], u->T[r][2], u->T[r][3]);
		vunif(s_uW + r, u->W[r][0], u->W[r][1], u->W[r][2], u->W[r][3]);
	}
	if (s_projShader && st && st->proj0) {
		vunif(s_uTQ, u->TQ[0], u->TQ[1], u->TQ[2], u->TQ[3]);
		vunif(s_uTQ1, u->TQ1[0], u->TQ1[1], u->TQ1[2], u->TQ1[3]);
	}
	vunif(s_uCS, u->CS[0], u->CS[1], u->CS[2], u->CS[3]);
	vunif(s_uCM, u->CM[0], u->CM[1], u->CM[2], u->CM[3]);
	gx_shortcuts(u);
	set_buffer(verts, L->stride, s_gxEntries, s_gxPerm);
	return 1;
}

void pica_gx_projection(const float o[4][4], PicaGxUniforms* u)
{
	/* F = PICA projection (EFB pixels -> clip) x O (view -> EFB pixels, homogeneous) */
	const C3D_Mtx* m = &s_projEfb;
	for (int i = 0; i < 4; i++) {
		const float mi[4] = {m->r[i].x, m->r[i].y, m->r[i].z, m->r[i].w};
		for (int j = 0; j < 4; j++)
			u->F[i][j] = mi[0] * o[0][j] + mi[1] * o[1][j] + mi[2] * o[2][j] + mi[3] * o[3][j];
	}
}

/* Stages from `first` on pass the previous stage through, and the combiner buffer is not
 * updated. */
static void passthrough_from(int first)
{
	for (int i = first; i < PICA_MAX_STAGES; i++) C3D_TexEnvInit(C3D_GetTexEnv(i));
	C3D_TexEnvBufUpdate(C3D_Both, 0);
}

/* A GameCube alpha test passing two alpha ranges (a <= r1 or a >= r0, the sea's waves) is two
 * PICA draws, one per range: no pixel passes both, so the result is the same. */
static void second_pass(GPU_Primitive_t prim, int count, const void* idx, const PicaState* st)
{
	if (!st->alpha_test2) return;
	C3D_AlphaTest(true, (GPU_TESTFUNC)st->alpha_func2, st->alpha_ref2);
	s_stat[ST_DRAW]++;
	C3D_DrawElements(prim, count, C3D_UNSIGNED_SHORT, idx);
	C3D_AlphaTest(st->alpha_test, (GPU_TESTFUNC)st->alpha_func, st->alpha_ref);
}

/* Unchanged GPU state is not set again, piece by piece (as Mario Kart Wii 3DS's citro3d patch
 * does): citro3d re-sends every register group a setter touched, so a draw whose state differed
 * from the last one in any field used to re-send all six combiner stages, its textures (each bind
 * also empties the GPU's texture cache), the blend, depth, alpha and cull settings and the scissor.
 * Each group is compared with the last state sent (s_last, while s_last_valid: code that sets GPU
 * state on its own clears that). A texture unit also remembers the texture it was given and the
 * texture-content generation then (s_texGen: texels rewritten by the CPU or copied into VRAM). */
static C3D_Tex s_boundTex[PICA_MAX_TEX];
static uint32_t s_boundGen[PICA_MAX_TEX];
static uint8_t s_boundValid[PICA_MAX_TEX];
void pica_set_state_skip(int on) { s_skipOff = !on; }

static int effect_same(const PicaState* a, const PicaState* b)
{
	return a->alpha_test == b->alpha_test && a->alpha_func == b->alpha_func && a->alpha_ref == b->alpha_ref &&
	       a->cull == b->cull && a->depth_test == b->depth_test && a->depth_func == b->depth_func &&
	       a->depth_write == b->depth_write && a->write_mask == b->write_mask && a->blend == b->blend &&
	       a->blend_eq == b->blend_eq && a->blend_src == b->blend_src && a->blend_dst == b->blend_dst &&
	       a->blend_aeq == b->blend_aeq && a->blend_asrc == b->blend_asrc && a->blend_adst == b->blend_adst &&
	       a->blend_color == b->blend_color;
}

static void set_stage(int i, const PicaTevStage* s)
{
	C3D_TexEnv* env = C3D_GetTexEnv(i);
	env->srcRgb = s->src_rgb;
	env->srcAlpha = s->src_alpha;
	env->opRgb = s->op_rgb;
	env->opAlpha = s->op_alpha;
	env->funcRgb = s->func_rgb;
	env->funcAlpha = s->func_alpha;
	env->scaleRgb = s->scale_rgb;
	env->scaleAlpha = s->scale_alpha;
	env->color = s->color;
}

static void apply_state(const PicaState* st, uint32_t same)
{
	/* the caller's number (pica_draw_same): *st is what it was when s_last was set from it
	 * (s_last = *st, below, or the compare found them equal), so the compare would again */
	if (same && s_last_valid && st == s_sameSrc && same == s_sameState) return;
	s_sameSrc = st;
	s_sameState = same;
	const PicaState* L = s_last_valid ? &s_last : NULL;
	if (L && memcmp(st, L, sizeof *st) == 0) return;
	if (s_skipOff) L = NULL;
	if (!L) memset(s_boundValid, 0, sizeof s_boundValid);
	for (int u = 0; u < PICA_MAX_TEX; u++) {
		if (!st->tex[u]) continue;
		PicaTex* pt = st->tex[u];
		C3D_Tex* tex = &pt->tex;
		C3D_TexSetWrap(tex, (GPU_TEXTURE_WRAP_PARAM)st->wrap_s[u], (GPU_TEXTURE_WRAP_PARAM)st->wrap_t[u]);
		C3D_TexSetFilter(tex, st->linear[u] ? GPU_LINEAR : GPU_NEAREST, st->linear[u] ? GPU_LINEAR : GPU_NEAREST);
		C3D_TexSetLodBias(tex, st->lod_bias[u] / 256.0f);
		pt->lastUse = s_frameNo;
		if (!pt->vram && !pt->rt && s_vramTex && s_ovl && s_in_frame) vram_try(pt);
		if (pt->vready && !pt->noVram) { /* the VRAM copy is queued ahead of this frame's draws */
			pt->vramUse = s_frameNo;
			pt->vtex = pt->tex;
			pt->vtex.data = pt->vram;
			tex = &pt->vtex;
		}
		if (s_smallTexReady &&
		    (s_gpuTest == 8 || (s_gpuTest == 7 && (uintptr_t)tex->data - 0x1F000000u >= 0x600000u)))
			tex = &s_smallTex; /* GPU tests 7, 8: in place of a texture outside VRAM, or of any */
		if (u == 0 && st->proj0) { /* (s, t) divided by q (texcoord0w) per pixel */
			s_projTex = *tex;
			s_projTex.param = (s_projTex.param & ~GPU_TEXTURE_MODE(7)) | GPU_TEXTURE_MODE(GPU_TEX_PROJECTION);
			tex = &s_projTex;
		}
		if (s_boundValid[u] && s_boundGen[u] == s_texGen && memcmp(tex, &s_boundTex[u], sizeof *tex) == 0) continue;
		C3D_TexBind(u, tex);
		s_stat[ST_TEXBIND]++;
		s_boundTex[u] = *tex;
		s_boundGen[u] = s_texGen;
		s_boundValid[u] = 1;
	}
	if (!L || st->proj0 != L->proj0) C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bPj, st->proj0 != 0);
	/* Depth is the GameCube's own (-z/w); a decal drawn on the surface under it (the goop layer)
	 * is moved toward the camera by a few steps of the PICA's float24 depth, which is coarser near
	 * 1 than the GameCube's 24-bit fixed point and lets the two fight. */
	if (!L || st->depth_bias != L->depth_bias) C3D_DepthMap(true, -1.0f, st->depth_bias ? -1.0f / 16384.0f : 0.0f);
	/* Combiner stages: those that changed, then pass-through for stages the last state used and
	 * this one does not (the rest are pass-through already). */
	const int n = st->nstages < PICA_MAX_STAGES ? st->nstages : PICA_MAX_STAGES;
	const int ln = L ? (L->nstages < PICA_MAX_STAGES ? L->nstages : PICA_MAX_STAGES) : -1;
	if (n == 0) {
		if (ln != 0) {
			C3D_TexEnv* env = C3D_GetTexEnv(0);
			C3D_TexEnvInit(env);
			C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
			C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
			const int to = L ? (ln > 1 ? ln : 1) : PICA_MAX_STAGES;
			for (int i = 1; i < to; i++) C3D_TexEnvInit(C3D_GetTexEnv(i));
			C3D_TexEnvBufUpdate(C3D_Both, 0);
		}
	} else {
		for (int i = 0; i < n; i++)
			if (i >= ln || memcmp(&st->stage[i], &L->stage[i], sizeof st->stage[i]) != 0) {
				set_stage(i, &st->stage[i]);
				s_stat[ST_STAGE]++;
			}
		const int to = L ? (ln > n ? ln : n) : PICA_MAX_STAGES;
		for (int i = n; i < to; i++) C3D_TexEnvInit(C3D_GetTexEnv(i));
		if (ln <= 0 || st->buf_rgb != L->buf_rgb || st->buf_alpha != L->buf_alpha) {
			C3D_TexEnvBufUpdate(C3D_RGB, st->buf_rgb);
			C3D_TexEnvBufUpdate(C3D_Alpha, st->buf_alpha);
		}
		if (ln <= 0 || st->buf_color != L->buf_color) C3D_TexEnvBufColor(st->buf_color);
	}
	if (!L || st->scissor != L->scissor || st->sc_x0 != L->sc_x0 || st->sc_y0 != L->sc_y0 || st->sc_x1 != L->sc_x1 ||
	    st->sc_y1 != L->sc_y1) {
		if (s_gpuTest == 1 || s_gpuTest == 4 || s_gpuTest == 5 || s_gpuTest == 6 || s_drawCut) {
			C3D_SetScissor(GPU_SCISSOR_NORMAL, 0, 0, 8, 8);
		} else if (st->scissor) {
			/* The EFB texture holds the EFB with its bottom row first (pica_efb_uv); scissor
			 * coordinates are its pixels. */
			u32 l = st->sc_x0 * EFB_TEX_W / PICA_EFB_WIDTH, r = (st->sc_x1 * EFB_TEX_W + PICA_EFB_WIDTH - 1) / PICA_EFB_WIDTH;
			u32 b = (PICA_EFB_HEIGHT - st->sc_y1) * EFB_TEX_H / PICA_EFB_HEIGHT;
			u32 t = ((PICA_EFB_HEIGHT - st->sc_y0) * EFB_TEX_H + PICA_EFB_HEIGHT - 1) / PICA_EFB_HEIGHT;
			C3D_SetScissor(GPU_SCISSOR_NORMAL, l, b, r, t);
		} else {
			C3D_SetScissor(GPU_SCISSOR_DISABLE, 0, 0, 0, 0);
		}
	}
	if (!L || !effect_same(st, L)) {
		C3D_AlphaTest(st->alpha_test, (GPU_TESTFUNC)st->alpha_func, st->alpha_ref);
		C3D_CullFace((GPU_CULLMODE)st->cull);
		C3D_DepthTest(st->depth_test, (GPU_TESTFUNC)st->depth_func,
		              (GPU_WRITEMASK)((st->write_mask & GPU_WRITE_COLOR) | (st->depth_write ? GPU_WRITE_DEPTH : 0)));
		if (st->blend) {
			C3D_AlphaBlend((GPU_BLENDEQUATION)st->blend_eq, (GPU_BLENDEQUATION)st->blend_aeq, (GPU_BLENDFACTOR)st->blend_src,
			               (GPU_BLENDFACTOR)st->blend_dst, (GPU_BLENDFACTOR)st->blend_asrc, (GPU_BLENDFACTOR)st->blend_adst);
			C3D_BlendingColor(st->blend_color);
		} else
			C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	}
	s_last = *st;
	s_last_valid = 1;
}

void pica_draw_triangles(const PicaVtx* verts, const uint16_t* idx, uint32_t nidx, const PicaState* st)
{
	const uint32_t same = s_sameOff ? 0 : pica_draw_same;
	pica_draw_same = 0;
	if (!s_in_frame || nidx == 0 || test_skips(SKIP_CPU)) return;
	apply_state(st, same);
	cpu_mode();
	set_buffer(verts, sizeof(PicaVtx), 5, 0x43210);
	s_frameDraws++;
	s_stat[ST_DRAW]++;
	s_fx[FX_CPU]++;
	C3D_DrawElements(GPU_TRIANGLES, (int)nidx, C3D_UNSIGNED_SHORT, idx);
	second_pass(GPU_TRIANGLES, (int)nidx, idx, st);
	gpu_check("CPU-path draw", NULL, nidx, st);
	maybe_split();
}

void pica_clear(uint32_t rgba, uint32_t depth24)
{
	s_clear_rgba = rgba;
	s_clear_depth = depth24 & 0xFFFFFF;
	if (s_in_frame) {
		C3D_RenderTargetClear(s_efb, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
		draw_on(s_efb);
		s_last_valid = 0;
		gpu_check("EFB clear", NULL, 0, NULL);
	} else {
		s_clear_pending = 1;
	}
}

void pica_clear_rect(float x, float y, float w, float h, uint32_t rgba, uint32_t depth24)
{
	if (!pica_init()) return;
	if (!s_in_frame) frame_begin();
	if (test_skips(SKIP_RECT)) return;
	s_fx[FX_RECT]++;
	s_fx[FX_RECT_PX] += (uint32_t)(w * h * ((float)s_efb->frameBuf.width / PICA_EFB_WIDTH) *
	                               ((float)s_efb->frameBuf.height / PICA_EFB_HEIGHT));
	PicaVtx* v;
	uint16_t* idx;
	uint32_t keep = s_dropped;
	int ok = pica_alloc(4, 6, &v, &idx);
	s_dropped = keep; /* not a game draw */
	if (!ok) return;
	const float z = (float)(depth24 & 0xFFFFFF) / 16777215.0f;
	const float cx[4] = {x, x + w, x + w, x}, cy[4] = {y, y, y + h, y + h};
	for (int i = 0; i < 4; i++) {
		memset(&v[i], 0, sizeof v[i]);
		v[i].x = cx[i];
		v[i].y = cy[i];
		v[i].z = z;
		v[i].w = 1.0f;
	}
	static const uint16_t q[6] = {0, 1, 2, 0, 2, 3};
	memcpy(idx, q, sizeof q);
	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	C3D_TexEnvSrc(env, C3D_Both, GPU_CONSTANT, 0, 0);
	C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
	/* 0xRRGGBBAA -> the constant's 0xAABBGGRR */
	C3D_TexEnvColor(env, (rgba >> 24) | ((rgba >> 8) & 0xFF00) | ((rgba << 8) & 0xFF0000) | (rgba << 24));
	passthrough_from(1);
	C3D_SetScissor(GPU_SCISSOR_DISABLE, 0, 0, 0, 0);
	C3D_CullFace(GPU_CULL_NONE);
	C3D_AlphaTest(false, GPU_ALWAYS, 0);
	C3D_DepthTest(true, GPU_ALWAYS, GPU_WRITE_ALL);
	C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	cpu_mode();
	set_buffer(v, sizeof(PicaVtx), 5, 0x43210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, idx);
	s_last_valid = 0;
	gpu_check("EFB rectangle clear", NULL, 6, NULL);
}

/* The GPU reads the rings from memory: write back what the CPU wrote since the last time. */
static void flush_rings(void)
{
	if (s_vused > s_vflushed) cache_flush(s_vring[s_half] + s_vflushed, s_vused - s_vflushed);
	if (s_iused > s_iflushed) cache_flush(s_iring[s_half] + s_iflushed, s_iused - s_iflushed);
	s_vflushed = s_vused;
	s_iflushed = s_iused;
}

void pica_efb_uv(float x, float y, float* s, float* t)
{
	*s = x * (1.0f / PICA_EFB_WIDTH);
	*t = 1.0f - y * (1.0f / PICA_EFB_HEIGHT);
}

static void* copy_arena_alloc(uint32_t bytes, PicaTex* t)
{
	uint32_t n = 1;
	while (n * COPY_UNIT < bytes) n <<= 1;
	if (!s_copyArena || n > COPY_UNITS) return NULL;
	/* First fit at any unit (textures need far less than 4 KB alignment): placed only at multiples of
	 * their own size, a 512 KB copy found no room in an arena with 512 KB free (Sirena Beach). */
	for (uint32_t at = 0; at + n <= COPY_UNITS;) {
		uint32_t k = 0;
		while (k < n && !s_copyUnitUsed[at + k]) k++;
		if (k < n) {
			at += k + 1;
			continue;
		}
		memset(&s_copyUnitUsed[at], 1, n);
		t->arenaUnit = (uint16_t)at;
		t->arenaUnits = (uint16_t)n;
		s_copyUnits += n;
		if (s_copyUnits > s_copyUnitsPeak) s_copyUnitsPeak = s_copyUnits;
		return s_copyArena + at * COPY_UNIT;
	}
	return NULL;
}

static void copy_tex_mem_free(PicaTex* t)
{
	if (t->arenaUnits) {
		memset(&s_copyUnitUsed[t->arenaUnit], 0, t->arenaUnits);
		s_copyUnits -= t->arenaUnits;
		t->arenaUnits = 0;
	} else if (t->tex.data && t->copyLinear) {
		LIN_LOCK();
		linearFree(t->tex.data);
		LIN_UNLOCK();
		t->copyLinear = 0;
	} else if (t->tex.data) {
		vramFree(t->tex.data);
	}
	t->tex.data = NULL;
}

/* The nearest power of two to v (by ratio), 8 to max. */
static uint32_t near_pow2(float v, uint32_t max)
{
	uint32_t p = 8;
	while (p < max && v > (float)p * 1.41421356f) p <<= 1;
	return p;
}

void pica_copy_tex_layout(float x, float y, float w, float h, int half, int exact, float yscale, uint16_t* tw,
                          uint16_t* th, float win[4])
{
	const float kx = (float)EFB_TEX_W / PICA_EFB_WIDTH, ky = (float)EFB_TEX_H / PICA_EFB_HEIGHT / yscale;
	if (exact) {
		/* The rectangle's EFB texels as they are, in a window of the EFB texture of power-of-two
		 * sides: drawn back into the EFB and copied again, it comes back unchanged. */
		int tx0 = (int)(x * kx), ty0 = (int)(y * ky);
		int tx1 = (int)ceilf((x + w) * kx), ty1 = (int)ceilf((y + h) * ky);
		tx1 = tx1 > EFB_TEX_W ? EFB_TEX_W : tx1;
		ty1 = ty1 > EFB_TEX_H ? EFB_TEX_H : ty1;
		uint32_t pw = 8, ph = 8;
		while ((int)pw < tx1 - tx0 && pw < EFB_TEX_W) pw <<= 1;
		while ((int)ph < ty1 - ty0 && ph < EFB_TEX_H) ph <<= 1;
		if (tx0 + (int)pw > EFB_TEX_W) tx0 = EFB_TEX_W - (int)pw;
		if (ty0 + (int)ph > EFB_TEX_H) ty0 = EFB_TEX_H - (int)ph;
		*tw = (uint16_t)pw;
		*th = (uint16_t)ph;
		win[0] = tx0 / kx;
		win[1] = ty0 / ky;
		win[2] = pw / kx;
		win[3] = ph / ky;
		return;
	}
	/* The rectangle stretched over the texture: at the EFB texture's resolution, or the GX copy's
	 * own size when smaller. */
	const float ew = w * kx, eh = h * ky;
	const float gw = half ? w * 0.5f : w, gh = half ? h * 0.5f : h;
	*tw = (uint16_t)near_pow2(ew < gw ? ew : gw, EFB_TEX_W);
	*th = (uint16_t)near_pow2(eh < gh ? eh : gh, EFB_TEX_H);
	win[0] = x;
	win[1] = y;
	win[2] = w;
	win[3] = h;
}

int pica_copy_tex_fits(const PicaTex* t, uint32_t tw, uint32_t th, int rgb565)
{
	return t && t->tex.width == tw && t->tex.height == th && (t->tex.fmt == GPU_RGB565) == (rgb565 != 0);
}

PicaTex* pica_copy_tex_create(uint32_t tw, uint32_t th, int rgb565)
{
	if (!pica_init()) return NULL;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	const GPU_TEXCOLOR fmt = rgb565 ? GPU_RGB565 : GPU_RGBA8;
	const uint32_t bytes = tw * th * (rgb565 ? 2u : 4u);
	void* mem = copy_arena_alloc(bytes, t);
	if (!mem) mem = vramAlloc(bytes);
	if (!mem) {
		vram_make_room(bytes);
		mem = vramAlloc(bytes);
	}
	if (!mem) {
		/* No VRAM left (the arena full, the scene's textures in use). Not linear FCRAM: citro3d makes
		 * render targets in VRAM only (C3D_RenderTargetCreateFromTex returns none for other memory),
		 * so that fallback lost the copy every time, and draws sampling it read the destination's
		 * memory, which the GPU never wrote: on a console whatever the heap held there (Sirena Beach's
		 * Phantamanta tinted the screen by its alpha copy: noise in the manta's colour). The caller
		 * tries a smaller texture. Logged once per size. */
		static uint32_t s_said[8];
		const uint32_t key = tw << 16 | th << 1 | (rgb565 ? 1u : 0u);
		int k = 0;
		while (k < 8 && s_said[k] && s_said[k] != key) k++;
		if (k < 8 && !s_said[k]) {
			s_said[k] = key;
			fprintf(stderr, "[pica] no VRAM for a %lux%lu EFB-copy texture (%lu KB free, arena %lu of %lu KB used)\n",
			        (unsigned long)tw, (unsigned long)th, (unsigned long)(vramSpaceFree() >> 10),
			        (unsigned long)(s_copyUnits * (COPY_UNIT >> 10)), (unsigned long)(COPY_ARENA_BYTES >> 10));
		}
		free(t);
		return NULL;
	}
	/* A 2D texture's mode and filters as the EFB texture's, at mem. */
	t->tex = s_efbTex;
	t->tex.data = mem;
	t->tex.fmt = fmt;
	t->tex.size = bytes;
	t->tex.width = (u16)tw;
	t->tex.height = (u16)th;
	t->tex.lodParam = 0;
	t->rt = C3D_RenderTargetCreateFromTex(&t->tex, GPU_TEXFACE_2D, 0, -1);
	if (!t->rt) {
		copy_tex_mem_free(t);
		free(t);
		return NULL;
	}
	t->bytes = bytes;
	s_copyTexCount++;
	return t;
}

void pica_copy_arena(uint32_t* usedKB, uint32_t* peakKB, uint32_t* totalKB)
{
	*usedKB = s_copyUnits * (COPY_UNIT >> 10);
	*peakKB = s_copyUnitsPeak * (COPY_UNIT >> 10);
	*totalKB = s_copyArena ? COPY_ARENA_BYTES >> 10 : 0;
}

/* Draws the rectangle (x, y, w, h) of the EFB texture (EFB pixels) over the whole of the bound
 * target, as the present pass (the whole EFB) and EFB copies (their source rectangle) do.
 * alpha_one: alpha 1 instead of the EFB's (formats without alpha). */
static void draw_efb_quad_ch(int alpha_one, C3D_Tex* src, int chan, float x, float y, float w, float h);
static void draw_efb_quad(int alpha_one, C3D_Tex* src)
{
	draw_efb_quad_ch(alpha_one, src, -1, 0.0f, 0.0f, PICA_EFB_WIDTH, PICA_EFB_HEIGHT);
}

/* chan: -1 the texels as they are; 0-3 that channel (r, g, b, a) in all four, as GX reads a
 * one-channel copy (R8 or I8 as I8: intensity and alpha); 5 red in rgb and alpha kept (RA4,
 * RA8 as IA4, IA8). */
static void draw_efb_quad_ch(int alpha_one, C3D_Tex* src, int chan, float x, float y, float w, float h)
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
		memset(v[i].tc, 0, sizeof v[i].tc);
		pica_efb_uv(x + corners[i][0] * (w / PICA_EFB_WIDTH), y + corners[i][1] * (h / PICA_EFB_HEIGHT), &v[i].tc[0][0],
		            &v[i].tc[0][1]);
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
	if (chan >= 0 && chan <= 3) {
		static const int rgbOp[4] = {GPU_TEVOP_RGB_SRC_R, GPU_TEVOP_RGB_SRC_G, GPU_TEVOP_RGB_SRC_B, GPU_TEVOP_RGB_SRC_ALPHA};
		static const int aOp[4] = {GPU_TEVOP_A_SRC_R, GPU_TEVOP_A_SRC_G, GPU_TEVOP_A_SRC_B, GPU_TEVOP_A_SRC_ALPHA};
		if (chan == 3 && alpha_one) { /* alpha of an EFB without one: 1 */
			C3D_TexEnvSrc(env, C3D_Both, GPU_CONSTANT, 0, 0);
		} else {
			C3D_TexEnvSrc(env, C3D_Both, GPU_TEXTURE0, 0, 0);
			C3D_TexEnvOpRgb(env, (GPU_TEVOP_RGB)rgbOp[chan], 0, 0);
			C3D_TexEnvOpAlpha(env, (GPU_TEVOP_A)aOp[chan], 0, 0);
		}
	} else if (chan == 5) {
		C3D_TexEnvOpRgb(env, GPU_TEVOP_RGB_SRC_R, 0, 0);
	}
	passthrough_from(1);
	C3D_TexBind(0, src);
	C3D_SetScissor(GPU_SCISSOR_DISABLE, 0, 0, 0, 0);
	C3D_CullFace(GPU_CULL_NONE);
	C3D_AlphaTest(false, GPU_ALWAYS, 0);
	C3D_DepthTest(false, GPU_ALWAYS, GPU_WRITE_ALL & ~GPU_WRITE_DEPTH);
	C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	cpu_mode();
	set_buffer(v, sizeof(PicaVtx), 5, 0x43210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, idx);
	s_last_valid = 0;
}

void pica_efb_copy_to_ch(PicaTex* dst, int alpha_one, int chan, const float win[4], float yscale)
{
	if (!dst || !dst->rt || !pica_init()) return;
	const float x = win[0], y = win[1], w = win[2], h = win[3];
	dst->wx = x;
	dst->wy = y;
	dst->ww = w;
	dst->wh = h;
	if (!s_in_frame) frame_begin();
	if (test_skips(SKIP_COPY)) return;
	s_fx[FX_COPY]++;
	s_fx[FX_COPY_PX] += (uint32_t)dst->rt->frameBuf.width * dst->rt->frameBuf.height;
	/* A draw pass, not a texture copy: it stays in the frame's command list (no CPU wait) and
	 * can fix alpha, as the GameCube's copy does for EFB formats without alpha. */
	draw_on(dst->rt);
	draw_efb_quad_ch(alpha_one, &s_efbTex, chan, x, y / yscale, w, h / yscale);
	draw_on(s_efb);
	{
		char what[96];
		snprintf(what, sizeof what, "EFB copy into texture %p (%ux%u, fmt %d)", (void*)dst, dst->tex.width,
		         dst->tex.height, (int)dst->tex.fmt);
		gpu_check(what, NULL, 6, NULL);
	}
	s_copies++;
	if (s_nFrameCopies < MAX_VIEW_COPIES) s_frameCopies[s_nFrameCopies++] = dst;
}

void pica_efb_depth_copy_to(PicaTex* dst, const float win[4], float yscale)
{
	if (!dst || !dst->rt || !pica_init()) return;
	const float x = win[0], y = win[1], w = win[2], h = win[3];
	dst->wx = x;
	dst->wy = y;
	dst->ww = w;
	dst->wh = h;
	if (!s_in_frame) frame_begin();
	if (test_skips(SKIP_COPY)) return;
	s_fx[FX_COPY]++;
	s_fx[FX_COPY_PX] += (uint32_t)dst->rt->frameBuf.width * dst->rt->frameBuf.height;
	draw_on(dst->rt);
	draw_efb_quad_ch(0, &s_efbDepthTex, -1, x, y / yscale, w, h / yscale);
	draw_on(s_efb);
	s_copies++;
	if (s_nFrameCopies < MAX_VIEW_COPIES) s_frameCopies[s_nFrameCopies++] = dst;
}

void pica_present(void)
{
	if (!pica_init()) return;
	if (!s_in_frame) frame_begin();

	s_endTimed = 1;
	s_endT = svcGetSystemTick();
	/* The EFB texture as one quad over the whole top screen. */
	draw_on(s_top);
	set_cpu_proj(&s_projTop);
	C3D_Tex* show = &s_efbTex;
	if (s_view > 0 && s_view <= s_nLastFrameCopies) show = &s_lastFrameCopies[s_view - 1]->tex;
	draw_efb_quad(1, show);
	end_mark(PICA_END_PASS);
	/* No ring flush: the frame's end writes back the whole linear heap (ovl_frame_end, and
	 * C3D_FrameEnd(0) without overlap), rings included. Flushed again here by range, the ring's
	 * 0.5-1 MB took 3-4 ms; the whole heap takes about 1 (Delfino Airstrip, 17 Oct). */
	s_vflushed = s_vused;
	s_iflushed = s_iused;
	end_mark(PICA_END_RINGS);
	memcpy(s_lastFrameCopies, s_frameCopies, sizeof s_frameCopies);
	s_nLastFrameCopies = s_nFrameCopies;
	s_nFrameCopies = 0;
	{
		const uint32_t used = cmd_words_used();
		if (used > s_cmdPeakWords) s_cmdPeakWords = used;
	}
	s_where = "waiting for the GPU to finish the frame's earlier parts";
	drain_parts();
	end_mark(PICA_END_PARTS);
	s_where = "C3D_FrameEnd (submitting the frame)";
	frame_end();
	if (!s_ovl) end_mark(PICA_END_SUBMIT);
	s_endTimed = 0;
	s_where = "between frames (game code)";
	s_frameDraws = 0;
	s_checkFramesInc();
	s_in_frame = 0;
	s_last_valid = 0;
	s_clear_pending = 1; /* the GameCube clears the EFB as it copies it out */
}

/* EFB copy readback. The CPU reads some copies back (the goop map: TPollutionLayer reads its
 * R8 copy to tell where the floor is polluted), but copies live in VRAM snapshots. A copy asked
 * for here is fetched into linear memory by a GPU transfer queued after its frame's draws
 * (overlapped frames only), and once that frame has finished, converted to the GX tiled layout
 * of a one-byte format (8x4 texel tiles) and written to the copy's destination: two frames
 * late. Several copies to one destination in a frame: the last one is written. */
typedef struct {
	PicaTex* tex;
	uint8_t* dest;
	uint16_t dw, dh, tilesPerRow;
	uint8_t chan; /* 0-3: r, g, b, a; 4: luminance */
	uint16_t tw, th; /* the copy texture's size (RGBA8) */
	float x, y, w, h; /* the source rectangle, EFB pixels */
	float wx, wy, ww, wh; /* the texture's EFB window (its latest copy's) */
	uint8_t* buf;
} Readback;
#define MAX_READBACK 8
/* Asked for this frame, queued with the frame the GPU runs, arrived (to convert) */
static Readback s_rbCur[MAX_READBACK], s_rbRun[MAX_READBACK], s_rbArrived[MAX_READBACK];
static int s_nrbCur, s_nrbRun, s_nrbArrived;
static uint32_t s_rbDone, s_rbSkipped;

void pica_efb_readback(PicaTex* t, void* dest, int dw, int dh, int tilesPerRow, int chan, float x, float y, float w,
                       float h)
{
	if (!t || !dest || dw <= 0 || dh <= 0 || w <= 0 || h <= 0 || t->tex.fmt != GPU_RGBA8 || t->ww <= 0 || t->wh <= 0)
		return;
	if (!s_ovl) {
		if (!s_rbSkipped++) fprintf(stderr, "[pica] EFB copy readback needs overlapped frames: skipped\n");
		return;
	}
	Readback* r = NULL;
	for (int i = 0; i < s_nrbCur; i++)
		if (s_rbCur[i].dest == dest) r = &s_rbCur[i];
	if (!r) {
		if (s_nrbCur == MAX_READBACK) return;
		r = &s_rbCur[s_nrbCur++];
	}
	r->tex = t;
	r->dest = (uint8_t*)dest;
	r->dw = (uint16_t)dw;
	r->dh = (uint16_t)dh;
	r->tilesPerRow = (uint16_t)tilesPerRow;
	r->chan = (uint8_t)chan;
	r->tw = t->tex.width;
	r->th = t->tex.height;
	r->x = x;
	r->y = y;
	r->w = w;
	r->h = h;
	r->wx = t->wx;
	r->wy = t->wy;
	r->ww = t->ww;
	r->wh = t->wh;
	r->buf = NULL;
}

static void readback_queue(void)
{
	/* A GPU test (pixels cut to 8x8, the GX writes muted, ...): the copies hold nothing real, so
	 * they must not reach the game's memory (the goop map would lose its goop; its copies are left
	 * out too: pica_gpu_test_seen). */
	if (s_gpuTest) {
		s_nrbCur = s_nrbRun = 0;
		return;
	}
	for (int i = 0; i < s_nrbCur; i++) {
		Readback* r = &s_rbCur[i];
		const uint32_t size = (uint32_t)r->tw * r->th * 4;
		LIN_LOCK();
		r->buf = (uint8_t*)linearAlloc(size);
		LIN_UNLOCK();
		if (!r->buf) continue;
		/* Without the whole-heap flush, lines the CPU left dirty here could be written back over
		 * what the GPU copies in. */
		if (!s_heapFlush) cache_invalidate(r->buf, size);
		GX_TextureCopy((u32*)r->tex->tex.data, 0, (u32*)r->buf, 0, size, 8);
	}
	memcpy(s_rbRun, s_rbCur, sizeof(Readback) * (size_t)s_nrbCur);
	s_nrbRun = s_nrbCur;
	s_nrbCur = 0;
}

/* A one-byte GX texture (8x4 tiles) as a PGM image, for looking at what the readback writes.
 * Off unless asked for (pica_set_readback_dumps): each file cost about a second of SD writes. */
static int s_rbDumps, s_rbDumpAll;
void pica_set_readback_dumps(int on) { s_rbDumps = on; }
/* Each map's next readback written as goop-<its address>.pgm (autorun.txt goop_dump_at). */
void pica_request_goop_dumps(void) { s_rbDumpAll = 1; }
static void rb_dump(const char* tag, const Readback* r)
{
	if (!s_rbDumps && !s_rbDumpAll) return;
	char path[64];
	snprintf(path, sizeof path, "sdmc:/sms/goop-%s.pgm", tag);
	FILE* f = fopen(path, "wb");
	if (!f) return;
	fprintf(f, "P5 %u %u 255\n", r->dw, r->dh);
	for (int y = 0; y < r->dh; y++)
		for (int x = 0; x < r->dw; x++)
			fputc(r->dest[((uint32_t)(y >> 2) * r->tilesPerRow + (x >> 3)) * 32 + (y & 3) * 8 + (x & 7)], f);
	fclose(f);
	fprintf(stderr, "[pica] goop map written to %s\n", path);
}

/* Each destination column's byte within a row of the copy texture's 8x8 tiles (readback_rows). */
#define RB_MAX_W 1024 /* GX copies are at most 1024 wide */
static uint32_t s_rbCol[RB_MAX_W];

/* Writes a readback's texels into the game's memory, in GX's one-byte tiled layout (8x4 texels
 * a 32-byte tile). The texture holds its EFB window with the top row at t = 1, which the PICA
 * stores first (pica_tex_upload writes the image's bottom row first): memory row v is EFB row
 * wy + v * wh / th. The column mapping is the same on every row, so each column's byte in a tile
 * row is worked out once (s_rbCol) and a texel costs a table read, a byte read and a share of a
 * 32-bit store; rows that sample the source row of the row above copy it. Per texel float math
 * took 34 ms a frame for Delfino Airstrip's 512x512 goop map. */
static void readback_rows(const Readback* r)
{
	static const int byteOf[4] = {3, 2, 1, 0}; /* RGBA8 texels are stored A, B, G, R */
	const int lum = r->chan > 3;
	const int dw = r->dw < RB_MAX_W ? r->dw : RB_MAX_W, dh = r->dh, tw = r->tw, th = r->th;
	const uint32_t rowBytes = (uint32_t)tw * 8 * 4, tileRow = (uint32_t)r->tilesPerRow * 32;
	const uint32_t chanByte = lum ? 0 : (uint32_t)byteOf[r->chan];
	const float sx = r->w / r->dw, sy = r->h / r->dh;
	const float ku = tw / r->ww, kv = th / r->wh;
	for (int x = 0; x < dw; x++) {
		int u = (int)((r->x + (x + 0.5f) * sx - r->wx) * ku);
		u = u < 0 ? 0 : u > tw - 1 ? tw - 1 : u;
		const int ux = u & 7;
		s_rbCol[x] = ((uint32_t)(u >> 3) * 64 + (uint32_t)((ux & 1) | (ux & 2) << 1 | (ux & 4) << 2)) * 4 + chanByte;
	}
	int lastV = -1;
	const uint8_t* lastRow = NULL;
	for (int y = 0; y < dh; y++) {
		int v = (int)((r->y + (y + 0.5f) * sy - r->wy) * kv);
		v = v < 0 ? 0 : v > th - 1 ? th - 1 : v;
		uint8_t* drow = r->dest + (uint32_t)(y >> 2) * tileRow + (uint32_t)(y & 3) * 8;
		int x = 0;
		if (v == lastV) {
			for (; x + 8 <= dw; x += 8) memcpy(drow + (x >> 3) * 32, lastRow + (x >> 3) * 32, 8);
			for (; x < dw; x++) drow[(x >> 3) * 32 + (x & 7)] = lastRow[(x >> 3) * 32 + (x & 7)];
			continue;
		}
		const int vy = v & 7;
		const uint8_t* trow = r->buf + (uint32_t)(v >> 3) * rowBytes + (uint32_t)((vy & 1) << 1 | (vy & 2) << 2 | (vy & 4) << 3) * 4;
		if (lum) {
			for (; x < dw; x++) {
				const uint8_t* px = trow + s_rbCol[x];
				drow[(x >> 3) * 32 + (x & 7)] = (uint8_t)((px[3] * 77 + px[2] * 150 + px[1] * 29) >> 8);
			}
		} else {
			for (; x + 8 <= dw; x += 8) {
				const uint32_t* c = s_rbCol + x;
				const uint32_t lo = trow[c[0]] | (uint32_t)trow[c[1]] << 8 | (uint32_t)trow[c[2]] << 16 | (uint32_t)trow[c[3]] << 24;
				const uint32_t hi = trow[c[4]] | (uint32_t)trow[c[5]] << 8 | (uint32_t)trow[c[6]] << 16 | (uint32_t)trow[c[7]] << 24;
				uint8_t* d = drow + (x >> 3) * 32;
				memcpy(d, &lo, 4); /* little-endian: texel x first */
				memcpy(d + 4, &hi, 4);
			}
			for (; x < dw; x++) drow[(x >> 3) * 32 + (x & 7)] = trow[s_rbCol[x]];
		}
		lastV = v;
		lastRow = drow;
	}
}

/* A depth copy's readback (chan 6) as GX's Z24X8: 4x4 texel tiles of 64 bytes, the first 32 the
 * texels' X and depth's high byte, the next 32 its middle and low bytes. The copy texture's texels
 * hold the depth buffer's bytes as they were (low, middle, high, stencil). */
static void readback_rows_z24(const Readback* r)
{
	const int dw = r->dw, dh = r->dh, tw = r->tw, th = r->th;
	const uint32_t rowBytes = (uint32_t)tw * 8 * 4, tileRow = (uint32_t)r->tilesPerRow * 64;
	const float sx = r->w / r->dw, sy = r->h / r->dh;
	const float ku = tw / r->ww, kv = th / r->wh;
	for (int y = 0; y < dh; y++) {
		int v = (int)((r->y + (y + 0.5f) * sy - r->wy) * kv);
		v = v < 0 ? 0 : v > th - 1 ? th - 1 : v;
		const int vy = v & 7;
		const uint8_t* trow = r->buf + (uint32_t)(v >> 3) * rowBytes + (uint32_t)((vy & 1) << 1 | (vy & 2) << 2 | (vy & 4) << 3) * 4;
		uint8_t* drow = r->dest + (uint32_t)(y >> 2) * tileRow + (uint32_t)(y & 3) * 8;
		for (int x = 0; x < dw; x++) {
			int u = (int)((r->x + (x + 0.5f) * sx - r->wx) * ku);
			u = u < 0 ? 0 : u > tw - 1 ? tw - 1 : u;
			const int ux = u & 7;
			const uint8_t* px = trow + ((uint32_t)(u >> 3) * 64 + (uint32_t)((ux & 1) | (ux & 2) << 1 | (ux & 4) << 2)) * 4;
			uint8_t* d = drow + (x >> 2) * 64 + (x & 3) * 2;
			d[0] = 0xFF;
			d[1] = px[2];
			d[32] = px[1];
			d[33] = px[0];
		}
	}
}

/* The queue that ran the last frame has finished: its copies are in their buffers. */
static void readback_arrived(void)
{
	memcpy(s_rbArrived, s_rbRun, sizeof(Readback) * (size_t)s_nrbRun);
	s_nrbArrived = s_nrbRun;
	s_nrbRun = 0;
}

static void readback_finish(void)
{
	for (int i = 0; i < s_nrbArrived; i++) {
		Readback* r = &s_rbArrived[i];
		if (!r->buf) continue;
		end_mark(PICA_END_RB_CONVERT);
		cache_invalidate(r->buf, (uint32_t)r->th * r->tw * 4);
		end_mark(PICA_END_RB_INVAL);
		if (r->chan == 6) {  /* a depth copy */
			readback_rows_z24(r);
			LIN_LOCK();
			linearFree(r->buf);
			LIN_UNLOCK();
			r->buf = NULL;
			s_rbDone++;
			continue;
		}
		const uint32_t bytes = (uint32_t)((r->dh + 3) / 4) * r->tilesPerRow * 32;
		uint32_t before = 0, after = 0, maxv = 0;
		const int logIt = s_rbDone % 150 == 0;
		if (logIt)
			for (uint32_t k = 0; k < bytes; k++) before += r->dest[k] > 100;
		if (s_rbDone == 0) rb_dump("cpu", r);
		readback_rows(r);
		if (s_rbDumpAll) {
			char tag[16];
			snprintf(tag, sizeof tag, "%08lx", (unsigned long)(uintptr_t)r->dest);
			rb_dump(tag, r);
		} else if (s_rbDumps && (s_rbDone == 0 || s_rbDone == 1 || s_rbDone == 30 || s_rbDone == 300)) {
			char tag[16];
			snprintf(tag, sizeof tag, "gpu%u", (unsigned)s_rbDone);
			rb_dump(tag, r);
		}
		if (logIt) {
			for (uint32_t k = 0; k < bytes; k++) {
				after += r->dest[k] > 100;
				if (r->dest[k] > maxv) maxv = r->dest[k];
			}
			fprintf(stderr, "[pica] readback %u: %p %ux%u chan %u, texels over 100: %lu before, %lu after (max %lu)\n",
			        (unsigned)s_rbDone, (void*)r->dest, r->dw, r->dh, r->chan, (unsigned long)before,
			        (unsigned long)after, (unsigned long)maxv);
		}
		LIN_LOCK();
		linearFree(r->buf);
		LIN_UNLOCK();
		r->buf = NULL;
		s_rbDone++;
	}
	if (s_nrbArrived) s_rbDumpAll = 0;
	s_nrbArrived = 0;
}

uint32_t pica_readbacks(void) { return s_rbDone; }

uint32_t pica_copies(void) { return s_copies; }

int pica_cycle_view(void)
{
	s_view = s_view >= s_nLastFrameCopies ? 0 : s_view + 1;
	return s_view;
}

int pica_view(void) { return s_view; }
int pica_view_count(void) { return s_nLastFrameCopies; }

uint32_t pica_dropped(void) { return s_dropped; }

void pica_drop_reasons(uint32_t* ring, uint32_t* cmdbuf)
{
	*ring = s_dropRing;
	*cmdbuf = s_dropCmd;
}

uint32_t pica_cmd_peak(void)
{
	const uint32_t p = s_cmdPeakWords * 4u;
	s_cmdPeakWords = 0;
	return p;
}

uint32_t pica_cmd_capacity(void) { return CMDBUF_BYTES; }

uint32_t pica_linear_free(void)
{
	LIN_LOCK();
	const uint32_t n = (uint32_t)linearSpaceFree();
	LIN_UNLOCK();
	return n;
}

/* ---- the GX back end's render thread ------------------------------------------------------ */
static LightEvent s_evt[4];

int pica_thread_start(void (*fn)(void*), void* arg)
{
	LightEvent_Init(&s_evt[0], RESET_ONESHOT);
	LightEvent_Init(&s_evt[1], RESET_ONESHOT);
	s32 prio = 0x30;
	svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
	if (prio > 0x19) prio--; /* just above the thread that starts it (the game thread) */
	/* Core 2: the New 3DS's second application core, otherwise idle. Then core 1 (the system
	 * core), with up to 80% of its time given to the application. */
	if (threadCreate(fn, arg, 256u << 10, prio, 2, true)) return 2;
	if (R_SUCCEEDED(APT_SetAppCpuTimeLimit(80)) && threadCreate(fn, arg, 256u << 10, prio, 1, true)) return 1;
	return -1;
}
/* The GX back end's GX thread: on core 1 (the system core, up to 80% of its time for the
 * application), just above the thread that starts it; events 2 and 3. Returns 1, or -1. */
int pica_gx_core = 1; /* autorun.txt gx_thread_core (profiling in Azahar): another core for it */
int pica_gx_thread_start(void (*fn)(void*), void* arg)
{
	LightEvent_Init(&s_evt[2], RESET_ONESHOT);
	LightEvent_Init(&s_evt[3], RESET_ONESHOT);
	s32 prio = 0x30;
	svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
	if (prio > 0x19) prio--;
	if (pica_gx_core != 1 && threadCreate(fn, arg, 256u << 10, prio, pica_gx_core, true)) return 1;
	if (R_SUCCEEDED(APT_SetAppCpuTimeLimit(80)) && threadCreate(fn, arg, 256u << 10, prio, 1, true)) return 1;
	return -1;
}
void pica_event_wait(int i) { LightEvent_Wait(&s_evt[i]); }
void pica_event_signal(int i) { LightEvent_Signal(&s_evt[i]); }
/* A thread polling for another core's work, between checks: an address arbitration nobody waits
 * on. It does nothing on hardware but a kernel call; in Azahar, which runs the cores in turn, a
 * slice at a time, it ends the caller's slice there, so the other cores' work shows up at once
 * instead of at the slice's end (the game thread polled up to ~2 ms for the GX thread's frame). */
void pica_poll_pause(void)
{
	static s32 nobody;
	syncArbitrateAddress(&nobody, ARBITRATION_SIGNAL, 1);
}

/* ---- hardware self-test ------------------------------------------------------------------ */
/* Runs the pieces of a game frame one frame at a time and logs each: a piece the GPU never
 * finishes leaves the next frame unable to start, which is reported (with a timeout) instead of
 * hanging the way the game's first frame did on a New 3DS. */
static int selftest_wait(const char* after)
{
	u64 t0 = osGetTime();
	while (!frame_begin_ex(C3D_FRAME_NONBLOCK)) {
		if (osGetTime() - t0 > 3000) {
			fprintf(stderr, "[gputest] %s: the GPU never finished (HUNG)\n", after);
			s_where = "GPU self-test: hung";
			return 0;
		}
		svcSleepThread(1000000);
	}
	fprintf(stderr, "[gputest] %s: ok (%lu ms)\n", after, (unsigned long)(osGetTime() - t0));
	fflush(stderr);
	return 1;
}

static void selftest_end(void)
{
	flush_rings();
	frame_end();
	s_in_frame = 0;
	s_last_valid = 0;
}

/* A vertex-colour quad (blue) with every gx_pass GX-mode input in the buffer: position, normal,
 * colour, three texture coordinates, matrix index (stride 56). Sets the uniforms prog declares:
 * identity matrices, no texture coordinates, colour = vertex colour. gx: the `cpu` bool is false. */
static void selftest_quad56(shaderProgram_s* prog, int gx)
{
	void* vb;
	uint16_t* ib;
	if (!pica_alloc_raw(4 * 56, 6, &vb, &ib)) {
		fprintf(stderr, "[gputest] no ring space\n");
		return;
	}
	const float cx[4] = {320, 576, 576, 320}, cy[4] = {224, 224, 352, 352};
	for (int i = 0; i < 4; i++) {
		uint8_t* v = (uint8_t*)vb + i * 56;
		memset(v, 0, 56);
		float* f = (float*)v;
		f[0] = cx[i];
		f[1] = cy[i];
		f[5] = 1.0f; /* normal z */
		v[26] = 255; /* colour: blue, alpha 255 */
		v[27] = 255;
		/* texture coordinates (28..51) and the matrix index (52, float) stay 0 */
	}
	static const uint16_t q[6] = {0, 1, 2, 0, 2, 3};
	memcpy(ib, q, sizeof q);
	PicaState st;
	memset(&st, 0, sizeof st);
	st.write_mask = 0x0F;
	st.depth_func = GPU_ALWAYS;
	st.alpha_func = GPU_ALWAYS;
	st.cull = GPU_CULL_NONE;
	st.blend_eq = st.blend_aeq = GPU_BLEND_ADD;
	st.blend_src = st.blend_asrc = GPU_ONE;
	st.blend_dst = st.blend_adst = GPU_ZERO;
	s_last_valid = 0;
	apply_state(&st, 0);
	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	AttrInfo_AddLoader(ai, 0, GPU_FLOAT, 3);
	AttrInfo_AddLoader(ai, 1, GPU_FLOAT, 3);
	AttrInfo_AddLoader(ai, 2, GPU_UNSIGNED_BYTE, 4);
	AttrInfo_AddLoader(ai, 3, GPU_FLOAT, 2);
	AttrInfo_AddLoader(ai, 4, GPU_FLOAT, 2);
	AttrInfo_AddLoader(ai, 5, GPU_FLOAT, 2);
	AttrInfo_AddLoader(ai, 6, GPU_FLOAT, 1);
	s_mode = MODE_NONE; /* the attribute setup is the test's own */
	s_gxLayoutValid = 0;
	s_gxMatricesLoaded = 0;
	s_fIsCpu = 0;
	shaderInstance_s* vs = prog->vertexShader;
	if (gx) C3D_BoolUnifSet(GPU_VERTEX_SHADER, shaderInstanceGetUniformLocation(vs, "cpu"), false);
	PicaGxUniforms u;
	memset(&u, 0, sizeof u);
	static const float ident[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	pica_gx_projection(ident, &u);
	int l;
	if ((l = shaderInstanceGetUniformLocation(vs, "F")) >= 0)
		for (int r = 0; r < 4; r++) C3D_FVUnifSet(GPU_VERTEX_SHADER, l + r, u.F[r][0], u.F[r][1], u.F[r][2], u.F[r][3]);
	static const char* const rowsNames[2] = {"P", "N"};
	for (int k = 0; k < 2; k++)
		if ((l = shaderInstanceGetUniformLocation(vs, rowsNames[k])) >= 0)
			for (int r = 0; r < PICA_GX_MTX_ROWS; r++) {
				const int c = r % 3;
				C3D_FVUnifSet(GPU_VERTEX_SHADER, l + r, c == 0, c == 1, c == 2, 0.0f);
			}
	static const char* const zeroNames[2] = {"T", "W"};
	for (int k = 0; k < 2; k++)
		if ((l = shaderInstanceGetUniformLocation(vs, zeroNames[k])) >= 0)
			for (int r = 0; r < 6; r++) C3D_FVUnifSet(GPU_VERTEX_SHADER, l + r, 0, 0, 0, 0);
	if ((l = shaderInstanceGetUniformLocation(vs, "CS")) >= 0) C3D_FVUnifSet(GPU_VERTEX_SHADER, l, 1, 1, 1, 1);
	if ((l = shaderInstanceGetUniformLocation(vs, "CM")) >= 0) C3D_FVUnifSet(GPU_VERTEX_SHADER, l, 0, 0, 0, 0);
	if ((l = shaderInstanceGetUniformLocation(vs, "AMB")) >= 0) C3D_FVUnifSet(GPU_VERTEX_SHADER, l, 1, 1, 1, 1);
	set_buffer(vb, 56, 7, 0x6543210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, ib);
}

int pica_selftest(void)
{
	if (!pica_init()) return 0;
	fprintf(stderr, "[gputest] start\n");
	if (!selftest_wait("start")) return 0;
	/* 1: clear the EFB target (memory fill), nothing drawn */
	selftest_end();
	if (!selftest_wait("1 EFB clear")) return 0;
	/* 2: a constant-colour quad on the CPU vertex path */
	pica_clear_rect(64.0f, 64.0f, 256.0f, 128.0f, 0xFF0000FF, 0);
	selftest_end();
	if (!selftest_wait("2 CPU-path quad")) return 0;
	/* 3a: gx_pass's GX mode with every input read from the vertex buffer (no fixed attributes) */
	selftest_quad56(&s_prog, 1);
	selftest_end();
	if (!selftest_wait("3a gx_pass shader (GX mode), all inputs from the buffer")) return 0;
	/* 3b: a vertex-colour quad on the GPU transform path, laid out as the game's first draw
	 * (position f32x3, colour u8x4: stride 16), identity matrices */
	{
		void* vb;
		uint16_t* ib;
		if (pica_alloc_raw(4 * 16, 6, &vb, &ib)) {
			const float cx[4] = {320, 576, 576, 320}, cy[4] = {64, 64, 192, 192};
			for (int i = 0; i < 4; i++) {
				float* f = (float*)((uint8_t*)vb + i * 16);
				f[0] = cx[i];
				f[1] = cy[i];
				f[2] = 0.0f;
				uint8_t* c = (uint8_t*)vb + i * 16 + 12;
				c[0] = 0;
				c[1] = 255;
				c[2] = 0;
				c[3] = 255;
			}
			static const uint16_t q[6] = {0, 1, 2, 0, 2, 3};
			memcpy(ib, q, sizeof q);
			PicaGxLayout L;
			memset(&L, 0, sizeof L);
			L.stride = 16;
			L.off_nrm = L.off_mtx = -1;
			L.off_tex[0] = L.off_tex[1] = L.off_tex[2] = -1;
			L.off_clr = 12;
			L.fixed_idx = 0;
			PicaGxUniforms u;
			memset(&u, 0, sizeof u);
			static const float ident[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
			pica_gx_projection(ident, &u);
			for (int i = 0; i < 4; i++) u.CS[i] = 1.0f;
			float rows[PICA_GX_MTX_ROWS][4];
			memset(rows, 0, sizeof rows);
			for (int r = 0; r < PICA_GX_MTX_ROWS; r++) rows[r][r % 3] = 1.0f;
			PicaState st;
			memset(&st, 0, sizeof st);
			st.write_mask = 0x0F;
			st.depth_func = GPU_ALWAYS;
			st.alpha_func = GPU_ALWAYS;
			st.cull = GPU_CULL_NONE;
			st.blend_eq = st.blend_aeq = GPU_BLEND_ADD;
			st.blend_src = st.blend_asrc = GPU_ONE;
			st.blend_dst = st.blend_adst = GPU_ZERO;
			pica_draw_gx(vb, 4, &L, ib, 6, &u, (1u << PICA_GX_MTX_ROWS) - 1, (const float(*)[4])rows, (const float(*)[4])rows, &st);
		} else {
			fprintf(stderr, "[gputest] 3: no ring space\n");
		}
	}
	selftest_end();
	if (!selftest_wait("3b GPU-transform quad with fixed inputs (as the game)")) return 0;
	/* 4: the present: the EFB texture drawn onto the top screen, then the display transfer */
	pica_present();
	if (!selftest_wait("4 present (EFB onto the top screen)")) return 0;
	selftest_end();
	if (!selftest_wait("5 empty frame")) return 0;
	selftest_end();
	fprintf(stderr, "[gputest] all steps finished\n");
	return 1;
}

/* ---- vertex benchmark ([vbench]) ------------------------------------------------------------- */
/* What a vertex costs the GPU on this console, apart from pixels: one grid mesh of tiny triangles
 * (half-pixel spacing, so almost nothing is filled) drawn with gx_pass in GX mode under several
 * setups, each in a frame of its own, timed by citro3d (C3D_GetDrawingTime). Each line gives the
 * frame's GPU ms and ns per index; the setups differ from the first in one thing each. */
enum { VB_GRID = 128, VB_VERTS = VB_GRID * VB_GRID, VB_IDX = (VB_GRID - 1) * (VB_GRID - 1) * 6 };
enum { VB_UNIQUE_TRIS = 10922 };
/* room for the S lines' strips: a strip each band row, two indices a quad column plus two to join */
enum { VB_STRIP = 49152 };
static GPU_Primitive_t s_vbPrim = GPU_TRIANGLES; /* the primitive vb_run draws */

struct VbField { int reg, fmt, count, bytes; };
static C3D_Tex s_vbTex[2];  /* setup G3's two textures */

/* Attribute loaders for fields in memory order (no gaps), the rest of the GX registers fixed. */
static int vb_attribs(const struct VbField* f, int nf, u64* perm)
{
	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	u64 p = 0;
	int used = 0;
	for (int i = 0; i < nf; i++) {
		if (f[i].reg < 0) { /* padding: 4-16 bytes skipped */
			p |= (u64)(0xB + f[i].bytes / 4) << (4 * i);
			continue;
		}
		int a = AttrInfo_AddLoader(ai, f[i].reg, (GPU_FORMATS)f[i].fmt, f[i].count);
		if (a < 0) return 0;
		p |= (u64)a << (4 * i);
		used |= 1 << f[i].reg;
	}
	for (int r = 0; r < NUM_GX_REGS; r++)
		if (!(used & (1 << r))) {
			int a = AttrInfo_AddFixed(ai, r);
			if (a < 0) return 0;
			C3D_FixedAttribSet(a, 0.0f, 0.0f, r == REG_NRM ? 1.0f : 0.0f, r == REG_NRM ? 0.0f : 1.0f);
		}
	*perm = p;
	return nf;
}

/* The uniforms for the mesh: positions in EFB pixels times `pscale`, light along +z, texture unit 0
 * from its own coordinate. */
static void vb_uniforms(float pscale, int lit, int tex)
{
	PicaGxUniforms u;
	memset(&u, 0, sizeof u);
	static const float ident[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	pica_gx_projection(ident, &u);
	for (int r = 0; r < 4; r++) C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uF + r, u.F[r][0], u.F[r][1], u.F[r][2], u.F[r][3]);
	for (int r = 0; r < PICA_GX_MTX_ROWS; r++) {
		const int c = r % 3;
		C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uP + r, (c == 0) * pscale, (c == 1) * pscale, (c == 2) * pscale, 0.0f);
		C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uN + r, c == 0, c == 1, c == 2, 0.0f);
	}
	for (int r = 0; r < 6; r++) {
		C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uT + r, tex && r == 0 ? 1.0f / 1024 : 0, tex && r == 1 ? 1.0f / 1024 : 0, 0, 0);
		C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uW + r, 0, 0, 0, 0);
	}
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uCS, 1, 1, 1, 1);
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uCM, 0, 0, 0, 0);
	/* lit: GX lighting with one light far along +z, as the old fixed shade (0.35 + 0.65 N.L) */
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uAMB, 0.35f, 0.35f, 0.35f, 1.0f);
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uLP, 0, 0, 1e6f, 1.0f);
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uLC, 0.65f, 0.65f, 0.65f, 0.0f);
	C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uLF, 0, 1, 0, 0);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bCpu, false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bPj, false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bLit, lit != 0);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bL[0], lit != 0);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bL[1], false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bA[0], false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bA[1], false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bS[0], false);
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bS[1], false);
	for (int k = 0; k < 3; k++) {
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bTu[k], tex && k == 0);
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bTd[k], tex && k == 0);
	}
}

/* Waits for the frame just ended; its GPU ms, or -1 when it never finished. */
static double vb_wait(void)
{
	const double before = s_gpuMsSum;
	u64 t0 = osGetTime();
	while (!frame_begin_ex(C3D_FRAME_NONBLOCK)) {
		if (osGetTime() - t0 > 3000) return -1.0;
		svcSleepThread(500000);
	}
	return s_gpuMsSum - before;
}

/* One setup: `reps` passes over the index list, each in draws of `per_draw` indices (0: one draw
 * a pass); `touch` changes a uniform before every draw, as the game's draws do. */
static void vb_run(const char* name, const void* verts, int stride, const struct VbField* f, int nf,
                   const uint16_t* idx, int nidx, int reps, int per_draw, int touch, float pscale, int lit, int tex,
                   GPU_CULLMODE cull)
{
	u64 perm;
	if (!vb_attribs(f, nf, &perm)) {
		fprintf(stderr, "[vbench] %-44s attribute setup failed\n", name);
		return;
	}
	PicaState st;
	memset(&st, 0, sizeof st);
	st.write_mask = 0x0F;
	st.depth_func = GPU_ALWAYS;
	st.alpha_func = GPU_ALWAYS;
	st.cull = cull;
	st.blend_eq = st.blend_aeq = GPU_BLEND_ADD;
	st.blend_src = st.blend_asrc = GPU_ONE;
	st.blend_dst = st.blend_adst = GPU_ZERO;
	s_last_valid = 0;
	apply_state(&st, 0);
	vb_uniforms(pscale, lit, tex);
	s_bufValid = 0;
	set_buffer(verts, stride, nf, perm);
	const int chunk = per_draw > 0 ? per_draw : nidx;
	int draws = 0;
	for (int r = 0; r < reps; r++)
		for (int at = 0; at < nidx; at += chunk) {
			if (!cmd_room()) break;
			if (touch == 1) C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uCM, 0, 0, 0, (float)(draws & 1) * (1.0f / 512));
			if (touch == 2) /* a matrix's rows, as each object's draw loads PNMTX0 */
				for (int r = 0; r < 3; r++) {
					const float k = 1.0f + (float)(draws & 1) * (1.0f / 4096);
					C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uP + r, (r == 0) * k * pscale, (r == 1) * k * pscale,
					              (r == 2) * k * pscale, 0.0f);
					C3D_FVUnifSet(GPU_VERTEX_SHADER, s_uN + r, r == 0, r == 1, r == 2, 0.0f);
				}
			if (touch == 3 && s_vbTex[0].data) { /* a texture bound and the combiner set, as a material change */
				C3D_TexBind(0, &s_vbTex[draws & 1]);
				C3D_TexEnv* env = C3D_GetTexEnv(0);
				C3D_TexEnvInit(env);
				if (draws & 1) {
					C3D_TexEnvSrc(env, C3D_Both, GPU_TEXTURE0, GPU_PRIMARY_COLOR, 0);
					C3D_TexEnvFunc(env, C3D_Both, GPU_MODULATE);
				} else {
					C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
					C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
				}
			}
			const int n = nidx - at < chunk ? nidx - at : chunk;
			C3D_DrawElements(s_vbPrim, n, C3D_UNSIGNED_SHORT, idx + at);
			draws++;
		}
	selftest_end();
	const double ms = vb_wait();
	const double n = (double)nidx * reps;
	if (ms < 0)
		fprintf(stderr, "[vbench] %-44s the GPU never finished\n", name);
	else
		fprintf(stderr, "[vbench] %-44s %6.2f ms  %5.0f ns/index  (%d draws, %.0fK indices, stride %d)\n", name, ms,
		        ms * 1e6 / n, draws, n / 1000.0, stride);
	fflush(stderr);
}

/* I1: the bring-up shader (position and colour in, the same out) over 16-byte vertices (f32
 * position, u8 colour), its projection uniform the EFB projection: what fewer outputs save. */
static void vb_run_min(const char* name, const void* verts, const uint16_t* idx, int nidx, int reps)
{
	DVLB_s* dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
	if (!dvlb) return;
	shaderProgram_s prog;
	shaderProgramInit(&prog);
	shaderProgramSetVsh(&prog, &dvlb->DVLE[0]);
	C3D_BindProgram(&prog);
	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	AttrInfo_AddLoader(ai, 0, GPU_FLOAT, 3);
	AttrInfo_AddLoader(ai, 1, GPU_UNSIGNED_BYTE, 4);
	PicaGxUniforms u;
	memset(&u, 0, sizeof u);
	static const float ident[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	pica_gx_projection(ident, &u);
	const int l = shaderInstanceGetUniformLocation(prog.vertexShader, "projection");
	for (int r = 0; r < 4 && l >= 0; r++) C3D_FVUnifSet(GPU_VERTEX_SHADER, l + r, u.F[r][0], u.F[r][1], u.F[r][2], u.F[r][3]);
	PicaState st;
	memset(&st, 0, sizeof st);
	st.write_mask = 0x0F;
	st.depth_func = GPU_ALWAYS;
	st.alpha_func = GPU_ALWAYS;
	st.cull = GPU_CULL_NONE;
	st.blend_eq = st.blend_aeq = GPU_BLEND_ADD;
	st.blend_src = st.blend_asrc = GPU_ONE;
	st.blend_dst = st.blend_adst = GPU_ZERO;
	s_last_valid = 0;
	apply_state(&st, 0);
	s_bufValid = 0;
	set_buffer(verts, 16, 2, 0x10);
	for (int r = 0; r < reps; r++) C3D_DrawElements(GPU_TRIANGLES, nidx, C3D_UNSIGNED_SHORT, idx);
	selftest_end();
	const double ms = vb_wait();
	fprintf(stderr, "[vbench] %-44s %6.2f ms  %5.0f ns/index  (%d draws, %.0fK indices, stride 16)\n", name, ms,
	        ms * 1e6 / ((double)nidx * reps), reps, nidx * reps / 1000.0);
	fflush(stderr);
	C3D_BindProgram(&s_prog);
	shaderProgramFree(&prog);
	DVLB_Free(dvlb);
	s_mode = MODE_NONE;
	s_gxLayoutValid = 0;
	s_gxMatricesLoaded = 0;
	s_fIsCpu = 0;
}

void pica_vertex_bench(void)
{
	if (!pica_init() || s_ovl) return;
	/* game-like vertices: f32 position, normal, three f32 coordinates, u8 colour and index (56) */
	struct V56 { float pos[3], nrm[3]; uint8_t clr[4]; float t0[2], t1[2], t2[2]; uint8_t mtx[4]; };
	/* compact: s16 position (x8, w unused), s8 normal, u8 colour, s16 coordinate, u8 index (24) */
	struct V24 { int16_t pos[4]; int8_t nrm[4]; uint8_t clr[4]; int16_t t0[2]; uint8_t mtx[4]; };
	static const struct VbField f56[] = {{REG_POS, GPU_FLOAT, 3, 12}, {REG_NRM, GPU_FLOAT, 3, 12},
	                                     {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}, {REG_TA, GPU_FLOAT, 2, 8},
	                                     {REG_TB, GPU_FLOAT, 2, 8}, {REG_TC, GPU_FLOAT, 2, 8},
	                                     {REG_IDX, GPU_UNSIGNED_BYTE, 4, 4}};
	static const struct VbField f24[] = {{REG_POS, GPU_SHORT, 4, 8}, {REG_NRM, GPU_BYTE, 4, 4},
	                                     {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}, {REG_TA, GPU_SHORT, 2, 4},
	                                     {REG_IDX, GPU_UNSIGNED_BYTE, 4, 4}};
	/* the game's packed resident layout: f32 position, s8 normal, u8 colour, s16 coordinate (24) */
	struct V24m { float pos[3]; int8_t nrm[4]; uint8_t clr[4]; int16_t t0[2]; };
	/* the game's usual decoded layout without matrix indices: f32 position, normal, colour, one coordinate (36) */
	struct V36 { float pos[3], nrm[3]; uint8_t clr[4]; float t0[2]; };
	/* position and colour only (16): I1, I2 */
	struct V16 { float pos[3]; uint8_t clr[4]; };
	static const struct VbField f16[] = {{REG_POS, GPU_FLOAT, 3, 12}, {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}};
	static const struct VbField f24m[] = {{REG_POS, GPU_FLOAT, 3, 12}, {REG_NRM, GPU_BYTE, 4, 4},
	                                      {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}, {REG_TA, GPU_SHORT, 2, 4}};
	static const struct VbField f36[] = {{REG_POS, GPU_FLOAT, 3, 12}, {REG_NRM, GPU_FLOAT, 3, 12},
	                                     {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}, {REG_TA, GPU_FLOAT, 2, 8}};
	_Static_assert(sizeof(struct V56) == 56 && sizeof(struct V24) == 24 && sizeof(struct V24m) == 24 &&
	               sizeof(struct V36) == 36, "vbench vertex sizes");
	const size_t uniqueVerts = VB_UNIQUE_TRIS * 3;
	struct V56* lin56 = (struct V56*)linearAlloc(VB_VERTS * sizeof(struct V56));
	struct V24* lin24 = (struct V24*)linearAlloc(VB_VERTS * sizeof(struct V24));
	struct V56* vram56 = (struct V56*)vramAlloc(VB_VERTS * sizeof(struct V56));
	struct V24* vram24 = (struct V24*)vramAlloc(VB_VERTS * sizeof(struct V24));
	struct V56* uniq56 = (struct V56*)linearAlloc(uniqueVerts * sizeof(struct V56));
	struct V24m* lin24m = (struct V24m*)linearAlloc(VB_VERTS * sizeof(struct V24m));
	struct V36* lin36 = (struct V36*)linearAlloc(VB_VERTS * sizeof(struct V36));
	struct V16* lin16 = (struct V16*)linearAlloc(VB_VERTS * sizeof(struct V16));
	uint16_t* idx = (uint16_t*)linearAlloc(VB_IDX * 2);
	uint16_t* seq = (uint16_t*)linearAlloc(uniqueVerts * 2);
	uint16_t* idxw = (uint16_t*)linearAlloc(VB_IDX * 2);
	uint16_t* strip = (uint16_t*)linearAlloc(VB_STRIP * 2);
	uint8_t* linL = (uint8_t*)linearAlloc(VB_VERTS * 64);
	if (!lin56 || !lin24 || !vram56 || !vram24 || !uniq56 || !lin24m || !lin36 || !lin16 || !idx || !seq || !idxw ||
	    !strip || !linL) {
		fprintf(stderr, "[vbench] not enough memory; skipped\n");
		goto out;
	}
	fprintf(stderr, "[vbench] start: %d-vertex grid, %d indices a pass, triangles about 1/8 pixel\n", VB_VERTS, VB_IDX);
	fflush(stderr);
	for (int y = 0; y < VB_GRID; y++)
		for (int x = 0; x < VB_GRID; x++) {
			const float px = 288.0f + x * 0.5f, py = 160.0f + y * 0.5f;
			struct V56* a = &lin56[y * VB_GRID + x];
			memset(a, 0, sizeof *a);
			a->pos[0] = px;
			a->pos[1] = py;
			a->nrm[2] = 1.0f;
			a->clr[0] = (uint8_t)(x * 2);
			a->clr[1] = (uint8_t)(y * 2);
			a->clr[2] = a->clr[3] = 255;
			a->t0[0] = (float)x;
			a->t0[1] = (float)y;
			struct V24* b = &lin24[y * VB_GRID + x];
			memset(b, 0, sizeof *b);
			b->pos[0] = (int16_t)(px * 8);
			b->pos[1] = (int16_t)(py * 8);
			b->nrm[2] = 127;
			memcpy(b->clr, a->clr, 4);
			b->t0[0] = (int16_t)x;
			b->t0[1] = (int16_t)y;
			struct V24m* c = &lin24m[y * VB_GRID + x];
			memcpy(c->pos, a->pos, 12);
			c->nrm[0] = c->nrm[1] = c->nrm[3] = 0;
			c->nrm[2] = 64;
			memcpy(c->clr, a->clr, 4);
			c->t0[0] = (int16_t)x;
			c->t0[1] = (int16_t)y;
			struct V36* d = &lin36[y * VB_GRID + x];
			memcpy(d->pos, a->pos, 12);
			memcpy(d->nrm, a->nrm, 12);
			memcpy(d->clr, a->clr, 4);
			memcpy(d->t0, a->t0, 8);
			memcpy(lin16[y * VB_GRID + x].pos, a->pos, 12);
			memcpy(lin16[y * VB_GRID + x].clr, a->clr, 4);
		}
	{
		int n = 0;
		for (int y = 0; y + 1 < VB_GRID; y++)
			for (int x = 0; x + 1 < VB_GRID; x++) {
				const uint16_t i0 = (uint16_t)(y * VB_GRID + x), i1 = i0 + 1, i2 = i0 + VB_GRID, i3 = i2 + 1;
				idx[n++] = i0, idx[n++] = i2, idx[n++] = i1;
				idx[n++] = i1, idx[n++] = i2, idx[n++] = i3;
			}
		/* the same triangles with no vertex shared (the first VB_UNIQUE_TRIS of them) */
		for (size_t i = 0; i < uniqueVerts; i++) {
			uniq56[i] = lin56[idx[i]];
			seq[i] = (uint16_t)i;
		}
	}
	GSPGPU_FlushDataCache(lin56, VB_VERTS * sizeof(struct V56));
	GSPGPU_FlushDataCache(lin24, VB_VERTS * sizeof(struct V24));
	GSPGPU_FlushDataCache(uniq56, uniqueVerts * sizeof(struct V56));
	GSPGPU_FlushDataCache(lin24m, VB_VERTS * sizeof(struct V24m));
	GSPGPU_FlushDataCache(lin36, VB_VERTS * sizeof(struct V36));
	GSPGPU_FlushDataCache(lin16, VB_VERTS * sizeof(struct V16));
	GSPGPU_FlushDataCache(idx, VB_IDX * 2);
	GSPGPU_FlushDataCache(seq, uniqueVerts * 2);
	s_mode = MODE_NONE;
	s_gxLayoutValid = 0;
	s_gxMatricesLoaded = 0;
	s_fIsCpu = 0;
	if (!selftest_wait("vbench setup")) goto out;
	/* VRAM copies by the GPU (the CPU cannot write VRAM: a memcpy there was a data abort). They are
	 * queued in this frame, ahead of its draw list, and are done before setup C reads them. */
	GX_TextureCopy((u32*)lin56, 0, (u32*)vram56, 0, VB_VERTS * sizeof(struct V56), 8);
	GX_TextureCopy((u32*)lin24, 0, (u32*)vram24, 0, VB_VERTS * sizeof(struct V24), 8);
	const int R = 3;
	vb_run("A  f32 56B, FCRAM, lit + tex0 (as the game)", lin56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
	vb_run("A' again (repeatability)", lin56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
	vb_run("B  shortest shader (no light, no tex)", lin56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 0, 0, GPU_CULL_NONE);
	vb_run("C  vertices in VRAM", vram56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
	vb_run("D  s16/s8 24B, FCRAM", lin24, 24, f24, 5, idx, VB_IDX, R, 0, 0, 1.0f / 8, 1, 1, GPU_CULL_NONE);
	vb_run("E  s16/s8 24B, VRAM", vram24, 24, f24, 5, idx, VB_IDX, R, 0, 0, 1.0f / 8, 1, 1, GPU_CULL_NONE);
	vb_run("D2 f32 pos + s8/s16 24B, FCRAM (game packed)", lin24m, 24, f24m, 4, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1,
	       GPU_CULL_NONE);
	vb_run("D3 f32 36B, FCRAM (game decoded)", lin36, 36, f36, 4, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
	vb_run("D3s as D3, shortest shader (no light, no tex)", lin36, 36, f36, 4, idx, VB_IDX, R, 0, 0, 1.0f, 0, 0,
	       GPU_CULL_NONE);
	vb_run("D3l as D3, light only", lin36, 36, f36, 4, idx, VB_IDX, R, 0, 0, 1.0f, 1, 0, GPU_CULL_NONE);
	vb_run("D3t as D3, tex0 only", lin36, 36, f36, 4, idx, VB_IDX, R, 0, 0, 1.0f, 0, 1, GPU_CULL_NONE);
	vb_run("D4 as D2, draws of 252, uniform set each", lin24m, 24, f24m, 4, idx, VB_IDX, R, 252, 1, 1.0f, 1, 1,
	       GPU_CULL_NONE);
	vb_run("D5 as D3, draws of 252, uniform set each", lin36, 36, f36, 4, idx, VB_IDX, R, 252, 1, 1.0f, 1, 1,
	       GPU_CULL_NONE);
	vb_run("I2 f32 pos + colour 16B, game shader (6 outputs)", lin16, 16, f16, 2, idx, VB_IDX, R, 0, 0, 1.0f, 0, 0,
	       GPU_CULL_NONE);
	vb_run_min("I1 the same, a 2-output shader", lin16, idx, VB_IDX, R);
	vb_run("F  no vertex shared (no reuse)", uniq56, 56, f56, 7, seq, (int)uniqueVerts, R * 3, 0, 0, 1.0f, 1, 1,
	       GPU_CULL_NONE);
	vb_run("G  draws of 252 indices, uniform set each", lin56, 56, f56, 7, idx, VB_IDX, R, 252, 1, 1.0f, 1, 1,
	       GPU_CULL_NONE);
	vb_run("G2 as G, a matrix's 6 rows set each", lin56, 56, f56, 7, idx, VB_IDX, R, 252, 2, 1.0f, 1, 1, GPU_CULL_NONE);
	if (C3D_TexInit(&s_vbTex[0], 64, 64, GPU_RGBA8) && C3D_TexInit(&s_vbTex[1], 64, 64, GPU_RGBA8)) {
		memset(s_vbTex[0].data, 0x80, s_vbTex[0].size);
		memset(s_vbTex[1].data, 0xC0, s_vbTex[1].size);
		GSPGPU_FlushDataCache(s_vbTex[0].data, s_vbTex[0].size);
		GSPGPU_FlushDataCache(s_vbTex[1].data, s_vbTex[1].size);
		vb_run("G3 as G, a texture bound + combiner set each", lin56, 56, f56, 7, idx, VB_IDX, R, 252, 3, 1.0f, 1, 1,
		       GPU_CULL_NONE);
		C3D_TexDelete(&s_vbTex[0]);
		C3D_TexDelete(&s_vbTex[1]);
		memset(s_vbTex, 0, sizeof s_vbTex);
		passthrough_from(0);
	}
	vb_run("H  every triangle culled (back faces)", lin56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1,
	       GPU_CULL_BACK_CCW);
	vb_run("H' every triangle culled (front faces)", lin56, 56, f56, 7, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1,
	       GPU_CULL_FRONT_CCW);
	/* W: D3's triangles in vertical bands of w quads, each band top to bottom: a vertex's second use
	 * (by the next row of quads) comes 6w indices after its first, so the bands at which the cost
	 * drops give the size of the GPU's post-transform vertex cache (rows the full width, as D3:
	 * each vertex shaded about twice; narrow enough bands: about once). */
	vb_run("D3 again (rows the full width)", lin36, 36, f36, 4, idx, VB_IDX, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
	{
		static const int widths[] = {2, 3, 4, 6, 8, 12, 16, 32};
		for (size_t k = 0; k < sizeof widths / sizeof widths[0]; k++) {
			const int w = widths[k];
			int n = 0;
			for (int x0 = 0; x0 + 1 < VB_GRID; x0 += w)
				for (int y = 0; y + 1 < VB_GRID; y++)
					for (int x = x0; x < x0 + w && x + 1 < VB_GRID; x++) {
						const uint16_t i0 = (uint16_t)(y * VB_GRID + x), i1 = i0 + 1, i2 = i0 + VB_GRID, i3 = i2 + 1;
						idxw[n++] = i0, idxw[n++] = i2, idxw[n++] = i1;
						idxw[n++] = i1, idxw[n++] = i2, idxw[n++] = i3;
					}
			GSPGPU_FlushDataCache(idxw, VB_IDX * 2);
			char name[64];
			snprintf(name, sizeof name, "W%-2d D3 in bands %d quads wide", w, w);
			vb_run(name, lin36, 36, f36, 4, idxw, n, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
		}
	}
	/* S: D3's triangles as triangle strips (one index a triangle instead of three), compare the ms
	 * with D3's: S1 one strip a row, all joined into one draw; S8/S16 strips across bands of 8 and
	 * 16 quads (as W8/W16), joined the same way. */
	{
		static const int widths[] = {VB_GRID - 1, 8, 16};
		for (size_t k = 0; k < sizeof widths / sizeof widths[0]; k++) {
			const int w = widths[k];
			int n = 0;
			for (int x0 = 0; x0 + 1 < VB_GRID; x0 += w)
				for (int y = 0; y + 1 < VB_GRID; y++) {
					const int x1 = x0 + w < VB_GRID - 1 ? x0 + w : VB_GRID - 1;
					if (n + 2 * (x1 - x0 + 1) + 2 > VB_STRIP) break;
					if (n) { /* join: the last index again, then this strip's first */
						strip[n] = strip[n - 1];
						n++;
						strip[n++] = (uint16_t)(y * VB_GRID + x0);
					}
					for (int x = x0; x <= x1; x++) {
						strip[n++] = (uint16_t)(y * VB_GRID + x);
						strip[n++] = (uint16_t)((y + 1) * VB_GRID + x);
					}
				}
			GSPGPU_FlushDataCache(strip, (size_t)n * 2);
			char name[64];
			if (k == 0) snprintf(name, sizeof name, "S1 D3 as strips, one a row (one draw)");
			else snprintf(name, sizeof name, "S%-2d D3 as strips across bands of %d", w, w);
			s_vbPrim = GPU_TRIANGLE_STRIP;
			vb_run(name, lin36, 36, f36, 4, strip, n, R, 0, 0, 1.0f, 1, 1, GPU_CULL_NONE);
			s_vbPrim = GPU_TRIANGLES;
		}
	}
	/* L: what each part of a layout costs, from D3's 36 bytes (f32 position and normal, u8 colour,
	 * f32 coordinate) to A's 56 and the game's: the matrix-index block the game stores (12 bytes, 4
	 * read and 8 skipped by a padding entry), more coordinates, and A padded to 64 bytes. */
	{
		enum { P_POS, P_NRM, P_CLR, P_T0, P_T1, P_T2, P_IDX, P_PAD4, P_PAD8 };
		static const struct VbField part[] = {{REG_POS, GPU_FLOAT, 3, 12}, {REG_NRM, GPU_FLOAT, 3, 12},
		                                      {REG_CLR, GPU_UNSIGNED_BYTE, 4, 4}, {REG_TA, GPU_FLOAT, 2, 8},
		                                      {REG_TB, GPU_FLOAT, 2, 8}, {REG_TC, GPU_FLOAT, 2, 8},
		                                      {REG_IDX, GPU_UNSIGNED_BYTE, 4, 4}, {-1, 0, 0, 4}, {-1, 0, 0, 8}};
		static const struct { const char* name; int n, parts[9]; } lay[] = {
		    {"L1 D3 + matrix index (40B)", 5, {P_POS, P_NRM, P_CLR, P_T0, P_IDX}},
		    {"L2 D3 + the game's matrix block (48B, 8 skipped)", 6, {P_POS, P_NRM, P_CLR, P_T0, P_IDX, P_PAD8}},
		    {"L3 D3 + 4 bytes skipped (40B)", 5, {P_POS, P_NRM, P_CLR, P_T0, P_PAD4}},
		    {"L4 D3 + coordinate 1 (44B)", 5, {P_POS, P_NRM, P_CLR, P_T0, P_T1}},
		    {"L5 D3 + coordinates 1, 2 (52B)", 6, {P_POS, P_NRM, P_CLR, P_T0, P_T1, P_T2}},
		    {"L6 A (56B, as line A)", 7, {P_POS, P_NRM, P_CLR, P_T0, P_T1, P_T2, P_IDX}},
		    {"L7 A + 8 bytes skipped (64B)", 8, {P_POS, P_NRM, P_CLR, P_T0, P_T1, P_T2, P_IDX, P_PAD8}},
		    {"L8 position, colour, coordinate (24B, unlit)", 3, {P_POS, P_CLR, P_T0}},
		    {"L9 position, normal (24B)", 2, {P_POS, P_NRM}},
		};
		for (size_t k = 0; k < sizeof lay / sizeof lay[0]; k++) {
			struct VbField f[9];
			int stride = 0;
			for (int j = 0; j < lay[k].n; j++) {
				f[j] = part[lay[k].parts[j]];
				stride += f[j].bytes;
			}
			for (int v = 0; v < VB_VERTS; v++) {
				const struct V56* a = &lin56[v];
				uint8_t* o = linL + (size_t)v * stride;
				for (int j = 0; j < lay[k].n; j++) {
					const void* src = NULL;
					switch (lay[k].parts[j]) {
					case P_POS: src = a->pos; break;
					case P_NRM: src = a->nrm; break;
					case P_CLR: src = a->clr; break;
					case P_T0: src = a->t0; break;
					case P_T1: src = a->t1; break;
					case P_T2: src = a->t2; break;
					case P_IDX: src = a->mtx; break;
					}
					if (src) memcpy(o, src, (size_t)f[j].bytes);
					else memset(o, 0, (size_t)f[j].bytes);
					o += f[j].bytes;
				}
			}
			GSPGPU_FlushDataCache(linL, (size_t)VB_VERTS * stride);
			const int lit = lay[k].parts[1] == P_NRM;
			vb_run(lay[k].name, linL, stride, f, lay[k].n, idx, VB_IDX, R, 0, 0, 1.0f, lit, lay[k].n > 2, GPU_CULL_NONE);
		}
	}
	fprintf(stderr, "[vbench] done\n");
out:
	if (lin56) linearFree(lin56);
	if (lin24) linearFree(lin24);
	if (vram56) vramFree(vram56);
	if (vram24) vramFree(vram24);
	if (uniq56) linearFree(uniq56);
	if (lin24m) linearFree(lin24m);
	if (lin36) linearFree(lin36);
	if (lin16) linearFree(lin16);
	if (idx) linearFree(idx);
	if (seq) linearFree(seq);
	if (idxw) linearFree(idxw);
	if (strip) linearFree(strip);
	if (linL) linearFree(linL);
	/* the GPU state, buffer and uniforms here were the benchmark's own */
	s_mode = MODE_NONE;
	s_gxLayoutValid = 0;
	s_gxMatricesLoaded = 0;
	s_fIsCpu = 0;
	s_bufValid = 0;
	s_last_valid = 0;
	if (s_in_frame) selftest_end(); /* an empty frame: the last timed one has finished */
}

/* ---- per-draw GPU check (pica_set_gpu_check) -------------------------------------------------- */
/* Each draw is submitted on its own and waited for (2 s timeout), so a draw the GPU never finishes
 * is named in the log with its vertex layout, textures and combiner stages. Very slow; diagnosis
 * only. */
void pica_set_gpu_check(int on) { s_check = on; }

static void describe_state(const PicaState* st)
{
	fprintf(stderr, "[gpucheck]   state: depth test %u func %u write %u, blend %u eq %u src %u dst %u / a eq %u src %u dst %u, "
	        "mask %02x, cull %u, scissor %u (%u,%u)-(%u,%u), alpha test %u func %u ref %u, stages %u, buf %02x/%02x %08lx\n",
	        st->depth_test, st->depth_func, st->depth_write, st->blend, st->blend_eq, st->blend_src, st->blend_dst,
	        st->blend_aeq, st->blend_asrc, st->blend_adst, st->write_mask, st->cull, st->scissor, st->sc_x0, st->sc_y0,
	        st->sc_x1, st->sc_y1, st->alpha_test, st->alpha_func, st->alpha_ref, st->nstages, st->buf_rgb, st->buf_alpha,
	        (unsigned long)st->buf_color);
	for (int u = 0; u < PICA_MAX_TEX; u++) {
		const PicaTex* t = st->tex[u];
		if (!t) continue;
		fprintf(stderr, "[gpucheck]   texture %d: %ux%u fmt %d data %p (phys %08lx) %s, wrap %u/%u, %s\n", u,
		        t->tex.width, t->tex.height, (int)t->tex.fmt, t->tex.data,
		        (unsigned long)osConvertVirtToPhys(t->tex.data), t->rt ? "EFB copy" : "uploaded", st->wrap_s[u],
		        st->wrap_t[u], st->linear[u] ? "linear" : "nearest");
	}
	for (int i = 0; i < st->nstages && i < PICA_MAX_STAGES; i++) {
		const PicaTevStage* g = &st->stage[i];
		fprintf(stderr, "[gpucheck]   stage %d: src %03x/%03x op %03x/%03x fn %u/%u scale %u/%u const %08lx\n", i,
		        g->src_rgb, g->src_alpha, g->op_rgb, g->op_alpha, g->func_rgb, g->func_alpha, g->scale_rgb,
		        g->scale_alpha, (unsigned long)g->color);
	}
}

static void gpu_check(const char* kind, const PicaGxLayout* L, uint32_t nidx, const PicaState* st)
{
	if (!s_check || !s_in_frame) return;
	s_checkDraws++;
	flush_rings();
	frame_end();
	s_in_frame = 0;
	s_last_valid = 0;
	u64 t0 = osGetTime();
	while (!frame_begin_ex(C3D_FRAME_NONBLOCK)) {
		if (osGetTime() - t0 > 2000) {
			fprintf(stderr, "[gpucheck] HUNG on draw %u of frame %u (%lu checked): %s, %lu indices\n",
			        (unsigned)s_frameDraws, (unsigned)s_checkFrames, (unsigned long)s_checkDraws, kind, (unsigned long)nidx);
			if (L)
				fprintf(stderr, "[gpucheck]   layout: stride %u, normal %d, colour %d, tex %d %d %d, matrix %d (fixed %d)\n",
				        L->stride, L->off_nrm, L->off_clr, L->off_tex[0], L->off_tex[1], L->off_tex[2], L->off_mtx,
				        L->fixed_idx);
			if (st) describe_state(st);
			fflush(stderr);
			s_where = "GPU check: a draw hung the GPU (see log)";
			s_check = 0;
			for (;;) svcSleepThread(1000000000LL); /* the GPU is gone; keep the log as it is */
		}
		svcSleepThread(200000);
	}
}

/* Most vertex-ring bytes one frame used since the last call (one ring half's capacity:
 * pica_ring_capacity). */
uint32_t pica_ring_peak(void)
{
	s_ringPeak = s_ringPeakNow;
	s_ringPeakNow = 0;
	return s_ringPeak;
}
uint32_t pica_ring_capacity(void) { return VTX_RING_BYTES; }

int pica_kernel_threads(void)
{
	Handle rl;
	ResourceLimitType what = RESLIMIT_THREAD;
	s64 cur = -1;
	if (R_FAILED(svcGetResourceLimit(&rl, CUR_PROCESS_HANDLE)))
		return -1;
	svcGetResourceLimitCurrentValues(&cur, rl, &what, 1);
	svcCloseHandle(rl);
	return (int)cur;
}
