/* PICA200 draw path for the GX back end: see pica_draw.h. */
#include "pica_draw.h"

#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gx_pass_shbin.h"

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
static int s_uF, s_uP, s_uN, s_uT, s_uW, s_uCS, s_uCM, s_uLD, s_bCpu;
static int s_bLit, s_bTu[3], s_bTd[3], s_bPj; /* the GX shader's per-draw shortcuts */
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

/* pica_request_stop: quitting. Every entry point goes through pica_init, so nothing new reaches
 * the GPU after this, and what is in flight finishes before the screens are shut down. */
static volatile int s_stopping;
void pica_request_stop(void) { s_stopping = 1; }

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
#define COPY_ARENA_BYTES (2u << 20)
#define COPY_UNIT 4096u
#define COPY_UNITS (COPY_ARENA_BYTES / COPY_UNIT)
static uint8_t* s_copyArena;
static uint8_t s_copyUnitUsed[COPY_UNITS];
static uint32_t s_copyUnits, s_copyUnitsPeak;

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
	{
		shaderInstance_s* vs = s_prog.vertexShader;
		s_uF = shaderInstanceGetUniformLocation(vs, "F");
		s_uP = shaderInstanceGetUniformLocation(vs, "P");
		s_uN = shaderInstanceGetUniformLocation(vs, "N");
		s_uT = shaderInstanceGetUniformLocation(vs, "T");
		s_uW = shaderInstanceGetUniformLocation(vs, "W");
		s_uCS = shaderInstanceGetUniformLocation(vs, "CS");
		s_uCM = shaderInstanceGetUniformLocation(vs, "CM");
		s_uLD = shaderInstanceGetUniformLocation(vs, "LD");
		s_bCpu = shaderInstanceGetUniformLocation(vs, "cpu");
		s_bLit = shaderInstanceGetUniformLocation(vs, "lit");
		s_bPj = shaderInstanceGetUniformLocation(vs, "pj");
		for (int k = 0; k < 3; k++) {
			char n[4] = {'t', 'u', (char)('0' + k), 0};
			s_bTu[k] = shaderInstanceGetUniformLocation(vs, n);
			n[1] = 'd';
			s_bTd[k] = shaderInstanceGetUniformLocation(vs, n);
		}
	}

	if (!C3D_TexInitVRAM(&s_efbTex, EFB_TEX_W, EFB_TEX_H, GPU_RGBA8)) goto fail;
	s_efb = C3D_RenderTargetCreateFromTex(&s_efbTex, GPU_TEXFACE_2D, 0, GPU_RB_DEPTH24_STENCIL8);
	if (!s_efb) goto fail;
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
	uint16_t arenaUnit, arenaUnits; /* EFB copy textures: their run in s_copyArena (0 units: own VRAM) */
	float wx, wy, ww, wh;           /* EFB copy textures: the EFB window (EFB pixels) of their latest copy */
	uint32_t lastUse, vramUse, vcopyFrame, vtryFrame; /* frame numbers (s_frameNo) */
};

/* A texture freed while a frame may still be drawing with it waits for two frame starts. */
#define MAX_DEFERRED 256
static PicaTex* s_dead[2][MAX_DEFERRED];
static int s_ndead[2];

static void tex_count(const PicaTex* t, int sign);
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
 * them when it could not be made: at most 6 (the back end's kMaxEfbCopies), one more while a
 * replaced one waits out its two frames, each at most a whole-EFB texture. */
#define COPY_TEX_BYTES (EFB_TEX_W * EFB_TEX_H * 4u)
static uint32_t copy_reserve(void)
{
	const int left = 7 - s_copyTexCount;
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

static void tex_flush(PicaTex* t)
{
	GSPGPU_FlushDataCache(t->tex.data, t->bytes);
	if (t->vram && !t->noVram) vram_copy(t);
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
	t->bytes = w * h * bpp;
	tex_count(t, 1);
	return t;
}

PicaTex* pica_tex_create(uint32_t w, uint32_t h) { return tex_create(w, h, GPU_RGBA8, 4); }
PicaTex* pica_tex_create_fmt(uint32_t w, uint32_t h, int fmt)
{
	static const GPU_TEXCOLOR kGpu[] = {GPU_RGBA8, GPU_RGB565, GPU_RGBA5551, GPU_RGBA4, GPU_LA8, GPU_L8};
	static const uint8_t kBpp[] = {4, 2, 2, 2, 2, 1};
	if (fmt < 0 || fmt > PICA_TF_L8) fmt = PICA_TF_RGBA8;
	PicaTex* t = tex_create(w, h, kGpu[fmt], kBpp[fmt]);
	if (t) t->fmt = fmt;
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
	if (t->fmt == PICA_TF_RGBA8)
		tex_swizzle((uint32_t*)t->tex.data, t->tex.width, t->tex.height, rgba);
	else
		tex_swizzle_small(t->tex.data, t->tex.width, t->tex.height, rgba, t->fmt);
	tex_flush(t);
}

PicaTex* pica_tex_create_mip(uint32_t w, uint32_t h, int levels, float lodBias)
{
	if (!pica_init() || levels < 1) return NULL;
	PicaTex* t = (PicaTex*)calloc(1, sizeof *t);
	if (!t) return NULL;
	const bool ok = tex_init(&t->tex, w, h, levels - 1, GPU_RGBA8);
	if (!ok) {
		free(t);
		return NULL;
	}
	C3D_TexSetFilterMipmap(&t->tex, GPU_LINEAR);
	C3D_TexSetLodBias(&t->tex, lodBias);
	t->bytes = C3D_TexCalcTotalSize(t->tex.size, levels - 1);
	tex_count(t, 1);
	return t;
}

void pica_tex_upload_level(PicaTex* t, int level, const uint8_t* rgba)
{
	if (!t || level > t->tex.maxLevel) return;
	tex_swizzle((uint32_t*)C3D_Tex2DGetImagePtr(&t->tex, level, NULL), t->tex.width >> level, t->tex.height >> level,
	            rgba);
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

/* GPU test (pica_cycle_gpu_test): 1 scissors every game draw to 8x8 pixels, so the GPU still
 * transforms every vertex but shades almost no pixels. Comparing its GPU time with the normal
 * mode's tells vertex work from pixel work. */
static int s_gpuTest;
int pica_cycle_gpu_test(void)
{
	s_gpuTest = (s_gpuTest + 1) % 2;
	s_last_valid = 0;
	return s_gpuTest;
}
int pica_gpu_test(void) { return s_gpuTest; }
void pica_set_gpu_test(int mode) { s_gpuTest = mode; }
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

static void readback_finish(void);
static void readback_queue(void);

static void ovl_frame_end(void)
{
	gxCmdQueue_s* q = &s_ovq[s_ovb];
	u32* buf;
	u32 size;
	if (C3Di_SplitFrame(&buf, &size)) GX_ProcessCommandList(buf, size * 4, GX_CMDLIST_FLUSH);
	GPUCMD_SetBuffer(NULL, 0, 0);
	end_mark(PICA_END_SUBMIT);
	{
		/* As C3D_FrameEnd(0): write back the whole linear heap. */
		extern u32 __ctru_linear_heap, __ctru_linear_heap_size;
		GSPGPU_FlushDataCache((void*)__ctru_linear_heap, __ctru_linear_heap_size);
	}
	end_mark(PICA_END_HEAP_FLUSH);
	s_where = "waiting for the GPU to finish the previous frame";
	ovl_wait_running();
	end_mark(PICA_END_WAIT);
	readback_finish(); /* the previous frame's copies are drawn: into the game's memory */
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
void pica_flush(const void* p, uint32_t bytes) { GSPGPU_FlushDataCache(p, bytes); }
/* Waits until the GPU has finished every frame handed to it. */
void pica_gpu_idle(void)
{
	if (s_ovl) ovl_wait_running();
	else if (s_queue) gxCmdQueueWait(s_queue, -1);
}

/* C3D_FrameDrawOn, without citro3d's in-frame check in overlapped frames. */
static void draw_on(C3D_RenderTarget* t)
{
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
		C3D_RenderTargetClear(s_efb, C3D_CLEAR_ALL, s_clear_rgba, s_clear_depth);
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

static void apply_state(const PicaState* st);
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
static inline void vunif(int id, float x, float y, float z, float w)
{
	const C3D_FVec* v = &C3D_FVUnif[GPU_VERTEX_SHADER][id];
	if (v->x != x || v->y != y || v->z != z || v->w != w) C3D_FVUnifSet(GPU_VERTEX_SHADER, id, x, y, z, w);
}

static inline int vzero(const float* v) { return v[0] == 0.f && v[1] == 0.f && v[2] == 0.f && v[3] == 0.f; }

/* The shader's shortcuts for this draw (gx_pass.v.pica): each only skips work whose result the
 * uniforms already fix. Unshaded draws skip the normal and the shade; a unit whose rows T are zero
 * has the coordinate 0 either way; a unit whose source is exactly its own attribute (unit N from
 * v(3+N), the usual case) skips building the source. */
static void gx_shortcuts(const PicaGxUniforms* u)
{
	C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bLit, u->LD[3] != 0.f);
	for (int k = 0; k < 3; k++) {
		const float* w0 = u->W[2 * k];
		const float* w1 = u->W[2 * k + 1];
		int direct;
		if (k < 2)
			direct = w0[0] == 0.f && w0[1] == 0.f && w0[2] == (k == 0 ? 1.f : 0.f) &&
			         w0[3] == (k == 1 ? 1.f : 0.f) && vzero(w1);
		else
			direct = vzero(w0) && w1[0] == 1.f && w1[1] == 0.f && w1[2] == 0.f && w1[3] == 0.f;
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bTu[k], !(vzero(u->T[2 * k]) && vzero(u->T[2 * k + 1])));
		C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bTd[k], direct);
	}
}

int pica_draw_gx(const void* verts, uint32_t nverts, const PicaGxLayout* L, const uint16_t* idx, uint32_t nidx,
                 const PicaGxUniforms* u, int matrices_dirty, const float (*pos_rows)[4], const float (*nrm_rows)[4],
                 const PicaState* st)
{
	(void)nverts;
	if (!s_in_frame || nidx == 0) return 1;
	if (s_mode != MODE_GX || !s_gxLayoutValid || memcmp(L, &s_gxLayout, sizeof *L) != 0) {
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
		matrices_dirty = matrices_dirty || !s_gxMatricesLoaded;
	}
	apply_state(st);
	if (matrices_dirty || !s_gxMatricesLoaded) {
		for (int r = 0; r < PICA_GX_MTX_ROWS; r++) {
			vunif(s_uP + r, pos_rows[r][0], pos_rows[r][1], pos_rows[r][2], pos_rows[r][3]);
			vunif(s_uN + r, nrm_rows[r][0], nrm_rows[r][1], nrm_rows[r][2], 0.0f);
		}
		s_gxMatricesLoaded = 1;
	}
	for (int r = 0; r < 4; r++) vunif(s_uF + r, u->F[r][0], u->F[r][1], u->F[r][2], u->F[r][3]);
	s_fIsCpu = 0;
	for (int r = 0; r < 6; r++) {
		vunif(s_uT + r, u->T[r][0], u->T[r][1], u->T[r][2], u->T[r][3]);
		vunif(s_uW + r, u->W[r][0], u->W[r][1], u->W[r][2], u->W[r][3]);
	}
	vunif(s_uCS, u->CS[0], u->CS[1], u->CS[2], u->CS[3]);
	vunif(s_uCM, u->CM[0], u->CM[1], u->CM[2], u->CM[3]);
	vunif(s_uLD, u->LD[0], u->LD[1], u->LD[2], u->LD[3]);
	gx_shortcuts(u);
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, verts, L->stride, s_gxEntries, s_gxPerm);
	s_frameDraws++;
	C3D_DrawElements(GPU_TRIANGLES, (int)nidx, C3D_UNSIGNED_SHORT, idx);
	second_pass(GPU_TRIANGLES, (int)nidx, idx, st);
	gpu_check("GPU-transform draw", L, nidx, st);
	maybe_split();
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
	C3D_DrawElements(prim, count, C3D_UNSIGNED_SHORT, idx);
	C3D_AlphaTest(st->alpha_test, (GPU_TESTFUNC)st->alpha_func, st->alpha_ref);
}

static void apply_state(const PicaState* st)
{
	if (s_last_valid && memcmp(st, &s_last, sizeof *st) == 0) return;
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
		if (u == 0 && st->proj0) { /* (s, t) divided by q (texcoord0w) per pixel */
			s_projTex = *tex;
			s_projTex.param = (s_projTex.param & ~GPU_TEXTURE_MODE(7)) | GPU_TEXTURE_MODE(GPU_TEX_PROJECTION);
			tex = &s_projTex;
		}
		C3D_TexBind(u, tex);
	}
	if (!s_last_valid || st->proj0 != s_last.proj0) C3D_BoolUnifSet(GPU_VERTEX_SHADER, s_bPj, st->proj0 != 0);
	/* Depth is the GameCube's own (-z/w); a decal drawn on the surface under it (the goop layer)
	 * is moved toward the camera by a few steps of the PICA's float24 depth, which is coarser near
	 * 1 than the GameCube's 24-bit fixed point and lets the two fight. */
	if (!s_last_valid || st->depth_bias != s_last.depth_bias)
		C3D_DepthMap(true, -1.0f, st->depth_bias ? -1.0f / 16384.0f : 0.0f);
	if (st->nstages == 0) {
		C3D_TexEnv* env = C3D_GetTexEnv(0);
		C3D_TexEnvInit(env);
		C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
		C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);
		passthrough_from(1);
	} else {
		for (int i = 0; i < st->nstages && i < PICA_MAX_STAGES; i++) {
			const PicaTevStage* s = &st->stage[i];
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
		passthrough_from(st->nstages);
		C3D_TexEnvBufUpdate(C3D_RGB, st->buf_rgb);
		C3D_TexEnvBufUpdate(C3D_Alpha, st->buf_alpha);
		C3D_TexEnvBufColor(st->buf_color);
	}
	C3D_AlphaTest(st->alpha_test, (GPU_TESTFUNC)st->alpha_func, st->alpha_ref);
	C3D_CullFace((GPU_CULLMODE)st->cull);
	if (s_gpuTest == 1 || s_drawCut) {
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
	C3D_DepthTest(st->depth_test, (GPU_TESTFUNC)st->depth_func,
	              (GPU_WRITEMASK)((st->write_mask & GPU_WRITE_COLOR) | (st->depth_write ? GPU_WRITE_DEPTH : 0)));
	if (st->blend) {
		C3D_AlphaBlend((GPU_BLENDEQUATION)st->blend_eq, (GPU_BLENDEQUATION)st->blend_aeq, (GPU_BLENDFACTOR)st->blend_src,
		               (GPU_BLENDFACTOR)st->blend_dst, (GPU_BLENDFACTOR)st->blend_asrc, (GPU_BLENDFACTOR)st->blend_adst);
		C3D_BlendingColor(st->blend_color);
	} else
		C3D_AlphaBlend(GPU_BLEND_ADD, GPU_BLEND_ADD, GPU_ONE, GPU_ZERO, GPU_ONE, GPU_ZERO);
	s_last = *st;
	s_last_valid = 1;
}

void pica_draw_triangles(const PicaVtx* verts, const uint16_t* idx, uint32_t nidx, const PicaState* st)
{
	if (!s_in_frame || nidx == 0) return;
	apply_state(st);
	cpu_mode();
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, verts, sizeof(PicaVtx), 5, 0x43210);
	s_frameDraws++;
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
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, v, sizeof(PicaVtx), 5, 0x43210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, idx);
	s_last_valid = 0;
	gpu_check("EFB rectangle clear", NULL, 6, NULL);
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

static void* copy_arena_alloc(uint32_t bytes, PicaTex* t)
{
	uint32_t n = 1;
	while (n * COPY_UNIT < bytes) n <<= 1;
	if (!s_copyArena || n > COPY_UNITS) return NULL;
	for (uint32_t at = 0; at + n <= COPY_UNITS; at += n) {
		uint32_t k = 0;
		while (k < n && !s_copyUnitUsed[at + k]) k++;
		if (k < n) continue;
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

void pica_copy_tex_layout(float x, float y, float w, float h, int half, int exact, uint16_t* tw, uint16_t* th,
                          float win[4])
{
	const float kx = (float)EFB_TEX_W / PICA_EFB_WIDTH, ky = (float)EFB_TEX_H / PICA_EFB_HEIGHT;
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
		fprintf(stderr, "[pica] no VRAM for a %lux%lu EFB-copy texture (%lu KB free)\n", (unsigned long)tw,
		        (unsigned long)th, (unsigned long)(vramSpaceFree() >> 10));
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
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, v, sizeof(PicaVtx), 5, 0x43210);
	C3D_DrawElements(GPU_TRIANGLES, 6, C3D_UNSIGNED_SHORT, idx);
	s_last_valid = 0;
}

void pica_efb_copy_to_ch(PicaTex* dst, int alpha_one, int chan, const float win[4])
{
	if (!dst || !dst->rt || !pica_init()) return;
	const float x = win[0], y = win[1], w = win[2], h = win[3];
	dst->wx = x;
	dst->wy = y;
	dst->ww = w;
	dst->wh = h;
	if (!s_in_frame) frame_begin();
	/* A draw pass, not a texture copy: it stays in the frame's command list (no CPU wait) and
	 * can fix alpha, as the GameCube's copy does for EFB formats without alpha. */
	draw_on(dst->rt);
	draw_efb_quad_ch(alpha_one, &s_efbTex, chan, x, y, w, h);
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
static Readback s_rbCur[MAX_READBACK], s_rbRun[MAX_READBACK];
static int s_nrbCur, s_nrbRun;
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
	/* Pixels cut to 8x8 (GPU test): the copies hold nothing real, so they must not reach the
	 * game's memory (the goop map would lose its goop). */
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
		GX_TextureCopy((u32*)r->tex->tex.data, 0, (u32*)r->buf, 0, size, 8);
	}
	memcpy(s_rbRun, s_rbCur, sizeof(Readback) * (size_t)s_nrbCur);
	s_nrbRun = s_nrbCur;
	s_nrbCur = 0;
}

/* A one-byte GX texture (8x4 tiles) as a PGM image, for looking at what the readback writes.
 * Off unless asked for (pica_set_readback_dumps): each file cost about a second of SD writes. */
static int s_rbDumps;
void pica_set_readback_dumps(int on) { s_rbDumps = on; }
static void rb_dump(const char* tag, const Readback* r)
{
	if (!s_rbDumps) return;
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

static void readback_finish(void)
{
	for (int i = 0; i < s_nrbRun; i++) {
		Readback* r = &s_rbRun[i];
		if (!r->buf) continue;
		end_mark(PICA_END_RB_CONVERT);
		GSPGPU_InvalidateDataCache(r->buf, (uint32_t)r->th * r->tw * 4);
		end_mark(PICA_END_RB_INVAL);
		const uint32_t bytes = (uint32_t)((r->dh + 3) / 4) * r->tilesPerRow * 32;
		uint32_t before = 0, after = 0, maxv = 0;
		const int logIt = s_rbDone % 150 == 0;
		if (logIt)
			for (uint32_t k = 0; k < bytes; k++) before += r->dest[k] > 100;
		if (s_rbDone == 0) rb_dump("cpu", r);
		readback_rows(r);
		if (s_rbDone == 0 || s_rbDone == 1 || s_rbDone == 30 || s_rbDone == 300) {
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
	s_nrbRun = 0;
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
static LightEvent s_evt[2];

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
void pica_event_wait(int i) { LightEvent_Wait(&s_evt[i]); }
void pica_event_signal(int i) { LightEvent_Signal(&s_evt[i]); }

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
	apply_state(&st);
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
	if ((l = shaderInstanceGetUniformLocation(vs, "LD")) >= 0) C3D_FVUnifSet(GPU_VERTEX_SHADER, l, 0, 0, 0, 0);
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, vb, 56, 7, 0x6543210);
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
			pica_draw_gx(vb, 4, &L, ib, 6, &u, 1, (const float(*)[4])rows, (const float(*)[4])rows, &st);
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
