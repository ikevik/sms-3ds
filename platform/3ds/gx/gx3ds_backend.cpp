// 3DS back end for the sms_gx front half (gx_api.cpp, gx_fifo.cpp, gx_vert.cpp, gx_trace.cpp).
//
// The front half turns every GX/GD call, write-gather-pipe byte and display list into register
// writes (g, the BP/CP/XF file) and decodes each primitive's vertices into a packed batch. It
// reaches the renderer only through the functions defined here (see gx_internal.h). This file
// is written for the PICA200:
//   * drawBatch() transforms each batch on the CPU as sms-pc-port's vertex shader does
//     (position matrix from XF memory, GX projection, GX viewport) into EFB pixels and hands
//     the triangles to platform/3ds/pica (citro3d);
//   * the rasterised colour is the channel's material or vertex colour, shaded by the normal
//     when lighting is on; depth test and blending follow the BP registers;
//   * GX textures are decoded and cached, and the TEV stages are translated into PICA texture
//     combiner stages (translateTev);
//   * display copies present the frame.
// Interface is sms-pc-port's (MIT), the implementation is independent.
#include "gx_internal.h"
#include "../pica/pica_draw.h"
#include "gx3ds_tevsym.h"
#include <malloc.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace gx {

// ---- globals the front half expects from its renderer -------------------------------------
uint32_t g_displayFrames = 0;
double g_presentWaited = 0, g_presentDrain = 0, g_flushSeconds = 0;
bool g_gxStats = false;

// Counters, readable from the 3DS side (platform/3ds/gx/gx3ds_backend.h).
struct Stats {
	uint32_t frames, batches, vertices, triangles, lines, points, copies;
};
static Stats s_stats;
static Stats s_frame;  // the frame in progress

// ---- the batch --------------------------------------------------------------------------
// The game thread (the GX front half) builds each batch here; flushBatch hands it to the renderer
// as a draw record (render thread, below), which draws it from its own copy (r_verts, r_idx).
static bool s_ready;
// Small mipmapped textures uploaded with their mip levels (the texture cache; GX3DS_SetMips,
// settings.txt mipmaps).
static bool s_mipsOn = true;
extern "C" void GX3DS_SetMips(int on) { s_mipsOn = on != 0; }
static bool s_etc1On;  // settings.txt etc1_textures (textureForMap)
extern "C" void GX3DS_SetEtc1(int on) { s_etc1On = on != 0; }
// The ETC1 worker's queues (etcWorker: s_etcIn, then s_etcOut) and whether it runs: 1, or -1 when
// none could start (the encodes then run in place, as before).
static std::atomic<uint32_t> s_etcInHead{0}, s_etcInTail{0}, s_etcOutHead{0}, s_etcOutTail{0};
static int s_etcWorker;
static bool s_tevSymOn = true;  // settings.txt tev_compiler (tevSymImprove)
extern "C" void GX3DS_SetTevCompiler(int on) { s_tevSymOn = on != 0; }
static uint64_t s_arenaDrawVerts, s_arenaUploadBytes; // resident vertices drawn, stored (display-list arena)
// [perf]: the TEV cache's hits, misses and clears, and drawTriangles' TEV time by part: the
// registers read (readTev), the state, key and lookup, the translation (misses), the rest.
static uint64_t s_tevHits, s_tevMisses, s_tevClears, s_tevPartTicks[4];
static uint32_t s_tevSymChecks, s_tevSymCompiles, s_tevSymUsed;  // tevSymImprove: checks, compiles, compiled setups drawn
// Pixel metrics (pixMetricCount): counting since a clear (GX thread), its pixels passed and
// triangles; the value a draw-sync callback reads (game thread, streamTokenPoll).
static bool s_pmActive, s_pmInToken;
static uint32_t s_pmCount, s_pmTris, s_pmTokenValue;
static uint32_t s_tevEntries;
// Vertex reuse in the resident runs drawn (runReuse): what the GPU's post-transform vertex cache
// could save with duplicate vertices merged. Indices, vertices stored, then the run stats.
enum { RS_TRIS, RS_STRIP, RS_DISTINCT, RS_C8, RS_C16, RS_C32, RS_D8, RS_D16, RS_D32, RS_T16, RS_S16, kRunStat };
static uint64_t s_reuse[2 + kRunStat], s_reuse0[2 + kRunStat];
// Render thread: changes to anything a draw's TEV setup, textures and PICA state are worked out from
// (s_rsGen: BP, XF and konst registers, TLUTs, texture images, the copy destination, EFB copies;
// not XF memory, the matrices), the draws that came with none since the last draw, and the draws
// that took the last draw's setup (drawTriangles), for the [perf] lines.
static uint32_t s_rsGen, s_rsGenLastDraw, s_drawsSameState, s_drawsCounted, s_drawsFast;
// The draw state's generation (pica_draw_same, pica_draw.h): changed whenever drawTriangles works
// the PICA state out afresh for a change (not for a draw_state_check rebuild of an unchanged one,
// which the check compares) or a CPU-path draw changes its projection flag; so a draw with the same
// number gives pica_draw the same state and layout as the last, and its compares are skipped.
static uint32_t s_stGen = 1;
// settings.txt draw_state_check (GX3DS_SetDrawStateCheck): the draws that could take the last draw's
// setup work it out, compared with it (checked, differing).
static bool s_fpCheck;
static uint32_t s_fpChecked, s_fpDiffer;
// Resident runs store each distinct vertex once (arenaUpload; GX3DS_SetMergeDuplicates), and those
// drawn where order makes no difference have their triangles reordered for the GPU's vertex cache
// (GX3DS_SetReorder). Reordered runs drawn, and those of them drawn where order matters, for [perf].
static bool s_mergeDup = true, s_reorder = true;
static uint64_t s_reordDraws, s_reordUnsafe;
// Runs stored, by indices, by what decided their order (orderWhy): reordered, then kept for
// blending or a logic op, no depth test, no depth update, a compare that does not order.
static uint64_t s_reordWhy[5];
// Vertices drawn per frame by vertex format (VtxFmtBits): which layouts the GPU fetches.
static uint64_t s_fmtVerts[1u << 13], s_fmtVerts0[1u << 13];
static uint64_t s_perfArenaDraw0, s_perfArenaUp0;
// Vertices copied into batches instead: display-list runs not resident (appendDecoded) and decoded
// draws (uncached lists, direct GXBegin/End); of the runs, those that found no room or were too
// big for 16-bit indices (arenaUpload).
static uint64_t s_copyRunVerts, s_copyDirectVerts, s_aNoRoomVerts, s_aTooBigVerts, s_perfCopy0[4];
// The batch's packed vertices (layout: vtxFmtLayout(s_bfmt)). The front half decodes them straight
// into the render queue, where the batch's draw record will be (render thread, below): flushBatch
// writes the record's header in front of them and its indices after them, and nothing is copied.
// A batch that outgrows the queue's room moves to the heap (and is copied into its record).
static uint8_t* rcOpenBatch(size_t* cap);   // the queue room for a batch's vertices
static void rcBatchRoom(size_t bytes);      // waits until the batch may use that many bytes of it
static uint32_t s_batchToHeap;
class BatchBuf {
public:
	uint8_t* data() { return p_; }
	uint8_t* end() { return p_ + n_; }
	size_t size() const { return n_; }
	bool inRing() const { return ring_; }
	void resize(size_t sz) {
		ensure(sz);
		n_ = sz;
	}
	void insert(uint8_t*, const uint8_t* b, const uint8_t* e) {
		const size_t k = size_t(e - b), at = n_;
		ensure(at + k);
		memcpy(p_ + at, b, k);
		n_ = at + k;
	}
	void moveToHeap() {
		if (ring_) toHeap(n_);
	}
	void clear() {
		n_ = 0;
		open_ = ring_ = false;
	}

private:
	void open() {
		p_ = rcOpenBatch(&cap_);
		ring_ = p_ != nullptr;
		room_ = 0;
		if (!ring_) {
			p_ = heap_.data();
			cap_ = heap_.size();
		}
		open_ = true;
	}
	void toHeap(size_t need) {
		if (ring_) s_batchToHeap++;
		if (heap_.size() < need) heap_.resize(need + need / 2 + 4096);
		if (ring_ && n_) memcpy(heap_.data(), p_, n_);
		ring_ = false;
		p_ = heap_.data();
		cap_ = heap_.size();
	}
	void ensure(size_t sz) {
		if (!open_) open();
		if (sz > cap_) {
			toHeap(sz);
		} else if (ring_ && sz > room_) {
			room_ = sz + (16u << 10);
			if (room_ > cap_) room_ = cap_;
			rcBatchRoom(room_);
		}
	}
	uint8_t* p_ = nullptr;
	size_t n_ = 0, cap_ = 0, room_ = 0;
	bool ring_ = false, open_ = false;
	std::vector<uint8_t> heap_;
};
static BatchBuf s_vdata;
static std::vector<uint32_t> s_idx;    // indices into s_vdata
static uint32_t s_bcount;              // vertices in the batch
static uint32_t s_bstride;
static uint32_t s_bfmt;
static PrimClass s_bclass = PRIM_TRIS;

// A resident batch's indices: its runs' 16-bit indices where the arena stored them (GPU-readable,
// relative to the page), runs stored one after another making one span. Nothing is copied per frame.
struct IdxSpan {
	const uint16_t* idx;
	uint32_t n;
};
static std::vector<IdxSpan> s_bSpans;
static uint32_t s_bSpanIdx;        // indices in s_bSpans
static inline bool batchHasVertices() { return !s_idx.empty() || !s_bSpans.empty(); }
static uint8_t* s_bArena;          // the resident page the batch draws from (display-list arena below), or null
static uint32_t s_bmin, s_bmax;    // its vertices there: [s_bmin, s_bmax)
// A resident batch whose vertices carry matrix indices: the matrices its runs use (arenaRunMtx), as
// its page window may also hold other runs' vertices, which it does not draw.
static uint8_t s_bPnMax;           // highest position matrix index
static uint8_t s_bTexMtx[8];       // each coordinate's texture matrix index, 0xFF where it varies

// The render side: its copy of the register file (bp, kreg, xfReg, xfMem, texImage, copyDest and
// tlutMem are kept; the rest is not used), and the batch being drawn.
static State rs;
static const uint8_t* r_verts;     // r_count vertices of r_stride bytes, layout r_layout
static const uint16_t* r_idx;      // r_nidx indices (null while r_spans holds them)
static const IdxSpan* r_spans;     // a resident batch: r_nspans spans of indices relative to its page,
static uint32_t r_nspans, r_bmin;  // r_verts the page's first vertex; its vertices from r_bmin are drawn
static uint64_t s_spanBatches, s_spansDrawn, s_spanCopies;  // render side: resident batches, their spans, those copied
// Off (circle pad up at launch): every resident batch's indices are copied into the ring rebased to
// its first vertex, as before they were stored (the GPU took ~8 ms more a frame drawing them in
// place on Delfino Plaza; to compare).
static bool s_idxInPlace = true;
extern "C" void GX3DS_SetResidentIndicesInPlace(int on) { s_idxInPlace = on != 0; }
static uint32_t r_count, r_nidx, r_stride, r_fmt;
static PrimClass r_class = PRIM_TRIS;
static VtxFmtLayout r_layout;
static const uint8_t* r_arena;     // non-null: r_verts is in a resident page (drawn where it is)
static bool r_amtx;                // a resident batch with matrix indices: its runs' matrices follow
static uint8_t r_pnMax, r_texMtx[8];
static uint32_t s_rFrames;         // display copies the render side has done

// ---- profiler -------------------------------------------------------------------------------
// Per-frame time of each renderer phase, in ARM11 system ticks (268,111,856 Hz), averaged over
// kPerfFrames frames for the status screen and logged as [perf] lines. "other" is the rest of the
// frame: game logic, the GX front half (command parsing, vertex decoding) and waiting.
extern "C" void SMS3DS_TimelineEvent(int who, int what);  // prof_3ds.cpp: the frame timeline (autorun.txt timeline)
extern "C" unsigned SMS3DS_RankStats[2];  // prof_3ds.cpp: decomp patch 3ds-0031's check (checked, differing)
extern "C" int GXPC_FastListCheck;  // gx_fifo.cpp: state lists' register words checked (with draw_state_check)
extern "C" uint32_t GXPC_FastListStats[2];  // ... lists checked, differing
extern "C" volatile int GXPC_GxIdle;  // gx_fifo.cpp: the game thread hands over partial chunks while set
// The GX thread's zone (gx_internal.h, GX_ZONES builds), sampled by a thread on core 2 every 100 us
// (prof_3ds.cpp, autorun.txt gx_zones): the samples by zone, for the [perf] lines.
extern "C" {
volatile uint8_t GXPC_Zone;
volatile uint32_t GX3DS_ZoneHist[gx::GZ_COUNT];
extern const int GX3DS_ZoneCount = gx::GZ_COUNT;
int GX3DS_ZoneSampling;  // prof_3ds.cpp: the sampler runs
}  // prof_3ds.cpp: decomp patch 3ds-0031's check (checked, differing)
extern "C" uint64_t svcGetSystemTick(void);
extern "C" uint64_t etc1_encode_tex(void* dst, uint32_t w, uint32_t h, const uint8_t* rgba, int alpha);  // pica/etc1_enc.c
// The "other" breakdown: GX command parsing and display-list calls (gx_fifo.cpp, patch 0006; it
// includes our draw work done inside them), the emulated OS's idle time (no game thread runnable:
// waiting for the retrace), and beside them the main thread's bottom-screen status redraw, which
// shares the game thread's core (source_game/main.cpp adds to GX3DS_MainBusyTicks).
extern "C" uint64_t GXPC_CmdTicks;
extern "C" double port_idle_seconds(void);
extern "C" volatile uint64_t GX3DS_MainBusyTicks;
volatile uint64_t GX3DS_MainBusyTicks;
static uint64_t s_perfCmd0, s_perfMain0;
static double s_perfIdle0;
// PERF_BATCH: flushBatch's own work (the draw-time setup the phases above do not cover); PERF_APPEND:
// copying a cached display list's vertices and indices into the batch (appendDecoded).
enum PerfPhase {
	PERF_TEX, PERF_TEV, PERF_XFORM, PERF_SUBMIT, PERF_COPY, PERF_PRESENT, PERF_DECODE, PERF_BATCH, PERF_APPEND,
	PERF_QSTATE, PERF_QVERT, PERF_QIDX,  // flushBatch's queueing (render thread): registers, vertices, indices
	PERF_COUNT
};
// Vertices the game thread has put into batches (profile runs count them per object).
extern "C" uint32_t GX3DS_GameVerts;
uint32_t GX3DS_GameVerts;
// Profile runs (prof_3ds.cpp): while SMS3DS_ProfCutActive, the vertices of the object
// SMS3DS_ProfCutNow names are drawn with their pixels cut, in batches of their own.
extern "C" volatile int SMS3DS_ProfCutActive;
extern "C" int SMS3DS_ProfCutNow(void);
extern "C" void SMS3DS_ProfGxVerts(uint32_t n);  // profile runs: vertices credited to the draws' object
extern "C" int SMS3DS_ProfOn;
extern "C" volatile int SMS3DS_SampWaiting;  // profile runs' sampler: the game thread waiting here
static uint8_t s_bcut;  // the open batch's vertices have their pixels cut
// Running total of every phase, for the front half's display-list timers (patch 0007) to leave out.
extern "C" uint64_t GX3DS_OursTicks;
uint64_t GX3DS_OursTicks;
// The GX thread's waits, for the stream's next chunk and for the render thread: what the profile
// runs' GX-time credit leaves out (sms-pc-port patch 0034, native J3D stage 0).
extern "C" uint64_t GX3DS_GxWaitTicks;
uint64_t GX3DS_GxWaitTicks;
extern "C" uint64_t GXPC_DlTicks[4];
extern "C" uint64_t GXPC_DlFineTicks[8];  // patch 0032: list calls split finer, in profile runs
extern "C" uint32_t GXPC_DlFineCount[8];
extern "C" uint32_t GXPC_DlCalls, GXPC_DlHits, GXPC_DlVerts, GXPC_DlRecords, GXPC_DlNew, GXPC_DlUncached;
extern "C" uint32_t GXPC_J3DStats[16];  // native J3D (patch 0035, gx_fifo.cpp)
extern "C" uint32_t GXPC_J3DPkgStats[4];  // its compiled shapes (patch 0037): drawn from, made, not compilable, kept
extern "C" uint64_t GXPC_J3DTicks[8], GXPC_J3DHashed;
static uint32_t s_j3dSigs, s_j3dSigBad;  // verify mode (renderJ3DSig): native draws checked, differences
extern "C" uint64_t GXPC_DlHashed;
// Hashing by where the bytes are (patch 0021, and the texture cache below): outside the write stamps'
// window (64 MiB from MEM1's base) the caches cannot trust stamps and hash on every check.
extern "C" uint64_t GXPC_DlHashedOutside;
extern "C" uintptr_t GXPC_DlOutLo, GXPC_DlOutHi;
static uint64_t s_texHashed, s_texHashedOut, s_tlutHashed;
static uintptr_t s_texOutLo = ~uintptr_t(0), s_texOutHi;
extern "C" uint32_t GXPC_CullRuns, GXPC_CullVerts;  // patch 0014
extern "C" int GXPC_CullOff;
extern "C" float GXPC_CullFar;  // patch 0051: the profiler's far-geometry ceiling
extern "C" uint32_t GXPC_CullFarRuns, GXPC_CullFarVerts;
static uint32_t s_perfCullRuns0, s_perfCullVerts0;
extern "C" size_t GXPC_DlCacheBytes(void);
extern "C" uint32_t GXPC_J3DPkgBytes(void);  // sms-port patch 0053: the compiled shapes' memory
extern "C" __attribute__((weak)) int SMS3DS_MemReserveSpent(void);  // main.cpp: the heap's reserve given back
extern "C" __attribute__((weak)) void SMS3DS_MemReserveRestore(void);
extern "C" uint32_t GXPC_DlEntryCount(void);
extern "C" int GXPC_DlCacheClearNow(void);  // gx_fifo.cpp (sms-port patch 0052)
extern "C" uint32_t GXPC_DlFail[7];  // patch 0009: why cached lists could not be reused
static uint32_t s_perfDlFail0[7];
extern "C" uint32_t GXPC_CmdCount[7];  // patch 0010: commands parsed, by kind
static uint32_t s_perfCmdCount0[7];
extern "C" void GXPC_SetStream(uint8_t* (*swap)(uint8_t*, uint32_t, uint32_t*), void (*token)(uint16_t));  // patch 0017
extern "C" uint32_t GXPC_StreamParse(const uint8_t* p, uint32_t n, uint32_t* need);
extern "C" void GXPC_StreamFlush(void);
extern "C" void port_irq_kick(void);                 // the emulated OS (port_os.h): an interrupt source has work
extern "C" void port_irq_add_source(void (*fn)(void));
extern void (*drawSyncCallback)(uint16_t);  // gx_fifo.cpp
extern "C" uint8_t* port_mem1_base;  // port_runtime.cpp: the emulated MEM1
extern "C" uint32_t port_mem1_size;
extern "C" uint32_t GXPC_DrawSyncCalls;          // patch 0011
extern "C" uint64_t GXPC_DrawSyncTicks;
extern "C" unsigned long long port_os_switches(void);
static uint32_t s_perfDsCalls0;
static uint64_t s_perfDsTicks0;
static unsigned long long s_perfSwitches0;
static uint64_t s_perfWait0;
extern "C" int port_os_live_hosts(void);  // patch 0012
extern "C" int port_os_stack_report(char* out, int n);  // patch 0029
extern "C" uint32_t GXPC_StateBlockCalls[3];  // patch 0031: written as blocks, checked against the calls, calls
extern "C" unsigned long __ctru_heap_size; // libctru: the process heap's size
static uint64_t s_perfDl0[4], s_perfDlHashed0;
static uint32_t s_perfDlCalls0, s_perfDlHits0, s_perfDlVerts0, s_perfDlRecords0, s_perfDlNew0, s_perfDlUncached0;
static uint64_t s_perfAcc[PERF_COUNT], s_perfFrameStart, s_perfWindowStart;
static uint32_t s_perfFrames, s_perfBatches, s_perfVerts;
static float s_perfAvgMs[PERF_COUNT + 2];  // phases, then frame and other
static float s_perfAvgBatches, s_perfAvgVerts;
static const uint32_t kPerfFrames = 60;
static uint64_t s_decodeT0;  // primitiveBegin -> primitiveEnd: the GX front half decoding vertices
static const double kTicksPerMs = 268111.856;
// The game thread's phases go to s_gAcc and reach s_perfAcc with the display copy's record; with
// the render thread, its phases are not the game thread's (GX3DS_OursTicks).
static thread_local bool t_render;
static uint64_t s_gAcc[PERF_COUNT];
static inline void perfAdd(PerfPhase ph, uint64_t ticks) {
	if (t_render) {
		s_perfAcc[ph] += ticks;
		return;
	}
	s_gAcc[ph] += ticks;
	GX3DS_OursTicks += ticks;
}
// The [perf] breakdown by phase (tex, tev, xform, submit, decode, batch, append, the queueing)
// reads the system tick several times a draw, thousands of times a frame: 261 ns each on a New
// 3DS, 1.3 us in Azahar. Off unless settings.txt says perf_timers = on (GX3DS_SetPerfTimers); the
// frame, the threads' busy and idle time and the waits are always timed.
static bool s_perfTimers;
static inline uint64_t perfTick() { return s_perfTimers ? svcGetSystemTick() : 0; }
struct PerfScope {
	PerfPhase ph;
	uint64_t t0;
	explicit PerfScope(PerfPhase p) : ph(p), t0(perfTick()) {}
	~PerfScope() {
		if (s_perfTimers) perfAdd(ph, svcGetSystemTick() - t0);
	}
};
// A phase without the phases nested in it.
struct PerfScopeExcl {
	PerfPhase ph;
	uint64_t t0, o0;
	explicit PerfScopeExcl(PerfPhase p) : ph(p), t0(perfTick()), o0(GX3DS_OursTicks) {}
	~PerfScopeExcl() {
		if (s_perfTimers) perfAdd(ph, (svcGetSystemTick() - t0) - (GX3DS_OursTicks - o0));
	}
};

// Why batches fell back to the CPU path: batches and vertices per reason over the profiler window,
// logged with the [perf] lines.
enum GpuFallback {
	FB_TRACE, FB_NOCLIP, FB_SIZE, FB_PNMTX, FB_TWOCOLOURS, FB_TEXGEN_TYPE, FB_TEXGEN_PROJ, FB_TEXGEN_SRC,
	FB_TEXMTX_VARIES, FB_POST_NORMALISE, FB_POST_PROJ, FB_LAYOUT, FB_LIGHTS, FB_COUNT
};
static const char* const kFallbackName[FB_COUNT] = {
	"trace", "clip-off", "size", "pos-matrix>27", "two-colours", "texgen-type", "texgen-proj", "texgen-src",
	"texmtx-varies", "post-normalise", "post-proj", "layout", "lights"};
static uint32_t s_fbBatches[FB_COUNT], s_fbVerts[FB_COUNT];

static bool gpuFail(GpuFallback why, uint32_t n) {
	s_fbBatches[why]++;
	s_fbVerts[why] += n;
	return false;
}

static size_t arenaPageCount();
static size_t arenaBytes();
static uint32_t arenaResets();
static uint32_t arenaFullCount();
static void arenaFrameEnd(bool force = false);
static int debugColorMode();  // SELECT's mode: 0 material, 1 debug colours, 2 opaque
static bool renderThreaded();    // the render thread runs (else records are drawn as they are queued)
static uint64_t gameWaitTicks(); // the profiler window's game-thread wait for the render thread
static void renderThreadPerf(double frames);

static uint64_t s_perfEnd0[PICA_END_COUNT];
// Streamed GX (the GX thread, below): its numbers for the profiler window.
static bool streamOn();
static void streamPerf(uint64_t& gameWait, double perMs, bool print);

static void texInventory(double frames);  // [perf]: textures drawn from outside VRAM (below)
static void perfHostMemory();  // [perf]: the host side's caches on the heap (below)
static void perfFrameEnd() {
	const uint64_t now = svcGetSystemTick();
	if (!s_perfWindowStart) {
		s_perfWindowStart = now;
		pica_end_ticks(s_perfEnd0);
		s_perfCmd0 = GXPC_CmdTicks;
		s_perfMain0 = GX3DS_MainBusyTicks;
		s_perfIdle0 = port_idle_seconds();
		memset(s_perfAcc, 0, sizeof s_perfAcc);
		s_perfFrames = s_perfBatches = s_perfVerts = 0;
		return;
	}
	if (++s_perfFrames < kPerfFrames) return;
	const double frames = s_perfFrames, total = double(now - s_perfWindowStart);
	// ours: the renderer's time on the game thread (with the render thread, the front-half phases
	// and waiting for the render thread; with the GX thread as well, only waiting for that).
	double front = 0;
	for (int i = 0; i < PERF_COUNT; i++) {
		s_perfAvgMs[i] = float(double(s_perfAcc[i]) / frames / kTicksPerMs);
		if (!renderThreaded() || i == PERF_DECODE || i == PERF_BATCH || i == PERF_APPEND || i >= PERF_QSTATE) front += s_perfAcc[i];
	}
	front += double(gameWaitTicks());
	uint64_t streamGameWait = 0;
	streamPerf(streamGameWait, 0, false);
	const double ours = streamOn() ? double(streamGameWait) : front;
	s_perfAvgMs[PERF_COUNT] = float(total / frames / kTicksPerMs);
	s_perfAvgMs[PERF_COUNT + 1] = float((total - ours) / frames / kTicksPerMs);
	s_perfAvgBatches = float(s_perfBatches / frames);
	s_perfAvgVerts = float(s_perfVerts / frames);
	const double waitMs = double(pica_wait_ticks() - s_perfWait0) / frames / kTicksPerMs;
	s_perfWait0 = pica_wait_ticks();
	static double s_gpuMs0;
	static uint32_t s_splits0;
	const double splits = (pica_splits() - s_splits0) / frames;
	s_splits0 = pica_splits();
	const double gpuMs = (pica_gpu_ms_sum() - s_gpuMs0) / frames;
	s_gpuMs0 = pica_gpu_ms_sum();
	fprintf(stderr, "[perf]   gpu wait (%s): %.1f ms/frame, counted in the renderer's own time | GPU drawing "
	        "%.1f ms/frame after the last part | %.1f parts sent mid-frame, %lu lost completions so far | test: %s, "
	        "colours %s\n", pica_overlap() ? "frame end, frames overlapped" : "C3D_FrameBegin", waitMs, gpuMs, splits, (unsigned long)pica_lost_completions(), pica_gpu_test() == 1 ? "pixels cut to 8x8" : pica_gpu_test() == 2 ? "positions collapsed" : pica_gpu_test() == 3 ? "no draws" : "off",
	        debugColorMode() == 1 ? "debug (no textures)" : debugColorMode() == 2 ? "opaque (no blending)" : "material");
	fprintf(stderr,
	        "[perf] frame %.1f ms (%.1f fps) | tex %.2f tev %.2f xform %.2f submit %.2f copy %.2f present %.2f | "
	        "decode %.1f batch %.2f append %.2f other %.1f | %.0f batches %.0f verts/frame%s\n",
	        s_perfAvgMs[PERF_COUNT], 1000.0 / s_perfAvgMs[PERF_COUNT], s_perfAvgMs[PERF_TEX], s_perfAvgMs[PERF_TEV],
	        s_perfAvgMs[PERF_XFORM], s_perfAvgMs[PERF_SUBMIT], s_perfAvgMs[PERF_COPY], s_perfAvgMs[PERF_PRESENT],
	        s_perfAvgMs[PERF_DECODE], s_perfAvgMs[PERF_BATCH], s_perfAvgMs[PERF_APPEND], s_perfAvgMs[PERF_COUNT + 1],
	        s_perfAvgBatches, s_perfAvgVerts, s_perfTimers ? "" : " | phases not timed (perf_timers off)");
	{
		// present split up: pica_present's frame end
		uint64_t end[PICA_END_COUNT];
		double ms[PICA_END_COUNT];
		pica_end_ticks(end);
		for (int i = 0; i < PICA_END_COUNT; i++) {
			ms[i] = double(end[i] - s_perfEnd0[i]) / frames / kTicksPerMs;
			s_perfEnd0[i] = end[i];
		}
		fprintf(stderr,
		        "[perf]   present: pass %.2f, ring flush %.2f, parts wait %.2f, submit %.2f, heap flush %.2f, gpu wait "
		        "%.2f, readback invalidate %.2f convert %.2f fetch %.2f, run %.2f ms/frame\n",
		        ms[PICA_END_PASS], ms[PICA_END_RINGS], ms[PICA_END_PARTS], ms[PICA_END_SUBMIT], ms[PICA_END_HEAP_FLUSH],
		        ms[PICA_END_WAIT], ms[PICA_END_RB_INVAL], ms[PICA_END_RB_CONVERT], ms[PICA_END_RB_FETCH],
		        ms[PICA_END_RUN]);
	}
	{
		// The TEV cache (drawTriangles): hits, misses and clears, and its time by part.
		static uint64_t h0, m0, c0, p0[4];
		const double hits = double(s_tevHits - h0) / frames, misses = double(s_tevMisses - m0) / frames;
		double part[4];
		for (int i = 0; i < 4; i++) {
			part[i] = double(s_tevPartTicks[i] - p0[i]) / frames / kTicksPerMs;
			p0[i] = s_tevPartTicks[i];
		}
		static uint32_t sd0, dc0, df0;
		fprintf(stderr, "[perf]   TEV cache per frame: %.0f hits, %.0f misses, %.1f clears (%u entries) | its time: read %.2f, "
		        "state+key+lookup %.2f, translate (misses) %.2f, rest %.2f ms | draws with nothing changed since the last "
		        "draw: %.0f of %.0f, %.0f of them with its TEV setup and state as they were\n", hits, misses,
		        double(s_tevClears - c0) / frames, unsigned(s_tevEntries), part[0], part[1], part[2], part[3],
		        double(s_drawsSameState - sd0) / frames, double(s_drawsCounted - dc0) / frames,
		        double(s_drawsFast - df0) / frames);
		if (s_fpCheck) {
			fprintf(stderr, "[perf]   draw state check: %lu draws so far worked out where the last draw's setup would have "
			        "been taken, %lu differing\n", (unsigned long)s_fpChecked, (unsigned long)s_fpDiffer);
			fprintf(stderr, "[perf]   scene sound ranking check: %u distances so far reused and worked out again, %u "
			        "differing\n", SMS3DS_RankStats[0], SMS3DS_RankStats[1]);
			fprintf(stderr, "[perf]   state lists' register words check: %u lists so far replayed both ways, %u differing\n",
			        (unsigned)GXPC_FastListStats[0], (unsigned)GXPC_FastListStats[1]);
		}
		static uint32_t sc0, sp0, su0;
		if (s_tevSymChecks != sc0 || s_tevSymCompiles != sp0)
			fprintf(stderr, "[perf]   TEV compiler: %u setups checked, %u compiled, %u compiled ones drawn (per %.0f frames)\n",
			        s_tevSymChecks - sc0, s_tevSymCompiles - sp0, s_tevSymUsed - su0, frames);
		sc0 = s_tevSymChecks;
		sp0 = s_tevSymCompiles;
		su0 = s_tevSymUsed;
		sd0 = s_drawsSameState;
		dc0 = s_drawsCounted;
		df0 = s_drawsFast;
		h0 = s_tevHits;
		m0 = s_tevMisses;
		c0 = s_tevClears;
	}
	char fb[256];
	int at = 0;
	for (int i = 0; i < FB_COUNT && at < int(sizeof fb) - 48; i++)
		if (s_fbBatches[i])
			at += snprintf(fb + at, sizeof fb - at, " %s %.1f/%.0fv", kFallbackName[i], s_fbBatches[i] / frames,
			               s_fbVerts[i] / frames);
	if (at) fprintf(stderr, "[perf]   cpu fallbacks per frame (batches/verts):%s\n", fb);
	{
		// other = idle + GX commands outside our phases + status screen + the game itself
		const double perMs = 1.0 / frames / kTicksPerMs;
		const double cmd = double(GXPC_CmdTicks - s_perfCmd0), mainBusy = double(GX3DS_MainBusyTicks - s_perfMain0);
		const double idleMs = (port_idle_seconds() - s_perfIdle0) * 1000.0 / frames;
		const double cmdOnly = cmd > front ? (cmd - front) * perMs : 0.0;
		const double otherMs = s_perfAvgMs[PERF_COUNT + 1], mainMs = mainBusy * perMs;
		// Wall-clock times on a shared core: the status screen's redraw can preempt the game thread
		// inside any of the others, so it is reported beside them, not subtracted.
		if (streamOn()) {
			fprintf(stderr,
			        "[perf]   other %.1f = idle %.1f + game %.1f ms/frame (game thread; its GX calls only write "
			        "commands; status screen redraw, main thread: %.1f)\n",
			        otherMs, idleMs, otherMs - idleMs, mainMs);
			streamPerf(streamGameWait, perMs, true);
		} else {
			fprintf(stderr,
			        "[perf]   other %.1f = idle %.1f + gx commands %.1f + game %.1f ms/frame (gx commands incl. our draw "
			        "work: %.1f; status screen redraw, main thread: %.1f)\n",
			        otherMs, idleMs, cmdOnly, otherMs - idleMs - cmdOnly, cmd * perMs, mainMs);
		}
		// gx commands = display lists (validate a cached one, replay it, record one, run one uncached)
		// + everything else the parser runs (the game's direct GX commands)
		double dl[4], dlSum = 0;
		for (int i = 0; i < 4; i++) {
			dl[i] = double(GXPC_DlTicks[i] - s_perfDl0[i]) * perMs;
			dlSum += dl[i];
			s_perfDl0[i] = GXPC_DlTicks[i];
		}
		fprintf(stderr,
		        "[perf]   gx commands %.1f = display lists: validate %.1f + replay %.1f + record %.1f + uncached %.1f; "
		        "direct %.1f ms/frame\n",
		        cmdOnly, dl[0], dl[1], dl[2], dl[3], cmdOnly - dlSum);
		fprintf(stderr,
		        "[perf]   display lists per frame: %.0f calls, %.0f cached (%.0f verts), %.1f recorded (%.1f new), %.1f "
		        "uncached, %.0f KB hashed | cache %u entries, %u KB\n",
		        (GXPC_DlCalls - s_perfDlCalls0) / frames, (GXPC_DlHits - s_perfDlHits0) / frames,
		        (GXPC_DlVerts - s_perfDlVerts0) / frames, (GXPC_DlRecords - s_perfDlRecords0) / frames,
		        (GXPC_DlNew - s_perfDlNew0) / frames, (GXPC_DlUncached - s_perfDlUncached0) / frames,
		        double(GXPC_DlHashed - s_perfDlHashed0) / frames / 1024.0, unsigned(GXPC_DlEntryCount()),
		        unsigned(GXPC_DlCacheBytes() >> 10));
		{
			static uint64_t ft0[8];
			static uint32_t fc0[8];
			double ft[8], fc[8], any = 0;
			for (int i = 0; i < 8; i++) {
				ft[i] = double(GXPC_DlFineTicks[i] - ft0[i]) * perMs;
				fc[i] = double(GXPC_DlFineCount[i] - fc0[i]) / frames;
				ft0[i] = GXPC_DlFineTicks[i];
				fc0[i] = GXPC_DlFineCount[i];
				any += ft[i];
			}
			if (any > 0)
				fprintf(stderr,
				        "[perf]   display-list calls, finer (ms/frame): lookup %.2f | validate: compare %.2f (%.1f arrays a call), "
				        "write stamps %.2f, hash %.2f (%.1f hashed) | replay: state-only %.2f (%.0f lists, %.0f commands; %.0f lists "
				        "as register words), draw lists: off-screen test %.2f (%.0f lists, %.0f runs, %.0f off-screen), appends "
				        "%.2f\n",
				        ft[0], ft[1], fc[2] + fc[0] > 0 ? fc[5] / (fc[2] + fc[0]) : 0.0, ft[2], ft[3], fc[6], ft[4], fc[0], fc[1],
				        fc[7], ft[5], fc[2], fc[3], fc[4], ft[6]);
		}
		{
			// Native J3D (patch 0035): ops drawn natively and replayed as calls, by reason.
			static uint32_t j0[16];
			double j[16], any = 0;
			for (int i = 0; i < 16; i++) {
				j[i] = (GXPC_J3DStats[i] - j0[i]) / frames;
				j0[i] = GXPC_J3DStats[i];
				any += j[i];
			}
			if (any > 0)
				fprintf(stderr,
				        "[perf]   native J3D per frame: materials %.1f native (%.1f changing nothing), %.1f replayed | shapes %.1f "
				        "native, %.1f replayed (no entry %.1f, state %.1f, data %.1f, not resident %.1f, vertex format %.1f) | "
				        "matrix updates %.1f, runs culled %.1f, hand-overs %.1f | checks %.1f, differences %.1f | render side "
				        "verified %.1f, differences so far %lu\n",
				        j[0], j[14], j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8], j[13], j[12], j[9], j[10], j[11], j[15],
				        (unsigned long)s_j3dSigBad);
			static uint32_t p0[3];
			double pk[3];
			for (int i = 0; i < 3; i++) {
				pk[i] = (GXPC_J3DPkgStats[i] - p0[i]) / frames;
				p0[i] = GXPC_J3DPkgStats[i];
			}
			if (any > 0)
				fprintf(stderr,
				        "[perf]   native J3D compiled shapes per frame: %.1f shapes drawn from them, %.1f made, %.1f not "
				        "compilable; %lu kept\n",
				        pk[0], pk[1], pk[2], (unsigned long)GXPC_J3DPkgStats[3]);
			static uint64_t t0[8], h0;
			double t[8], tall = 0;
			for (int i = 0; i < 8; i++) {
				t[i] = double(GXPC_J3DTicks[i] - t0[i]) * perMs;
				t0[i] = GXPC_J3DTicks[i];
				tall += t[i];
			}
			const double kb = double(GXPC_J3DHashed - h0) / frames / 1024.0;
			h0 = GXPC_J3DHashed;
			if (tall > 0)  // profile runs
				fprintf(stderr,
				        "[perf]   native J3D's GX time (ms/frame, profile runs): materials: lookup %.2f, flush and record %.2f | "
				        "shapes: lookup %.2f, state compared %.2f, data checked %.2f (%.0f KB hashed), matrices %.2f, runs %.2f "
				        "| hand-overs %.2f | %.1f us a material, %.1f us a shape\n",
				        t[0], t[1], t[2], t[3], t[4], kb, t[5], t[6], t[7], j[0] > 0 ? (t[0] + t[1]) * 1000.0 / j[0] : 0.0,
				        j[2] > 0 ? (t[2] + t[3] + t[4] + t[5] + t[6]) * 1000.0 / j[2] : 0.0);
		}
		{
			static uint64_t do0, th0, to0, tl0;
			const uintptr_t base = uintptr_t(physToPtr(32)) - 32;
			fprintf(stderr,
			        "[perf]   hashing per frame: display lists %.0f KB outside the write-stamp window (%08lx-%08lx; the "
			        "window is %08lx-%08lx) | textures %.0f KB, %.0f KB of it outside (%08lx-%08lx), TLUTs %.0f KB\n",
			        double(GXPC_DlHashedOutside - do0) / frames / 1024.0,
			        (unsigned long)(GXPC_DlOutHi ? GXPC_DlOutLo : 0), (unsigned long)GXPC_DlOutHi, (unsigned long)base,
			        (unsigned long)(base + (64u << 20)), double(s_texHashed - th0) / frames / 1024.0,
			        double(s_texHashedOut - to0) / frames / 1024.0, (unsigned long)(s_texOutHi ? s_texOutLo : 0),
			        (unsigned long)s_texOutHi, double(s_tlutHashed - tl0) / frames / 1024.0);
			do0 = GXPC_DlHashedOutside;
			th0 = s_texHashed;
			to0 = s_texHashedOut;
			tl0 = s_tlutHashed;
		}
		{
			static uint32_t far0[2];
			fprintf(stderr, "[perf]   off-screen culling: %.1f display-list draws (%.0f verts) skipped per frame%s | farther than "
			        "%.0f: %.1f (%.0f verts)\n",
			        (GXPC_CullRuns - s_perfCullRuns0) / frames, (GXPC_CullVerts - s_perfCullVerts0) / frames,
			        GXPC_CullOff ? " (off: L held at launch)" : "", double(GXPC_CullFar), (GXPC_CullFarRuns - far0[0]) / frames,
			        (GXPC_CullFarVerts - far0[1]) / frames);
			far0[0] = GXPC_CullFarRuns;
			far0[1] = GXPC_CullFarVerts;
		}
		static const char* const kFail[7] = {"not-valid", "size", "vcd", "vat", "mtx", "array", "changed"};
		char fl[160];
		int fat = 0;
		for (int i = 0; i < 7; i++) {
			const uint32_t n = GXPC_DlFail[i] - s_perfDlFail0[i];
			s_perfDlFail0[i] = GXPC_DlFail[i];
			if (n) fat += snprintf(fl + fat, sizeof fl - fat, " %s %.1f", kFail[i], n / frames);
		}
		if (fat) fprintf(stderr, "[perf]   display lists not reusable per frame:%s\n", fl);
		double cc[7];
		for (int i = 0; i < 7; i++) {
			cc[i] = (GXPC_CmdCount[i] - s_perfCmdCount0[i]) / frames;
			s_perfCmdCount0[i] = GXPC_CmdCount[i];
		}
		fprintf(stderr,
		        "[perf]   commands parsed per frame: %.0f primitives (%.0f in uncached lists), %.0f XF writes, %.0f "
		        "indexed XF loads, %.0f BP, %.0f CP, %.0f nested list calls\n",
		        cc[0], cc[1], cc[2], cc[3], cc[4], cc[5], cc[6]);
		fprintf(stderr, "[perf]   draw-sync callbacks: %.1f per frame, %.1f ms inside (wall clock) | emulated thread "
		        "switches: %.1f per frame | %d host threads, %d kernel threads | heap %lu KB in use, arena %lu of %lu KB\n",
		        (GXPC_DrawSyncCalls - s_perfDsCalls0) / frames, double(GXPC_DrawSyncTicks - s_perfDsTicks0) * perMs,
		        double(port_os_switches() - s_perfSwitches0) / frames, port_os_live_hosts(), pica_kernel_threads(),
		        (unsigned long)mallinfo().uordblks >> 10, (unsigned long)mallinfo().arena >> 10, __ctru_heap_size >> 10);
		perfHostMemory();
		s_perfDsCalls0 = GXPC_DrawSyncCalls;
		s_perfDsTicks0 = GXPC_DrawSyncTicks;
		s_perfSwitches0 = port_os_switches();
		{
			char st[1024];
			if (port_os_stack_report(st, sizeof st))
				fprintf(stderr, "[perf]   emulated threads' stacks, most used of the host stack in KB (!: into its last "
				        "2 KB, or past it):%s\n", st);
		}
		uint32_t ringPeak = pica_ring_peak();
		uint32_t dropRing, dropCmd;
		pica_drop_reasons(&dropRing, &dropCmd);
		fprintf(stderr, "[perf]   vertex ring: peak %u KB of %u KB per frame | GPU commands: peak %u KB of %u KB | "
		        "%lu draws dropped so far (ring full %lu, command buffer full %lu) | linear free %u KB\n",
		        unsigned(ringPeak >> 10), unsigned(pica_ring_capacity() >> 10), unsigned(pica_cmd_peak() >> 10),
		        unsigned(pica_cmd_capacity() >> 10), (unsigned long)pica_dropped(),
		        (unsigned long)dropRing, (unsigned long)dropCmd, unsigned(pica_linear_free() >> 10));
		{
			static uint32_t st0[6];
			uint32_t st[6];
			pica_state_stats(st);
			fprintf(stderr, "[perf]   GPU state sent per frame: %.0f draws, %.0f texture binds, %.0f combiner stages, %.0f "
			        "matrix rows, %.0f other uniform vectors, %.0f bool uniforms\n", (st[5] - st0[5]) / frames,
			        (st[0] - st0[0]) / frames, (st[1] - st0[1]) / frames, (st[2] - st0[2]) / frames,
			        (st[3] - st0[3]) / frames, (st[4] - st0[4]) / frames);
			memcpy(st0, st, sizeof st0);
		}
		{
			static uint32_t fx0[7];
			uint32_t fx[7];
			pica_fixed_stats(fx);
			double d[7];
			for (int i = 0; i < 7; i++) d[i] = (fx[i] - fx0[i]) / frames;
			fprintf(stderr, "[perf]   GPU work besides GX-mode draws per frame: %.1f CPU-path draws | %.1f EFB copies "
			        "(%.0f K pixels) | %.1f copies' rectangle clears (%.0f K pixels) | %.1f EFB fills | %.1f render-target "
			        "switches\n", d[0], d[1], d[2] / 1000, d[3], d[4] / 1000, d[5], d[6]);
			memcpy(fx0, fx, sizeof fx0);
		}
		{
			uint32_t vKB, vN, lKB, lN, vFree, aUsed, aPeak, aSize;
			pica_tex_stats(&vKB, &vN, &lKB, &lN, &vFree);
			pica_copy_arena(&aUsed, &aPeak, &aSize);
			fprintf(stderr, "[perf]   textures: %u with a VRAM copy (%u KB), %u in FCRAM only (%u KB) | VRAM free %u KB | "
			        "%lu copies into VRAM, %lu evicted, %lu animated kept in FCRAM so far | EFB copies %u KB of %u (peak %u), "
			        "%lu KB of texture VRAM given to them so far\n",
			        unsigned(vN), unsigned(vKB), unsigned(lN), unsigned(lKB), unsigned(vFree), (unsigned long)pica_vram_copies(),
			        (unsigned long)pica_vram_evicted(), (unsigned long)pica_vram_animated(), unsigned(aUsed), unsigned(aSize),
			        unsigned(aPeak), (unsigned long)(pica_vram_given_to_copies() >> 10));
			uint32_t eN, eKB, eEnc;
			float eMs, eDb;
			pica_etc1_stats(&eN, &eKB, &eEnc, &eMs, &eDb);
			if (eN || eEnc)
				fprintf(stderr, "[perf]   ETC1 textures: %u (%u KB) | %lu encoded so far, in %.0f ms (%s), %.1f dB against the decoded "
				        "texels | %lu waiting for it\n",
				        unsigned(eN), unsigned(eKB), (unsigned long)eEnc, double(eMs),
				        s_etcWorker > 0 ? "on the worker, core 2, its waits included" : "in the render thread", double(eDb),
				        (unsigned long)(s_etcInHead.load() - s_etcInTail.load()));
			texInventory(frames);
		}
		renderThreadPerf(frames);
		static uint64_t sb0, sd0, sc0;
		fprintf(stderr, "[perf]   resident vertices: %.0f drawn from %u pages (%u KB with their indices) per frame, %.0f KB "
		        "stored per frame | %lu resets, %lu times full%s | resident batches %.0f/frame in %.0f index spans, %.0f "
		        "copied into the ring (more than one span)\n", double(s_arenaDrawVerts - s_perfArenaDraw0) / frames,
		        unsigned(arenaPageCount()), unsigned(arenaBytes() >> 10),
		        double(s_arenaUploadBytes - s_perfArenaUp0) / frames / 1024.0, (unsigned long)arenaResets(),
		        (unsigned long)arenaFullCount(), arenaEnabled() ? "" : " (off: Up held at launch)",
		        (s_spanBatches - sb0) / frames, (s_spansDrawn - sd0) / frames, (s_spanCopies - sc0) / frames);
		sb0 = s_spanBatches;
		sd0 = s_spansDrawn;
		sc0 = s_spanCopies;
		{
			double r[2 + kRunStat];
			for (int j = 0; j < 2 + kRunStat; j++) {
				r[j] = double(s_reuse[j] - s_reuse0[j]) / frames;
				s_reuse0[j] = s_reuse[j];
			}
			fprintf(stderr, "[perf]   vertex reuse in resident runs per frame: %.0f indices, %.0f triangles (%.0f%% continuing "
			        "a strip), %.0f vertices stored%s, %.0f distinct | vertices shaded with a FIFO cache of 8/16/32: as "
			        "recorded %.0f/%.0f/%.0f, duplicates merged %.0f/%.0f/%.0f, merged and reordered (cache 16) %.0f\n",
			        r[0], r[2 + RS_TRIS], r[2 + RS_TRIS] > 0 ? 100.0 * r[2 + RS_STRIP] / r[2 + RS_TRIS] : 0.0, r[1],
			        s_mergeDup ? " (duplicates merged)" : " (duplicates kept: circle pad right at launch)",
			        r[2 + RS_DISTINCT], r[2 + RS_C8], r[2 + RS_C16], r[2 + RS_C32], r[2 + RS_D8], r[2 + RS_D16],
			        r[2 + RS_D32], r[2 + RS_T16]);
			const double why = double(s_reordWhy[0] + s_reordWhy[1] + s_reordWhy[2] + s_reordWhy[3] + s_reordWhy[4]);
			fprintf(stderr, "[perf]   as stored (cache 16): %.0f vertices shaded per frame | runs stored so far, by indices: "
			        "reordered %.0f%%, kept for blending/logic op %.0f%%, no depth test %.0f%%, no depth update %.0f%%, "
			        "compare not ordering %.0f%%\n", r[2 + RS_S16], why > 0 ? 100.0 * s_reordWhy[0] / why : 0.0,
			        why > 0 ? 100.0 * s_reordWhy[1] / why : 0.0, why > 0 ? 100.0 * s_reordWhy[2] / why : 0.0,
			        why > 0 ? 100.0 * s_reordWhy[3] / why : 0.0, why > 0 ? 100.0 * s_reordWhy[4] / why : 0.0);
			static uint64_t rd0, ru0;
			fprintf(stderr, "[perf]   reordered runs drawn per frame: %.1f, %.1f of them where order matters (blending, no "
			        "depth test or update, compare equal/always)%s\n", double(s_reordDraws - rd0) / frames,
			        double(s_reordUnsafe - ru0) / frames,
			        !s_mergeDup || !s_reorder ? " | reordering off (A or circle pad right held at launch)" : "");
			rd0 = s_reordDraws;
			ru0 = s_reordUnsafe;
		}
		{
			// The vertex formats drawn, most vertices first.
			uint32_t top[8];
			double tv[8];
			int nt = 0;
			double all = 0;
			for (uint32_t f = 0; f < (1u << 13); f++) {
				const double v = double(s_fmtVerts[f] - s_fmtVerts0[f]) / frames;
				s_fmtVerts0[f] = s_fmtVerts[f];
				if (v <= 0) continue;
				all += v;
				int at;
				if (nt < 8) at = nt++;
				else if (v > tv[7]) at = 7;
				else continue;
				while (at > 0 && tv[at - 1] < v) {
					top[at] = top[at - 1];
					tv[at] = tv[at - 1];
					at--;
				}
				top[at] = f;
				tv[at] = v;
			}
			char line[640];
			int o = 0;
			for (int i = 0; i < nt; i++) {
				const uint32_t f = top[i];
				char d[64];
				int k = snprintf(d, sizeof d, "%s%s%s", (f & VF_NRM) ? ((f & VF_NBT) ? "nbt " : "nrm ") : "",
				                 (f & VF_CLR0) ? "c0 " : "", (f & (VF_CLR0 << 1)) ? "c1 " : "");
				for (int t = 0; t < 8; t++)
					if (f & (VF_TEX0 << t)) k += snprintf(d + k, sizeof d - k, "t%d ", t);
				if (f & VF_MTX) k += snprintf(d + k, sizeof d - k, "mtx ");
				o += snprintf(line + o, sizeof line - o, "%s pos %s(%uB) %.0f", i ? " |" : "", d,
				              unsigned(vtxFmtLayout(f).stride), tv[i]);
			}
			fprintf(stderr, "[perf]   vertices drawn by format per frame (%.0f in all): %s\n", all, line);
		}
		{
			const uint64_t c[4] = {s_copyRunVerts, s_copyDirectVerts, s_aNoRoomVerts, s_aTooBigVerts};
			double d[4];
			for (int i = 0; i < 4; i++) {
				d[i] = double(c[i] - s_perfCopy0[i]) / frames;
				s_perfCopy0[i] = c[i];
			}
			fprintf(stderr, "[perf]   copied into batches instead: %.0f verts/frame of display-list runs (no room %.0f, "
			        "over 65535 %.0f, the rest lines and points), %.0f decoded (uncached lists, direct draws) | "
			        "linear free %u KB\n", d[0], d[2], d[3], d[1], unsigned(pica_linear_free() >> 10));
		}
		s_perfArenaDraw0 = s_arenaDrawVerts;
		s_perfArenaUp0 = s_arenaUploadBytes;
		s_perfDlCalls0 = GXPC_DlCalls;
		s_perfDlHits0 = GXPC_DlHits;
		s_perfDlVerts0 = GXPC_DlVerts;
		s_perfDlRecords0 = GXPC_DlRecords;
		s_perfDlNew0 = GXPC_DlNew;
		s_perfDlUncached0 = GXPC_DlUncached;
		s_perfDlHashed0 = GXPC_DlHashed;
		s_perfCullRuns0 = GXPC_CullRuns;
		s_perfCullVerts0 = GXPC_CullVerts;
		s_perfCmd0 = GXPC_CmdTicks;
		s_perfMain0 = GX3DS_MainBusyTicks;
		s_perfIdle0 = port_idle_seconds();
	}
	memset(s_fbBatches, 0, sizeof s_fbBatches);
	memset(s_fbVerts, 0, sizeof s_fbVerts);
	s_perfWindowStart = now;
	memset(s_perfAcc, 0, sizeof s_perfAcc);
	s_perfFrames = s_perfBatches = s_perfVerts = 0;
}

// ---- stage 1 renderer: CPU transform + flat colour ---------------------------------------
static inline float xff(int r) {
	float f;
	memcpy(&f, &rs.xfReg[r], 4);
	return f;
}

// GX rows per EFB row: 1, except in a pass whose viewport reaches past the EFB's last row, which
// is drawn squeezed into the EFB's rows (the goop maps, 512 rows tall in Pianta Village and on
// Delfino Airstrip, drawn in a 512x512 viewport and copied back texel for texel: the GameCube's
// EFB has 528 rows). Its scissor, copy and clear rectangles are squeezed the same way.
static float efbRowScale() {
	const float bottom = xff(XFR_VIEWPORT + 4) - 342.0f + fabsf(xff(XFR_VIEWPORT + 1));
	return bottom > PICA_EFB_HEIGHT + 2.0f ? bottom / float(PICA_EFB_HEIGHT) : 1.0f;
}
// XF memory's normal matrices: 3x3 each, matrix k's rows at 0x400 + 9 k (GXLoadNrmMtxImm's 0x400 + 3 id).
// (sms-pc-port's renderer copies them to word 256 of its own block, gx_render.cpp; XF memory itself
// has nothing there: until 2026-10-23 every lit draw here read zero normals.)
static constexpr uint32_t kXfNrm = 0x400;
static inline const float* xfMtx(uint32_t word) {  // XF memory as floats
	return reinterpret_cast<const float*>(&rs.xfMem[word]);
}

// GX compare functions (NEVER, LESS, EQUAL, LEQUAL, GREATER, NEQUAL, GEQUAL, ALWAYS) as GPU_TESTFUNC.
static const uint8_t kTestFunc[8] = {0, 4, 2, 5, 6, 3, 7, 1};
// GX blend factors as GPU_BLENDFACTOR; the colour slots differ between source and destination.
static const uint8_t kBlendSrc[8] = {0, 1, 4, 5, 6, 7, 8, 9};  // ZERO ONE DSTCLR INVDSTCLR SRCA INVSRCA DSTA INVDSTA
static const uint8_t kBlendDst[8] = {0, 1, 2, 3, 6, 7, 8, 9};  // ZERO ONE SRCCLR INVSRCCLR SRCA INVSRCA DSTA INVDSTA

// Depth (BP 0x40), blending and write masks (0x41, and 0x42's constant destination alpha), as
// sms-pc-port's renderer maps them (gx_render.cpp). Logic ops are not mapped (drawn unblended).
static void currentPicaState(PicaState& st) {
	enum { EQ_ADD = 0, EQ_REVSUB = 2, F_ZERO = 0, F_ONE = 1, F_DSTA = 8, F_INVDSTA = 9, F_CONSTA = 12 };
	const uint32_t z = rs.bp[BP_ZMODE], c = rs.bp[BP_CMODE0], c1 = rs.bp[BP_CMODE1];
	st.depth_test = z & 1;
	st.depth_func = kTestFunc[(z >> 1) & 7];
	st.depth_write = (z & 1) && ((z >> 4) & 1);
	// A perspective surface tested (less / lequal) but not written lies on what is under it (the
	// sea over the seabed, decals): moved toward the camera a little, as it fights in distant bands
	// with the PICA's coarser float24 depth otherwise.
	if ((z & 1) && !((z >> 4) & 1) && (((z >> 1) & 7) == 1 || ((z >> 1) & 7) == 3) && !(rs.xfReg[XFR_PROJ + 6] & 1))
		st.depth_bias = 1;
	// Culling (BP 0x00 bits 14-15, hardware encoding: 1 back, 2 front, 3 all). GX front faces are
	// clockwise on screen: clockwise in the PICA's y-up clip space, where its front is
	// counter-clockwise.
	const uint32_t cull = (rs.bp[BP_GENMODE] >> 14) & 3;
	st.cull = cull == 1 ? 1 /* GPU_CULL_FRONT_CCW */ : cull == 2 ? 2 /* GPU_CULL_BACK_CCW */ : 0;
	const bool hasAlpha = (rs.bp[BP_PE_CONTROL] & 7) == 1;  // GX_PF_RGBA6_Z24
	const bool colorUpd = (c >> 3) & 1, alphaUpd = ((c >> 4) & 1) && hasAlpha;
	const bool dstAlpha = (c1 >> 8) & 1;  // the EFB alpha written is a constant
	st.write_mask = uint8_t((colorUpd ? 7 : 0) | (alphaUpd || (dstAlpha && hasAlpha) ? 8 : 0));
	// Without EFB alpha, destination alpha reads as 1.
	auto noDstAlpha = [&](uint8_t f) -> uint8_t {
		if (hasAlpha) return f;
		return f == F_DSTA ? F_ONE : f == F_INVDSTA ? F_ZERO : f;
	};
	// Scissor (BP 0x20/0x21: coordinates + 342), off when it covers the EFB.
	{
		const uint32_t tl = rs.bp[BP_SCISSOR_TL], br = rs.bp[BP_SCISSOR_BR];
		const float rows = efbRowScale();
		int top = int(float(int(tl & 0x7FF) - 342) / rows), left = int((tl >> 12) & 0x7FF) - 342;
		int bottom = int(ceilf(float(int(br & 0x7FF) - 342 + 1) / rows)), right = int((br >> 12) & 0x7FF) - 342 + 1;
		auto clampi = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
		left = clampi(left, PICA_EFB_WIDTH), right = clampi(right, PICA_EFB_WIDTH);
		top = clampi(top, PICA_EFB_HEIGHT), bottom = clampi(bottom, PICA_EFB_HEIGHT);
		if (right < left) right = left;
		if (bottom < top) bottom = top;
		st.scissor = !(left == 0 && top == 0 && right == PICA_EFB_WIDTH && bottom >= PICA_EFB_HEIGHT);
		if (st.scissor) {
			st.sc_x0 = uint16_t(left);
			st.sc_y0 = uint16_t(top);
			st.sc_x1 = uint16_t(right);
			st.sc_y1 = uint16_t(bottom);
		}
	}
	const uint8_t aSrc = dstAlpha ? F_CONSTA : F_ONE, aDst = F_ZERO;
	st.blend_color = uint32_t(c1 & 0xFF) << 24;
	st.blend = 0;
	if (c & 1) {
		st.blend = 1;
		if ((c >> 11) & 1) {  // subtract: dst - src
			st.blend_eq = EQ_REVSUB;
			st.blend_src = st.blend_dst = F_ONE;
			st.blend_aeq = dstAlpha ? EQ_ADD : EQ_REVSUB;
			st.blend_asrc = dstAlpha ? aSrc : F_ONE;
			st.blend_adst = dstAlpha ? aDst : F_ONE;
		} else {
			st.blend_eq = st.blend_aeq = EQ_ADD;
			st.blend_src = noDstAlpha(kBlendSrc[(c >> 8) & 7]);
			st.blend_dst = noDstAlpha(kBlendDst[(c >> 5) & 7]);
			st.blend_asrc = dstAlpha ? aSrc : st.blend_src;
			st.blend_adst = dstAlpha ? aDst : st.blend_dst;
		}
	} else if (dstAlpha) {  // colour as is, alpha the constant
		st.blend = 1;
		st.blend_eq = st.blend_aeq = EQ_ADD;
		st.blend_src = F_ONE;
		st.blend_dst = F_ZERO;
		st.blend_asrc = aSrc;
		st.blend_adst = aDst;
	}
}

static inline uint8_t clamp255(float v) { return v <= 0.f ? 0 : v >= 255.f ? 255 : uint8_t(v + 0.5f); }

// Debug colours (GX3DS_SetDebugColors, SELECT on the 3DS side): each draw gets a colour from a
// hash of what it draws (format, size, first vertex, matrix), so the same model keeps its
// colour from frame to frame, and every surface is shaded by its view-space normal. Alpha and
// blending are left alone so fades stay transparent.
static volatile bool s_debugColors;
// Opaque mode (GX3DS_SetDebugColors 2): blending and the alpha test off, to tell what a draw's
// colour is from how it is blended.
static volatile bool s_opaque;
static int debugColorMode() { return s_debugColors ? 1 : s_opaque ? 2 : 0; }
// One-frame trace (GX3DS_TraceNextFrame, L on the 3DS side): every draw, copy and clear of the
// next frame, in order, to the log.
static volatile bool s_traceArm;
static bool s_tracing;
// A traced frame goes to its own file (sdmc:/sms/trace-N.txt), with the TEV dump of every setup it
// draws; otherwise (no SD card) to the log.
static FILE* s_traceFile;
static unsigned s_traceNo;
static std::vector<unsigned> s_traceDumped;  // TEV dumps written to the current trace file
static inline FILE* tf() { return s_traceFile ? s_traceFile : stderr; }
// GX3DS_SetVerboseLog: the once-per-kind [tevdump], [tev] and [copy] lines. Off by default: on a
// New 3DS each new TEV setup's dump cost several ms of SD writes on the game thread.
static bool s_verbose;
static unsigned s_traceDraw;
static unsigned s_skipTev, s_skipTevTo;  // draws of these renderer-log TEV setups (#N) left out (GX3DS_SetSkipTev; 0: none)
// Draws before and after the frame's first EFB-to-texture copy, and the copies (status screen).
static bool s_afterTexCopy;
static uint32_t s_drawsBeforeCopy, s_drawsAfterCopy, s_texCopies;
static uint32_t s_lastBefore, s_lastAfter, s_lastCopies;

static void debugColor(uint32_t pm, const uint8_t* v0, uint32_t stride, uint32_t nidx, uint8_t out[3]) {
	uint32_t h = 2166136261u;
	auto mix = [&](uint32_t x) { h = (h ^ x) * 16777619u; };
	mix(r_fmt);
	mix(nidx);
	mix(pm);
	for (uint32_t i = 0; i < stride && i < 16; i++) mix(v0[i]);
	// A bright colour: one channel high, one low, one from the hash.
	static const uint8_t kOrder[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	const uint8_t* o = kOrder[(h >> 8) % 6];
	out[o[0]] = 255;
	out[o[1]] = uint8_t(40 + (h >> 16) % 80);
	out[o[2]] = uint8_t(h >> 24);
}

// ---- stage 2: textures ---------------------------------------------------------------------
// GX textures are decoded to RGBA8 (gx3ds_texdecode.cpp), resampled to a power of two from 8
// to kMaxTexSide, and kept in a cache keyed like sms-pc-port's (address, format, size, TLUT).
// The data is hashed once per frame to catch changes. Textures that are EFB copies are not
// supported yet: draws sampling them are skipped.
void decodeTexture(const uint8_t* src, uint32_t fmt, uint32_t w, uint32_t h, const uint8_t* tlut, uint32_t tlutFmt,
                   uint8_t* out);

static const uint32_t kMaxTexSide = 256;
static const uint32_t kTexBudget = 8u << 20;

struct TexKey {
	const uint8_t* ptr;
	uint32_t fmt, w, h, tlutAddr, tlutFmt;
	bool operator==(const TexKey& o) const {
		return ptr == o.ptr && fmt == o.fmt && w == o.w && h == o.h && tlutAddr == o.tlutAddr && tlutFmt == o.tlutFmt;
	}
};
struct TexKeyHash {
	size_t operator()(const TexKey& k) const {
		return size_t(uintptr_t(k.ptr)) ^ (k.fmt << 28) ^ (k.w << 16) ^ k.h ^ (k.tlutAddr << 8);
	}
};
struct TexEntry {
	PicaTex* tex = nullptr;
	uint64_t dataHash = 0, tlutHash = 0;
	uint32_t checkedFrame = ~0u, usedFrame = 0;
	uint32_t hashedFrame = 0, hashedClock = 0;  // when the data was last hashed (write-stamp clock)
	int levels = 1;                              // mip levels uploaded
	uint32_t draws = 0, draws0 = 0;              // draws that sampled it, so far and at the last report
	uint32_t uploadFrame = 0;                    // when it was last uploaded
	uint32_t serial = 0;                         // which upload it holds (s_texSerial), for etcApply
};
static std::unordered_map<TexKey, TexEntry, TexKeyHash> s_texCache;
static uint32_t s_texBytes;
static std::vector<uint8_t> s_decodeBuf, s_scaleBuf;
static uint32_t s_texUploads, s_copyTexDraws, s_texSerial;

// settings.txt etc1_textures: the ETC1 encodes, ~20 ms a texture on a New 3DS, ran here in the
// render thread as each stage's textures were first drawn, 30 to 70 of them (1-1.7 s): every stage
// started with seconds of 60-100 ms frames (issue #1). They run on a worker thread instead (core 2
// at the lowest priority, the render thread's idle time), from the game's texels; the texture is
// drawn in its plain format meanwhile and swapped for its ETC1 version at a later draw (etcApply).
struct EtcJob {
	TexKey key;
	uint32_t pw, ph, serial;
	uint64_t dataHash;  // the upload's: its result counts only while the entry holds that upload
	bool alpha;         // ETC1A4
	uint8_t* out;       // the encode (the worker's malloc), null: none
	uint64_t err, ticks;
};
static const uint32_t kEtcJobs = 128;
static EtcJob s_etcIn[kEtcJobs], s_etcOut[kEtcJobs];  // render thread -> worker -> render thread

static void etcWorker(void*) {
	for (;;) {
		const uint32_t t = s_etcInTail.load(std::memory_order_relaxed);
		if (t == s_etcInHead.load(std::memory_order_acquire)) {
			pica_worker_wait();
			continue;
		}
		EtcJob j = s_etcIn[t % kEtcJobs];
		s_etcInTail.store(t + 1, std::memory_order_release);
		j.out = nullptr;
		// The texels as the upload hashed them, before the decode and after the encode: a new stage
		// may have put something else there meanwhile.
		const uint8_t* ptr = j.key.ptr;
		const uint32_t w = j.key.w, h = j.key.h, bytes = texLevelBytes(14, w, h);
		uint8_t* rgba = hashBytes(ptr, bytes) == j.dataHash ? (uint8_t*)malloc(size_t(w) * h * 4) : nullptr;
		uint8_t* scaled = nullptr;
		if (rgba) {
			decodeTexture(ptr, 14, w, h, nullptr, 0, rgba);
			const uint8_t* src = rgba;
			if (j.pw != w || j.ph != h) {  // nearest resample to the power-of-two size, as textureForMap
				scaled = (uint8_t*)malloc(size_t(j.pw) * j.ph * 4);
				src = scaled;
				for (uint32_t y = 0; scaled && y < j.ph; y++) {
					const uint8_t* row = rgba + size_t(y * h / j.ph) * w * 4;
					for (uint32_t x = 0; x < j.pw; x++) memcpy(scaled + (size_t(y) * j.pw + x) * 4, row + (x * w / j.pw) * 4, 4);
				}
			}
			if (src) j.out = (uint8_t*)malloc(size_t(j.pw) * j.ph / (j.alpha ? 1 : 2));
			if (j.out) {
				const uint64_t t0 = svcGetSystemTick();
				j.err = etc1_encode_tex(j.out, j.pw, j.ph, src, j.alpha);
				j.ticks = svcGetSystemTick() - t0;
				if (hashBytes(ptr, bytes) != j.dataHash) {
					free(j.out);
					j.out = nullptr;
				}
			}
		}
		free(scaled);
		free(rgba);
		const uint32_t o = s_etcOutHead.load(std::memory_order_relaxed);
		if (o - s_etcOutTail.load(std::memory_order_acquire) < kEtcJobs) {
			s_etcOut[o % kEtcJobs] = j;
			s_etcOutHead.store(o + 1, std::memory_order_release);
		} else {
			free(j.out);
		}
	}
}

// Whether a texture's encode can go to the worker (started the first time it is asked).
static bool etcRoom() {
	if (!s_etcWorker) s_etcWorker = pica_worker_start(etcWorker) ? 1 : -1;
	return s_etcWorker > 0 && s_etcInHead.load(std::memory_order_relaxed) - s_etcInTail.load(std::memory_order_acquire) < kEtcJobs;
}

static void etcQueue(const TexKey& key, uint32_t pw, uint32_t ph, uint64_t dataHash, uint32_t serial, bool alpha) {
	const uint32_t i = s_etcInHead.load(std::memory_order_relaxed);
	s_etcIn[i % kEtcJobs] = EtcJob{key, pw, ph, serial, dataHash, alpha, nullptr, 0, 0};
	s_etcInHead.store(i + 1, std::memory_order_release);
	pica_worker_signal();
}

// The worker's encodes into the texture cache (before a draw looks its textures up): each entry
// still holding the upload it was made from gets its ETC1 texture, the plain one freed.
static void etcApply() {
	const uint32_t head = s_etcOutHead.load(std::memory_order_acquire);
	for (uint32_t i = s_etcOutTail.load(std::memory_order_relaxed); i != head; i++) {
		EtcJob& j = s_etcOut[i % kEtcJobs];
		auto it = s_texCache.find(j.key);
		if (j.out && it != s_texCache.end() && it->second.serial == j.serial && it->second.dataHash == j.dataHash) {
			TexEntry& e = it->second;
			if (PicaTex* t = pica_tex_create_fmt(j.pw, j.ph, j.alpha ? PICA_TF_ETC1A4 : PICA_TF_ETC1)) {
				pica_tex_upload_raw(t, j.out);
				pica_etc1_note(j.pw, j.ph, j.alpha, j.err, j.ticks);
				s_texBytes -= pica_tex_bytes(e.tex);
				pica_tex_free(e.tex);
				e.tex = t;
				s_texBytes += pica_tex_bytes(t);
			}
		}
		free(j.out);
		s_etcOutTail.store(i + 1, std::memory_order_release);
	}
}
static std::unordered_map<PicaTex*, uint32_t> s_texAvg;  // average RGBA of each uploaded texture

// EFB copies: each destination address the game copies the EFB to gets a texture holding a
// window of the EFB around the source rectangle of its latest copy (platform/3ds/pica,
// pica_copy_tex_layout), replaced when a copy needs another size or format. A draw sampling
// that address maps its texture coordinates through the rectangle into the window (copyMap).
struct EfbCopy {
	const void* dest;
	PicaTex* tex;
	float x, y, w, h;  // source rectangle in EFB pixels
	float win[4];      // the texture's EFB window (x, y, w, h), its top row at t = 1
	uint32_t usedFrame;
	uint32_t clock;  // writeClock() at the latest copy: CPU writes to dest after it make the snapshot stale
	uint32_t fmt;    // copy texture format (1, 7-10: one byte a texel, a map such as the goop's)
	bool reduced;    // at half its size: no VRAM was left for its own (see the copy below)
	uint32_t triedFrame;  // when a coarser texture than its own last tried for its own size
};
static std::vector<EfbCopy> s_efbCopies;
// Bianco Hills copies to 7 destinations a frame (5 goop maps, a 256x256 RGB5A3 copy, the screen
// texture), Noki Bay to 11 (8 goop maps of 32x32 to 256x256): with fewer kept, some were replaced
// every frame, and a goop map whose copy texture goes is drawn from its memory, two frames behind
// (its last frame's cleaning lost: the goop never went). The copy arena (pica_draw.c) holds them.
static const size_t kMaxEfbCopies = 16;
static const uintptr_t kStampPage = 4096;  // the write stamps' granule (sms-pc-port's gx_fifo.cpp, kStampShift)

// Logs each distinct kind of EFB copy once (destination, rectangle, format, flags).
static void logCopyKind(const EfbCopy* c, uint32_t ctrl, uint32_t pix) {
	if (!s_verbose) return;
	static std::vector<uint64_t> seen;
	const uint32_t fmt = ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7);
	uint64_t key = (uint64_t(uintptr_t(c->dest)) << 32) ^ (uint64_t(ctrl) << 8) ^ pix ^
	               (uint64_t(uint32_t(c->w)) << 20) ^ (uint64_t(uint32_t(c->h)) << 40);
	for (uint64_t k : seen)
		if (k == key) return;
	if (seen.size() < 64) seen.push_back(key);
	fprintf(stderr,
	        "[copy] dest %p rect %d,%d %dx%d fmt %u intensity %d half %d clear %d (AR %04x GB %04x) efb-pixfmt %u%s\n",
	        c->dest, int(c->x), int(c->y), int(c->w), int(c->h), fmt, ((ctrl >> 15) & 3) == 3, (ctrl >> 9) & 1,
	        (ctrl >> 11) & 1, rs.bp[BP_CLEAR_AR] & 0xFFFF, rs.bp[BP_CLEAR_GB] & 0xFFFF, pix,
	        pix == 3 ? (fmt == 6 ? " (Z copy: from the depth buffer, read back)" : " (Z copy: not supported, colour copied)") : "");
}

// GX coordinates (s, t) of a copy to its texture's: (s * as + bs, t * at + bt).
static inline void copyMap(const EfbCopy* c, float& as, float& bs, float& at, float& bt) {
	as = c->w / c->win[2];
	bs = (c->x - c->win[0]) / c->win[2];
	at = -c->h / c->win[3];
	bt = 1.f - (c->y - c->win[1]) / c->win[3];
}

// A profiler ceiling (profile_steps = trades): the texture copies of colour formats left out, with
// their clears (the mirror's and the screen texture's in Delfino Plaza); one-byte maps still copied.
static volatile int s_skipColourCopies;
// A profiler step (profile_steps = lighting): 1 the shader's lights drawn without attenuation, 2 no
// lighting in the shader (each lit draw's factor flat, as without normals).
static volatile int s_lightMode;

static EfbCopy* findEfbCopy(const void* p) {
	for (EfbCopy& c : s_efbCopies)
		if (c.dest == p) return &c;
	return nullptr;
}

// A line about a copy destination, once per destination (a few of them).
static void logCopyOnce(const void* d, const char* what, unsigned tw, unsigned th) {
	static const void* s_said[16];
	int i = 0;
	while (i < 16 && s_said[i] && s_said[i] != d) i++;
	if (i == 16 || s_said[i]) return;
	s_said[i] = d;
	fprintf(stderr, "[copy] dest %p (a %ux%u copy texture): %s\n", d, tw, th, what);
}

// A copy with no texture at all: its destination zeroed once (draws read the memory there), so
// what they sample is nothing rather than whatever the heap held, as on Azahar, whose memory starts
// zeroed. The GX size of the copy's format, so nothing past the destination is touched.
static void lostCopy(const void* d, uint32_t cfmt, bool half, float rw, float rh, unsigned tw, unsigned th) {
	static const void* s_zeroed[16];
	int i = 0;
	while (i < 16 && s_zeroed[i] && s_zeroed[i] != d) i++;
	if (i == 16 || s_zeroed[i]) return;
	s_zeroed[i] = d;
	static const uint8_t kTexFmt[13] = {0, 1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 3, 3};  // copy format -> GX texture format
	const uint32_t dw = uint32_t(half ? rw / 2 : rw), dh = uint32_t(half ? rh / 2 : rh);
	const uint32_t bytes = cfmt < 13 ? texLevelBytes(kTexFmt[cfmt], dw, dh) : 0;
	if (bytes) {
		memset(const_cast<void*>(d), 0, bytes);
		memoryWritten(d, bytes);
	}
	fprintf(stderr, "[copy] dest %p: no VRAM for a %ux%u copy texture, nor half that: lost, its %lu bytes zeroed once\n", d, tw,
	        th, (unsigned long)bytes);
}

static inline uint32_t pow2Side(uint32_t v) {
	uint32_t p = 8;
	while (p < v && p < kMaxTexSide) p <<= 1;
	return p;
}

static void evictTextures() {
	while (s_texBytes > kTexBudget) {
		auto victim = s_texCache.end();
		for (auto it = s_texCache.begin(); it != s_texCache.end(); ++it)
			if (it->second.usedFrame != s_rFrames &&
			    (victim == s_texCache.end() || it->second.usedFrame < victim->second.usedFrame))
				victim = it;
		if (victim == s_texCache.end()) return;  // everything is in use this frame
		s_texBytes -= pica_tex_bytes(victim->second.tex);
		pica_tex_free(victim->second.tex);
		s_texCache.erase(victim);
	}
}

// GX I8 (8x4 tiles of 8-bit intensity) straight into a PICA LA8 texture of pw x ph (nearest
// resample), one pass, no RGBA step: the PICA's tiles are 8x8 in Morton order with rows from the
// image's bottom (as pica_tex_upload stores them). L = A = I is how GX samples I8 (I, I, I, I).
static inline uint32_t morton8(uint32_t x, uint32_t y) {
	return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}

static void i8ToLa8(const uint8_t* src, uint32_t w, uint32_t h, uint16_t* dst, uint32_t pw, uint32_t ph);
static void uploadI8(const uint8_t* src, uint32_t w, uint32_t h, PicaTex* t, uint32_t pw, uint32_t ph) {
	uint16_t* dst = static_cast<uint16_t*>(pica_tex_data(t));
	if (!dst) return;
	i8ToLa8(src, w, h, dst, pw, ph);
	pica_tex_written(t);
}
// An I8 texture as LA8 (I, I) in the PICA's layout, resampled (nearest) to pw x ph.
static void i8ToLa8(const uint8_t* src, uint32_t w, uint32_t h, uint16_t* dst, uint32_t pw, uint32_t ph) {
	const uint32_t tileCols = (w + 7) / 8;
	static std::vector<uint32_t> colSrc, colDst;  // per output column: source offset within a tile row, destination
	colSrc.resize(pw);
	colDst.resize(pw);
	for (uint32_t x = 0; x < pw; x++) {
		const uint32_t sx = x * w / pw;
		colSrc[x] = (sx >> 3) * 32 + (sx & 7);
		colDst[x] = (x >> 3) * 64;
	}
	for (uint32_t ry = 0; ry < ph; ry++) {
		const uint32_t sy = (ph - 1 - ry) * h / ph;  // stored row ry holds image row ph - 1 - ry
		const uint8_t* srow = src + size_t(sy >> 2) * tileCols * 32 + (sy & 3) * 8;
		uint16_t* drow = dst + size_t(ry >> 3) * (pw >> 3) * 64;
		const uint32_t my = ry & 7;
		for (uint32_t x = 0; x < pw; x++) {
			const uint16_t i = srow[colSrc[x]];
			drow[colDst[x] + morton8(x & 7, my)] = uint16_t(i | i << 8);
		}
	}
}

// The smallest PICA texel format that holds decoded RGBA8 texels as they are (L8 and LA8 when
// grey, 565 when opaque, 5551 when alpha is only 0 or 255), or with no loss beyond the GX data's
// own precision (RGBA4 for RGB5A3 data with translucent texels, whose colour is 4 bits there).
// RGBA8 when nothing smaller fits.
static int smallTexFormat(const uint8_t* px, size_t n, uint32_t fmt, uint32_t tlutFmt) {
	bool grey = true, opaque = true, alpha1 = true;
	for (size_t i = 0; i < n; i++, px += 4) {
		const uint8_t a = px[3];
		if (a != 255) {
			opaque = false;
			if (a != 0) alpha1 = false;
		}
		if (px[0] != px[1] || px[1] != px[2]) grey = false;
		if (!grey && !opaque && !alpha1) break;
	}
	if (grey) return opaque ? PICA_TF_L8 : PICA_TF_LA8;
	if (opaque) return PICA_TF_RGB565;
	if (alpha1) return PICA_TF_RGBA5551;
	const bool ci = fmt == 8 || fmt == 9 || fmt == 10;
	if (fmt == 5 || (ci && tlutFmt == 2)) return PICA_TF_RGBA4;
	return PICA_TF_RGBA8;
}

// [perf]: the textures drawn in the window that are not in VRAM, by texel format and mip levels,
// and the ones drawn most (draws a frame, with their size): which ones a VRAM copy would help.
// Render thread (perfFrameEnd).
static void texInventory(double frames) {
	static const char* const kFmt[] = {"RGBA8", "RGB565", "RGBA5551", "RGBA4", "LA8", "L8", "ETC1", "ETC1A4"};
	const int kFmts = int(sizeof kFmt / sizeof kFmt[0]);  // (PICA_TF_*: the ETC1 two overran a table of six, crash dump 48)
	struct Row {
		uint32_t w, h, kb, gxFmt;
		int fmt, levels, where;
		double draws;
	};
	std::vector<Row> rows;
	double kb[8][2] = {}, n[8][2] = {}, kept = 0, keptKB = 0;
	static_assert(sizeof kb / sizeof kb[0] == sizeof kFmt / sizeof kFmt[0], "a row per PICA_TF_* format");
	for (auto& kv : s_texCache) {
		TexEntry& e = kv.second;
		const uint32_t d = e.draws - e.draws0;
		e.draws0 = e.draws;
		if (!e.tex || !d) continue;
		const int where = pica_tex_where(e.tex);
		if (where == 1) continue;
		const int f = pica_tex_format(e.tex);
		if (f < 0 || f >= kFmts) continue;
		const uint32_t k = pica_tex_bytes(e.tex) >> 10;
		if (where == 2) {
			kept++;
			keptKB += k;
		}
		kb[f][e.levels > 1] += k;
		n[f][e.levels > 1]++;
		rows.push_back(Row{kv.first.w, kv.first.h, k, kv.first.fmt, f, e.levels, where, d / frames});
	}
	char line[600];
	int at = 0;
	for (int f = 0; f < kFmts; f++)
		for (int m = 0; m < 2; m++)
			if (n[f][m])
				at += snprintf(line + at, sizeof line - at, "%s %s%s %.0f (%.0f KB)", at ? "," : "", kFmt[f],
				               m ? " mipmapped" : "", n[f][m], kb[f][m]);
	fprintf(stderr, "[perf]   textures drawn from outside VRAM:%s | %.0f of them (%.0f KB) kept out (rewritten often)\n",
	        at ? line : " none", kept, keptKB);
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.draws * a.kb > b.draws * b.kb; });
	at = 0;
	for (size_t i = 0; i < rows.size() && i < 14; i++)
		at += snprintf(line + at, sizeof line - at, "%s %ux%u GX%u->%s%s %u KB %.1f/frame%s", i ? " |" : "", unsigned(rows[i].w),
		               unsigned(rows[i].h), unsigned(rows[i].gxFmt), kFmt[rows[i].fmt],
		               rows[i].levels > 1 ? " mip" : "", unsigned(rows[i].kb), rows[i].draws, rows[i].where == 2 ? " kept" : "");
	if (at) fprintf(stderr, "[perf]   outside VRAM, most drawn (draws a frame x size):%s\n", line);
}

// The PICA texture for GX texture map `map`, or null (no data, out of memory). *copy is set
// when the map is an EFB copy.
static PicaTex* textureForMap(int map, const EfbCopy** copy) {
	const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, map)], tlutReg = rs.bp[bpTexReg(BP_TX_TLUT, map)];
	uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1;
	const uint32_t fmt = (img0 >> 20) & 15;
	const uint8_t* ptr = rs.texImage[map];
	if (!ptr) return nullptr;
	*copy = nullptr;
	if (EfbCopy* c = findEfbCopy(ptr)) {
		// The CPU rewrote the destination since the copy (a new stage loaded its own data there,
		// as the goop map of the next level): the memory, not the old snapshot. Write stamps are
		// per 4 KiB page (sms-pc-port's gx_fifo.cpp), so only pages the texture covers whole count
		// (all of its pages when it covers none): its first and last pages hold other heap data
		// too, whose cache flushes dropped the snapshot for memory the GPU never wrote (a 256x256
		// copy in Bianco Hills, every frame: its draws read the destination's zeroed memory).
		const uint32_t fmt0 = (img0 >> 20) & 15;
		const uintptr_t a = uintptr_t(ptr), e = a + texLevelBytes(fmt0, w, h);
		const uintptr_t ia = (a + kStampPage - 1) & ~uintptr_t(kStampPage - 1), ie = e & ~uintptr_t(kStampPage - 1);
		const bool stamped = trustWriteStamps() && unwrittenSince(ptr, 0, 0) && !unwrittenSince(ptr, e - a, c->clock);
		const bool inner = stamped && (ie <= ia || !unwrittenSince(reinterpret_cast<const void*>(ia), ie - ia, c->clock));
		if (stamped && !inner && s_verbose) {  // once per destination
			static const void* s_said[8];
			int i = 0;
			while (i < 8 && s_said[i] && s_said[i] != ptr) i++;
			if (i < 8 && !s_said[i]) {
				s_said[i] = ptr;
				fprintf(stderr, "[copy] dest %p: CPU writes to its edge pages only (other data): copy kept\n", ptr);
			}
		}
		if (inner) {
			if (s_tracing)
				fprintf(tf(), "[trace]   copy to %p (fmt %u) dropped: its memory was written after the copy (copy clock %lu, now %lu)\n",
				        ptr, unsigned(c->fmt), (unsigned long)c->clock, (unsigned long)writeClock());
			pica_tex_free(c->tex);
			s_efbCopies.erase(s_efbCopies.begin() + (c - s_efbCopies.data()));
			s_rsGen++;
		} else {
			c->usedFrame = s_rFrames;
			s_copyTexDraws++;
			*copy = c;
			return c->tex;
		}
	}
	// Mip levels: a mipmapped texture (min filter with mipmaps, max LOD of 1 or more) up to the
	// PICA's 8x8, the levels the GPU may pick. Uploaded with them when level 0 is blank (the sea's
	// distant waves in sea.bmd, seen only through the smaller levels), and (s_mipsOn) when it has 3
	// levels or more and is at most 128x256: a tiled texture seen far off at level 0 thrashes the
	// GPU's texture cache (Delfino Plaza's wave.bti, 128x256 I4 with 5 levels: ~11 ms of GPU a frame
	// with the sea in view, drawn twice for its two alpha ranges) and shimmers. The bigger ones
	// (map.bmd's 128x1024 with 2 levels) stay at one: mip textures are RGBA8, and linear memory runs
	// short in the bigger stages (Delfino Plaza's 39 mipmapped textures would take 4.5 MB).
	const uint32_t mode0 = rs.bp[bpTexReg(BP_TX_MODE0, map)], mode1 = rs.bp[bpTexReg(BP_TX_MODE1, map)];
	int mipLevels = 1;
	if (((mode0 >> 5) & 3) != 0 && fmt != 1 && fmt <= 6 && (w & (w - 1)) == 0 && (h & (h - 1)) == 0) {
		const int maxLod = int(((mode1 >> 8) & 0xFF) >> 4);
		while (mipLevels <= maxLod && (w >> mipLevels) >= 8 && (h >> mipLevels) >= 8) mipLevels++;
	}
	const bool ci = fmt == 8 || fmt == 9 || fmt == 10;
	TexKey key{ptr, fmt, w, h, ci ? (tlutReg & 0x3FF) : 0, ci ? ((tlutReg >> 10) & 3) : 0};
	const uint8_t* tlut = ci ? rs.tlutMem + ((tlutReg & 0x3FF) << 9) : nullptr;
	uint32_t tlutBytes = fmt == 8 ? 32 : fmt == 9 ? 512 : 0x8000;
	if (tlut && ((tlutReg & 0x3FF) << 9) + tlutBytes > TMEM_TLUT_SIZE) tlutBytes = TMEM_TLUT_SIZE - ((tlutReg & 0x3FF) << 9);

	TexEntry& e = s_texCache[key];
	e.usedFrame = s_rFrames;
	e.draws++;
	if (e.tex && e.checkedFrame == s_rFrames) return e.tex;
	const uint32_t bytes = texLevelBytes(fmt, w, h);
	// As sms-pc-port's cache: when no write stamp (GXPC_InvalidateRange below) touched the data
	// since it was last hashed, trust it, but still hash every kRecheckFrames frames for writes
	// nothing reported. TLUTs live in TMEM, rewritten by register loads: always hashed (small).
	const uint64_t th = tlut ? hashBytes(tlut, tlutBytes) : 0;
	if (tlut) s_tlutHashed += tlutBytes;
	e.checkedFrame = s_rFrames;
	const bool stamps = trustWriteStamps();
	if (e.tex && th == e.tlutHash && stamps && s_rFrames - e.hashedFrame < kRecheckFrames &&
	    unwrittenSince(ptr, bytes, e.hashedClock))
		return e.tex;
	// Data the write stamps say the game rewrote (streamed textures, such as THP movie frames every
	// frame) is uploaded without hashing it first: it has almost always changed. Memory outside the
	// stamped range (unwrittenSince(p, 0, 0) false) is always hashed.
	const bool rewritten = e.tex && stamps && th == e.tlutHash && unwrittenSince(ptr, 0, 0) &&
	                       !unwrittenSince(ptr, bytes, e.hashedClock);
	e.hashedClock = writeClock();
	e.hashedFrame = s_rFrames;
	const uint64_t dh = rewritten ? 0 : hashBytes(ptr, bytes);
	if (!rewritten) {
		s_texHashed += bytes;
		if (!unwrittenSince(ptr, 0, 0)) {
			s_texHashedOut += bytes;
			if (uintptr_t(ptr) < s_texOutLo) s_texOutLo = uintptr_t(ptr);
			if (uintptr_t(ptr) + bytes > s_texOutHi) s_texOutHi = uintptr_t(ptr) + bytes;
		}
	}
	if (!rewritten && e.tex && dh == e.dataHash && th == e.tlutHash) return e.tex;

	const uint32_t pw = pow2Side(w), ph = pow2Side(h);
	const bool la8 = fmt == 1;  // I8: decoded straight into an LA8 texture (I, I), as GX samples it
	int levels = 1;
	if (mipLevels > 1) {
		bool blank = true;
		for (uint32_t i = 0; i < bytes && blank; i++) blank = ptr[i] == 0;
		if (blank || (s_mipsOn && mipLevels >= 3 && w * h <= 128u * 256u)) levels = mipLevels;
	}
	// Decoded first (every level of a mipmapped one, one after another in s_decodeBuf), to pick the
	// texel format from what they hold: the smallest that holds every texel as the RGBA8 decode
	// (smallTexFormat), for the mip levels as for a single one.
	const uint8_t* src = nullptr;
	int tf = PICA_TF_RGBA8;
	size_t levelOff[16] = {0};
	bool etcLater = false, etcAlpha = false;  // its ETC1 encode on the worker (etcQueue, after the upload)
	if (!la8) {
		size_t total = 0;
		for (int l = 0; l < levels; l++) {
			levelOff[l] = total;
			total += size_t(w >> l) * (h >> l) * 4;
		}
		s_decodeBuf.resize(total);
		const uint8_t* lp = ptr;
		for (int l = 0; l < levels; l++) {
			if (l) lp += texLevelBytes(fmt, w >> (l - 1), h >> (l - 1));
			decodeTexture(lp, fmt, w >> l, h >> l, tlut, key.tlutFmt, s_decodeBuf.data() + levelOff[l]);
		}
		src = s_decodeBuf.data();
		tf = smallTexFormat(src, total / 4, fmt, key.tlutFmt);
		// settings.txt etc1_textures = on: CMPR textures (already 4 bits a texel, lossy) as ETC1, a
		// quarter of RGB565's memory, or ETC1A4 with alpha, half of RGBA5551's: more of them fit in
		// VRAM, at a second lossy step. Not for data rewritten lately (each upload is an encode). On
		// the worker when it has room, the plain format until its encode is in; in place without one.
		if (s_etc1On && fmt == 14 && (!e.tex || s_rFrames - e.uploadFrame > 300)) {
			etcAlpha = !(tf == PICA_TF_RGB565 || tf == PICA_TF_L8);
			if (dh && levels == 1 && etcRoom()) etcLater = true;
			else if (s_etcWorker < 0) tf = etcAlpha ? PICA_TF_ETC1A4 : PICA_TF_ETC1;
		}
	} else {
		tf = PICA_TF_LA8;
	}
	if (e.tex && (e.levels != levels || pica_tex_format(e.tex) != tf)) {
		s_texBytes -= pica_tex_bytes(e.tex);
		pica_tex_free(e.tex);
		e.tex = nullptr;
	}
	if (!e.tex) {
		e.levels = levels;
		e.tex = levels > 1 ? pica_tex_create_mip(pw, ph, levels, 0.f, tf)
		        : la8      ? pica_tex_create_la8(pw, ph)
		                   : pica_tex_create_fmt(pw, ph, tf);
		if (!e.tex) {
			s_texCache.erase(key);
			return nullptr;
		}
		s_texBytes += pica_tex_bytes(e.tex);
	}
	if (la8) {
		uploadI8(ptr, w, h, e.tex, pw, ph);
		e.dataHash = dh;
		e.tlutHash = th;
		e.uploadFrame = s_rFrames;
		e.serial = ++s_texSerial;
		s_texUploads++;
		evictTextures();
		return e.tex;
	}
	if (pw != w || ph != h) {  // nearest resample to the power-of-two size
		s_scaleBuf.resize(size_t(pw) * ph * 4);
		for (uint32_t y = 0; y < ph; y++) {
			const uint8_t* row = s_decodeBuf.data() + size_t(y * h / ph) * w * 4;
			uint8_t* o = s_scaleBuf.data() + size_t(y) * pw * 4;
			for (uint32_t x = 0; x < pw; x++) memcpy(o + x * 4, row + (x * w / pw) * 4, 4);
		}
		src = s_scaleBuf.data();
	}
	if (s_tracing || s_traceArm) {  // average colour, for the draw trace
		uint64_t sum[4] = {0, 0, 0, 0};
		const size_t px = size_t(pw) * ph;
		for (size_t i = 0; i < px; i++)
			for (int c = 0; c < 4; c++) sum[c] += src[i * 4 + c];
		s_texAvg[e.tex] = uint32_t(sum[0] / px) << 24 | uint32_t(sum[1] / px) << 16 | uint32_t(sum[2] / px) << 8 |
		                  uint32_t(sum[3] / px);
	}
	if (levels > 1) {
		pica_tex_upload_level(e.tex, 0, src);
		for (int l = 1; l < levels; l++) pica_tex_upload_level(e.tex, l, s_decodeBuf.data() + levelOff[l]);
	} else {
		pica_tex_upload(e.tex, src);
	}
	e.dataHash = dh;
	e.tlutHash = th;
	e.uploadFrame = s_rFrames;
	e.serial = ++s_texSerial;
	s_texUploads++;
	if (etcLater) {  // drawn from FCRAM until then: a VRAM copy given back seconds later left gaps in VRAM
		pica_tex_fcram_only(e.tex);
		etcQueue(key, pw, ph, dh, e.serial, etcAlpha);
	}
	evictTextures();
	return e.tex;
}

// For the draw trace: whether map's texture still holds what the game's data makes of it (its
// level 0 as the GPU reads it, against a fresh conversion), whether that data changed since it
// was uploaded (the cache missed a write), and where the texture lives.
static void traceTexCheck(FILE* out, int u, int map, PicaTex* t) {
	const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, map)], tlutReg = rs.bp[bpTexReg(BP_TX_TLUT, map)];
	const uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1, fmt = (img0 >> 20) & 15;
	const uint8_t* ptr = rs.texImage[map];
	if (!t || !ptr) return;
	static std::vector<PicaTex*> s_checked;  // each texture once per traced frame
	static uint32_t s_checkedFrame = ~0u;
	if (s_checkedFrame != s_rFrames) {
		s_checked.clear();
		s_checkedFrame = s_rFrames;
	}
	if (std::find(s_checked.begin(), s_checked.end(), t) != s_checked.end()) return;
	s_checked.push_back(t);
	const bool ci = fmt == 8 || fmt == 9 || fmt == 10;
	const TexKey key{ptr, fmt, w, h, ci ? (tlutReg & 0x3FF) : 0, ci ? ((tlutReg >> 10) & 3) : 0};
	const uint8_t* tlut = ci ? rs.tlutMem + ((tlutReg & 0x3FF) << 9) : nullptr;
	auto it = s_texCache.find(key);
	const char* data = "not in the cache";
	if (it != s_texCache.end() && it->second.tex == t)
		data = it->second.dataHash == 0                          ? "data not hashed at upload"
		       : hashBytes(ptr, texLevelBytes(fmt, w, h)) == it->second.dataHash ? "data unchanged since upload"
		                                                                         : "DATA CHANGED since upload";
	const uint32_t pw = pow2Side(w), ph = pow2Side(h);
	static std::vector<uint8_t> rgba, scaled;
	static std::vector<uint16_t> la8;
	uint32_t diff, bytes, first;
	if (fmt == 1 && pica_tex_format(t) == PICA_TF_LA8) {
		la8.assign(size_t(pw) * ph, 0);
		i8ToLa8(ptr, w, h, la8.data(), pw, ph);
		diff = pica_tex_check(t, nullptr, la8.data(), &bytes, &first);
	} else {
		rgba.resize(size_t(w) * h * 4);
		decodeTexture(ptr, fmt, w, h, tlut, key.tlutFmt, rgba.data());
		const uint8_t* src = rgba.data();
		if (pw != w || ph != h) {
			scaled.resize(size_t(pw) * ph * 4);
			for (uint32_t y = 0; y < ph; y++)
				for (uint32_t x = 0; x < pw; x++)
					memcpy(&scaled[(size_t(y) * pw + x) * 4], &rgba[(size_t(y * h / ph) * w + x * w / pw) * 4], 4);
			src = scaled.data();
		}
		diff = pica_tex_check(t, src, nullptr, &bytes, &first);
	}
	const int where = pica_tex_where(t);
	fprintf(out, "[trace]   unit %d: texture %s at %08lx%s, %s, level 0 %s", u,
	        where == 1 ? "drawn from its VRAM copy" : where == 2 ? "kept in FCRAM" : "in FCRAM",
	        (unsigned long)pica_tex_phys(t, where == 1), where == 1 ? " (VRAM)" : "", data,
	        diff ? "DIFFERS" : bytes ? "matches the data" : "not checked");
	if (diff) fprintf(out, ": %lu of %lu bytes, first at +%lx", (unsigned long)diff, (unsigned long)bytes, (unsigned long)first);
	fprintf(out, " (PICA format %d)\n", pica_tex_format(t));
}

// For the draw trace: the average colour (and smallest alpha) of map's texture as the game's data
// decodes now.
static uint8_t s_traceMinA;
static uint32_t traceTexAvg(int map) {
	s_traceMinA = 0;
	const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, map)], tlutReg = rs.bp[bpTexReg(BP_TX_TLUT, map)];
	const uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1, fmt = (img0 >> 20) & 15;
	const uint8_t* ptr = rs.texImage[map];
	if (!ptr || findEfbCopy(ptr)) return 0;
	const bool ci = fmt == 8 || fmt == 9 || fmt == 10;
	const uint8_t* tlut = ci ? rs.tlutMem + ((tlutReg & 0x3FF) << 9) : nullptr;
	static std::vector<uint8_t> buf;
	buf.resize(size_t(w) * h * 4);
	decodeTexture(ptr, fmt, w, h, tlut, (tlutReg >> 10) & 3, buf.data());
	uint64_t sum[4] = {0, 0, 0, 0};
	uint8_t minA = 255;
	const size_t px = size_t(w) * h;
	for (size_t i = 0; i < px; i++) {
		for (int c = 0; c < 4; c++) sum[c] += buf[i * 4 + c];
		minA = std::min(minA, buf[i * 4 + 3]);
	}
	s_traceMinA = minA;
	return uint32_t(sum[0] / px) << 24 | uint32_t(sum[1] / px) << 16 | uint32_t(sum[2] / px) << 8 | uint32_t(sum[3] / px);
}

// ---- stage 3: TEV -> PICA texture combiners ------------------------------------------------
// The GX TEV state is read from BP registers as sms-pc-port's shader generator reads it
// (gx_shader.cpp: buildKey/genFS) and translated into PICA combiner stages. Each GX stage
// computes, per channel (colour, alpha), r = d +/- lerp(a, b, c) + bias, times a scale, into
// one of four registers (PREV, C0-C2); the PICA has six stages that see the previous stage's
// output, a combiner buffer (one stage delayed), one constant colour per stage, the vertex
// colour and three textures. The translation tracks where each GX register's channel lives:
//   * a constant known now (the BP_TEV_REG value it starts with, or a stage whose inputs are
//     all constant, folded on the CPU);
//   * PREVIOUS, the output of the last PICA stage;
//   * the combiner buffer, where a live value is saved before a stage overwrites PREVIOUS.
// Each GX stage channel becomes one to three PICA operations (REPLACE / MODULATE / ADD /
// ADD_SIGNED / SUBTRACT / INTERPOLATE / MULTIPLY_ADD, and the stage scale); with several,
// each one's result is kept in PREVIOUS for the next, and a GX stage takes that many PICA
// stages. A setup that cannot be expressed falls back to the stage-2 rule (first texture x
// vertex colour) and is logged once.
enum { CH_RGB = 0, CH_A = 1 };
enum : uint8_t { SRC_PRIMARY = 0x00, SRC_TEX0 = 0x03, SRC_BUFFER = 0x0D, SRC_CONSTANT = 0x0E, SRC_PREVIOUS = 0x0F };
enum : uint8_t {
	FN_REPLACE = 0, FN_MODULATE = 1, FN_ADD = 2, FN_ADD_SIGNED = 3, FN_INTERPOLATE = 4, FN_SUBTRACT = 5,
	FN_MULTIPLY_ADD = 8
};

// A combiner input for one channel: a constant (rgb, or v[0] for alpha) or a PICA source with
// the component it reads (0-3: r, g, b, a replicated; 4: the colour as is, rgb only). mid: the
// result of the channel's first operation (PREVIOUS in its second).
struct TevSrc {
	bool k;
	int16_t v[3];
	uint8_t src, comp;
	bool inv;  // 1 - x
	bool mid;
};
static TevSrc tsConst(int r, int g, int b) {
	return TevSrc{true, {int16_t(r), int16_t(g), int16_t(b)}, 0, 0, false, false};
}
static TevSrc tsConst1(int a) { return tsConst(a, a, a); }
static TevSrc tsSrc(uint8_t src, uint8_t comp) { return TevSrc{false, {0, 0, 0}, src, comp, false, false}; }
static bool tsSame(const TevSrc& a, const TevSrc& b, int nc) {
	if (a.k != b.k || a.inv != b.inv || a.mid != b.mid) return false;
	if (a.k) {
		for (int i = 0; i < nc; i++)
			if (a.v[i] != b.v[i]) return false;
		return true;
	}
	return a.src == b.src && a.comp == b.comp;
}
static bool tsAll(const TevSrc& s, int nc, int v) {
	if (!s.k) return false;
	for (int i = 0; i < nc; i++)
		if (s.v[i] != v) return false;
	return true;
}

struct TevOp {
	uint8_t func, n, scale;
	TevSrc s[3];
};
// What one channel of a GX stage becomes: folded to a constant, or one to three PICA operations.
static const int kMaxOps = 3;
struct ChanPlan {
	bool folded;
	int16_t fv[3];
	int nops;
	TevOp op[kMaxOps];
	const char* fail;
	bool softScale;    // scale 1/2, drawn at 1
	bool softCompare;  // compare drawn as a steep ramp
};

static inline int tevLerp(int a, int b, int c) {
	int c2 = c + (c >> 7);
	return (a * (256 - c2) + b * c2 + 128) >> 8;
}

// GX stage arithmetic on constants (as genFS computes it).
static int tevFold(int a, int b, int c, int d, uint32_t bias, uint32_t sub, uint32_t scale, uint32_t clampv) {
	int r = sub ? d - tevLerp(a & 255, b & 255, c & 255) : d + tevLerp(a & 255, b & 255, c & 255);
	r += bias == 1 ? 128 : bias == 2 ? -128 : 0;
	if (scale == 1) r *= 2;
	else if (scale == 2) r *= 4;
	else if (scale == 3) r >>= 1;
	if (clampv) return r < 0 ? 0 : r > 255 ? 255 : r;
	return r < -1024 ? -1024 : r > 1023 ? 1023 : r;
}

// GX compare stage on constants: d + (a OP b ? c : 0), per genFS. mode: scale << 1 | op. The
// GR8/GR16/GR24 forms compare the colour inputs' r(, g, b); for the alpha channel those are not
// known here (nc == 1): -1.
static int tevCompareFold(const int16_t* a, const int16_t* b, int c, int d, int i, int nc, uint32_t mode,
                          uint32_t clampv) {
	bool cond;
	if (mode >= 6) {
		cond = mode == 6 ? a[i] > b[i] : a[i] == b[i];
	} else {
		if (nc == 1) return -1;
		int x = a[0] & 255, y = b[0] & 255;
		if (mode >= 2) x |= (a[1] & 255) << 8, y |= (b[1] & 255) << 8;
		if (mode >= 4) x |= (a[2] & 255) << 16, y |= (b[2] & 255) << 16;
		cond = (mode & 1) ? x == y : x > y;
	}
	int r = d + (cond ? c : 0);
	if (clampv) return r < 0 ? 0 : r > 255 ? 255 : r;
	return r < -1024 ? -1024 : r > 1023 ? 1023 : r;
}

// Scale 1/2 as an operation of its own (else drawn at 1); off for a retry when the stages run out.
static bool s_tevHalfExact = true;

static ChanPlan planChannel(TevSrc A, TevSrc B, TevSrc C, TevSrc D, int nc, uint32_t bias, uint32_t sub,
                            uint32_t scale, uint32_t clampv) {
	ChanPlan p;
	memset(&p, 0, sizeof p);
	for (TevSrc* s : {&A, &B, &C})
		if (s->k)
			for (int i = 0; i < 3; i++) s->v[i] &= 255;
	auto op = [&](uint8_t fn, std::initializer_list<TevSrc> srcs) {
		TevOp& o = p.op[p.nops++];
		o.func = fn;
		o.n = 0;
		for (const TevSrc& s : srcs) o.s[o.n++] = s;
	};
	auto failed = [&](const char* why) {
		p.fail = why;
		return p;
	};
	if (bias == 3) {  // compare: d + (a OP b ? c : 0), no scale
		const uint32_t mode = scale << 1 | sub;
		if (A.k && B.k && C.k && D.k) {
			p.folded = true;
			for (int i = 0; i < nc; i++) {
				const int r = tevCompareFold(A.v, B.v, C.v[i], D.v[i], i, nc, mode, clampv);
				if (r < 0 && nc == 1) return failed("alpha compare of colour inputs");
				p.fv[i] = int16_t(r);
			}
			return p;
		}
		// Never true (0 > x), or adding nothing: d.
		if ((!sub && tsAll(A, nc, 0)) || tsAll(C, nc, 0)) {
			if (D.k) {
				p.folded = true;
				for (int i = 0; i < nc; i++)
					p.fv[i] = int16_t(clampv ? (D.v[i] < 0 ? 0 : D.v[i] > 255 ? 255 : D.v[i]) : D.v[i]);
			} else {
				op(FN_REPLACE, {D});
			}
			return p;
		}
		// a > b ? 1 : 0 as a steep ramp, clamp(32 * (a - b)): the GR8-GR24 forms compare red
		// (the inputs are usually replicated), RGB8/A8 each component.
		if (!sub) {
			TevSrc a = A, b = B;
			for (TevSrc* x : {&a, &b}) {
				if (mode >= 6) continue;
				if (nc == 1) return failed("alpha compare of colour inputs");
				if (x->k) x->v[1] = x->v[2] = x->v[0];
				else if (x->comp == 4) x->comp = 0;
			}
			TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(nc == 3 ? 4 : 3));
			mid.mid = true;
			op(FN_SUBTRACT, {a, b});
			p.op[0].scale = 2;
			op(FN_ADD, {mid, mid});
			p.op[1].scale = 2;
			p.softCompare = true;
			// d + c * ramp
			const bool c1 = tsAll(C, nc, 255), d0 = tsAll(D, nc, 0);
			if (c1 && !d0) op(FN_ADD, {mid, D});
			else if (!c1 && d0) op(FN_MODULATE, {mid, C});
			else if (!c1) op(FN_MULTIPLY_ADD, {mid, C, D});
			return p;
		}
		return failed("compare stage");
	}
	// Inputs the lerp ignores.
	if (tsAll(C, nc, 0)) B = tsConst1(0);
	else if (tsAll(C, nc, 255)) A = tsConst1(0);
	if (A.k && B.k && C.k && D.k) {
		p.folded = true;
		for (int i = 0; i < nc; i++) p.fv[i] = int16_t(tevFold(A.v[i], B.v[i], C.v[i], D.v[i], bias, sub, scale, clampv));
		return p;
	}
	const uint8_t outScale = scale == 1 ? 1 : scale == 2 ? 2 : 0;
	p.softScale = scale == 3;

	// lerp(a, b, c) as: nothing, X, X * Y, or a real interpolation.
	enum { LZ, LX, LXY, LERP } lf;
	TevSrc X{}, Y{};
	if (tsAll(C, nc, 0)) { lf = tsAll(A, nc, 0) ? LZ : LX; X = A; }
	else if (tsAll(C, nc, 255)) { lf = tsAll(B, nc, 0) ? LZ : LX; X = B; }
	else if (tsAll(A, nc, 0) && tsAll(B, nc, 0)) lf = LZ;
	else if (tsAll(A, nc, 0)) { lf = LXY; X = B; Y = C; }
	else if (tsAll(B, nc, 0)) { lf = LXY; X = A; Y = C; Y.inv = !Y.inv; }
	else if (tsSame(A, B, nc)) { lf = LX; X = A; }
	else lf = LERP;
	auto kval = [](const TevSrc& s, int i) { return s.inv ? 255 - s.v[i] : s.v[i]; };

	// A constant added to d +/- lerp, per component: the bias, and parts of the lerp below.
	int off[3];
	for (int i = 0; i < 3; i++) off[i] = i < nc ? (bias == 1 ? 128 : bias == 2 ? -128 : 0) : 0;
	if (D.k && lf == LERP && A.k && B.k) {
		// d +/- lerp(a, b, c) with constant a, b, d: lerp(d +/- a, d +/- b, c), when both fit.
		TevSrc A2 = A, B2 = B;
		bool fits = true;
		for (int i = 0; i < nc; i++) {
			const int dv = D.v[i] + off[i];
			const int a = sub ? dv - A.v[i] : dv + A.v[i], b = sub ? dv - B.v[i] : dv + B.v[i];
			fits &= a >= 0 && a <= 255 && b >= 0 && b <= 255;
			A2.v[i] = int16_t(a);
			B2.v[i] = int16_t(b);
		}
		if (fits) {
			A = A2;
			B = B2;
			D = tsConst1(0);
			memset(off, 0, sizeof off);
			sub = 0;
		}
	}
	if (lf == LERP && C.k && (A.k || B.k)) {
		// A constant c and one constant end: x * y plus a constant.
		for (int i = 0; i < nc; i++) {
			const int k = B.k ? tevLerp(0, B.v[i], C.v[i]) : tevLerp(A.v[i], 0, C.v[i]);
			off[i] += sub ? -k : k;
		}
		if (B.k) {
			X = A;
			Y = C;
			Y.inv = true;
		} else {
			X = B;
			Y = C;
		}
		lf = LXY;
	}
	if (lf == LXY && X.k && Y.k) {  // a constant product
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) K.v[i] = int16_t(tevLerp(0, kval(X, i), kval(Y, i)));
		lf = LX;
		X = K;
	}
	if (lf == LXY) {  // a product with 1: the other factor
		auto one = [&](const TevSrc& s) {
			if (!s.k) return false;
			for (int i = 0; i < nc; i++)
				if (kval(s, i) != 255) return false;
			return true;
		};
		if (one(X)) {
			lf = LX;
			X = Y;
		} else if (one(Y)) {
			lf = LX;
		}
	}
	if (lf == LX && X.k && !D.k) {  // a constant lerp joins the constant
		for (int i = 0; i < nc; i++) off[i] += sub ? -X.v[i] : X.v[i];
		lf = LZ;
	}

	TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(nc == 3 ? 4 : 3));
	mid.mid = true;
	// The lerp alone, as one operation (for LXY and LERP).
	auto lerpOp = [&] {
		if (lf == LXY) op(FN_MODULATE, {X, Y});
		else op(FN_INTERPOLATE, {B, A, C});
	};
	auto fold0 = [&] {
		p.folded = true;
		memset(p.fv, 0, sizeof p.fv);
	};
	auto finish = [&]() -> ChanPlan {
		if (p.nops) p.op[p.nops - 1].scale = outScale;
		if (p.softScale && s_tevHalfExact && p.nops && p.nops < kMaxOps) {  // scale 1/2: x * 1/2
			op(FN_MODULATE, {mid, tsConst1(128)});
			p.softScale = false;
		}
		// Constants under 1 - x are folded.
		for (int k = 0; k < p.nops; k++)
			for (int j = 0; j < p.op[k].n; j++) {
				TevSrc& s = p.op[k].s[j];
				if (s.k && s.inv) {
					for (int i = 0; i < 3; i++) s.v[i] = int16_t(255 - s.v[i]);
					s.inv = false;
				}
			}
		return p;
	};

	if (D.k && !sub && lf != LZ) {  // lerp + k, k of mixed sign within +/-1/2: lerp + (k + 1/2) - 1/2
		bool neg = false, pos = false, fits = true;
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) {
			const int v = D.v[i] + off[i];
			neg |= v < 0;
			pos |= v > 0;
			fits &= v >= -128 && v <= 127;
			K.v[i] = int16_t(v + 128);
		}
		if (neg && pos && fits) {
			if (lf == LX) op(FN_ADD_SIGNED, {X, K});
			else {
				lerpOp();
				op(FN_ADD_SIGNED, {mid, K});
			}
			return finish();
		}
		if (neg && pos) {  // past +/-1/2: the positive part added, then the negative part taken off
			TevSrc KP = tsConst1(0), KN = tsConst1(0);
			for (int i = 0; i < nc; i++) {
				const int v = D.v[i] + off[i];
				KP.v[i] = int16_t(v > 0 ? (v > 255 ? 255 : v) : 0);
				KN.v[i] = int16_t(v < 0 ? (v < -255 ? 255 : -v) : 0);
			}
			if (lf == LX) op(FN_ADD, {X, KP});
			else {
				lerpOp();
				op(FN_ADD, {mid, KP});
			}
			op(FN_SUBTRACT, {mid, KN});
			return finish();
		}
	}
	if (D.k) {  // the constant joins a constant d
		bool neg = false, pos = false, over = false;
		for (int i = 0; i < nc; i++) {
			int v = D.v[i] + off[i];
			D.v[i] = int16_t(v > 255 ? 255 : v);
			if (v < 0) neg = true;
			if (v > 0) pos = true;
			if (v > 255) over = true;
			off[i] = 0;
		}
		if (over && sub) return failed("constant d over 1 minus x");
		if (neg) {  // lerp - k
			if (pos || sub) return failed("constant d of mixed sign");
			TevSrc K = D;
			for (int i = 0; i < nc; i++) K.v[i] = int16_t(-K.v[i]);
			if (lf == LZ) fold0();
			else if (lf == LX) op(FN_SUBTRACT, {X, K});
			else {
				lerpOp();
				op(FN_SUBTRACT, {mid, K});
			}
			return finish();
		}
	}
	if (!sub && lf == LERP && A.k && B.k && !D.k && !D.mid && D.src == SRC_PREVIOUS) {
		// d + lerp(ka, kb, c) + k with d in PREVIOUS: d + ka * (1 - c), + kb * c, + k; d is read
		// first (before an operation overwrites it), one constant per operation.
		bool kp = false, kn = false;
		for (int i = 0; i < nc; i++) kp |= off[i] > 0, kn |= off[i] < 0;
		if (!(kp && kn)) {
			TevSrc ci = C;
			ci.inv = !ci.inv;
			op(FN_MULTIPLY_ADD, {ci, A, D});
			op(FN_MULTIPLY_ADD, {C, B, mid});
			if (kp || kn) {
				TevSrc K = tsConst1(0);
				for (int i = 0; i < nc; i++) K.v[i] = int16_t(off[i] < 0 ? -off[i] : off[i]);
				op(kp ? FN_ADD : FN_SUBTRACT, {mid, K});
			}
			return finish();
		}
	}
	const bool dz = tsAll(D, nc, 0);
	bool offPos = false, offNeg = false, offHalf = true;
	for (int i = 0; i < nc; i++) {
		offPos |= off[i] > 0;
		offNeg |= off[i] < 0;
		offHalf &= off[i] == off[0];
	}
	if (offPos || offNeg) {  // a varying d and a constant
		if (lf == LZ && offPos && offNeg) {  // d + k, k of mixed sign: d + (k + 1/2) - 1/2
			TevSrc K = tsConst1(0);
			for (int i = 0; i < nc; i++) {
				const int v = off[i] + 128;
				if (v < 0 || v > 255) return failed("constant of mixed sign with a varying d");
				K.v[i] = int16_t(v);
			}
			op(FN_ADD_SIGNED, {D, K});
			return finish();
		}
		if (offPos && offNeg) return failed("constant of mixed sign with a varying d");
		offHalf &= off[0] == 128 || off[0] == -128;
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) K.v[i] = int16_t(off[i] < 0 ? -off[i] : off[i]);
		const uint8_t addK = offPos ? FN_ADD : FN_SUBTRACT;
		if (lf == LZ) {
			op(addK, {D, K});
		} else if (offHalf && !sub && offNeg) {  // d + lerp - 1/2
			if (lf == LX) op(FN_ADD_SIGNED, {X, D});
			else {
				lerpOp();
				op(FN_ADD_SIGNED, {mid, D});
			}
		} else if (offHalf && sub && offPos) {  // d - lerp + 1/2 ~ d + (1 - lerp) - 1/2
			if (lf == LX) {
				TevSrc nx = X;
				nx.inv = !nx.inv;
				op(FN_ADD_SIGNED, {D, nx});
			} else {
				lerpOp();
				TevSrc nm = mid;
				nm.inv = true;
				op(FN_ADD_SIGNED, {D, nm});
			}
		} else if (!sub && lf == LXY) {  // d + x * y, then the constant
			op(FN_MULTIPLY_ADD, {X, Y, D});
			op(addK, {mid, K});
		} else if (lf == LX) {  // d +/- x, then the constant
			op(sub ? FN_SUBTRACT : FN_ADD, sub ? std::initializer_list<TevSrc>{D, X} : std::initializer_list<TevSrc>{X, D});
			op(addK, {mid, K});
		} else {
			return failed("bias with a varying d");
		}
		return finish();
	}
	if (!sub) {
		switch (lf) {
		case LZ: if (dz) fold0(); else op(FN_REPLACE, {D}); break;
		case LX: if (dz) op(FN_REPLACE, {X}); else op(FN_ADD, {X, D}); break;
		case LXY: if (dz) op(FN_MODULATE, {X, Y}); else op(FN_MULTIPLY_ADD, {X, Y, D}); break;
		case LERP:
			op(FN_INTERPOLATE, {B, A, C});
			if (!dz) op(FN_ADD, {mid, D});
			break;
		}
	} else {
		switch (lf) {
		case LZ: if (dz) fold0(); else op(FN_REPLACE, {D}); break;
		case LX: if (dz) fold0(); else op(FN_SUBTRACT, {D, X}); break;
		case LXY:
		case LERP:
			if (dz) {
				fold0();
				break;
			}
			lerpOp();
			op(FN_SUBTRACT, {D, mid});
			break;
		}
	}
	return finish();
}

// Logs each distinct unsupported TEV pattern once.
static void tevLogOnce(const char* why, uint32_t stage, uint32_t nStages, uint32_t ce, uint32_t ae) {
	if (!s_verbose) return;
	static std::vector<uint64_t> seen;
	uint64_t key = (uint64_t(uintptr_t(why)) << 32) ^ (uint64_t(ce) << 8) ^ (uint64_t(ae) << 24) ^ stage;
	for (uint64_t k : seen)
		if (k == key) return;
	if (seen.size() >= 256) return;
	seen.push_back(key);
	fprintf(stderr, "[tev] %s: stage %u of %u, colour env %06x, alpha env %06x\n", why, stage, nStages, ce, ae);
}

// The GX TEV state of the draw, decoded. Textures are (map, coordinate) pairs; PICA texture
// units are given to the pairs the translation samples (unitPair).
static const int kMaxPairs = 8;
struct TevSetup {
	uint32_t numStages;
	uint32_t colorEnv[16], alphaEnv[16], order[16], ksel[16], swap[4];
	int stagePair[16];  // texture pair each stage samples (-1: none)
	int nPairs;
	int pairMap[kMaxPairs], pairCoord[kMaxPairs];
	int rasRgbChan, rasAChan;  // colour channels the vertex colour's rgb and alpha carry (0, 1)
	bool rgbVarying[2], aVarying[2];  // channel colour / alpha differs per vertex (vertex colour or lighting)
	int16_t rasConst[2][4];  // otherwise: its colour (material register)
	bool chanSpec[2];        // channel colour is specular lighting (attenuation function spec)
	int16_t pairConst[kMaxPairs][4];  // a pair read as a constant (pairWhite): its rgba (white, or its average)
	uint32_t pairTexels[kMaxPairs];   // texture size of each pair (0: none or an EFB copy)
	uint8_t pairFmt[kMaxPairs];       // its GX texture format
	int nUnits;              // set by the translation
	int unitPair[PICA_MAX_TEX];
	int16_t symErrOld, symErrNew;  // tevSymImprove's checks (-1: none): the translation's error, the compiled one's
	bool symUsed;                  // the compiled one drawn
};

static void readTev(TevSetup& t) {
	memset(&t, 0, sizeof t);
	t.symErrOld = t.symErrNew = -1;
	t.numStages = ((rs.bp[BP_GENMODE] >> 10) & 15) + 1;
	for (int p = 0; p < kMaxPairs; p++)
		for (int i = 0; i < 4; i++) t.pairConst[p][i] = 255;
	for (uint32_t s = 0; s < t.numStages; s++) {
		t.colorEnv[s] = rs.bp[BP_TEV_COLOR_ENV + 2 * s] & 0xFFFFFF;
		t.alphaEnv[s] = rs.bp[BP_TEV_COLOR_ENV + 2 * s + 1] & 0xFFFFFF;
		t.order[s] = (rs.bp[BP_TREF + (s >> 1)] >> ((s & 1) * 12)) & 0x3FF;
		const uint32_t ks = rs.bp[BP_TEV_KSEL + (s >> 1)];
		t.ksel[s] = (s & 1) ? ((ks >> 14) & 0x3FF) : ((ks >> 4) & 0x3FF);
	}
	for (int i = 0; i < 4; i++)
		t.swap[i] = (rs.bp[BP_TEV_KSEL + 2 * i] & 15) | (rs.bp[BP_TEV_KSEL + 2 * i + 1] & 15) << 4;
	// Colour channels: per-vertex, or one material colour for the draw.
	for (int c = 0; c < 2; c++) {
		const uint32_t cctrl = rs.xfReg[XFR_COLOR0CTRL + c], actrl = rs.xfReg[XFR_ALPHA0CTRL + c], mat = rs.xfReg[XFR_MAT0 + c];
		const bool hasClr = (r_fmt & (VF_CLR0 << c)) != 0, hasNrm = (r_fmt & VF_NRM) != 0;
		t.rgbVarying[c] = ((cctrl & 1) && hasClr) || (((cctrl >> 1) & 1) && hasNrm);
		t.aVarying[c] = ((actrl & 1) && hasClr) || (((actrl >> 1) & 1) && hasNrm);
		t.chanSpec[c] = ((cctrl >> 1) & 1) && hasNrm && ((cctrl >> 9) & 3) == 1;
		t.rasConst[c][0] = int16_t(mat >> 24);
		t.rasConst[c][1] = int16_t((mat >> 16) & 0xFF);
		t.rasConst[c][2] = int16_t((mat >> 8) & 0xFF);
		t.rasConst[c][3] = int16_t(mat & 0xFF);
	}
	bool needRgb[2] = {false, false}, needA[2] = {false, false};
	for (uint32_t s = 0; s < t.numStages; s++) {
		const uint32_t ord = t.order[s], map = ord & 7, coord = (ord >> 3) & 7, chan = (ord >> 7) & 7;
		t.stagePair[s] = -1;
		if ((ord >> 6) & 1) {
			int p = 0;
			while (p < t.nPairs && (t.pairMap[p] != int(map) || t.pairCoord[p] != int(coord))) p++;
			if (p == t.nPairs && p == kMaxPairs) {
				p = -1;  // more pairs than tracked: the stage reads white
			} else if (p == t.nPairs) {
				t.pairMap[p] = int(map);
				t.pairCoord[p] = int(coord);
				t.nPairs++;
			}
			t.stagePair[s] = p;
		}
		// Which parts (rgb, alpha) of which channel the stage reads, through its swap table.
		if (chan <= 1) {
			const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s], sw = t.swap[ae & 3];
			auto part = [&](uint32_t comp) { (comp == 3 ? needA : needRgb)[chan] = true; };
			for (int i = 0; i < 4; i++) {
				uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c == 10)
					for (int k = 0; k < 3; k++) part((sw >> (2 * k)) & 3);
				if (c == 11 || a == 5) part((sw >> 6) & 3);
			}
		}
	}
	// The vertex colour carries the rgb of a channel whose rgb is read (one that varies per vertex
	// first), and the alpha of one whose alpha is read.
	auto pick = [](const bool need[2], const bool varying[2]) {
		for (int c = 0; c < 2; c++)
			if (need[c] && varying[c]) return c;
		return need[1] && !need[0] ? 1 : 0;
	};
	t.rasRgbChan = pick(needRgb, t.rgbVarying);
	t.rasAChan = pick(needA, t.aVarying);
}

// GX registers PREV, C0-C2 by channel: where the value lives.
enum : uint8_t { LOC_CONST, LOC_PREV, LOC_BUF, LOC_LOST };
struct TevLoc {
	uint8_t kind;
	int16_t v[3];  // LOC_CONST: rgb, or v[0] for alpha
};

static inline int sext11(uint32_t v) { return int(v << 21) >> 21; }

static inline uint8_t clampByte(int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); }

// The channel (CH_RGB, CH_A) of PREVIOUS / the buffer a source reads.
static inline int srcChannel(const TevSrc& s) { return s.comp == 3 ? CH_A : CH_RGB; }

// A texture pair as a combiner source until it gets a texture unit.
enum : uint8_t { SRC_PAIR = 0x10 };

// Translates the TEV setup into st's combiner stages and gives texture units to the pairs it
// samples (t.nUnits, t.unitPair). pairWhite[p]: pair p has no texture (reads as white).
// Returns false (and logs) when the setup cannot be expressed.
static bool translateTev(TevSetup& t, const bool pairWhite[kMaxPairs], PicaState& st) {
	const uint32_t N = t.numStages;
	t.nUnits = 0;
	int unitOfPair[kMaxPairs];
	for (int i = 0; i < kMaxPairs; i++) unitOfPair[i] = -1;
	// Copies (a register moved unchanged into another) are propagated: later stages read the
	// original while it holds, so the copy is often dead (and dropped below).
	uint32_t CE[16], AE[16];
	{
		int al[2][4] = {{0, 1, 2, 3}, {0, 1, 2, 3}};  // per channel: the register each one equals
		for (uint32_t s = 0; s < N; s++) {
			uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s];
			for (int i = 0; i < 4; i++) {
				const uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c < 8) ce = (ce & ~(15u << (4 * i))) | uint32_t(al[c & 1][c >> 1] * 2 + (c & 1)) << (4 * i);
				if (a < 4) ae = (ae & ~(7u << (4 + 3 * i))) | uint32_t(al[CH_A][a]) << (4 + 3 * i);
			}
			CE[s] = ce;
			AE[s] = ae;
			// A copy: no lerp, no bias or scale, d a register's colour / alpha.
			const bool cCopy = ((ce >> 4) & 0xFFF) == 0xFFF && (ce & 15) < 8 && !(ce & 1) && ((ce >> 16) & 3) == 0 &&
			                   ((ce >> 20) & 3) == 0;
			const bool aCopy = ((ae >> 7) & 0x1FF) == 0x1FF && ((ae >> 4) & 7) < 4 && ((ae >> 16) & 3) == 0 &&
			                   ((ae >> 20) & 3) == 0;
			const int dst[2] = {int((ce >> 22) & 3), int((ae >> 22) & 3)};
			const int from[2] = {cCopy ? int((ce & 15) >> 1) : dst[0], aCopy ? int((ae >> 4) & 7) : dst[1]};
			for (int ch = 0; ch < 2; ch++) {
				for (int x = 0; x < 4; x++)
					if (al[ch][x] == dst[ch]) al[ch][x] = x;
				al[ch][dst[ch]] = from[ch];
			}
		}
	}
	// Liveness: bit 2 * reg + channel, live after each stage.
	uint8_t liveAfter[16];
	{
		uint8_t live = uint8_t(1u << (2 * ((CE[N - 1] >> 22) & 3) + CH_RGB) |
		                       1u << (2 * ((AE[N - 1] >> 22) & 3) + CH_A));
		for (int s = int(N) - 1; s >= 0; s--) {
			liveAfter[s] = live;
			const uint32_t ce = CE[s], ae = AE[s];
			live &= uint8_t(~(1u << (2 * ((ce >> 22) & 3) + CH_RGB)));
			live &= uint8_t(~(1u << (2 * ((ae >> 22) & 3) + CH_A)));
			for (int i = 0; i < 4; i++) {
				uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c < 8) live |= uint8_t(1u << (2 * (c >> 1) + (c & 1)));
				if (a < 4) live |= uint8_t(1u << (2 * a + CH_A));
			}
		}
	}
	// Values read whole later: bit 2 * reg + channel, after each stage, read as some stage's d
	// before being overwritten. Inputs a, b, c take a value's low 8 bits, as does the output.
	uint8_t dReadAfter[16];
	{
		uint8_t rd = 0;
		for (int s = int(N) - 1; s >= 0; s--) {
			dReadAfter[s] = rd;
			const uint32_t ce = CE[s], ae = AE[s], dc = ce & 15, da = (ae >> 4) & 7;
			rd &= uint8_t(~(1u << (2 * ((ce >> 22) & 3) + CH_RGB)));
			rd &= uint8_t(~(1u << (2 * ((ae >> 22) & 3) + CH_A)));
			if (dc < 8) rd |= uint8_t(1u << (2 * (dc >> 1) + (dc & 1)));
			if (da < 4) rd |= uint8_t(1u << (2 * da + CH_A));
		}
	}

	TevLoc loc[4][2];
	for (int r = 0; r < 4; r++) {
		const uint32_t ra = rs.bp[BP_TEV_REG + 2 * r], bg = rs.bp[BP_TEV_REG + 2 * r + 1];
		loc[r][CH_RGB] = TevLoc{LOC_CONST, {int16_t(sext11(ra & 0x7FF)), int16_t(sext11((bg >> 12) & 0x7FF)),
		                                    int16_t(sext11(bg & 0x7FF))}};
		loc[r][CH_A] = TevLoc{LOC_CONST, {int16_t(sext11((ra >> 12) & 0x7FF)), 0, 0}};
	}
	int prevOcc[2] = {-1, -1}, bufOcc[2] = {-1, -1};
	// The buffer's initial colour as a constant: set, its value, and whether a value was saved
	// over it.
	bool bufColorSet[2] = {false, false}, bufTouched[2] = {false, false};
	TevSrc bufColor[2];
	int n = 0;  // PICA stages emitted
	int curTex = -1;  // texture pair `tex` holds (-1: white), as genFS keeps it between stages
	static const int k8[8] = {255, 223, 191, 159, 127, 95, 63, 31};
	static const int nc[2] = {3, 1};

	for (uint32_t s = 0; s < N; s++) {
		const uint32_t ce = CE[s], ae = AE[s], ord = t.order[s];
		const uint32_t chan = (ord >> 7) & 7;
		const uint32_t rasSwap = t.swap[ae & 3], texSwap = t.swap[(ae >> 2) & 3];
		if (t.stagePair[s] >= 0) curTex = t.stagePair[s];
		const char* softFail = nullptr;

		// konst
		const uint32_t kc = t.ksel[s] & 31, ka = (t.ksel[s] >> 5) & 31;
		auto kreg = [](uint32_t i, int comp) {  // comp 0-3: r, g, b, a
			const uint32_t ra = rs.kreg[2 * i], bg = rs.kreg[2 * i + 1];
			return int(comp == 0 ? ra & 0xFF : comp == 1 ? (bg >> 12) & 0xFF : comp == 2 ? bg & 0xFF : (ra >> 12) & 0xFF);
		};
		TevSrc konstC, konstA;
		if (kc < 8) konstC = tsConst1(k8[kc]);
		else if (kc >= 12 && kc < 16) konstC = tsConst(kreg(kc - 12, 0), kreg(kc - 12, 1), kreg(kc - 12, 2));
		else if (kc >= 16) konstC = tsConst1(kreg(kc & 3, int((kc - 16) >> 2)));
		else konstC = tsConst1(255);
		if (ka < 8) konstA = tsConst1(k8[ka]);
		else if (ka >= 16) konstA = tsConst1(kreg(ka & 3, int((ka - 16) >> 2)));
		else konstA = tsConst1(255);

		// A texture or vertex colour input through a swap table. comp: 4 colour, 3 alpha.
		auto swapped = [&](uint8_t src, uint32_t swap, int comp) -> TevSrc {
			const int sr = swap & 3, sg = (swap >> 2) & 3, sb = (swap >> 4) & 3, sa = (swap >> 6) & 3;
			if (comp == 3) return tsSrc(src, uint8_t(sa));
			if (sr == 0 && sg == 1 && sb == 2) return tsSrc(src, 4);
			if (sr == sg && sg == sb) return tsSrc(src, uint8_t(sr));
			softFail = "swap table permutes colour";
			return tsSrc(src, 4);
		};
		auto texIn = [&](int comp) -> TevSrc {
			if (curTex < 0) return tsConst1(255);
			if (pairWhite[curTex]) {  // read as a constant, through the swap table
				const int16_t* k = t.pairConst[curTex];
				if (comp == 3) return tsConst1(k[(texSwap >> 6) & 3]);
				return tsConst(k[texSwap & 3], k[(texSwap >> 2) & 3], k[(texSwap >> 4) & 3]);
			}
			return swapped(uint8_t(SRC_PAIR + curTex), texSwap, comp);
		};
		auto rasIn = [&](int comp) -> TevSrc {
			if (chan > 1) {
				if (chan != 7) softFail = "bump alpha (indirect) colour";
				return tsConst1(0);
			}
			// The components read; each comes from the vertex colour (when it carries that part of
			// this channel) or is constant for the draw.
			const int sr = rasSwap & 3, sg = (rasSwap >> 2) & 3, sb = (rasSwap >> 4) & 3, sa = (rasSwap >> 6) & 3;
			const int* cs = comp == 3 ? &sa : nullptr;
			const int rgbc[3] = {sr, sg, sb};
			bool fromVtx = false, isConst = true;
			for (int k = 0; k < (comp == 3 ? 1 : 3); k++) {
				const int c = cs ? *cs : rgbc[k];
				const bool alphaPart = c == 3;
				const bool carried = alphaPart ? int(chan) == t.rasAChan : int(chan) == t.rasRgbChan;
				const bool varying = alphaPart ? t.aVarying[chan] : t.rgbVarying[chan];
				if (carried) fromVtx = true;
				if (carried || varying) isConst = false;
			}
			if (isConst) {
				const int16_t* c = t.rasConst[chan];
				return comp == 3 ? tsConst1(c[sa]) : tsConst(c[sr], c[sg], c[sb]);
			}
			// A specular channel the vertex colour does not carry: black, as it is away from
			// highlights (the other channel's colour would light the model up).
			if (!fromVtx && comp != 3 && t.chanSpec[chan]) return tsConst1(0);
			if (!fromVtx) softFail = "stages read both colour channels";
			return swapped(SRC_PRIMARY, rasSwap, comp);
		};
		bool lost = false;
		// Register reg's channel ch, read as comp (4: colour, 3: alpha replicated).
		auto regIn = [&](int reg, int ch, int comp) -> TevSrc {
			const TevLoc& l = loc[reg][ch];
			switch (l.kind) {
			case LOC_CONST: return ch == CH_A ? tsConst1(l.v[0]) : tsConst(l.v[0], l.v[1], l.v[2]);
			case LOC_PREV: return tsSrc(SRC_PREVIOUS, uint8_t(comp));
			case LOC_BUF: return tsSrc(SRC_BUFFER, uint8_t(comp));
			default: lost = true; return tsConst1(0);
			}
		};
		auto colorIn = [&](uint32_t c) -> TevSrc {
			if (c < 8) return regIn(int(c >> 1), (c & 1) ? CH_A : CH_RGB, (c & 1) ? 3 : 4);
			switch (c) {
			case 8: return texIn(4);
			case 9: return texIn(3);
			case 10: return rasIn(4);
			case 11: return rasIn(3);
			case 12: return tsConst1(255);
			case 13: return tsConst1(128);
			case 14: return konstC;
			default: return tsConst1(0);
			}
		};
		auto alphaIn = [&](uint32_t a) -> TevSrc {
			if (a < 4) return regIn(int(a), CH_A, 3);
			switch (a) {
			case 4: return texIn(3);
			case 5: return rasIn(3);
			case 6: return konstA;
			default: return tsConst1(0);
			}
		};

		// (255 + lerp) * 2, unclamped, read only through its low 8 bits: on the console 510 + 2 lerp
		// wraps to 2 lerp - 2, which water surfaces rely on (the file select's sea and Bianco Hills'
		// river compute their alpha as (1 + a) * 2 for 2a). PICA stages clamp it to 1, which washes
		// the water out white; drawn as 2 lerp (exact for lerp 1-128). d: the stage's d input.
		auto rasIs255 = [&](int comp) {  // the vertex colour input is 255 for the whole draw
			if (chan > 1) return false;
			const int sw[4] = {int(rasSwap & 3), int((rasSwap >> 2) & 3), int((rasSwap >> 4) & 3), int((rasSwap >> 6) & 3)};
			for (int k = comp == 3 ? 3 : 0; k < (comp == 3 ? 4 : 3); k++) {
				const int c = sw[k];
				if ((c == 3 ? t.aVarying[chan] : t.rgbVarying[chan]) || t.rasConst[chan][c] != 255) return false;
			}
			return true;
		};
		auto wrapD = [&](uint32_t env, int ch, TevSrc d) {
			const uint32_t dArg = ch == CH_A ? (env >> 4) & 7 : env & 15;
			if ((env >> 16 & 0x3F) != 0x10 || (dReadAfter[s] >> (2 * ((env >> 22) & 3) + ch) & 1)) return d;
			const bool is255 = tsAll(d, ch == CH_A ? 1 : 3, 255) ||
			                   (ch == CH_A ? dArg == 5 && rasIs255(3) : (dArg == 10 && rasIs255(4)) || (dArg == 11 && rasIs255(3)));
			if (!is255) return d;
			tevLogOnce("(255 + x) * 2 wraps: drawn as 2x", s, N, ce, ae);
			return tsConst1(0);
		};

		// A channel whose result nothing reads is not computed.
		const bool dead[2] = {!(liveAfter[s] >> (2 * ((ce >> 22) & 3) + CH_RGB) & 1),
		                      !(liveAfter[s] >> (2 * ((ae >> 22) & 3) + CH_A) & 1)};
		ChanPlan plan[2];
		memset(plan, 0, sizeof plan);
		if (!dead[CH_RGB])
			plan[CH_RGB] = planChannel(colorIn((ce >> 12) & 15), colorIn((ce >> 8) & 15), colorIn((ce >> 4) & 15),
		                           wrapD(ce, CH_RGB, colorIn(ce & 15)), 3, (ce >> 16) & 3, (ce >> 18) & 1, (ce >> 20) & 3,
		                           (ce >> 19) & 1);
		if (!dead[CH_A] && ((ae >> 16) & 3) == 3 && ((ae >> 20) & 3) < 3) {
			// An alpha compare in the R8 / GR16 / BGR24 modes compares the colour inputs a and b
			// (here by their red): as the A8 mode on those.
			TevSrc ca = colorIn((ce >> 12) & 15), cb = colorIn((ce >> 8) & 15);
			for (TevSrc* x : {&ca, &cb})
				if (!x->k && x->comp == 4) x->comp = 0;
			plan[CH_A] = planChannel(ca, cb, alphaIn((ae >> 7) & 7), alphaIn((ae >> 4) & 7), 1, 3, (ae >> 18) & 1, 3,
			                         (ae >> 19) & 1);
		} else if (!dead[CH_A]) {
			plan[CH_A] = planChannel(alphaIn((ae >> 13) & 7), alphaIn((ae >> 10) & 7), alphaIn((ae >> 7) & 7),
			                         wrapD(ae, CH_A, alphaIn((ae >> 4) & 7)), 1, (ae >> 16) & 3, (ae >> 18) & 1,
			                         (ae >> 20) & 3, (ae >> 19) & 1);
		}
		if (lost) {
			tevLogOnce("register value lost (no room in the combiner buffer)", s, N, ce, ae);
			return false;
		}
		for (int ch = 0; ch < 2; ch++) {
			if (plan[ch].fail) {
				tevLogOnce(plan[ch].fail, s, N, ce, ae);
				return false;
			}
			if (plan[ch].softScale) tevLogOnce("scale 1/2 (drawn at 1)", s, N, ce, ae);
			if (plan[ch].softCompare) tevLogOnce("compare drawn as a ramp", s, N, ce, ae);
		}
		if (softFail) tevLogOnce(softFail, s, N, ce, ae);

		// An operation with two different constants reads one from the combiner buffer's initial
		// colour while the buffer holds it (from PICA stage 1, until a value is saved there), or
		// else loads one into PREVIOUS first: a one-op channel becomes two.
		bool noSave[2] = {false, false};
		for (int ch = 0; ch < 2; ch++) {
			ChanPlan& p = plan[ch];
			for (int k = 0; k < p.nops; k++) {
				TevSrc ks[3];
				int nk = 0;
				for (int j = 0; j < p.op[k].n; j++) {
					if (!p.op[k].s[j].k) continue;
					bool dup = false;
					for (int i = 0; i < nk; i++) dup |= tsSame(ks[i], p.op[k].s[j], nc[ch]);
					if (!dup) ks[nk++] = p.op[k].s[j];
				}
				if (nk < 2) continue;
				// Not while this stage needs the buffer to keep the register in PREVIOUS.
				const int dch = int(ch == CH_RGB ? (ce >> 22) & 3 : (ae >> 22) & 3), po = prevOcc[ch];
				const bool needSave = po >= 0 && dch != po && (liveAfter[s] >> (2 * po + ch) & 1);
				if (nk == 2 && n >= 1 && !bufTouched[ch] && !needSave &&
				    (!bufColorSet[ch] || tsSame(bufColor[ch], ks[0], nc[ch]))) {
					TevSrc b = tsSrc(SRC_BUFFER, uint8_t(ch == CH_RGB ? 4 : 3));
					for (int j = 0; j < p.op[k].n; j++)
						if (p.op[k].s[j].k && tsSame(p.op[k].s[j], ks[0], nc[ch])) p.op[k].s[j] = b;
					bufColor[ch] = ks[0];
					bufColorSet[ch] = true;
					noSave[ch] = true;
					continue;
				}
				if (nk > 2 || k != 0 || p.nops >= kMaxOps) {
					tevLogOnce("too many constants in one stage", s, N, ce, ae);
					return false;
				}
				TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(ch == CH_RGB ? 4 : 3));
				mid.mid = true;
				TevOp main = p.op[0];
				for (int j = 0; j < main.n; j++)
					if (main.s[j].k && tsSame(main.s[j], ks[0], nc[ch])) main.s[j] = mid;
				for (int i = p.nops; i > 1; i--) p.op[i] = p.op[i - 1];
				p.op[0] = TevOp{FN_REPLACE, 1, 0, {ks[0]}};
				p.op[1] = main;
				p.nops++;
				break;
			}
		}

		const int dest[2] = {int((ce >> 22) & 3), int((ae >> 22) & 3)};
		// A register copied onto itself (it is in PREVIOUS already) needs no PICA stage, nor does
		// a copy of the register in PREVIOUS that is not read afterwards (PREVIOUS is renamed).
		bool noop[2] = {false, false}, rename[2] = {false, false};
		for (int ch = 0; ch < 2; ch++) {
			const ChanPlan& p = plan[ch];
			const TevSrc& x = p.op[0].s[0];
			const bool copy = !p.folded && p.nops == 1 && p.op[0].func == FN_REPLACE && p.op[0].scale == 0 && !x.k &&
			                  !x.mid && !x.inv && x.src == SRC_PREVIOUS && x.comp == (ch == CH_RGB ? 4 : 3) && prevOcc[ch] >= 0;
			rename[ch] = copy && prevOcc[ch] != dest[ch] && !(liveAfter[s] >> (2 * prevOcc[ch] + ch) & 1);
			noop[ch] = dead[ch] || (copy && prevOcc[ch] == dest[ch]) || rename[ch];
			if (noop[ch]) plan[ch].nops = 0;
		}
		const bool computes[2] = {!plan[CH_RGB].folded && !noop[CH_RGB], !plan[CH_A].folded && !noop[CH_A]};
		// The PICA stages this GX stage takes: the channels' operations end together, the
		// shorter one passing PREVIOUS through first.
		int nst = 0;
		for (int ch = 0; ch < 2; ch++)
			if (computes[ch] && plan[ch].nops > nst) nst = plan[ch].nops;
		if (nst == 0 && (computes[0] || computes[1])) nst = 1;
		const int first[2] = {nst - plan[CH_RGB].nops, nst - plan[CH_A].nops};  // stage of each channel's first op
		// An alpha-only stage (one operation on constants, the vertex colour or textures) goes
		// into the last PICA stage's alpha when that one passes alpha through and nothing needs
		// the alpha it passed.
		bool mergeA = false;
		if (nst == 1 && !computes[CH_RGB] && computes[CH_A] && plan[CH_A].nops == 1 && n >= 1 &&
		    !((st.buf_alpha >> (n - 1)) & 1)) {
			const PicaTevStage& q = st.stage[n - 1];
			const TevOp& o = plan[CH_A].op[0];
			bool srcOk = true, usesK = false, rgbK = false;
			for (int j = 0; j < o.n; j++) {
				const TevSrc& x = o.s[j];
				if (x.k) usesK = true;
				else if (x.mid || x.src == SRC_PREVIOUS || x.src == SRC_BUFFER) srcOk = false;
			}
			for (int j = 0; j < 3; j++) rgbK |= ((q.src_rgb >> (4 * j)) & 15) == SRC_CONSTANT;
			const int po = prevOcc[CH_A];
			const bool keepNone = po < 0 || po == dest[CH_A] || !((liveAfter[s] >> (2 * po + CH_A)) & 1);
			mergeA = srcOk && keepNone && !(usesK && rgbK) && q.src_alpha == SRC_PREVIOUS && q.op_alpha == 0 &&
			         q.func_alpha == FN_REPLACE && q.scale_alpha == 0;
		}
		const int stagesNeeded = mergeA ? 0 : nst;
		if (n + stagesNeeded > PICA_MAX_STAGES) {
			tevLogOnce("more than 6 combiner stages", s, N, ce, ae);
			return false;
		}

		// Values in PREVIOUS that this stage overwrites but later stages read go to the buffer:
		// the stage that produced them (n - 1) updates it, and it shows from stage n + 1 on.
		bool saved[2] = {false, false};
		const int prevBefore[2] = {prevOcc[0], prevOcc[1]};
		for (int ch = 0; ch < 2; ch++) {
			if (!computes[ch]) continue;
			const int r = prevOcc[ch];
			if (r < 0) continue;
			if (dest[ch] != r && (liveAfter[s] >> (2 * r + ch) & 1)) {
				const int b = bufOcc[ch];
				const bool bufFree = b < 0 || b == dest[ch] || !(liveAfter[s] >> (2 * b + ch) & 1);
				if (n < 1 || n - 1 > 3 || !bufFree || noSave[ch]) {
					tevLogOnce("no room to keep a register value", s, N, ce, ae);
					return false;
				}
				if (b >= 0) loc[b][ch].kind = LOC_LOST;
				if (ch == CH_RGB) st.buf_rgb |= uint8_t(1u << (n - 1));
				else st.buf_alpha |= uint8_t(1u << (n - 1));
				loc[r][ch].kind = LOC_BUF;
				bufOcc[ch] = r;
				saved[ch] = true;
				bufTouched[ch] = true;
			} else {
				loc[r][ch].kind = LOC_LOST;
			}
			prevOcc[ch] = -1;
		}
		// With several PICA stages, the later ones' inputs must still be what this GX stage read:
		// the buffer only of a channel not saved to now. PREVIOUS read there, of a channel an
		// earlier one overwrites, is read from the buffer instead: saved there by the stage
		// before (n - 1), it shows from stage n + 1 on.
		if (nst > 1) {
			bool bufRead[2] = {false, false};  // the buffer's present value, read by a later operation
			for (int ch = 0; ch < 2; ch++)
				for (int k = 0; k < plan[ch].nops; k++) {
					if (first[ch] + k < 1) continue;
					const TevOp& o = plan[ch].op[k];
					for (int j = 0; j < o.n; j++)
						if (!o.s[j].k && !o.s[j].mid && o.s[j].src == SRC_BUFFER) bufRead[srcChannel(o.s[j])] = true;
				}
			for (int ch = 0; ch < 2; ch++)
				for (int k = 0; k < plan[ch].nops; k++) {
					const int w = first[ch] + k;
					if (w < 1) continue;
					TevOp& o = plan[ch].op[k];
					for (int j = 0; j < o.n; j++) {
						TevSrc& x = o.s[j];
						if (x.k || x.mid) continue;
						const int xc = srcChannel(x);
						if (x.src == SRC_BUFFER && saved[xc]) {
							tevLogOnce("value overwritten between two combiner stages", s, N, ce, ae);
							return false;
						}
						if (x.src != SRC_PREVIOUS || !plan[xc].nops || first[xc] >= w) continue;
						const int r = prevBefore[xc];
						if (!saved[xc]) {
							const int b = bufOcc[xc];
							const bool bufFree = b < 0 || b == r || b == dest[xc] || !(liveAfter[s] >> (2 * b + xc) & 1);
							if (r < 0 || n < 1 || n - 1 > 3 || !bufFree || bufRead[xc] || noSave[xc]) {
								tevLogOnce("value overwritten between two combiner stages", s, N, ce, ae);
								return false;
							}
							if (b >= 0 && b != r) loc[b][xc].kind = LOC_LOST;
							if (xc == CH_RGB) st.buf_rgb |= uint8_t(1u << (n - 1));
							else st.buf_alpha |= uint8_t(1u << (n - 1));
							if (loc[r][xc].kind == LOC_PREV || loc[r][xc].kind == LOC_LOST) loc[r][xc].kind = LOC_BUF;
							bufOcc[xc] = r;
							saved[xc] = true;
							bufTouched[xc] = true;
						}
						x.src = SRC_BUFFER;
					}
				}
		}

		auto opRgb = [](const TevSrc& x) -> uint32_t {
			static const uint8_t k[5] = {0x4, 0x8, 0xC, 0x2, 0x0};
			return (x.k ? 0u : k[x.comp]) | (x.inv ? 1u : 0u);
		};
		auto opA = [](const TevSrc& x) -> uint32_t {
			static const uint8_t k[4] = {0x2, 0x4, 0x6, 0x0};
			return (x.k ? 0u : k[x.comp > 3 ? 3 : x.comp]) | (x.inv ? 1u : 0u);
		};
		// Emits the PICA stage(s).
		for (int w = 0; w < nst; w++) {
			PicaTevStage& ps = mergeA ? st.stage[n - 1] : st.stage[n++];
			const PicaTevStage kept = ps;
			if (!mergeA) memset(&ps, 0, sizeof ps);
			int16_t kv[4] = {0, 0, 0, 0};
			for (int ch = 0; ch < 2; ch++) {
				const ChanPlan& p = plan[ch];
				const TevOp* o = nullptr;
				if (computes[ch] && w >= first[ch]) o = &p.op[w - first[ch]];
				uint32_t srcs = SRC_PREVIOUS, ops = 0;
				uint8_t func = FN_REPLACE, scl = 0;
				if (o) {
					srcs = 0;
					for (int j = 0; j < 3; j++) {
						const TevSrc& x = j < o->n ? o->s[j] : o->s[0];
						uint32_t src = x.k ? SRC_CONSTANT : x.src;
						if (src >= SRC_PAIR) {  // a texture: its unit, given on first use
							const int pr = int(src - SRC_PAIR);
							if (unitOfPair[pr] < 0) {
								if (t.nUnits == PICA_MAX_TEX) {
									tevLogOnce("more than 3 textures sampled", s, N, ce, ae);
									return false;
								}
								unitOfPair[pr] = t.nUnits;
								t.unitPair[t.nUnits++] = pr;
							}
							src = SRC_TEX0 + uint32_t(unitOfPair[pr]);
						}
						srcs |= src << (4 * j);
						ops |= (ch == CH_RGB ? opRgb(x) : opA(x)) << (4 * j);
						if (x.k) {
							if (ch == CH_RGB) memcpy(kv, x.v, sizeof x.v);
							else kv[3] = x.v[0];
						}
					}
					func = o->func;
					scl = o->scale;
				}
				if (ch == CH_RGB) {
					ps.src_rgb = uint16_t(srcs);
					ps.op_rgb = uint16_t(ops);
					ps.func_rgb = func;
					ps.scale_rgb = scl;
				} else {
					ps.src_alpha = uint16_t(srcs);
					ps.op_alpha = uint16_t(ops);
					ps.func_alpha = func;
					ps.scale_alpha = scl;
				}
			}
			ps.color = uint32_t(clampByte(kv[0])) | uint32_t(clampByte(kv[1])) << 8 |
			           uint32_t(clampByte(kv[2])) << 16 | uint32_t(clampByte(kv[3])) << 24;
			if (mergeA) {  // the colour half stays as it was
				ps.src_rgb = kept.src_rgb;
				ps.op_rgb = kept.op_rgb;
				ps.func_rgb = kept.func_rgb;
				ps.scale_rgb = kept.scale_rgb;
				ps.color = (kept.color & 0x00FFFFFFu) | (ps.color & 0xFF000000u);
			}
		}
		// Where the stage's results live now.
		for (int ch = 0; ch < 2; ch++) {
			const int r = dest[ch];
			if (rename[ch]) {
				loc[prevOcc[ch]][ch].kind = LOC_LOST;
				if (bufOcc[ch] == r) bufOcc[ch] = -1;
				loc[r][ch].kind = LOC_PREV;
				prevOcc[ch] = r;
			}
			if (noop[ch]) continue;
			if (bufOcc[ch] == r) bufOcc[ch] = -1;
			if (computes[ch]) {
				loc[r][ch].kind = LOC_PREV;
				prevOcc[ch] = r;
			} else {
				if (prevOcc[ch] == r) prevOcc[ch] = -1;
				loc[r][ch] = TevLoc{LOC_CONST, {plan[ch].fv[0], plan[ch].fv[1], plan[ch].fv[2]}};
			}
		}
	}

	// The output: the last stage's colour and alpha registers, moved into PREVIOUS if needed.
	const int outReg[2] = {int((CE[N - 1] >> 22) & 3), int((AE[N - 1] >> 22) & 3)};
	// A constant output goes into the last stage when that stage passes its channel through.
	if (n > 0) {
		PicaTevStage& last = st.stage[n - 1];
		for (int ch = 0; ch < 2; ch++) {
			const TevLoc& l = loc[outReg[ch]][ch];
			if (prevOcc[ch] == outReg[ch] || l.kind != LOC_CONST) continue;
			uint16_t& src = ch == CH_RGB ? last.src_rgb : last.src_alpha;
			const uint16_t opv = ch == CH_RGB ? last.op_rgb : last.op_alpha;
			const uint8_t fn = ch == CH_RGB ? last.func_rgb : last.func_alpha, sc = ch == CH_RGB ? last.scale_rgb : last.scale_alpha;
			// A computed alpha there is not read after the last stage: the constant replaces it,
			// unless the stage's colour reads its constant's alpha.
			bool rgbReadsKA = false;
			for (int j = 0; j < 3; j++)
				rgbReadsKA |= ((last.src_rgb >> (4 * j)) & 15) == SRC_CONSTANT && ((last.op_rgb >> (4 * j)) & 14) == 2;
			const bool passes = src == SRC_PREVIOUS && opv == 0 && fn == FN_REPLACE && sc == 0;
			if (!passes && (ch != CH_A || rgbReadsKA)) continue;
			if (!passes) {
				last.op_alpha = 0;
				last.func_alpha = FN_REPLACE;
				last.scale_alpha = 0;
			}
			src = SRC_CONSTANT;
			if (ch == CH_RGB)
				last.color = (last.color & 0xFF000000u) | uint32_t(clampByte(l.v[0])) | uint32_t(clampByte(l.v[1])) << 8 |
				             uint32_t(clampByte(l.v[2])) << 16;
			else
				last.color = (last.color & 0x00FFFFFFu) | uint32_t(clampByte(l.v[0])) << 24;
			prevOcc[ch] = outReg[ch];
		}
	}
	if (n == 0 || prevOcc[CH_RGB] != outReg[CH_RGB] || prevOcc[CH_A] != outReg[CH_A]) {
		if (n == PICA_MAX_STAGES) {
			tevLogOnce("more than 6 combiner stages", N - 1, N, CE[N - 1], AE[N - 1]);
			return false;
		}
		PicaTevStage& ps = st.stage[n++];
		memset(&ps, 0, sizeof ps);
		int16_t kv[4] = {0, 0, 0, 0};
		for (int ch = 0; ch < 2; ch++) {
			const TevLoc& l = loc[outReg[ch]][ch];
			uint8_t src = SRC_PREVIOUS;
			if (l.kind == LOC_LOST) {
				tevLogOnce("output value lost", N - 1, N, CE[N - 1], AE[N - 1]);
				return false;
			}
			if (l.kind == LOC_BUF) src = SRC_BUFFER;
			if (l.kind == LOC_CONST) {
				src = SRC_CONSTANT;
				if (ch == CH_RGB) memcpy(kv, l.v, sizeof l.v);
				else kv[3] = l.v[0];
			}
			if (ch == CH_RGB) ps.src_rgb = src;
			else ps.src_alpha = src;
		}
		ps.color = uint32_t(clampByte(kv[0])) | uint32_t(clampByte(kv[1])) << 8 | uint32_t(clampByte(kv[2])) << 16 |
		           uint32_t(clampByte(kv[3])) << 24;
	}
	if (bufColorSet[CH_RGB])
		st.buf_color |= uint32_t(clampByte(bufColor[CH_RGB].v[0])) | uint32_t(clampByte(bufColor[CH_RGB].v[1])) << 8 |
		                uint32_t(clampByte(bufColor[CH_RGB].v[2])) << 16;
	if (bufColorSet[CH_A]) st.buf_color |= uint32_t(clampByte(bufColor[CH_A].v[0])) << 24;
	st.nstages = uint8_t(n);
	return true;
}

static void resetTev(PicaState& st, TevSetup& t) {
	memset(st.stage, 0, sizeof st.stage);
	st.buf_rgb = st.buf_alpha = 0;
	st.buf_color = 0;
	st.nstages = 0;
	t.nUnits = 0;
}

// translateTev, and when the setup cannot be expressed, approximations of it, in order: scale
// 1/2 drawn at 1; past three textures, one of them read as white; the first k stages in full
// and the rest passing the colour through (keeping their alpha, then not), for the largest k.
static bool translateTevFallback(TevSetup& tev, const bool pwhite[kMaxPairs], PicaState& st) {
	if (translateTev(tev, pwhite, st)) return true;
	const TevSetup full = tev;
	bool ok = false;
	s_tevHalfExact = false;
	resetTev(st, tev);
	if (translateTev(tev, pwhite, st)) {
		tevLogOnce("scale 1/2 drawn at 1 (stages)", 0, full.numStages, full.colorEnv[0], full.alphaEnv[0]);
		ok = true;
	}
	bool white[kMaxPairs];
	memcpy(white, pwhite, sizeof white);
	int nTex = 0;
	for (int p = 0; p < full.nPairs; p++) nTex += !pwhite[p];
	// Past three textures, one read as its average colour (pairConst): small ones first (lighting
	// ramps, up to 32x32), then trailing stages dropped, then any texture (an EFB copy last).
	int order[kMaxPairs], nOrder = 0;
	for (int p = 0; p < full.nPairs; p++)
		if (!pwhite[p]) order[nOrder++] = p;
	std::sort(order, order + nOrder, [&](int a, int b) {
		const uint32_t sa = full.pairTexels[a] ? full.pairTexels[a] : ~0u, sb = full.pairTexels[b] ? full.pairTexels[b] : ~0u;
		return sa != sb ? sa < sb : a > b;
	});
	// Selectors: pairs a compare reads, or a lerp's factor between two inputs. Averaged, they
	// would pick one side everywhere (often the dark one); they read as white instead.
	bool selector[kMaxPairs] = {};
	for (uint32_t s = 0; s < full.numStages; s++) {
		const int pr = full.stagePair[s];
		if (pr < 0) continue;
		const uint32_t ce = full.colorEnv[s], ae = full.alphaEnv[s];
		const uint32_t ca = (ce >> 12) & 15, cb = (ce >> 8) & 15, cc = (ce >> 4) & 15;
		const uint32_t aa = (ae >> 13) & 7, ab = (ae >> 10) & 7, ac = (ae >> 7) & 7;
		if (((ce >> 16) & 3) == 3 && (ca == 8 || ca == 9 || cb == 8 || cb == 9)) selector[pr] = true;
		if (((ae >> 16) & 3) == 3 && (aa == 4 || ab == 4)) selector[pr] = true;
		if ((cc == 8 || cc == 9) && ca != 15 && cb != 15 && ca != cb) selector[pr] = true;
		if (ac == 4 && aa != 7 && ab != 7 && aa != ab) selector[pr] = true;
	}
	// small: intensity textures (ramps, detail) up to 32x32, not selectors; else any texture.
	auto averaged = [&](bool small) {
		for (int i = 0; i < nOrder && !ok && nTex > PICA_MAX_TEX; i++) {
			const int p = order[i];
			const bool intensity = full.pairFmt[p] <= 3;
			if (small && !(intensity && !selector[p] && full.pairTexels[p] && full.pairTexels[p] <= 1024)) continue;
			white[p] = true;
			tev = full;
			if (selector[p])
				for (int c = 0; c < 4; c++) tev.pairConst[p][c] = 255;
			resetTev(st, tev);
			if ((ok = translateTev(tev, white, st)))
				tevLogOnce(selector[p] ? "drawn with a texture read as white" : "drawn with a texture read as its average",
				           uint32_t(p), full.numStages, full.colorEnv[0], full.alphaEnv[0]);
			else
				white[p] = false;
		}
	};
	averaged(true);
	// The first k stages in full, the rest passing a register through. Which register: one
	// holding colour from a colour texture (not an intensity ramp or mask) as the first k stages
	// leave it, the material's output register first, then the latest written; else the output
	// register, else the one the k-th stage writes.
	const uint32_t outC = (full.colorEnv[full.numStages - 1] >> 22) & 3, outA = (full.alphaEnv[full.numStages - 1] >> 22) & 3;
	for (int k = int(full.numStages) - 1; k >= 1 && !ok; k--) {
		bool coloured[4] = {false, false, false, false};
		int lastWrite[4] = {-1, -1, -1, -1};
		for (int s = 0; s < k; s++) {
			const uint32_t ce = full.colorEnv[s];
			const int pr = full.stagePair[s];
			bool c = false;
			for (int in = 0; in < 4; in++) {
				const uint32_t v = (ce >> (4 * in)) & 15;
				if (v < 8) c |= coloured[v >> 1];
				else if (v == 8 && pr >= 0 && !white[pr] && full.pairFmt[pr] > 3) c = true;
			}
			const int d = int((ce >> 22) & 3);
			coloured[d] = c;
			lastWrite[d] = s;
		}
		uint32_t cand[6];
		int nc = 0;
		if (coloured[outC]) cand[nc++] = outC;
		for (int n = 0; n < 4; n++) {  // the other coloured registers, latest written first
			int best = -1;
			for (int r = 0; r < 4; r++) {
				bool used = false;
				for (int q = 0; q < nc; q++) used |= cand[q] == uint32_t(r);
				if (!used && coloured[r] && (best < 0 || lastWrite[r] > lastWrite[best])) best = r;
			}
			if (best >= 0) cand[nc++] = uint32_t(best);
		}
		if (!nc) {
			cand[nc++] = outC;
			const uint32_t lc = (full.colorEnv[k - 1] >> 22) & 3;
			if (lc != outC) cand[nc++] = lc;
		}
		for (int ci = 0; ci < nc && !ok; ci++) {
			for (int keepAlpha = 1; keepAlpha >= 0 && !ok; keepAlpha--) {
				tev = full;
				const uint32_t keepC = cand[ci], keepA = keepAlpha ? outA : (full.alphaEnv[k - 1] >> 22) & 3;
				for (uint32_t s = uint32_t(k); s < full.numStages; s++) {
					tev.colorEnv[s] = 0x08FFF0 | 2 * keepC;
					if (!keepAlpha) tev.alphaEnv[s] = 0x08FF80 | keepA << 4;
				}
				resetTev(st, tev);
				if ((ok = translateTev(tev, white, st)))
					tevLogOnce(keepAlpha ? "drawn with the colour of its first stages" : "drawn with its first stages only",
					           uint32_t(k), full.numStages, full.colorEnv[0], full.alphaEnv[0]);
			}
		}
	}
	averaged(false);
	s_tevHalfExact = true;
	if (ok) {
		memcpy(tev.colorEnv, full.colorEnv, sizeof tev.colorEnv);  // as the game set them, for the dump
		memcpy(tev.alphaEnv, full.alphaEnv, sizeof tev.alphaEnv);
	}
	return ok;
}

// The setup checked against GX's arithmetic (gx3ds_tevsym.cpp, tevSymError) and, when the
// translation is off by more than kTevSymAccept, compiled through its algebra (tevSymCompile);
// the more accurate one is drawn. The decision is kept per setup with its constants left out
// (tevStructKey): later setups that differ only there (colours animated every frame) reuse the
// translation without a check, or compile with the variant chosen, without the search.
static const int kTevSymAccept = 24, kTevSymMargin = 8;  // thousandths (as tevSymError)
static std::unordered_map<uint64_t, int8_t> s_tevSymChoice;  // structure -> variant (-1: the translation)

static uint64_t tevStructKey(const TevSetup& t, const bool pwhite[kMaxPairs]) {
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { h = (h ^ x) * 1099511628211ull; };
	mix(t.numStages);
	for (uint32_t s = 0; s < t.numStages; s++) {
		mix(t.colorEnv[s]);
		mix(t.alphaEnv[s]);
		mix(t.order[s]);
		mix(t.ksel[s]);
	}
	for (int i = 0; i < 4; i++) mix(t.swap[i]);
	uint32_t flags = 0;
	for (int p = 0; p < t.nPairs; p++) flags |= uint32_t(pwhite[p]) << p;
	for (int c = 0; c < 2; c++)
		flags |= uint32_t(t.rgbVarying[c] | t.aVarying[c] << 1 | t.chanSpec[c] << 2) << (8 + 4 * c);
	mix(flags | uint32_t(t.nPairs) << 16 | uint32_t(t.rasRgbChan) << 20 | uint32_t(t.rasAChan) << 22);
	return h;
}

static void tevSymImprove(TevSetup& tev, const bool pwhite[kMaxPairs], PicaState& st, bool& translated) {
	tev.symErrOld = tev.symErrNew = -1;
	tev.symUsed = false;
	if (tev.numStages < 2 || tev.nPairs > kTevSymPairs) return;
	TevSymIn in;
	memset(&in, 0, sizeof in);
	in.numStages = int(tev.numStages);
	for (uint32_t s = 0; s < tev.numStages; s++) {
		in.ce[s] = tev.colorEnv[s];
		in.ae[s] = tev.alphaEnv[s];
		in.order[s] = tev.order[s];
		in.ksel[s] = tev.ksel[s];
		in.stagePair[s] = tev.stagePair[s];
	}
	for (int i = 0; i < 4; i++) in.swap[i] = uint8_t(tev.swap[i]);
	for (int r = 0; r < 4; r++) {
		const uint32_t ra = rs.bp[BP_TEV_REG + 2 * r], bg = rs.bp[BP_TEV_REG + 2 * r + 1];
		in.reg[r][0] = int16_t(sext11(ra & 0x7FF));
		in.reg[r][1] = int16_t(sext11((bg >> 12) & 0x7FF));
		in.reg[r][2] = int16_t(sext11(bg & 0x7FF));
		in.reg[r][3] = int16_t(sext11((ra >> 12) & 0x7FF));
		const uint32_t ka = rs.kreg[2 * r], kb = rs.kreg[2 * r + 1];
		in.konst[r][0] = uint8_t(ka & 0xFF);
		in.konst[r][1] = uint8_t((kb >> 12) & 0xFF);
		in.konst[r][2] = uint8_t(kb & 0xFF);
		in.konst[r][3] = uint8_t((ka >> 12) & 0xFF);
	}
	in.nPairs = tev.nPairs;
	for (int p = 0; p < tev.nPairs; p++) {
		in.pairConst[p] = pwhite[p];
		for (int c = 0; c < 4; c++) in.pairValue[p][c] = tev.pairConst[p][c];
	}
	in.rasRgbChan = tev.rasRgbChan;
	in.rasAChan = tev.rasAChan;
	for (int c = 0; c < 2; c++) {
		in.rgbVarying[c] = tev.rgbVarying[c];
		in.aVarying[c] = tev.aVarying[c];
		in.chanSpec[c] = tev.chanSpec[c];
		for (int k = 0; k < 4; k++) in.rasConst[c][k] = tev.rasConst[c][k];
	}
	TevSymOut out;
	auto adopt = [&] {
		memcpy(st.stage, out.stage, sizeof st.stage);
		st.nstages = uint8_t(out.nstages);
		st.buf_rgb = out.buf_rgb;
		st.buf_alpha = out.buf_alpha;
		st.buf_color = out.buf_color;
		tev.nUnits = out.nUnits;
		for (int u = 0; u < PICA_MAX_TEX; u++) tev.unitPair[u] = u < out.nUnits ? out.unitPair[u] : 0;
		translated = true;
		tev.symUsed = true;
		s_tevSymUsed++;
	};
	const uint64_t key = tevStructKey(tev, pwhite);
	auto it = s_tevSymChoice.find(key);
	if (it != s_tevSymChoice.end()) {
		if (it->second < 0) return;
		if (tevSymCompileChoice(in, it->second, out)) {
			adopt();
			return;
		}
	}
	TevSymOut old;
	memset(&old, 0, sizeof old);
	memcpy(old.stage, st.stage, sizeof old.stage);
	old.nstages = st.nstages;
	old.buf_rgb = st.buf_rgb;
	old.buf_alpha = st.buf_alpha;
	old.buf_color = st.buf_color;
	old.nUnits = tev.nUnits;
	for (int u = 0; u < PICA_MAX_TEX; u++) old.unitPair[u] = tev.unitPair[u];
	const int eOld = translated ? tevSymError(in, old) : 1 << 14;
	s_tevSymChecks++;
	tev.symErrOld = int16_t(eOld);
	int choice = -1;
	if (eOld > kTevSymAccept) {
		int eNew;
		s_tevSymCompiles++;
		if (tevSymCompile(in, out, &eNew, &choice) && eNew + kTevSymMargin < eOld) {
			tev.symErrNew = int16_t(eNew);
			adopt();
		} else {
			choice = -1;
		}
	}
	if (s_tevSymChoice.size() >= 4096) s_tevSymChoice.clear();
	s_tevSymChoice[key] = int8_t(choice);
}

// THP movies (THPPlayer's THPGXYuv2RgbSetup) convert Y, U, V textures to RGB through signed,
// unclamped TEV intermediates, which PICA stages (clamped to 0..1) cannot hold. Per component,
// the setup computes out = y + off + cu * u + cv * v (off, cu, cv from C0 and konst 0-1; green
// through the alpha channel); here d = off + cu * u + cv * v is built as 0.5 + d / 2 from two
// interpolations whose endpoints stay in range, and applied as two halves with ADD_SIGNED
// (same-sign steps, so the clamp in between is harmless).
static bool isThpSetup(const TevSetup& t) {
	return t.numStages == 4 && (t.colorEnv[0] & 0xFFFF) == 0xF8E2 && (t.colorEnv[1] & 0xFFFF) == 0xF8E0 &&
	       (t.colorEnv[2] & 0xFFFF) == 0xF8C0 && (t.colorEnv[3] & 0xFFFF) == 0x10EF && t.stagePair[0] >= 0 &&
	       t.stagePair[1] >= 0 && t.stagePair[2] >= 0 && t.stagePair[0] != t.stagePair[1] &&
	       t.stagePair[1] != t.stagePair[2] && t.stagePair[0] != t.stagePair[2];
}

static void translateThp(TevSetup& t, PicaState& st) {
	// Units 0-2: U, V, Y (the textures of stages 0-2).
	t.nUnits = 3;
	for (int u = 0; u < 3; u++) t.unitPair[u] = t.stagePair[u];
	auto kc = [](uint32_t i, int comp) {  // konst i, comp 0-3: r, g, b, a (0..1)
		const uint32_t ra = rs.kreg[2 * i], bg = rs.kreg[2 * i + 1];
		return float(comp == 0 ? ra & 0xFF : comp == 1 ? (bg >> 12) & 0xFF : comp == 2 ? bg & 0xFF : (ra >> 12) & 0xFF) /
		       255.f;
	};
	const uint32_t ra = rs.bp[BP_TEV_REG + 2], bg = rs.bp[BP_TEV_REG + 3];  // C0
	const float c0r = sext11(ra & 0x7FF) / 255.f, c0b = sext11(bg & 0x7FF) / 255.f, a0 = sext11((ra >> 12) & 0x7FF) / 255.f;
	const float off[3] = {2 * c0r, a0, 2 * c0b};
	const float cu[3] = {2 * kc(0, 0), -kc(0, 3), 2 * kc(0, 2)};
	const float cv[3] = {2 * kc(1, 0), -kc(1, 3), 2 * kc(1, 2)};
	uint8_t duLo[3], duHi[3], evLo[3], evHi[3];
	auto byte = [](float x) { return clampByte(int(x * 255.f + 0.5f)); };
	for (int i = 0; i < 3; i++) {
		const float beta = off[i] / 2 + 0.25f * cu[i];
		duLo[i] = byte(0.5f - 0.25f * cu[i]);
		duHi[i] = byte(0.5f + 0.25f * cu[i]);
		evLo[i] = byte(0.5f + beta);
		evHi[i] = byte(0.5f + beta + 0.5f * cv[i]);
	}
	auto rgb = [](const uint8_t c[3]) { return uint32_t(c[0]) | uint32_t(c[1]) << 8 | uint32_t(c[2]) << 16; };
	const uint8_t texU = SRC_TEX0, texV = SRC_TEX0 + 1, texY = SRC_TEX0 + 2;
	auto stage = [&](int i, uint8_t func, uint8_t s0, uint8_t s1, uint8_t s2, uint32_t color) {
		PicaTevStage& ps = st.stage[i];
		memset(&ps, 0, sizeof ps);
		ps.src_rgb = uint16_t(s0 | s1 << 4 | s2 << 8);
		ps.func_rgb = func;
		ps.src_alpha = SRC_CONSTANT;  // the setup's alpha is 0
		ps.func_alpha = FN_REPLACE;
		ps.color = color;
	};
	// 0-1: E = interpolate(evLo, evHi, v), kept in the buffer (from stage 3).
	stage(0, FN_REPLACE, SRC_CONSTANT, 0, 0, rgb(evLo));
	stage(1, FN_INTERPOLATE, SRC_CONSTANT, SRC_PREVIOUS, texV, rgb(evHi));
	// 2: D_u = interpolate(duLo, duHi, u), duLo from the buffer's initial colour.
	stage(2, FN_INTERPOLATE, SRC_CONSTANT, SRC_BUFFER, texU, rgb(duHi));
	// 3: 0.5 + d / 2 = D_u + E - 0.5, kept in the buffer (from stage 5).
	stage(3, FN_ADD_SIGNED, SRC_PREVIOUS, SRC_BUFFER, 0, 0);
	// 4-5: y + d / 2, then + d / 2.
	stage(4, FN_ADD_SIGNED, texY, SRC_PREVIOUS, 0, 0);
	stage(5, FN_ADD_SIGNED, SRC_PREVIOUS, SRC_BUFFER, 0, 0);
	st.nstages = 6;
	st.buf_rgb = uint8_t(1u << 1 | 1u << 3);
	st.buf_alpha = 0;
	st.buf_color = rgb(duLo);
}

// A movie on a surface: THP's four conversion stages (as isThpSetup) followed by the material's
// own stages (Delfino Plaza's warp gates, TModelGate: the movie inside the gate's shape, from a
// frame texture and a mask; 7 stages and 5 textures). The conversion alone takes all six PICA
// stages and all three textures, so the fallback dropped the stages after it, and with them the
// mask's alpha that the gate's alpha compare reads: the gates were invisible. Here the frame is
// converted to RGB on the CPU (thpRgbTexture) and the four stages become one that samples it,
// placed where the material first reads or writes PREV (the gate's frame and mask stages first,
// so that its last stage finds the movie in PREV and the two in the buffer).
static bool isThpPrefix(const TevSetup& t) {
	if (t.numStages <= 4) return false;
	TevSetup h = t;
	h.numStages = 4;
	if (!isThpSetup(h)) return false;
	for (int s = 0; s < 4; s++)  // the conversion writes PREV only, through swap table 0
		if (((t.colorEnv[s] >> 22) & 3) || ((t.alphaEnv[s] >> 22) & 3) || (t.alphaEnv[s] & 15)) return false;
	if (((t.alphaEnv[3] >> 4) & 0xFFF) != 0xFFF) return false;  // ... and alpha 0
	for (uint32_t s = 4; s < t.numStages; s++)
		for (int k = 0; k < 3; k++)
			if (t.stagePair[s] >= 0 && t.stagePair[s] == t.stagePair[k]) return false;
	return true;
}

// The frame of the planes isThpPrefix's stages read, as an RGB texture the size of the Y plane:
// out = y + off + cu * u + cv * v per component (as translateThp), clamped. Converted when the
// planes or the constants change, once a frame at most; the last few frames' textures are kept
// (the player's three texture sets take turns).
struct ThpRgb {
	const uint8_t* y;
	uint64_t hash;
	uint32_t frame, pw, ph;
	PicaTex* tex;
};
static ThpRgb s_thpRgb[4];
static std::vector<uint8_t> s_thpPlane[3], s_thpOut;

static PicaTex* thpRgbTexture(const TevSetup& t) {
	const int map[3] = {t.pairMap[t.stagePair[0]], t.pairMap[t.stagePair[1]], t.pairMap[t.stagePair[2]]};  // U, V, Y
	const uint8_t* src[3];
	uint32_t w[3], h[3];
	for (int i = 0; i < 3; i++) {
		const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, map[i])];
		src[i] = rs.texImage[map[i]];
		w[i] = (img0 & 0x3FF) + 1;
		h[i] = ((img0 >> 10) & 0x3FF) + 1;
		if (!src[i] || ((img0 >> 20) & 15) != 1) return nullptr;  // I8 planes
	}
	const uint32_t ra = rs.bp[BP_TEV_REG + 2], bg = rs.bp[BP_TEV_REG + 3];  // C0
	const uint32_t konst[4] = {rs.kreg[0], rs.kreg[1], rs.kreg[2], rs.kreg[3]};
	ThpRgb* e = nullptr;
	for (ThpRgb& c : s_thpRgb)
		if (c.tex && c.y == src[2]) e = &c;
	if (e && e->frame == s_rFrames) return e->tex;
	uint64_t hash = hashBytes(konst, sizeof konst) ^ (uint64_t(ra) << 32 | bg);
	for (int i = 0; i < 3; i++) hash = hash * 0x100000001B3ull ^ hashBytes(src[i], texLevelBytes(1, w[i], h[i]));
	const uint32_t pw = pow2Side(w[2]), ph = pow2Side(h[2]);
	if (e && e->hash == hash && e->pw == pw && e->ph == ph) {
		e->frame = s_rFrames;
		return e->tex;
	}
	if (!e) {  // the least recently used
		e = &s_thpRgb[0];
		for (ThpRgb& c : s_thpRgb)
			if (!c.tex || c.frame < e->frame) e = &c;
	}
	if (e->tex && (e->pw != pw || e->ph != ph)) {
		s_texBytes -= pica_tex_bytes(e->tex);
		pica_tex_free(e->tex);
		e->tex = nullptr;
	}
	if (!e->tex) {
		e->tex = pica_tex_create_fmt(pw, ph, PICA_TF_RGB565);
		if (!e->tex) return nullptr;
		s_texBytes += pica_tex_bytes(e->tex);
	}
	e->y = src[2];
	e->hash = hash;
	e->frame = s_rFrames;
	e->pw = pw;
	e->ph = ph;
	for (int i = 0; i < 3; i++) {
		s_thpPlane[i].resize(size_t(w[i]) * h[i] * 4);
		decodeTexture(src[i], 1, w[i], h[i], nullptr, 0, s_thpPlane[i].data());
	}
	auto kc = [&](uint32_t i, int comp) {  // konst i, comp 0-3: r, g, b, a (0..1)
		const uint32_t kra = konst[2 * i], kbg = konst[2 * i + 1];
		return float(comp == 0 ? kra & 0xFF : comp == 1 ? (kbg >> 12) & 0xFF : comp == 2 ? kbg & 0xFF : (kra >> 12) & 0xFF) /
		       255.f;
	};
	const float c0r = sext11(ra & 0x7FF) / 255.f, c0b = sext11(bg & 0x7FF) / 255.f, a0 = sext11((ra >> 12) & 0x7FF) / 255.f;
	const float off[3] = {2 * c0r, a0, 2 * c0b};
	const float cu[3] = {2 * kc(0, 0), -kc(0, 3), 2 * kc(0, 2)};
	const float cv[3] = {2 * kc(1, 0), -kc(1, 3), 2 * kc(1, 2)};
	// Per component, the u and v terms by texel value (with the offset and rounding in the u
	// table), in 1/255 units: out = y + tu[u] + tv[v].
	int16_t tu[3][256], tv[3][256];
	for (int c = 0; c < 3; c++)
		for (int i = 0; i < 256; i++) {
			tu[c][i] = int16_t(lroundf((off[c] + cu[c] * (i / 255.f)) * 255.f));
			tv[c][i] = int16_t(lroundf(cv[c] * (i / 255.f) * 255.f));
		}
	s_thpOut.resize(size_t(pw) * ph * 4);
	uint32_t lastSy = ~0u;
	for (uint32_t y = 0; y < ph; y++) {
		uint8_t* row = &s_thpOut[size_t(y) * pw * 4];
		const uint32_t sy = y * h[2] / ph;
		if (sy == lastSy) {  // the power-of-two height repeats a row
			memcpy(row, row - pw * 4, pw * 4);
			continue;
		}
		lastSy = sy;
		const uint8_t* yr = &s_thpPlane[2][size_t(sy) * w[2] * 4];
		const uint8_t* ur = &s_thpPlane[0][size_t(sy * h[0] / h[2]) * w[0] * 4];
		const uint8_t* vr = &s_thpPlane[1][size_t(sy * h[1] / h[2]) * w[1] * 4];
		for (uint32_t x = 0; x < pw; x++) {
			const uint32_t sx = x * w[2] / pw;
			const int Y = yr[sx * 4], U = ur[(sx * w[0] / w[2]) * 4], V = vr[(sx * w[1] / w[2]) * 4];
			uint8_t* o = row + x * 4;
			for (int c = 0; c < 3; c++) o[c] = clampByte(Y + tu[c][U] + tv[c][V]);
			o[3] = 255;
		}
	}
	pica_tex_upload(e->tex, s_thpOut.data());
	s_texUploads++;
	return e->tex;
}

// isThpPrefix's stages with the conversion as one stage sampling the RGB frame through the Y
// plane's pair (its texture made the frame by the caller).
static void thpPrefixToTexture(TevSetup& t) {
	const TevSetup o = t;
	uint32_t at = 4;  // the first stage that reads or writes PREV
	for (; at < o.numStages; at++) {
		const uint32_t ce = o.colorEnv[at], ae = o.alphaEnv[at];
		bool prev = ((ce >> 22) & 3) == 0 || ((ae >> 22) & 3) == 0;
		for (int i = 0; i < 4; i++) prev |= ((ce >> (4 * i)) & 15) <= 1 || ((ae >> (4 + 3 * i)) & 7) == 0;
		if (prev) break;
	}
	uint32_t n = 0;
	auto put = [&](uint32_t s) {
		t.colorEnv[n] = o.colorEnv[s];
		t.alphaEnv[n] = o.alphaEnv[s];
		t.order[n] = o.order[s];
		t.ksel[n] = o.ksel[s];
		t.stagePair[n] = o.stagePair[s];
		n++;
	};
	for (uint32_t s = 4; s < at; s++) put(s);
	t.colorEnv[n] = 0x88FFF;  // PREV.rgb = TEXC, clamped
	t.alphaEnv[n] = 0x8FFF0;  // PREV.a = 0
	t.order[n] = (o.order[2] & 0x3F) | 1u << 6 | 7u << 7;  // the Y plane's map and coordinate, no colour channel
	t.ksel[n] = o.ksel[2];
	t.stagePair[n] = o.stagePair[2];
	n++;
	for (uint32_t s = at; s < o.numStages; s++) put(s);
	t.numStages = n;
	t.pairFmt[o.stagePair[2]] = 6;  // a colour texture now (RGBA8 for the fallback's choices)
}

// The goop stamp (TPollutionCounterLayer::drawTexStamp, initGXforStamp): a texel of the stamp's
// shape counts where the height map lies strictly between two registers,
//   0: A_k = h > A_x ? 1 : 0   1: P.a = A_y > h ? A_k : 0   2: P = (stamp rgb or 1, stamp a * P.a)
// with h the height map's alpha. The generic translation runs out of stages and draws each
// compare as a 32x ramp, which leaves a stamp at one or two height steps nearly transparent
// (spraying cleans nothing). Here both compares run side by side, h - A_x in the colour half and
// A_y - h in the alpha half, each a full 256x ramp (one step saturates).
static bool isStampSetup(const TevSetup& t) {
	if (t.numStages != 3 || t.stagePair[0] < 0 || t.stagePair[0] != t.stagePair[1] || t.stagePair[2] < 0 ||
	    t.stagePair[2] == t.stagePair[0] || (t.swap[0] & 0xFF) != 0xE4)
		return false;
	auto f = [](uint32_t e, int lo, int n) { return (e >> lo) & ((1u << n) - 1); };
	const uint32_t a0 = t.alphaEnv[0], a1 = t.alphaEnv[1], a2 = t.alphaEnv[2], c2 = t.colorEnv[2];
	for (int s = 0; s < 3; s++)
		if (f(t.alphaEnv[s], 0, 4) != 0) return false;  // swap table 0 for both
	// 0: d 0, a TEXA, b A_x, c KONST (1), A8 greater, into A_k.
	const uint32_t k = f(a0, 22, 2), x = f(a0, 10, 3);
	if (f(a0, 4, 3) != 7 || f(a0, 13, 3) != 4 || x < 1 || x > 3 || f(a0, 7, 3) != 6 || f(t.ksel[0], 5, 5) != 0 ||
	    f(a0, 16, 2) != 3 || (f(a0, 20, 2) << 1 | f(a0, 18, 1)) != 6 || k == 0)
		return false;
	// 1: d 0, a A_y, b TEXA, c A_k, A8 greater, into PREV.
	const uint32_t y = f(a1, 13, 3);
	if (f(a1, 4, 3) != 7 || f(a1, 10, 3) != 4 || y < 1 || y > 3 || y == k || f(a1, 7, 3) != k || f(a1, 16, 2) != 3 ||
	    (f(a1, 20, 2) << 1 | f(a1, 18, 1)) != 6 || f(a1, 22, 2) != 0)
		return false;
	// 2: alpha TEXA * APREV, colour TEXC or ONE, into PREV.
	const bool mul = (f(a2, 10, 3) == 4 && f(a2, 7, 3) == 0) || (f(a2, 10, 3) == 0 && f(a2, 7, 3) == 4);
	if (f(a2, 4, 3) != 7 || f(a2, 13, 3) != 7 || !mul || f(a2, 16, 3) != 0 || f(a2, 20, 2) != 0 || f(a2, 22, 2) != 0)
		return false;
	const uint32_t ca = f(c2, 12, 4);
	return (c2 & 0xFFF) == 0xFFF && (ca == 8 || ca == 12) && f(c2, 16, 3) == 0 && f(c2, 20, 2) == 0 &&
	       f(c2, 22, 2) == 0;
}

static void translateStamp(TevSetup& t, PicaState& st) {
	// Units 0, 1: the height map, the stamp.
	t.nUnits = 2;
	t.unitPair[0] = t.stagePair[0];
	t.unitPair[1] = t.stagePair[2];
	auto regA = [](uint32_t r) { return clampByte(sext11((rs.bp[BP_TEV_REG + 2 * r] >> 12) & 0x7FF)); };
	const uint32_t ax = regA((t.alphaEnv[0] >> 10) & 7), ay = regA((t.alphaEnv[1] >> 13) & 7);
	const bool white = ((t.colorEnv[2] >> 12) & 15) == 12;
	const uint8_t hmap = SRC_TEX0, stamp = SRC_TEX0 + 1;
	enum : uint16_t { OPC_ALPHA = 2, OPA_RED = 2 };
	auto stage = [&](int i) -> PicaTevStage& {
		PicaTevStage& ps = st.stage[i];
		memset(&ps, 0, sizeof ps);
		return ps;
	};
	// 0: (h - A_x) * 4 in rgb, (A_y - h) * 4 in alpha.
	PicaTevStage& s0 = stage(0);
	s0.src_rgb = uint16_t(hmap | SRC_CONSTANT << 4);
	s0.op_rgb = OPC_ALPHA;
	s0.func_rgb = FN_SUBTRACT;
	s0.scale_rgb = 2;
	s0.src_alpha = uint16_t(SRC_CONSTANT | hmap << 4);
	s0.func_alpha = FN_SUBTRACT;
	s0.scale_alpha = 2;
	s0.color = ax | ax << 8 | ax << 16 | ay << 24;
	// 1-2: doubled and scaled by 4, twice: 256x, one height step saturates.
	for (int i = 1; i <= 2; i++) {
		PicaTevStage& s = stage(i);
		s.src_rgb = uint16_t(SRC_PREVIOUS | SRC_PREVIOUS << 4);
		s.src_alpha = uint16_t(SRC_PREVIOUS | SRC_PREVIOUS << 4);
		s.func_rgb = s.func_alpha = FN_ADD;
		s.scale_rgb = s.scale_alpha = 2;
	}
	// 3: the stamp's colour; both conditions.
	PicaTevStage& s3 = stage(3);
	s3.src_rgb = white ? SRC_CONSTANT : stamp;
	s3.color = white ? 0xFFFFFFu : 0;
	s3.src_alpha = uint16_t(SRC_PREVIOUS | SRC_PREVIOUS << 4);
	s3.op_alpha = uint16_t(OPA_RED);
	s3.func_alpha = FN_MODULATE;
	// 4: times the stamp's alpha.
	PicaTevStage& s4 = stage(4);
	s4.src_rgb = SRC_PREVIOUS;
	s4.src_alpha = uint16_t(SRC_PREVIOUS | stamp << 4);
	s4.func_alpha = FN_MODULATE;
	st.nstages = 5;
	st.buf_rgb = st.buf_alpha = 0;
	st.buf_color = 0;
}

// Logs each distinct TEV setup once (sdmc log), with its textures and translation, to find
// what the game draws with.
// Returns the setup's number in the dump (0: not numbered).
static unsigned tevDump(const TevSetup& t, const PicaState& st, bool ok, PicaTex* const ptex[kMaxPairs],
                        const EfbCopy* const pcopy[kMaxPairs], const char* extra) {
	static std::vector<uint64_t> seen;
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { h = (h ^ x) * 1099511628211ull; };
	mix(t.numStages);
	for (uint32_t s = 0; s < t.numStages; s++) {
		mix(t.colorEnv[s]);
		mix(t.alphaEnv[s]);
		mix(t.order[s]);
		mix(t.ksel[s]);
	}
	for (int i = 0; i < 4; i++) mix(t.swap[i]);
	const uint32_t numInd = (rs.bp[BP_GENMODE] >> 16) & 7;
	mix(numInd);
	mix(ok);
	mix(st.nstages);
	for (int p = 0; p < t.nPairs; p++)
		mix((pcopy[p] ? 2 : ptex[p] ? 1 : 0) | (rs.bp[bpTexReg(BP_TX_IMAGE0, t.pairMap[p])] >> 20 & 15) << 2);
	unsigned id = 0;
	for (size_t i = 0; i < seen.size() && !id; i++)
		if (seen[i] == h) id = unsigned(i + 1);
	if (!id) {
		if (seen.size() >= 400) return 0;
		seen.push_back(h);
		id = unsigned(seen.size());
	} else if (!s_tracing) {
		return id;
	}
	if (s_tracing) {  // each setup once per trace file
		if (std::find(s_traceDumped.begin(), s_traceDumped.end(), id) != s_traceDumped.end()) return id;
		s_traceDumped.push_back(id);
	}
	FILE* const out = s_tracing ? tf() : stderr;
	fprintf(out, "[tevdump] #%u frame %u: %u stages, %u indirect, %s (%u PICA stages, buf %x/%x %08x)\n",
	        id, s_rFrames, t.numStages, numInd, ok ? "translated" : "FALLBACK", st.nstages,
	        st.buf_rgb, st.buf_alpha, (unsigned)st.buf_color);
	if (t.symErrOld >= 0)
		fprintf(out, "[tevdump]   check: translation off by %d/1000%s\n", t.symErrOld,
		        t.symUsed ? ", drawn compiled instead" : t.symErrOld > kTevSymAccept ? ", compiling did no better" : "");
	if (t.symUsed && t.symErrNew >= 0) fprintf(out, "[tevdump]   check: compiled off by %d/1000\n", t.symErrNew);
	if (t.symUsed && t.symErrOld < 0) fprintf(out, "[tevdump]   check: drawn compiled, as decided for the same stages with other constants\n");
	for (uint32_t s = 0; s < t.numStages; s++)
		fprintf(out, "[tevdump]   gx %u: ce %06x ae %06x order %03x ksel %03x\n", s, t.colorEnv[s], t.alphaEnv[s],
		        t.order[s], t.ksel[s]);
	for (int p = 0; p < t.nPairs; p++) {
		const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, t.pairMap[p])];
		int unit = -1;
		for (int u = 0; u < t.nUnits; u++)
			if (t.unitPair[u] == p) unit = u;
		fprintf(out, "[tevdump]   texture %d: map %d coord %d fmt %u %ux%u%s, unit %d\n", p, t.pairMap[p], t.pairCoord[p],
		        (img0 >> 20) & 15, (img0 & 0x3FF) + 1, ((img0 >> 10) & 0x3FF) + 1,
		        pcopy[p] ? " EFB copy" : ptex[p] ? "" : " MISSING", unit);
	}
	fprintf(out, "[tevdump]   regs");
	for (int r = 0; r < 4; r++)
		fprintf(out, " %06x:%06x", rs.bp[BP_TEV_REG + 2 * r] & 0xFFFFFF, rs.bp[BP_TEV_REG + 2 * r + 1] & 0xFFFFFF);
	fprintf(out, " konst");
	for (int r = 0; r < 4; r++) fprintf(out, " %06x:%06x", rs.kreg[2 * r] & 0xFFFFFF, rs.kreg[2 * r + 1] & 0xFFFFFF);
	fprintf(out, " swap %02x %02x %02x %02x ras rgb/a %02d (varying %d %d)\n", t.swap[0], t.swap[1], t.swap[2], t.swap[3],
	        t.rasRgbChan * 10 + t.rasAChan, t.rgbVarying[0] | t.aVarying[0] << 1, t.rgbVarying[1] | t.aVarying[1] << 1);
	fprintf(out, "[tevdump]   %s\n", extra);
	for (int i = 0; i < st.nstages; i++) {
		const PicaTevStage& p = st.stage[i];
		fprintf(out, "[tevdump]   pica %d: src %03x/%03x op %03x/%03x fn %u/%u scale %u/%u const %08x\n", i, p.src_rgb,
		        p.src_alpha, p.op_rgb, p.op_alpha, p.func_rgb, p.func_alpha, p.scale_rgb, p.scale_alpha,
		        (unsigned)p.color);
	}
	return id;
}

// GX alpha compare (BP 0xF3: ref0, ref1, comp0, comp1, op) as PICA tests. The pass set over the
// 256 alpha values is one range ([r,255], [0,r], one value or all but one) for one test, or two
// end ranges ([0,a] and [b,255], the sea's wave layers) for two passes (pica_draw.c).
enum { kNever = 0, kAlways = 1, kEqual = 2, kNotEqual = 3, kLequal = 5, kGequal = 7 };  // GPU_TESTFUNC

static bool gxCompare(uint32_t c, uint32_t a, uint32_t ref) {
	switch (c) {
	case 0: return false;
	case 1: return a < ref;
	case 2: return a == ref;
	case 3: return a <= ref;
	case 4: return a > ref;
	case 5: return a != ref;
	case 6: return a >= ref;
	default: return true;
	}
}

static void alphaTestCompute(PicaState& st, uint32_t r);
// The alpha test for BP 0xF3 (alphaTestCompute): a function of the register alone, so the results
// of the last few values are kept. Working it out tries all 256 alpha values, and it ran for every
// batch (~6 us each, a few ms a frame on the render thread in Delfino Plaza).
static void alphaTest(PicaState& st) {
	const uint32_t r = rs.bp[BP_ALPHACOMPARE] & 0xFFFFFF;
	struct Memo {
		uint32_t r;  // the register, bit 31 set once filled
		uint8_t test, func, ref, test2, func2, ref2;
	};
	static Memo memo[16];
	Memo& m = memo[(r ^ (r >> 7) ^ (r >> 15)) & 15];
	if (m.r == (r | 0x80000000u)) {
		st.alpha_test = m.test, st.alpha_func = m.func, st.alpha_ref = m.ref;
		st.alpha_test2 = m.test2, st.alpha_func2 = m.func2, st.alpha_ref2 = m.ref2;
		return;
	}
	alphaTestCompute(st, r);
	m = Memo{r | 0x80000000u, st.alpha_test, st.alpha_func, st.alpha_ref, st.alpha_test2, st.alpha_func2, st.alpha_ref2};
}
static void alphaTestCompute(PicaState& st, uint32_t r) {
	const uint32_t ref0 = r & 0xFF, ref1 = (r >> 8) & 0xFF, c0 = (r >> 16) & 7, c1 = (r >> 19) & 7, op = (r >> 22) & 3;
	bool pass[256];
	int n = 0;
	for (uint32_t a = 0; a < 256; a++) {
		const bool p0 = gxCompare(c0, a, ref0), p1 = gxCompare(c1, a, ref1);
		pass[a] = op == 0 ? p0 && p1 : op == 1 ? p0 || p1 : op == 2 ? p0 != p1 : p0 == p1;
		n += pass[a];
	}
	st.alpha_test2 = 0;
	if (n == 256) {
		st.alpha_test = 0;
		st.alpha_func = kAlways;
		st.alpha_ref = 0;
		return;
	}
	st.alpha_test = 1;
	if (n == 0) {
		st.alpha_func = kNever;
		st.alpha_ref = 0;
		return;
	}
	int lo = 0, hi = 255;  // pass[0..lo) and (hi..255] are the end runs
	while (lo < 256 && pass[lo]) lo++;
	while (hi >= 0 && pass[hi]) hi--;
	int mid = 0;  // passing values between the end runs
	for (int a = lo; a <= hi; a++) mid += pass[a];
	if (mid == 0 && lo == 0) {  // [hi+1, 255]
		st.alpha_func = kGequal;
		st.alpha_ref = uint8_t(hi + 1);
	} else if (mid == 0 && hi == 255) {  // [0, lo-1]
		st.alpha_func = kLequal;
		st.alpha_ref = uint8_t(lo - 1);
	} else if (mid == 0) {  // [0, lo-1] and [hi+1, 255]
		st.alpha_func = kLequal;
		st.alpha_ref = uint8_t(lo - 1);
		st.alpha_test2 = 1;
		st.alpha_func2 = kGequal;
		st.alpha_ref2 = uint8_t(hi + 1);
	} else if (n == 1) {
		int v = 0;
		while (!pass[v]) v++;
		st.alpha_func = kEqual;
		st.alpha_ref = uint8_t(v);
	} else if (n == 255) {
		st.alpha_func = kNotEqual;
		st.alpha_ref = uint8_t(lo);
	} else {  // a middle range: the first compare alone, as before
		st.alpha_test = c0 != 7;
		st.alpha_func = kTestFunc[c0];
		st.alpha_ref = uint8_t(ref0);
	}
}

// Texture coordinate `coord` of a vertex: texgen (XF 0x1040 + coord) with its texture matrix,
// and the dual-texture post matrix when enabled, as sms-pc-port's vertex shader computes it.
// q: a projective coordinate is left undivided, (s q, t q), and q stored there (1 for others).
static void texCoord(int coord, const uint8_t* v, const VtxFmtLayout& L, uint32_t fmt, const uint8_t defTex[8],
                     float& u, float& w, float* q = nullptr) {
	const uint32_t tg = rs.xfReg[XFR_TEXGEN + coord];
	const uint32_t proj = (tg >> 1) & 1, form = (tg >> 2) & 1, type = (tg >> 4) & 7, row = (tg >> 7) & 31;
	float sv[4] = {0, 0, 1, 1};
	if (row == 0) {
		memcpy(sv, v, 12);
	} else if (row == 1 && (fmt & VF_NRM)) {
		memcpy(sv, v + L.nrm, 12);
	} else if (row >= 5 && row <= 12 && (fmt & (VF_TEX0 << (row - 5)))) {
		memcpy(sv, v + L.tex[row - 5], 8);
	}
	if (form == 0 && row < 5) sv[2] = 1.0f;  // AB11
	if (type != 0) {  // emboss and colour texgens: not supported, pass the source through
		u = sv[0];
		w = sv[1];
		return;
	}
	uint32_t tm = (fmt & VF_MTX) ? v[L.mtx + 1 + coord] : defTex[coord];
	const float* m = reinterpret_cast<const float*>(&rs.xfMem[tm * 4]);
	float t0 = m[0] * sv[0] + m[1] * sv[1] + m[2] * sv[2] + m[3] * sv[3];
	float t1 = m[4] * sv[0] + m[5] * sv[1] + m[6] * sv[2] + m[7] * sv[3];
	float t2 = proj ? m[8] * sv[0] + m[9] * sv[1] + m[10] * sv[2] + m[11] * sv[3] : 1.0f;
	if (rs.xfReg[XFR_DUALTEX] & 1) {
		const uint32_t post = rs.xfReg[XFR_POSTTEX + coord];
		if ((post >> 8) & 1) {
			float len = sqrtf(t0 * t0 + t1 * t1 + t2 * t2);
			if (len > 0.f) t0 /= len, t1 /= len, t2 /= len;
		}
		const float* p = reinterpret_cast<const float*>(&rs.xfMem[0x500 + (post & 63) * 4]);
		float a = p[0] * t0 + p[1] * t1 + p[2] * t2 + p[3];
		float b = p[4] * t0 + p[5] * t1 + p[6] * t2 + p[7];
		float c = p[8] * t0 + p[9] * t1 + p[10] * t2 + p[11];
		t0 = a, t1 = b, t2 = c;
	}
	if (q) {
		*q = t2;
	} else if (t2 != 0.f && t2 != 1.f) {
		t0 /= t2;
		t1 /= t2;
	}
	u = t0;
	w = t1;
}

// Clears the combiner stages a translation filled in.

// TEV translation cache. readTev/translateTev read only the registers hashed here (TEV envs,
// orders, konst selection and swap tables, colour registers and konst colours, channel controls
// and material colours), the vertex format, which texture pairs have a texture, and the PicaState
// built before them (blend, depth, alpha test, scissor). Draws repeat the same few setups every
// frame, so translating each setup once saves most of the per-draw TEV cost.
struct TevCacheEntry {
	TevSetup tev;
	PicaState st;
	bool translated;
	unsigned dumpId;
	uint32_t usedFrame;  // the render frame it was last found in (s_rFrames)
};
static std::unordered_map<uint64_t, TevCacheEntry> s_tevCache;

// [perf]: the host side's caches on the heap. After two hours of play a console ran out of heap at
// Noki Bay's Eely-Mouth (85 of 86 MB taken, 71 MB in use, the use some 18 MB up since boot): which
// of them grows. (Read on the render thread while the GX thread changes some: figures, not exact.)
static void perfHostMemory() {
	fprintf(stderr,
	        "[perf]   host memory: display-list cache %lu KB, compiled shapes %lu (%lu KB), TEV cache %u entries, "
	        "texture cache %u entries, texture decode buffers %u KB, EFB copies %u\n",
	        (unsigned long)(GXPC_DlCacheBytes() >> 10), (unsigned long)GXPC_J3DPkgStats[3], (unsigned long)(GXPC_J3DPkgBytes() >> 10),
	        unsigned(s_tevCache.size()), unsigned(s_texCache.size()),
	        unsigned((s_decodeBuf.capacity() + s_scaleBuf.capacity() + s_thpOut.capacity()) >> 10), unsigned(s_efbCopies.size()));
}
static const size_t kTevCacheMax = 1024;


static uint64_t tevCacheKey(const bool pwhite[kMaxPairs], int nPairs, const PicaState& pre, bool debug,
                            PicaTex* const ptex[kMaxPairs]) {
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { h = (h ^ x) * 1099511628211ull; };
	mix(rs.bp[BP_GENMODE]);
	for (int i = 0; i < 32; i++) mix(rs.bp[BP_TEV_COLOR_ENV + i]);
	for (int i = 0; i < 8; i++) mix(rs.bp[BP_TREF + i]);
	for (int i = 0; i < 8; i++) mix(rs.bp[BP_TEV_KSEL + i]);
	for (int i = 0; i < 8; i++) mix(rs.bp[BP_TEV_REG + i]);
	for (int i = 0; i < 8; i++) mix(rs.kreg[i]);
	for (int c = 0; c < 2; c++) {
		mix(rs.xfReg[XFR_COLOR0CTRL + c]);
		mix(rs.xfReg[XFR_ALPHA0CTRL + c]);
		mix(rs.xfReg[XFR_MAT0 + c]);
	}
	mix(r_fmt);
	uint32_t white = 0;
	for (int p = 0; p < nPairs; p++) white |= uint32_t(pwhite[p]) << p;
	mix(white | uint32_t(nPairs) << 16 | uint32_t(debug) << 24);
	if (nPairs - __builtin_popcount(white) > PICA_MAX_TEX)  // one may be read as its average: per texture
		for (int p = 0; p < nPairs; p++) mix(uint64_t(uintptr_t(ptex[p])));
	const uint8_t* b = reinterpret_cast<const uint8_t*>(&pre);
	for (size_t i = 0; i < sizeof pre; i += 8) {
		uint64_t w = 0;
		memcpy(&w, b + i, sizeof pre - i < 8 ? sizeof pre - i : 8);
		mix(w);
	}
	return h;
}

// ---- GPU vertex transform ---------------------------------------------------------------------
// The common case of drawTriangles' CPU loop, done by the vertex shader (shaders/gx_pass.v.pica,
// GX mode): the batch's packed vertices go to the GPU as they are, with XF position/normal
// matrix rows 0-29 as uniforms indexed per vertex, the GX projection and viewport composed into
// one matrix, and each texture unit's texgen, texture matrix, post matrix and EFB-copy rectangle
// composed into two rows. Batches it cannot express exactly (projected texture coordinates, a
// texture matrix index that varies per vertex, clipping off, ...) stay on the CPU path.
// XF memory changed since drawGpu last copied the matrix rows: position rows (s_posDirty, bit r
// for rows 0-29, 4 words each from word 0) and normal rows (s_nrmDirty, 3 words each from word 0x400).
// Only those rows are copied and their uniforms compared again: one matrix load used to recopy and
// recompare all 30 rows of both at the next draw.
static bool s_xfDirty = true;
static uint32_t s_posDirty = 0x3FFFFFFFu, s_nrmDirty = 0x3FFFFFFFu;
static inline uint32_t rowBits(uint32_t r0, uint32_t r1) {  // rows r0..r1 (r1 < 30)
	return ((2u << r1) - 1) & ~((1u << r0) - 1);
}
static inline void xfMemChanged(uint32_t w0, uint32_t w1) {  // words w0..w1
	if (w0 < 4 * PICA_GX_MTX_ROWS) s_posDirty |= rowBits(w0 >> 2, std::min<uint32_t>(w1, 4 * PICA_GX_MTX_ROWS - 1) >> 2);
	if (w1 >= kXfNrm && w0 < kXfNrm + 3 * PICA_GX_MTX_ROWS)
		s_nrmDirty |= rowBits((std::max<uint32_t>(w0, kXfNrm) - kXfNrm) / 3,
		                      (std::min<uint32_t>(w1, kXfNrm + 3 * PICA_GX_MTX_ROWS - 1) - kXfNrm) / 3);
	s_xfDirty = true;
}
static bool s_gpuTransform = true;  // GX3DS_SetGpuTransform: off draws everything on the CPU path
static float s_posRows[PICA_GX_MTX_ROWS][4], s_nrmRows[PICA_GX_MTX_ROWS][4];
static uint32_t s_gpuBatches, s_cpuBatches;


// GX's colour channels (XF COLORnCTRL, ALPHAnCTRL): the material (register, or the vertex colour:
// bit 0) and, with lighting on (bit 1), times clamp(ambient (register, or the vertex colour: bit 6)
// + each light in the mask (bits 2-5: lights 0-3, 11-14: 4-7): its colour x the diffuse term (bits
// 7-8: none 1, sign N.L, clamp max(N.L, 0)) x its attenuation (bits 9-10: 0 none and 2 directional
// 1, 1 specular, 3 spot)), with L from the vertex to the light, as Dolphin computes it. Light i's XF
// words (0x600 + 16 i): 3 colour, 4-6 angle attenuation, 7-9 distance attenuation, 10-12 position,
// 13-15 direction, in view space.
static inline uint32_t chanLights(uint32_t ctrl) { return ((ctrl >> 2) & 15) | ((ctrl >> 7) & 0xF0); }
static inline uint32_t chanDiffuse(uint32_t ctrl) { return (ctrl >> 7) & 3; }
static inline uint32_t chanAttn(uint32_t ctrl) { return (ctrl >> 9) & 3; }
struct GxLight {
	float col[4];
	float cosa[3], dista[3], pos[3], dir[3];
};
static void gxLight(int i, GxLight& l) {
	const uint32_t* w = &rs.xfMem[0x600 + 16 * i];
	for (int k = 0; k < 4; k++) l.col[k] = float((w[3] >> (24 - 8 * k)) & 0xFF) * (1.0f / 255.0f);
	memcpy(l.cosa, w + 4, 12);
	memcpy(l.dista, w + 7, 12);
	memcpy(l.pos, w + 10, 12);
	memcpy(l.dir, w + 13, 12);
}
// A lit part (colour, or alpha) that comes out at 1 whatever the normals: its ambient full (255)
// and its lights only adding (diffuse none or clamp; attenuation numerators are clamped at 0, so
// distance coefficients not below 0). J3D materials light the alpha that way (ALPHA0CTRL 0x0706
// under an ambient alpha of 255) next to a colour in another diffuse mode, which the GX shader
// cannot draw together: those went to the CPU path (Delfino Plaza: 118 batches a frame).
// A spot light so far from the model (view-space origin o) that it gives under 1/1024 to any vertex
// within 1e5 of the origin (wider than any stage), by its distance attenuation 1 / (k0 + k1 d + k2 d^2):
// none. Delfino Plaza's light 1 sits at (2e5, 5e5, 2e5) in view space under 1 / (1 + 1e-6 d^2), some
// 3e-6 of its colour; the GX shader has no distance attenuation, so it sent 87 batches a frame to
// the CPU path.
static bool lightNegligible(uint32_t ctrl, const GxLight& l, const float o[3]) {
	if (chanAttn(ctrl) != 3 || l.dista[0] < 0.f || l.dista[1] < 0.f || l.dista[2] < 0.f) return false;
	const float dx = l.pos[0] - o[0], dy = l.pos[1] - o[1], dz = l.pos[2] - o[2];
	const float d = sqrtf(dx * dx + dy * dy + dz * dz) - 1e5f;
	if (!(d > 0.f)) return false;
	const float num = fabsf(l.cosa[0]) + fabsf(l.cosa[1]) + fabsf(l.cosa[2]);  // the angle term's most, cos in 0..1
	const float den = l.dista[0] + l.dista[1] * d + l.dista[2] * d * d;
	const float c = fmaxf(fmaxf(l.col[0], l.col[1]), fmaxf(l.col[2], l.col[3]));
	return den > 0.f && num * c < den * (1.f / 1024.f);
}
static bool chanSaturated(uint32_t ctrl, uint32_t amb, bool alpha) {
	if ((ctrl >> 6) & 1) return false;  // ambient from the vertex colour
	if (alpha ? (amb & 0xFF) != 0xFF : (amb >> 8) != 0xFFFFFF) return false;
	if (chanDiffuse(ctrl) == 1) return false;  // sign: N.L may be below 0
	for (uint32_t m = chanLights(ctrl); m; m &= m - 1) {
		GxLight l;
		gxLight(__builtin_ctz(m), l);
		if (l.dista[0] < 0.f || l.dista[1] < 0.f || l.dista[2] < 0.f) return false;
	}
	return true;
}
// One light's diffuse x attenuation at view-space point p with unit normal n (null: no normal,
// so only a light without a diffuse term reaches it).
static float gxLightTerm(uint32_t ctrl, const GxLight& l, const float* p, const float* n) {
	const uint32_t attnFn = chanAttn(ctrl), dfn = chanDiffuse(ctrl);
	float L[3], attn = 1.0f;
	if (attnFn == 1) {  // specular: the position holds a direction
		const float d = sqrtf(l.pos[0] * l.pos[0] + l.pos[1] * l.pos[1] + l.pos[2] * l.pos[2]);
		for (int k = 0; k < 3; k++) L[k] = d > 0.f ? l.pos[k] / d : 0.f;
		float c = 0.f;
		if (n && n[0] * L[0] + n[1] * L[1] + n[2] * L[2] >= 0.f) {
			c = n[0] * l.dir[0] + n[1] * l.dir[1] + n[2] * l.dir[2];
			c = c > 0.f ? c : 0.f;
		}
		float dd[3] = {l.dista[0], l.dista[1], l.dista[2]};
		if (dfn != 0) {  // (Dolphin: the distance coefficients as they are with diffuse none, else normalised)
			const float m = sqrtf(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]);
			for (int k = 0; k < 3; k++) dd[k] = m > 0.f ? dd[k] / m : 0.f;
		}
		const float num = l.cosa[0] + l.cosa[1] * c + l.cosa[2] * c * c, den = dd[0] + dd[1] * c + dd[2] * c * c;
		attn = num > 0.f && den > 0.f ? num / den : 0.f;
	} else {
		for (int k = 0; k < 3; k++) L[k] = l.pos[k] - p[k];
		const float d2 = L[0] * L[0] + L[1] * L[1] + L[2] * L[2], d = sqrtf(d2);
		for (int k = 0; k < 3; k++) L[k] = d > 0.f ? L[k] / d : (n ? n[k] : 0.f);
		if (attnFn == 3) {  // spot
			float c = L[0] * l.dir[0] + L[1] * l.dir[1] + L[2] * l.dir[2];
			c = c > 0.f ? c : 0.f;
			const float num = l.cosa[0] + l.cosa[1] * c + l.cosa[2] * c * c, den = l.dista[0] + l.dista[1] * d + l.dista[2] * d2;
			attn = num > 0.f && den > 0.f ? num / den : 0.f;
		}
	}
	if (dfn == 0) return attn;
	if (!n) return 0.f;
	const float dot = n[0] * L[0] + n[1] * L[1] + n[2] * L[2];
	return attn * (dfn == 1 ? dot : (dot > 0.f ? dot : 0.f));
}
// A light whose attenuation is 1 everywhere (none, directional, or spot with angle (1 0 0) and
// distance (1 0 0) terms), or 0 everywhere (spot or specular with no angle term): the shader's.
static int gxLightFlat(uint32_t ctrl, const GxLight& l) {
	const uint32_t a = chanAttn(ctrl);
	if (a == 0 || a == 2) return 1;
	if (l.cosa[0] == 0.f && l.cosa[1] == 0.f && l.cosa[2] == 0.f) return 0;
	if (a == 3 && l.cosa[0] == 1.f && l.cosa[1] == 0.f && l.cosa[2] == 0.f && l.dista[0] == 1.f && l.dista[1] == 0.f &&
	    l.dista[2] == 0.f)
		return 1;
	return -1;
}
// The lighting factor of channel rgb ctrl cctrl and alpha ctrl actrl at view-space point p with unit
// normal n (null: none): what the material is multiplied by (1 where lighting is off). ambVtx: the
// vertex colour, for an ambient taken from it (null: white).
static void gxLighting(uint32_t cctrl, uint32_t actrl, uint32_t amb, uint32_t aamb, const float* p, const float* n,
                       const uint8_t* ambVtx, const uint8_t* aambVtx, float out[4]) {
	const uint32_t ctrls[2] = {cctrl, actrl};
	for (int part = 0; part < 2; part++) {
		const uint32_t ctrl = ctrls[part];
		const int c0 = part ? 3 : 0, c1 = part ? 4 : 3;
		if (!((ctrl >> 1) & 1)) {
			for (int c = c0; c < c1; c++) out[c] = 1.0f;
			continue;
		}
		const uint32_t a = part ? aamb : amb;
		const uint8_t* av = part ? aambVtx : ambVtx;
		for (int c = c0; c < c1; c++)
			out[c] = ((ctrl >> 6) & 1) ? (av ? av[c] * (1.0f / 255.0f) : 1.0f) : float((a >> (24 - 8 * c)) & 0xFF) * (1.0f / 255.0f);
		for (uint32_t m = chanLights(ctrl); m; m &= m - 1) {
			GxLight l;
			gxLight(__builtin_ctz(m), l);
			const float t = gxLightTerm(ctrl, l, p, n);
			for (int c = c0; c < c1; c++) out[c] += t * l.col[c];
		}
		for (int c = c0; c < c1; c++) out[c] = out[c] < 0.f ? 0.f : out[c] > 1.f ? 1.f : out[c];
	}
}

// The renderer log (GX3DS_SetVerboseLog): a lit channel's colours and lights, once per distinct set
// ([light] lines). A light's XF words (0x600 + 16 i): 3 colour, 4-6 angle attenuation, 7-9 distance
// attenuation, 10-12 position, 13-15 direction.
static void logLighting(int ch, int cha, bool hasNrm, bool hasClr, const VtxFmtLayout& L, uint32_t defPm) {
	static uint64_t seen[96];
	static int nSeen;
	const uint32_t cctrl = rs.xfReg[XFR_COLOR0CTRL + ch], actrl = rs.xfReg[XFR_ALPHA0CTRL + cha];
	const uint32_t mask = ((cctrl >> 2) & 15) | ((cctrl >> 7) & 0xF0);
	uint64_t key = (uint64_t(cctrl) << 32) ^ actrl ^ (uint64_t(rs.xfReg[XFR_MAT0 + ch]) << 7) ^ rs.xfReg[XFR_AMB0 + ch] * 31ull;
	for (int i = 0; i < 8; i++)
		if (mask & (1u << i)) key = key * 1099511628211ull ^ rs.xfMem[0x600 + 16 * i + 3];
	for (int i = 0; i < nSeen; i++)
		if (seen[i] == key) return;
	if (nSeen == 96) return;
	seen[nSeen++] = key;
	if (hasNrm && r_verts) {  // the first vertex: its normal, in view space, and N.L for the first light
		const uint8_t* v = r_verts + size_t(r_spans ? r_bmin : 0) * r_stride;
		const uint32_t pm = (r_fmt & VF_MTX) ? v[L.mtx] : defPm;
		float nr[3], p[3];
		memcpy(nr, v + L.nrm, 12);
		memcpy(p, v, 12);
		const float* nm = xfMtx(kXfNrm + pm * 3);
		const float* m = xfMtx(pm * 4);
		float vn[3], vp[3];
		for (int k = 0; k < 3; k++) {
			vn[k] = nm[3 * k] * nr[0] + nm[3 * k + 1] * nr[1] + nm[3 * k + 2] * nr[2];
			vp[k] = m[4 * k] * p[0] + m[4 * k + 1] * p[1] + m[4 * k + 2] * p[2] + m[4 * k + 3];
		}
		int l0 = 0;
		while (l0 < 8 && !(mask & (1u << l0))) l0++;
		float nl = 0.f;
		if (l0 < 8) {
			const float* lp = reinterpret_cast<const float*>(&rs.xfMem[0x600 + 16 * l0 + 10]);
			float d[3] = {lp[0] - vp[0], lp[1] - vp[1], lp[2] - vp[2]};
			const float dl = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), nlen = sqrtf(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
			if (dl > 0.f && nlen > 0.f) nl = (vn[0] * d[0] + vn[1] * d[1] + vn[2] * d[2]) / (dl * nlen);
		}
		fprintf(stderr, "[light] vertex 0: pm %u normal %.3g %.3g %.3g -> view %.3g %.3g %.3g, at %.4g %.4g %.4g, N.L %.3f\n", pm,
		        nr[0], nr[1], nr[2], vn[0], vn[1], vn[2], vp[0], vp[1], vp[2], nl);
		fprintf(stderr, "[light]   position rows %.3g %.3g %.3g / %.3g %.3g %.3g / %.3g %.3g %.3g, normal rows %.3g %.3g %.3g / %.3g %.3g %.3g / %.3g %.3g %.3g\n",
		        m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], nm[0], nm[1], nm[2], nm[3], nm[4], nm[5], nm[6], nm[7], nm[8]);
	}
	fprintf(stderr,
	        "[light] frame %u chan %d/%d ctrl %04x/%04x (mat from %s, amb from %s, diffuse %u, atten %s) mat %08x amb %08x "
	        "lights %02x%s%s\n",
	        s_rFrames, ch, cha, cctrl & 0x7FFF, actrl & 0x7FFF, (cctrl & 1) ? "vertex" : "reg", ((cctrl >> 6) & 1) ? "vertex" : "reg",
	        (cctrl >> 7) & 3, !((cctrl >> 9) & 1) ? "none" : ((cctrl >> 10) & 1) ? "spot" : "spec", rs.xfReg[XFR_MAT0 + ch],
	        rs.xfReg[XFR_AMB0 + ch], mask, hasNrm ? "" : " (no normals)", hasClr ? "" : " (no colours)");

	for (int i = 0; i < 8; i++) {
		if (!(mask & (1u << i))) continue;
		const uint32_t* w = &rs.xfMem[0x600 + 16 * i];
		float f[12];
		memcpy(f, w + 4, sizeof f);
		fprintf(stderr, "[light]   light %d colour %08x cos %.3g %.3g %.3g dist %.3g %.3g %.3g pos %.4g %.4g %.4g dir %.3g %.3g %.3g\n", i,
		        w[3], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11]);
	}
}

static bool drawGpu(const VtxFmtLayout& L, uint32_t n, uint32_t nidx, const TevSetup& tev, bool debug,
                    PicaTex* const utex[PICA_MAX_TEX], const EfbCopy* const ucopy[PICA_MAX_TEX],
                    const int ucoord[PICA_MAX_TEX], PicaState& st) {
	if (s_tracing) return gpuFail(FB_TRACE, n);
	if (rs.xfReg[0x05] != 0) return gpuFail(FB_NOCLIP, n);
	if (n == 0 || n > 0xFFFF) return gpuFail(FB_SIZE, n);
	const bool hasMtx = (r_fmt & VF_MTX) != 0, hasNrm = (r_fmt & VF_NRM) != 0;
	const uint32_t defPm = rs.xfReg[XFR_MATIDX_A] & 63;
	const uint32_t matA = rs.xfReg[XFR_MATIDX_A], matB = rs.xfReg[XFR_MATIDX_B];
	const uint8_t defTex[8] = {uint8_t((matA >> 6) & 63),  uint8_t((matA >> 12) & 63), uint8_t((matA >> 18) & 63),
	                           uint8_t((matA >> 24) & 63), uint8_t(matB & 63),         uint8_t((matB >> 6) & 63),
	                           uint8_t((matB >> 12) & 63), uint8_t((matB >> 18) & 63)};
	// Position matrix rows: the shader indexes rows idx..idx+2 of the 30 it holds.
	if (hasMtx && r_amtx) {
		if (r_pnMax > PICA_GX_MTX_ROWS - 3) return gpuFail(FB_PNMTX, n);
	} else if (hasMtx) {
		for (uint32_t i = 0; i < n; i++)
			if (r_verts[size_t(i) * r_stride + L.mtx] > PICA_GX_MTX_ROWS - 3) return gpuFail(FB_PNMTX, n);
	} else if (defPm > PICA_GX_MTX_ROWS - 3) {
		return gpuFail(FB_PNMTX, n);
	}

	// Colour channel, as the CPU loop: rgb of channel ch, alpha of channel cha.
	const int ch = debug ? 0 : tev.rasRgbChan, cha = debug ? 0 : tev.rasAChan;
	const uint32_t cctrl = rs.xfReg[XFR_COLOR0CTRL + ch], actrl = rs.xfReg[XFR_ALPHA0CTRL + cha],
	               mat = rs.xfReg[XFR_MAT0 + ch], amat = rs.xfReg[XFR_MAT0 + cha];
	const bool hasClr = (r_fmt & (VF_CLR0 << ch)) != 0, hasAClr = (r_fmt & (VF_CLR0 << cha)) != 0;
	const bool clrFromVtx = !debug && (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasAClr;
	if (clrFromVtx && alphaFromVtx && ch != cha) return gpuFail(FB_TWOCOLOURS, n);  // one colour attribute
	const int clrField = clrFromVtx ? ch : alphaFromVtx ? cha : -1;
	const uint32_t amb = rs.xfReg[XFR_AMB0 + ch], aamb = rs.xfReg[XFR_AMB0 + cha];
	const bool litC = ((cctrl >> 1) & 1) && !chanSaturated(cctrl, amb, false),
	           litA = ((actrl >> 1) & 1) && !chanSaturated(actrl, aamb, true);
	const bool lit = !debug && (litC || litA) && hasNrm && s_lightMode < 2;
	if (s_verbose && ((cctrl >> 1) & 1)) logLighting(ch, cha, hasNrm, hasClr, L, defPm);

	PicaGxUniforms u;
	memset(&u, 0, sizeof u);
	PicaGxLayout lay;
	memset(&lay, 0, sizeof lay);
	lay.stride = uint16_t(r_stride);
	lay.off_nrm = lay.off_clr = lay.off_mtx = -1;
	lay.off_tex[0] = lay.off_tex[1] = lay.off_tex[2] = -1;
	lay.fixed_idx = int16_t(defPm);
	if (hasMtx) lay.off_mtx = int16_t(L.mtx);
	if (clrField >= 0) lay.off_clr = int16_t(L.clr[clrField]);
	bool needNrm = lit;

	// Texture units: source selection and the composed rows.
	int slotAttr[3] = {-1, -1, -1};  // texture coordinate attribute (0-7) in source slots a, b, c
	bool pdvNo = false;              // a unit 1 or 2 that is not divided per vertex
	u.TQ1[3] = 1;                    // (divided by 1 when not projective)
	for (int un = 0; un < PICA_MAX_TEX; un++) {
		if (!utex[un]) continue;
		const int coord = ucoord[un];
		const uint32_t tg = rs.xfReg[XFR_TEXGEN + coord];
		const uint32_t proj = (tg >> 1) & 1, form = (tg >> 2) & 1, type = (tg >> 4) & 7, row = (tg >> 7) & 31;
		// A projective texgen (with gx_proj.v.pica, pica_gx_projective): unit 0's q goes out as
		// texcoord0w, divided per pixel; units 1 and 2's are divided per vertex, by one q row.
		const bool projQ = proj && type == 0;
		if (projQ && !pica_gx_projective()) return gpuFail(FB_TEXGEN_PROJ, n);

		float* w0 = u.W[2 * un];
		if (row == 0) {
			w0[0] = 1;
		} else if (row == 1 && hasNrm) {
			w0[1] = 1;
			needNrm = true;
		} else if (row >= 5 && row <= 12 && (r_fmt & (VF_TEX0 << (row - 5)))) {
			const int a = int(row - 5);
			int slot = 0;
			while (slot < 3 && slotAttr[slot] >= 0 && slotAttr[slot] != a) slot++;
			if (slot == 3) return gpuFail(FB_TEXGEN_SRC, n);
			slotAttr[slot] = a;
			if (slot < 2) w0[2 + slot] = 1;
			else u.W[2 * un + 1][0] = 1;
		} else if (type == 0) {
			return gpuFail(FB_TEXGEN_SRC, n);  // a constant through the texture matrix: not expressible here
		}
		// (Emboss and colour texgens with any other source: the CPU path passes its default (0, 0)
		// through, as zero weights do here.)
		float R[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};  // s, t, q
		if (type != 0) {
			// Emboss and colour texgens: the CPU path passes the source's (s, t) through (texCoord), so
			// does this, with no texture or post matrix.
			const float id[2][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}};
			memcpy(R, id, sizeof id);
			goto efbRect;
		}
		{
		// Texture matrix: the default index, or the per-vertex one when it is the same everywhere.
		uint32_t tm = defTex[coord];
		if (hasMtx && r_amtx) {
			tm = r_texMtx[coord];
			if (tm == 0xFF) return gpuFail(FB_TEXMTX_VARIES, n);
		} else if (hasMtx) {
			tm = r_verts[L.mtx + 1 + coord];
			for (uint32_t i = 1; i < n; i++)
				if (r_verts[size_t(i) * r_stride + L.mtx + 1 + coord] != tm) return gpuFail(FB_TEXMTX_VARIES, n);
		}
		const float* m = reinterpret_cast<const float*>(&rs.xfMem[tm * 4]);
		for (int r = 0; r < (projQ ? 3 : 2); r++) memcpy(R[r], m + 4 * r, 16);
		if (row >= 5 || form == 0)  // the source's third component is 1 (AB11; texture attributes)
			for (int r = 0; r < 3; r++) R[r][3] += R[r][2], R[r][2] = 0.f;
		// (t0, t1, t2) through the post matrix (t2 = 1 unless projective)
		if (rs.xfReg[XFR_DUALTEX] & 1) {
			const uint32_t post = rs.xfReg[XFR_POSTTEX + coord];
			if ((post >> 8) & 1) return gpuFail(FB_POST_NORMALISE, n);  // normalised before the post matrix
			const float* pm = reinterpret_cast<const float*>(&rs.xfMem[0x500 + (post & 63) * 4]);
			if (!projQ && (pm[8] != 0.f || pm[9] != 0.f)) return gpuFail(FB_POST_PROJ, n);  // varying third component: divided per vertex
			float Q[3][4];
			for (int r = 0; r < 3; r++)
				for (int k = 0; k < 4; k++)
					Q[r][k] = pm[4 * r] * R[0][k] + pm[4 * r + 1] * R[1][k] + pm[4 * r + 2] * R[2][k] + (k == 3 ? pm[4 * r + 3] : 0.f);
			if (!projQ) {  // a constant third component: divided here
				const float c = Q[2][3];
				if (c != 0.f && c != 1.f)
					for (int r = 0; r < 2; r++)
						for (int k = 0; k < 4; k++) Q[r][k] /= c;
				const float one[4] = {0, 0, 0, 1};
				memcpy(Q[2], one, sizeof one);
			}
			memcpy(R, Q, sizeof R);
		}
		}
	efbRect:
		if (const EfbCopy* cp = ucopy[un]) {  // through the copy's source rectangle into its texture's window
			float as, bs, at, bt;
			copyMap(cp, as, bs, at, bt);
			// (s / q) * a + b = (s a + q b) / q
			for (int k = 0; k < 4; k++) {
				R[0][k] = R[0][k] * as + R[2][k] * bs;
				R[1][k] = R[1][k] * at + R[2][k] * bt;
			}
		}
		memcpy(u.T[2 * un], R[0], 16);
		memcpy(u.T[2 * un + 1], R[1], 16);
		if (un == 0) memcpy(u.TQ, R[2], 16);
		if (un > 0) {  // units 1 and 2: projective ones divided per vertex, with one q row (TQ1, pdv)
			if (projQ) {
				bool same = true;
				if (u.pdv)
					for (int k = 0; k < 4; k++) same &= fabsf(R[2][k] - u.TQ1[k]) <= 1e-5f * (1.0f + fabsf(u.TQ1[k]));
				// all or none, one q row, and unit 0 projective (its flag turns the division on)
				if (pdvNo || !same || !st.proj0) return gpuFail(FB_TEXGEN_PROJ, n);
				memcpy(u.TQ1, R[2], 16);
				u.pdv = 1;
			} else {
				if (u.pdv) return gpuFail(FB_TEXGEN_PROJ, n);
				pdvNo = true;
			}
		}
		if (un == 0 && st.proj0 != projQ) {
			st.proj0 = projQ;
			if (++s_stGen == 0) s_stGen = 1;
		}
	}
	if (!utex[0] && st.proj0) {
		st.proj0 = false;
		if (++s_stGen == 0) s_stGen = 1;
	}
	for (int sl = 0; sl < 3; sl++)
		if (slotAttr[sl] >= 0) lay.off_tex[sl] = int16_t(L.tex[slotAttr[sl]]);
	if (needNrm && hasNrm) lay.off_nrm = int16_t(L.nrm);

	// Colour: vertex * CS + CM, shaded when lit.
	uint8_t dbg[3] = {0, 0, 0};
	if (debug) debugColor(hasMtx ? r_verts[L.mtx] : defPm, r_verts, r_stride, nidx, dbg);
	const float k255 = 1.0f / 255.0f;
	const float rgbMat[3] = {debug ? dbg[0] * k255 : float((mat >> 24) & 0xFF) * k255,
	                         debug ? dbg[1] * k255 : float((mat >> 16) & 0xFF) * k255,
	                         debug ? dbg[2] * k255 : float((mat >> 8) & 0xFF) * k255};
	for (int c = 0; c < 3; c++) {
		u.CS[c] = clrFromVtx ? 1.f : 0.f;
		u.CM[c] = clrFromVtx ? 0.f : rgbMat[c];
	}
	u.CS[3] = alphaFromVtx ? 1.f : 0.f;
	u.CM[3] = alphaFromVtx ? 0.f : float(amat & 0xFF) * k255;
	// GX lighting (gxLighting): with normals, in the shader (up to two lights, no specular ones; else
	// the CPU path); without, the lights' diffuse terms are 0 and the factor is one for the draw.
	if (!debug && (((cctrl >> 1) & 1) || ((actrl >> 1) & 1))) {
		if (!lit) {
			float f[4];
			const float origin[3] = {0, 0, 0};
			gxLighting(cctrl, actrl, amb, aamb, origin, nullptr, nullptr, nullptr, f);
			for (int c = 0; c < 4; c++) u.CS[c] *= f[c], u.CM[c] *= f[c];
		} else {  // (a part litC / litA leaves out comes out at 1: ambient 1, no lights)
			const uint32_t maskC = litC ? chanLights(cctrl) : 0, maskA = litA ? chanLights(actrl) : 0;
			// Why a lit draw goes to the CPU path, once per setup (renderer log).
			auto lightFail = [&](const char* why, int light) {
				if (s_verbose) {
					static uint64_t said[64];
					static int nSaid;
					const uint64_t key = uint64_t(cctrl) << 48 ^ uint64_t(actrl) << 32 ^ amb ^ uint64_t(aamb) << 7 ^ uintptr_t(why);
					bool seen = false;
					for (int i = 0; i < nSaid; i++) seen |= said[i] == key;
					if (!seen && nSaid < 64) {
						said[nSaid++] = key;
						GxLight l{};
						if (light >= 0) gxLight(light, l);
						fprintf(stderr, "[light] CPU path (%s): ctrl %04x/%04x amb %08x %08x, light %d cos %.3g %.3g %.3g dist %.3g %.3g %.3g pos %.4g %.4g %.4g\n",
						        why, cctrl & 0x7FFF, actrl & 0x7FFF, amb, aamb, light, l.cosa[0], l.cosa[1], l.cosa[2], l.dista[0], l.dista[1],
						        l.dista[2], l.pos[0], l.pos[1], l.pos[2]);
					}
				}
				return gpuFail(FB_LIGHTS, n);
			};
			if (((cctrl >> 6) & 1) && litC) return lightFail("ambient from the vertex colour", -1);
			if (((actrl >> 6) & 1) && litA) return lightFail("alpha ambient from the vertex colour", -1);
			for (int c = 0; c < 3; c++) u.AMB[c] = litC ? float((amb >> (24 - 8 * c)) & 0xFF) * k255 : 1.f;
			u.AMB[3] = litA ? float(aamb & 0xFF) * k255 : 1.f;
			int nl = 0;
			const uint8_t* v0 = r_verts + size_t(r_spans ? r_bmin : 0) * r_stride;
			const float* pmx = xfMtx((hasMtx ? v0[L.mtx] : defPm) * 4);
			const float org[3] = {pmx[3], pmx[7], pmx[11]};  // the model's origin in view space
			for (uint32_t m = maskC | maskA; m; m &= m - 1) {
				const int i = __builtin_ctz(m);
				GxLight l;
				gxLight(i, l);
				const bool inC = (maskC >> i) & 1, inA = (maskA >> i) & 1;
				if ((!inC || lightNegligible(cctrl, l, org)) && (!inA || lightNegligible(actrl, l, org))) continue;
				const uint32_t ctrl = inC ? cctrl : actrl;
				const int flat = gxLightFlat(ctrl, l);  // 1: no attenuation, 0: no light, -1: attenuated
				if (inC && inA && (chanAttn(cctrl) != chanAttn(actrl) || gxLightFlat(actrl, l) != flat))
					return lightFail("colour and alpha attenuation differ", i);
				if (flat == 0) continue;  // (no light from it)
				if (inC && inA && chanDiffuse(cctrl) != chanDiffuse(actrl)) return lightFail("colour and alpha diffuse differ", i);
				if (l.col[0] == 0.f && l.col[1] == 0.f && l.col[2] == 0.f && (!inA || l.col[3] == 0.f)) continue;
				if (nl == 2) return lightFail("more than 2 lights", i);
				memcpy(u.LP[nl], l.pos, 12);
				for (int c = 0; c < 3; c++) u.LC[nl][c] = inC ? l.col[c] : 0.f;
				u.LC[nl][3] = inA ? l.col[3] : 0.f;
				const uint32_t dfn = chanDiffuse(ctrl);
				u.LF[nl][0] = dfn == 2 ? 0.f : -1e20f;
				u.LF[nl][1] = dfn == 0 ? 0.f : 1.f;
				u.LF[nl][2] = dfn == 0 ? 1.f : 0.f;
				if (flat < 0) {  // attenuated: a spot light, or a specular one (its term replaces the diffuse one)
					const bool spec = chanAttn(ctrl) == 1;
					if (spec && dfn != 0) return lightFail("specular with a diffuse term", i);  // (GXSetChanCtrl never sets it)
					u.attn |= uint8_t((spec ? 0x10u : 1u) << nl);
					memcpy(u.LA[nl], l.cosa, 12);
					memcpy(u.LK[nl], l.dista, 12);
					memcpy(u.LD[nl], l.dir, 12);
					const float m = sqrtf(l.dista[0] * l.dista[0] + l.dista[1] * l.dista[1] + l.dista[2] * l.dista[2]);
					if (spec && dfn != 0 && m > 0.f)
						for (int k = 0; k < 3; k++) u.LK[nl][k] /= m;
				}
				nl++;
			}
			u.lit = 1;
			u.lights = uint8_t(nl);
			if (s_lightMode == 1) u.attn = 0;
		}
	}

	// View space -> EFB pixels and GameCube depth (homogeneous), as the CPU loop computes them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = rs.xfReg[XFR_PROJ + 6] & 1;
	const float rowScale = efbRowScale();
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1) / rowScale, vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = (xff(XFR_VIEWPORT + 4) - 342.0f) / rowScale,
	            vcz = xff(XFR_VIEWPORT + 5);
	const float kInvZ = 1.0f / 16777215.0f;
	float C[4][4];
	if (!ortho) {
		const float c[4][4] = {{p0, 0, p1, 0}, {0, p2, p3, 0}, {0, 0, p4, p5}, {0, 0, -1, 0}};
		memcpy(C, c, sizeof C);
	} else {
		const float c[4][4] = {{p0, 0, 0, p1}, {0, p2, 0, p3}, {0, 0, p4, p5}, {0, 0, 0, 1}};
		memcpy(C, c, sizeof C);
	}
	float O[4][4];
	for (int k = 0; k < 4; k++) {
		O[0][k] = vsx * C[0][k] + vcx * C[3][k];
		O[1][k] = vsy * C[1][k] + vcy * C[3][k];
		O[2][k] = (vsz * C[2][k] + vcz * C[3][k]) * kInvZ;
		O[3][k] = C[3][k];
	}
	pica_gx_projection(O, &u);

	// Matrix rows, refreshed where XF memory changed (renderState, renderXFWord).
	const uint32_t dirty = s_posDirty;
	if (s_xfDirty) {
		for (uint32_t m = s_posDirty; m; m &= m - 1) {
			const int r = __builtin_ctz(m);
			memcpy(s_posRows[r], &rs.xfMem[4 * r], 16);
		}
		for (uint32_t m = s_nrmDirty; m; m &= m - 1) {
			const int r = __builtin_ctz(m);
			memcpy(s_nrmRows[r], &rs.xfMem[kXfNrm + 3 * r], 12);
			s_nrmRows[r][3] = 0.f;
		}
		s_posDirty = s_nrmDirty = 0;
		s_xfDirty = false;
	}

	void* vb;
	uint16_t* ib;
	{
		PerfScope ps(PERF_XFORM);
		const bool inPlace = r_arena != nullptr;  // decoded into the ring and claimed: drawn where it is
		// A resident batch of one span draws its indices where they are stored; one of several
		// (its page's runs not stored one after another) copies them into the ring, as every draw
		// costs the GPU a pipeline drain and its vertex cache.
		const bool oneSpan = r_spans && r_nspans == 1 && s_idxInPlace;
		const bool rebase = r_spans && !s_idxInPlace;
		if (r_spans) {
			s_spanBatches++;
			s_spansDrawn += r_nspans;
			if (!oneSpan) s_spanCopies++;
		}
		if (!pica_alloc_raw(inPlace ? 0 : n * r_stride, oneSpan ? 0 : nidx, &vb, &ib)) return true;  // ring full: dropped
		if (inPlace) vb = const_cast<uint8_t*>(r_verts) + (rebase ? size_t(r_bmin) * r_stride : 0);
		else memcpy(vb, r_verts, size_t(n) * r_stride);
		if (oneSpan) {
			ib = const_cast<uint16_t*>(r_spans[0].idx);
		} else if (rebase) {
			uint16_t* o = ib;
			const uint16_t b = uint16_t(r_bmin);
			for (uint32_t i = 0; i < r_nspans; i++)
				for (uint32_t k = 0; k < r_spans[i].n; k++) *o++ = uint16_t(r_spans[i].idx[k] - b);
		} else if (r_spans) {
			uint16_t* o = ib;
			for (uint32_t i = 0; i < r_nspans; i++) {
				memcpy(o, r_spans[i].idx, size_t(r_spans[i].n) * 2);
				o += r_spans[i].n;
			}
		} else {
			memcpy(ib, r_idx, size_t(nidx) * 2);
		}
	}
	PerfScope ps(PERF_SUBMIT);
	pica_draw_same = s_stGen;
	if (!pica_draw_gx(vb, n, &lay, ib, nidx, &u, dirty, s_posRows, s_nrmRows, &st)) {
		if (dirty) {  // the rows' uniforms were not compared: again at the next draw
			s_posDirty |= dirty;
			s_xfDirty = true;
		}
		return gpuFail(FB_LAYOUT, n);
	}
	s_gpuBatches++;
	return true;
}

// The CPU path wants a resident batch as the others are: its indices in one list, from its window's
// first vertex (r_verts and r_count moved there).
static std::vector<uint16_t> r_flat;
static void flattenSpans() {
	r_flat.resize(r_nidx);
	uint16_t* o = r_flat.data();
	for (uint32_t s = 0; s < r_nspans; s++)
		for (uint32_t i = 0; i < r_spans[s].n; i++) *o++ = uint16_t(r_spans[s].idx[i] - r_bmin);
	r_idx = r_flat.data();
	r_verts += size_t(r_bmin) * r_stride;
	r_count -= r_bmin;
	r_spans = nullptr;
	r_nspans = 0;
	r_bmin = 0;
}

static void drawTriangles() {
	if (s_etcOutHead.load(std::memory_order_relaxed) != s_etcOutTail.load(std::memory_order_relaxed)) etcApply();
	if (((rs.bp[BP_GENMODE] >> 14) & 3) == 3) return;  // every face culled
	if (s_afterTexCopy) s_drawsAfterCopy++;
	else s_drawsBeforeCopy++;
	s_drawsCounted++;
	if (s_rsGen == s_rsGenLastDraw) s_drawsSameState++;
	s_rsGenLastDraw = s_rsGen;
	const VtxFmtLayout& L = r_layout;
	uint32_t n = r_count, nidx = uint32_t(r_nidx);

	s_perfBatches++;
	s_perfVerts += n - (r_spans ? r_bmin : 0);
	// The textures the TEV stages sample, per (map, coordinate) pair, and the combiner stages,
	// which give PICA texture units to the pairs they use (none in debug-colour mode).
	const bool debug = s_debugColors;
	uint64_t perfTev0 = perfTick(), tevT1 = perfTev0;
	// A draw that comes with no change to anything worked out below (the BP, XF and konst registers,
	// the TLUTs, the texture images and the EFB copies: s_rsGen; the vertex format; the frame, as
	// the textures are checked once a frame; the colour modes) takes the last draw's TEV setup,
	// textures and PICA state as they were: in Delfino Plaza 42% of the draws (a material's shapes
	// after its first), each saving the TEV read, the texture lookups, the state and its key. They
	// are worked out in place (s_fp), so taking them copies nothing; the CPU path, which changes the
	// state for its own draw, leaves them invalid.
	struct FastState {
		bool valid, debug, opaque;
		uint32_t gen, fmt, frame;
		TevSetup tev;
		PicaTex* ptex[kMaxPairs];
		const EfbCopy* pcopy[kMaxPairs];
		bool pwhite[kMaxPairs];
		PicaState st;
		bool translated;
		unsigned dumpId;
		PicaTex* utex[PICA_MAX_TEX];
		const EfbCopy* ucopy[PICA_MAX_TEX];
		int ucoord[PICA_MAX_TEX];
	};
	static FastState s_fp, s_fpOld;  // (s_fpOld: draw_state_check's copy)
	TevSetup& tev = s_fp.tev;
	PicaTex*(&ptex)[kMaxPairs] = s_fp.ptex;
	const EfbCopy*(&pcopy)[kMaxPairs] = s_fp.pcopy;
	bool(&pwhite)[kMaxPairs] = s_fp.pwhite;
	PicaState& st = s_fp.st;
	bool& translated = s_fp.translated;
	unsigned& dumpId = s_fp.dumpId;
	PicaTex*(&utex)[PICA_MAX_TEX] = s_fp.utex;
	const EfbCopy*(&ucopy)[PICA_MAX_TEX] = s_fp.ucopy;
	int(&ucoord)[PICA_MAX_TEX] = s_fp.ucoord;
	const bool same = s_fp.valid && s_fp.gen == s_rsGen && s_fp.fmt == r_fmt && s_fp.frame == s_rFrames &&
	                  s_fp.debug == debug && s_fp.opaque == s_opaque && !s_verbose && !s_tracing;
	if (same && !s_fpCheck) {
		s_drawsFast++;
	} else {
		if (same) s_fpOld = s_fp;
		else if (++s_stGen == 0) s_stGen = 1;
		memset(ptex, 0, sizeof ptex);
		memset(pcopy, 0, sizeof pcopy);
		translated = true;
		dumpId = 0;
		memset(utex, 0, sizeof utex);
		memset(ucopy, 0, sizeof ucopy);
		memset(ucoord, 0, sizeof ucoord);
		readTev(tev);
		{
			const uint64_t t = perfTick() - perfTev0;
			perfAdd(PERF_TEV, t);
			s_tevPartTicks[0] += t;
		}
		for (int p = 0; p < kMaxPairs; p++) pwhite[p] = true;
		if (!debug) {
			PerfScope ps(PERF_TEX);
			for (int p = 0; p < tev.nPairs; p++) {
				ptex[p] = textureForMap(tev.pairMap[p], &pcopy[p]);
				pwhite[p] = ptex[p] == nullptr;
				if (ptex[p] && !pcopy[p]) {
					const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, tev.pairMap[p])];
					tev.pairTexels[p] = ((img0 & 0x3FF) + 1) * (((img0 >> 10) & 0x3FF) + 1);
					tev.pairFmt[p] = uint8_t((img0 >> 20) & 15);
					auto a = s_texAvg.find(ptex[p]);
					if (a != s_texAvg.end())
						for (int i = 0; i < 4; i++) tev.pairConst[p][i] = int16_t((a->second >> (24 - 8 * i)) & 0xFF);
				}
			}
		}
		// A movie with stages of its own after the conversion (isThpPrefix): the frame as RGB.
		if (!debug && isThpPrefix(tev)) {
			const int pu = tev.stagePair[0], pv = tev.stagePair[1], py = tev.stagePair[2];
			if (ptex[pu] && ptex[pv] && ptex[py] && !pcopy[pu] && !pcopy[pv] && !pcopy[py])
				if (PicaTex* rgb = thpRgbTexture(tev)) {
					thpPrefixToTexture(tev);
					ptex[py] = rgb;
					pwhite[pu] = pwhite[pv] = true;  // no stage reads them now
					tevLogOnce("movie converted on the CPU for the stages after it", 4, tev.numStages, tev.colorEnv[0],
					           tev.alphaEnv[0]);
				}
		}
		perfTev0 = perfTick();

		memset(&st, 0, sizeof st);  // the PICA side compares states bytewise
		currentPicaState(st);
		alphaTest(st);
		const uint64_t tevKey = tevCacheKey(pwhite, tev.nPairs, st, debug, ptex);
		// The last batch's entry when the key repeats (the map's elements stay where they are until it
		// is cleared), else a lookup.
		static uint64_t s_lastTevKey;
		static const TevCacheEntry* s_lastTevEntry;
		const TevCacheEntry* hitEntry = nullptr;
		if (s_lastTevEntry && tevKey == s_lastTevKey) {
			hitEntry = s_lastTevEntry;
		} else {
			auto tevHit = s_tevCache.find(tevKey);
			if (tevHit != s_tevCache.end()) hitEntry = &tevHit->second;
		}
		const bool cached = hitEntry != nullptr;
		if (cached) {
			const_cast<TevCacheEntry*>(hitEntry)->usedFrame = s_rFrames;
			tev = hitEntry->tev;
			st = hitEntry->st;
			translated = hitEntry->translated;
			dumpId = hitEntry->dumpId;
			s_lastTevKey = tevKey;
			s_lastTevEntry = hitEntry;
		}
		tevT1 = perfTick();
		s_tevPartTicks[1] += tevT1 - perfTev0;
		if (cached) s_tevHits++;
		else s_tevMisses++;
		if (cached) {
		} else if (debug) {
			tev.nUnits = 0;
		} else if (isThpSetup(tev) && ptex[tev.stagePair[0]] && ptex[tev.stagePair[1]] && ptex[tev.stagePair[2]]) {
			translateThp(tev, st);
		} else if (isStampSetup(tev) && ptex[tev.stagePair[0]] && ptex[tev.stagePair[2]]) {
			translateStamp(tev, st);
		} else {
			translated = translateTevFallback(tev, pwhite, st);
			if (s_tevSymOn) tevSymImprove(tev, pwhite, st, translated);
		}
		if (!cached && !debug && !translated) {
			resetTev(st, tev);
			for (uint32_t s = 0; s < tev.numStages; s++) {
				const int pr = tev.stagePair[s];
				if (pr < 0 || !ptex[pr]) continue;
				PicaTevStage& ps = st.stage[0];
				const bool rasNull = ((tev.order[s] >> 7) & 7) == 7;
				ps.src_rgb = ps.src_alpha = uint16_t(SRC_TEX0) | (rasNull ? 0 : SRC_PRIMARY << 4);
				ps.func_rgb = ps.func_alpha = rasNull ? FN_REPLACE : FN_MODULATE;
				st.nstages = 1;
				tev.nUnits = 1;
				tev.unitPair[0] = pr;
				break;
			}
		}
		if (!cached) {
			const uint64_t t2 = perfTick();
			s_tevPartTicks[2] += t2 - tevT1;
			tevT1 = t2;
			if (s_tevCache.size() >= kTevCacheMax) {
				// The entries not found in the last two frames go (animated TEV colours make new keys
				// every frame); emptying the whole cache made every material miss again after it.
				for (auto it = s_tevCache.begin(); it != s_tevCache.end();)
					it = it->second.usedFrame + 2 < s_rFrames ? s_tevCache.erase(it) : std::next(it);
				if (s_tevCache.size() >= kTevCacheMax) s_tevCache.clear();
				s_lastTevEntry = nullptr;
				s_tevClears++;
			}
			TevCacheEntry& e = s_tevCache[tevKey];
			e.usedFrame = s_rFrames;
			s_lastTevKey = tevKey;
			s_lastTevEntry = &e;
			s_tevEntries = uint32_t(s_tevCache.size());
			e.tev = tev;
			e.st = st;
			e.translated = translated;
			e.dumpId = 0;
		}
		static const uint8_t kWrap[4] = {0, 2, 3, 2};  // clamp, repeat, mirror -> GPU_TEXTURE_WRAP_PARAM
		for (int u = 0; u < tev.nUnits; u++) {
			const int pr = tev.unitPair[u];
			utex[u] = ptex[pr];
			ucopy[u] = pcopy[pr];
			ucoord[u] = tev.pairCoord[pr];
			if (!utex[u]) continue;
			const uint32_t mode0 = rs.bp[bpTexReg(BP_TX_MODE0, tev.pairMap[pr])];
			st.tex[u] = utex[u];
			st.wrap_s[u] = kWrap[mode0 & 3];
			st.wrap_t[u] = kWrap[(mode0 >> 2) & 3];
			st.linear[u] = (mode0 >> 4) & 1;
			st.lod_bias[u] = int16_t(int8_t((mode0 >> 9) & 0xFF) * 8);  // s2.5 -> 1/256
			// A perspective draw reading a one-byte map copy is the goop layer on the ground.
			if (ucopy[u] && (ucopy[u]->fmt == 1 || (ucopy[u]->fmt >= 7 && ucopy[u]->fmt <= 10)) &&
			    !(rs.xfReg[XFR_PROJ + 6] & 1))
				st.depth_bias = 1;
		}
		if (!debug && ((!cached && s_verbose) || s_tracing)) {
			// Blend (BP 0x41), depth (0x40), alpha compare (0xF3), channel controls and the first
			// vertex's colours, for the dump.
			const VtxFmtLayout& L0 = r_layout;
			uint32_t c0 = 0, c1 = 0;
			if (r_count && (r_fmt & VF_CLR0)) memcpy(&c0, r_verts + L0.clr[0], 4);
			if (r_count && (r_fmt & (VF_CLR0 << 1))) memcpy(&c1, r_verts + L0.clr[1], 4);
			char extra[240];
			snprintf(extra, sizeof extra,
			         "clip %s cmode0 %06x zmode %06x alphacmp %06x chans %u ctrl %04x/%04x %04x/%04x mat %08x %08x amb %08x %08x "
			         "vtx clr %08x %08x (fmt %x)",
			         rs.xfReg[0x05] ? "off" : "on", rs.bp[BP_CMODE0] & 0xFFFFFF, rs.bp[BP_ZMODE] & 0xFFFFFF, rs.bp[BP_ALPHACOMPARE] & 0xFFFFFF,
			         rs.xfReg[XFR_NUMCHANS] & 3, rs.xfReg[XFR_COLOR0CTRL] & 0x7FFF, rs.xfReg[XFR_ALPHA0CTRL] & 0x7FFF,
			         rs.xfReg[XFR_COLOR0CTRL + 1] & 0x7FFF, rs.xfReg[XFR_ALPHA0CTRL + 1] & 0x7FFF, rs.xfReg[XFR_MAT0],
			         rs.xfReg[XFR_MAT0 + 1], rs.xfReg[XFR_AMB0], rs.xfReg[XFR_AMB0 + 1], __builtin_bswap32(c0),
			         __builtin_bswap32(c1), r_fmt);
			dumpId = tevDump(tev, st, translated, ptex, pcopy, extra);
			if (!cached || !s_tevCache[tevKey].dumpId) s_tevCache[tevKey].dumpId = dumpId;
		}
		if (s_opaque) {
			st.blend = 0;
			st.alpha_test = st.alpha_test2 = 0;
			st.alpha_func = 1;  // GPU_ALWAYS
		}
		// The check (settings.txt draw_state_check): a draw that could have taken the last draw's
		// setup works it out, and each part is compared with what it would have taken.
		if (same) {
			s_fpChecked++;
			const bool differs = memcmp(&tev, &s_fpOld.tev, sizeof tev) != 0 || memcmp(ptex, s_fpOld.ptex, sizeof ptex) != 0 ||
			                     memcmp(pcopy, s_fpOld.pcopy, sizeof pcopy) != 0 ||
			                     memcmp(pwhite, s_fpOld.pwhite, sizeof pwhite) != 0 || memcmp(&st, &s_fpOld.st, sizeof st) != 0 ||
			                     translated != s_fpOld.translated || memcmp(utex, s_fpOld.utex, sizeof utex) != 0 ||
			                     memcmp(ucopy, s_fpOld.ucopy, sizeof ucopy) != 0 ||
			                     memcmp(ucoord, s_fpOld.ucoord, sizeof ucoord) != 0;
			if (differs) s_fpDiffer++;
		}
		s_fp.valid = true;
		s_fp.gen = s_rsGen;
		s_fp.fmt = r_fmt;
		s_fp.frame = s_rFrames;
		s_fp.debug = debug;
		s_fp.opaque = s_opaque;
	}
	{
		const uint64_t t3 = perfTick();
		perfAdd(PERF_TEV, t3 - perfTev0);
		s_tevPartTicks[3] += t3 - tevT1;
	}
	if (s_skipTev && dumpId >= s_skipTev && dumpId <= s_skipTevTo) return;  // (GX3DS_SetSkipTev: a debugging aid)
	// A projective texgen on a unit other than 0 (Bowser's hot tub: the water's lookup of the screen
	// copy, ~14K vertices): that unit and unit 0 swap, as the PICA divides per pixel on unit 0 only
	// (otherwise the draw takes the CPU path). Once swapped, unit 0 is the projective one, so draws
	// taking this setup again leave it as it is.
	if (!debug && tev.nUnits > 1 && pica_gx_projective()) {
		auto projective = [&](int un) {
			if (!utex[un]) return false;
			const uint32_t tg = rs.xfReg[XFR_TEXGEN + ucoord[un]];
			return ((tg >> 1) & 1) && ((tg >> 4) & 7) == 0;
		};
		int pu = -1;
		for (int un = 1; un < tev.nUnits && pu < 0; un++)
			if (projective(un)) pu = un;
		if (pu > 0 && !projective(0)) {
			std::swap(utex[0], utex[pu]);
			std::swap(ucopy[0], ucopy[pu]);
			std::swap(ucoord[0], ucoord[pu]);
			std::swap(tev.unitPair[0], tev.unitPair[pu]);
			std::swap(st.tex[0], st.tex[pu]);
			std::swap(st.wrap_s[0], st.wrap_s[pu]);
			std::swap(st.wrap_t[0], st.wrap_t[pu]);
			std::swap(st.linear[0], st.linear[pu]);
			std::swap(st.lod_bias[0], st.lod_bias[pu]);
			auto swapSrc = [&](uint16_t& srcs) {
				uint16_t o = 0;
				for (int k = 0; k < 3; k++) {
					uint16_t v = (srcs >> (4 * k)) & 15;
					if (v == SRC_TEX0) v = uint16_t(SRC_TEX0 + pu);
					else if (v == SRC_TEX0 + pu) v = SRC_TEX0;
					o |= uint16_t(v << (4 * k));
				}
				srcs = o;
			};
			for (int i = 0; i < st.nstages; i++) {
				swapSrc(st.stage[i].src_rgb);
				swapSrc(st.stage[i].src_alpha);
			}
			if (++s_stGen == 0) s_stGen = 1;
		}
	}
	if (s_gpuTransform && drawGpu(L, n, nidx, tev, debug, utex, ucopy, ucoord, st)) return;
	s_cpuBatches++;
	s_fp.valid = false;  // (the CPU path changes the state for its own draw)
	if (r_spans) {
		flattenSpans();
		n = r_count;
	}
	// A projective texgen on unit 0 (the sea's lookup of the screen copy, which the sea's colour
	// comes from): its (s q, t q, q) go to the GPU, which divides per pixel (texture unit 0 in
	// projection mode). Divided here per vertex, the coordinates bend across a large polygon and
	// it shows the wrong part of the screen. q travels in unit 2's coordinate, so unit 2 must be free.
	if (utex[0] && !utex[2]) {
		const uint32_t tg0 = rs.xfReg[XFR_TEXGEN + ucoord[0]];
		const bool proj0 = ((tg0 >> 1) & 1) && ((tg0 >> 4) & 7) == 0;
		if (st.proj0 != proj0) {
			st.proj0 = proj0;
			if (++s_stGen == 0) s_stGen = 1;
		}
	}
	PicaVtx* out;
	uint16_t* oidx;
	// A traced frame draws everything on this path; when the ring is full, its draws still go to
	// the trace (transformed into a heap buffer), they are just not drawn.
	const bool submit = pica_alloc(n, nidx, &out, &oidx);
	static std::vector<PicaVtx> s_traceVtx;
	static std::vector<uint16_t> s_traceIdx;
	if (!submit) {
		if (!s_tracing) return;
		s_traceVtx.resize(n);
		s_traceIdx.resize(nidx);
		out = s_traceVtx.data();
		oidx = s_traceIdx.data();
	}
	const uint64_t perfXform0 = perfTick();

	// Projection (GXSetProjection) and viewport (GXSetViewport), as the GX shader reads them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = rs.xfReg[XFR_PROJ + 6] & 1;
	const float rowScale = efbRowScale();
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1) / rowScale, vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = (xff(XFR_VIEWPORT + 4) - 342.0f) / rowScale,
	            vcz = xff(XFR_VIEWPORT + 5);
	const float kInvZ = 1.0f / 16777215.0f;
	const bool noClip = rs.xfReg[0x05] != 0;

	// The colour channels the TEV stages read (0 or 1) for the rgb and the alpha: material
	// register or vertex colour, times GX's lighting where it is on (gxLighting).
	const int ch = debug ? 0 : tev.rasRgbChan, cha = debug ? 0 : tev.rasAChan;
	const uint32_t cctrl = rs.xfReg[XFR_COLOR0CTRL + ch], actrl = rs.xfReg[XFR_ALPHA0CTRL + cha], mat = rs.xfReg[XFR_MAT0 + ch],
	               amat = rs.xfReg[XFR_MAT0 + cha];
	const bool hasClr = (r_fmt & (VF_CLR0 << ch)) != 0, hasAClr = (r_fmt & (VF_CLR0 << cha)) != 0,
	           hasNrm = (r_fmt & VF_NRM) != 0, hasMtx = (r_fmt & VF_MTX) != 0;
	const bool clrFromVtx = (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasAClr;
	const bool lightOn = ((cctrl >> 1) & 1) || ((actrl >> 1) & 1);
	const uint32_t amb = rs.xfReg[XFR_AMB0 + ch], aamb = rs.xfReg[XFR_AMB0 + cha];
	const uint8_t matRGBA[4] = {uint8_t(mat >> 24), uint8_t(mat >> 16), uint8_t(mat >> 8), uint8_t(amat)};
	const uint32_t defPm = rs.xfReg[XFR_MATIDX_A] & 63;
	const uint32_t matA = rs.xfReg[XFR_MATIDX_A], matB = rs.xfReg[XFR_MATIDX_B];
	const uint8_t defTex[8] = {uint8_t((matA >> 6) & 63),  uint8_t((matA >> 12) & 63), uint8_t((matA >> 18) & 63),
	                           uint8_t((matA >> 24) & 63), uint8_t(matB & 63),         uint8_t((matB >> 6) & 63),
	                           uint8_t((matB >> 12) & 63), uint8_t((matB >> 18) & 63)};
	uint8_t dbg[3] = {0, 0, 0};
	if (debug) debugColor(hasMtx ? r_verts[L.mtx] : defPm, r_verts, r_stride, nidx, dbg);

	for (uint32_t i = 0; i < n; i++) {
		const uint8_t* v = r_verts + size_t(i) * r_stride;
		float pos[3];
		memcpy(pos, v, 12);
		uint32_t pm = hasMtx ? v[L.mtx] : defPm;
		const float* m = xfMtx(pm * 4);
		float vx = m[0] * pos[0] + m[1] * pos[1] + m[2] * pos[2] + m[3];
		float vy = m[4] * pos[0] + m[5] * pos[1] + m[6] * pos[2] + m[7];
		float vz = m[8] * pos[0] + m[9] * pos[1] + m[10] * pos[2] + m[11];
		float cx, cy, cz, cw;
		if (!ortho) {
			cx = p0 * vx + p1 * vz;
			cy = p2 * vy + p3 * vz;
			cz = p4 * vz + p5;
			cw = -vz;
		} else {
			cx = p0 * vx + p1;
			cy = p2 * vy + p3;
			cz = p4 * vz + p5;
			cw = 1.0f;
		}
		PicaVtx& o = out[i];
		o.x = vcx * cw + vsx * cx;
		o.y = vcy * cw + vsy * cy;
		o.z = (vcz * cw + vsz * cz) * kInvZ;
		// GX clipping off (XF 0x1005): nothing is clipped by depth, it is clamped. The PICA always
		// clips, so keep the depth inside its range (skies sit on or past the far plane).
		if (noClip && cw > 0.f) o.z = o.z < 0.f ? 0.f : o.z > cw ? cw : o.z;
		o.w = cw;

		float q0 = 1.f;
		for (int u = 0; u < PICA_MAX_TEX; u++) {
			float* tc = o.tc[u];
			if (!utex[u]) {
				tc[0] = tc[1] = 0.f;
				continue;
			}
			float q = 1.f;
			texCoord(ucoord[u], v, L, r_fmt, defTex, tc[0], tc[1], u == 0 && st.proj0 ? &q : nullptr);
			if (const EfbCopy* c = ucopy[u]) {  // through the copy's source rectangle into its texture's window
				float as, bs, at, bt;
				copyMap(c, as, bs, at, bt);
				tc[0] = tc[0] * as + bs * q;
				tc[1] = tc[1] * at + bt * q;
			}
			if (u == 0) q0 = q;
		}
		if (st.proj0) o.tc[2][0] = q0;

		const uint8_t* src = debug ? dbg : clrFromVtx ? v + L.clr[ch] : matRGBA;
		float f[4] = {1.f, 1.f, 1.f, 1.f};
		if (lightOn || (debug && hasNrm)) {
			float vn[3];
			const float* np = nullptr;
			if (hasNrm) {
				float nrm[3];
				memcpy(nrm, v + L.nrm, 12);
				const float* nm = xfMtx(kXfNrm + pm * 3);  // normal matrices: 3x3, word 0x400 + 3 * index
				vn[0] = nm[0] * nrm[0] + nm[1] * nrm[1] + nm[2] * nrm[2];
				vn[1] = nm[3] * nrm[0] + nm[4] * nrm[1] + nm[5] * nrm[2];
				vn[2] = nm[6] * nrm[0] + nm[7] * nrm[1] + nm[8] * nrm[2];
				const float len = sqrtf(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
				for (int k = 0; k < 3; k++) vn[k] = len > 0.f ? vn[k] / len : 0.f;
				np = vn;
			}
			if (debug) {  // debug colours: shaded by one light from up-front
				const float d = 0.3f * vn[0] + 0.6f * vn[1] + 0.74f * vn[2];
				f[0] = f[1] = f[2] = 0.35f + 0.65f * (d > 0.f ? d : 0.f);
			} else {
				const float vp[3] = {vx, vy, vz};
				gxLighting(cctrl, actrl, amb, aamb, vp, np, hasClr ? v + L.clr[ch] : nullptr, hasAClr ? v + L.clr[cha] : nullptr,
				           f);
			}
		}
		o.rgba[0] = clamp255(src[0] * f[0]);
		o.rgba[1] = clamp255(src[1] * f[1]);
		o.rgba[2] = clamp255(src[2] * f[2]);
		o.rgba[3] = clamp255((alphaFromVtx ? v[L.clr[cha] + 3] : matRGBA[3]) * f[3]);
	}
	memcpy(oidx, r_idx, size_t(nidx) * 2);

	if (s_tracing) {
		float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, z0 = 1e9f, z1 = -1e9f, w0 = 1e9f, w1 = -1e9f;
		unsigned nan = 0;
		for (uint32_t i = 0; i < n; i++) {
			const PicaVtx& o = out[i];
			if (o.w != o.w || o.x != o.x || o.y != o.y) nan++;
			w0 = o.w < w0 ? o.w : w0, w1 = o.w > w1 ? o.w : w1;
			if (o.w <= 0.f) continue;
			const float x = o.x / o.w, y = o.y / o.w, z = o.z / o.w;
			x0 = x < x0 ? x : x0, x1 = x > x1 ? x : x1, y0 = y < y0 ? y : y0, y1 = y > y1 ? y : y1;
			z0 = z < z0 ? z : z0, z1 = z > z1 ? z : z1;
		}
		fprintf(tf(),
		        "[trace] draw %u tev#%u v%u x %.0f..%.0f y %.0f..%.0f z %.4f..%.4f blend %u eq %u/%u f %u,%u/%u,%u bc %08lx "
		        "mask %x depth %u f%u w%u sc %u %u,%u-%u,%u atest %u f%u r%u%s stages %u tr %d units %d copy %d proj %u v0 %02x%02x%02x%02x\n",
		        s_traceDraw++, dumpId, n, x0, x1, y0, y1, z0, z1, st.blend, st.blend_eq, st.blend_aeq, st.blend_src, st.blend_dst,
		        st.blend_asrc, st.blend_adst, (unsigned long)st.blend_color, st.write_mask, st.depth_test, st.depth_func,
		        st.depth_write, st.scissor, st.sc_x0, st.sc_y0, st.sc_x1, st.sc_y1, st.alpha_test, st.alpha_func, st.alpha_ref,
		        st.alpha_test2 ? " +2nd" : "", st.nstages, translated, tev.nUnits, ucopy[0] != nullptr, st.proj0, n ? out[0].rgba[0] : 0, n ? out[0].rgba[1] : 0,
		        n ? out[0].rgba[2] : 0, n ? out[0].rgba[3] : 0);
		const float* m0 = xfMtx((hasMtx ? r_verts[L.mtx] : defPm) * 4);
		float p0f[3] = {0, 0, 0};
		if (n) memcpy(p0f, r_verts, 12);
		for (int u = 0; u < tev.nUnits; u++) {
			const int map = tev.pairMap[tev.unitPair[u]];
			const uint32_t img0 = rs.bp[bpTexReg(BP_TX_IMAGE0, map)], tl = rs.bp[bpTexReg(BP_TX_TLUT, map)];
			char bytes[40] = "";
			if (const uint8_t* tp = rs.texImage[map])
				for (int b = 0; b < 16; b++) snprintf(bytes + 2 * b, 3, "%02x", tp[b]);
			fprintf(tf(), "[trace]   unit %d: map %d mode %06lx/%06lx\n", u, map,
			        (unsigned long)(rs.bp[bpTexReg(BP_TX_MODE0, map)] & 0xFFFFFF), (unsigned long)(rs.bp[bpTexReg(BP_TX_MODE1, map)] & 0xFFFFFF));
			fprintf(tf(), "[trace]   unit %d: map %d fmt %u %ux%u at %p [%s] tlut %03x/%u avg rgba %08lx (min a %u) tc0 %g %g%s\n", u,
			        map, (img0 >> 20) & 15, (img0 & 0x3FF) + 1, ((img0 >> 10) & 0x3FF) + 1, (const void*)rs.texImage[map], bytes,
			        tl & 0x3FF, (tl >> 10) & 3, (unsigned long)traceTexAvg(map), unsigned(s_traceMinA),
			        n ? out[0].tc[u][0] : 0.f, n ? out[0].tc[u][1] : 0.f,
			        ucopy[u] ? " (EFB copy)" : utex[u] ? "" : " (none)");
			if (!ucopy[u] && utex[u]) traceTexCheck(tf(), u, map, utex[u]);
			// A 256x256 I8 texture (the goop layer's height map): saved once as a PGM image.
			if (!ucopy[u] && rs.texImage[map] && ((img0 >> 20) & 15) == 1 && (img0 & 0x3FF) == 255 &&
			    ((img0 >> 10) & 0x3FF) == 255) {
				static std::vector<const void*> saved;
				const uint8_t* tp = rs.texImage[map];
				if (std::find(saved.begin(), saved.end(), (const void*)tp) == saved.end()) {
					saved.push_back(tp);
					char path[64];
					snprintf(path, sizeof path, "sdmc:/sms/tex-%08lx.pgm", (unsigned long)(uintptr_t)tp);
					if (FILE* f = fopen(path, "wb")) {
						fprintf(f, "P5 256 256 255\n");
						for (int y = 0; y < 256; y++)
							for (int x = 0; x < 256; x++) fputc(tp[((y >> 2) * 32 + (x >> 3)) * 32 + (y & 3) * 8 + (x & 7)], f);
						fclose(f);
						fprintf(tf(), "[trace]     I8 texture saved to %s\n", path);
					}
				}
			}
			// An all-zero texture (the water's wave map): the 32 bytes before it (a .bti header
			// there?) and how much of its image is zero.
			if (!ucopy[u] && rs.texImage[map] && traceTexAvg(map) == 0) {
				const uint8_t* tp = rs.texImage[map];
				char hdr[72] = "";
				for (int b = 0; b < 32; b++) snprintf(hdr + 2 * b, 3, "%02x", tp[b - 32]);
				const uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1;
				uint32_t nz = 0, sz = w * h / 2;
				for (uint32_t b = 0; b < sz; b++) nz += tp[b] != 0;
				fprintf(tf(), "[trace]     zero texture: 32 bytes before [%s], %lu of the first %lu bytes not zero\n", hdr,
				        (unsigned long)nz, (unsigned long)sz);
			}
		}
		fprintf(tf(), "[trace]   w %g..%g nan %u v0 %g %g %g %g pos %g %g %g pm %u m %g %g %g %g / %g %g %g %g / %g %g %g %g proj %g %g %g %g %g %g %s vp %g %g %g %g %g %g fmt %x\n",
		        w0, w1, nan, n ? out[0].x : 0.f, n ? out[0].y : 0.f, n ? out[0].z : 0.f, n ? out[0].w : 0.f, p0f[0], p0f[1], p0f[2],
		        hasMtx ? r_verts[L.mtx] : defPm, m0[0], m0[1], m0[2], m0[3], m0[4], m0[5], m0[6], m0[7], m0[8], m0[9], m0[10],
		        m0[11], p0, p1, p2, p3, p4, p5, ortho ? "ortho" : "persp", vsx, vsy, vsz, vcx, vcy, vcz, r_fmt);
	}
	perfAdd(PERF_XFORM, perfTick() - perfXform0);
	{
		PerfScope ps(PERF_SUBMIT);
		if (submit) {
			pica_draw_same = s_stGen;
			pica_draw_triangles(out, oidx, nidx, &st);
		}
	}
}

// The PICA200 draw: r_verts/r_idx hold the batch, rs the register state.
static void drawBatch() {
	if (r_class == PRIM_TRIS && r_count) drawTriangles();  // lines and points: not yet
	switch (r_class) {
	case PRIM_TRIS:   s_frame.triangles += uint32_t(r_nidx / 3); break;
	case PRIM_LINES:  s_frame.lines += uint32_t(r_nidx / 2); break;
	case PRIM_POINTS: s_frame.points += uint32_t(r_nidx); break;
	}
	s_frame.vertices += r_count;
	s_frame.batches++;
}

// ---- render thread ---------------------------------------------------------------------------
// The renderer (textures, TEV translation, vertex setup, citro3d) runs on its own CPU core: the
// game thread queues records in a single-producer, single-consumer ring and goes on with the game.
// Records: the registers changed since the last one (the front half reports each write:
// bpTouched/xfTouched/tlutTouched, patch 0016), each batch with its vertices and 16-bit indices
// (a resident batch: its indices only), and each EFB copy. The game thread runs at most one frame
// ahead of the render thread. GX3DS_SetRenderThread(0) (D-pad left at launch) draws each record
// as it is queued, on the game thread.
enum RecType : uint32_t { RC_WRAP = 1, RC_STATE, RC_DRAW, RC_COPY, RC_J3DMAT, RC_J3DXF, RC_J3DSIG };
struct RecHdr {
	uint32_t type, size;  // size: the whole record, a multiple of 8
};
struct DrawRec {
	uint32_t cls, fmt, stride, count, nidx;
	const uint8_t* arena;  // resident: the batch's first vertex in its page; else the vertices follow
	uint8_t* ext;          // a batch too large for the ring: its vertices and indices, malloc'd
	VtxFmtLayout L;
	uint32_t kreg[8];      // the registers the front half also sets directly (gx_api.cpp)
	const uint8_t* texImage[8];
	void* copyDest;
	uint32_t stateOff;     // non-zero: the registers to apply first, at this offset in the record
	uint8_t cut;           // pixels cut (profile runs)
	uint8_t amtx;          // resident with matrix indices: pnMax and texMtx hold its runs' matrices
	uint8_t pnMax, texMtx[8];
	uint32_t nspans;       // non-zero: a resident batch, its data nspans IdxSpans (arena at the page's start)
	uint32_t bmin;         // its first vertex in the page
	uint8_t native;        // native J3D draws (kreg and texture images as the native ops leave them)
};
// Where a draw record's data starts (vertices, then indices; or nothing, with ext).
static const uint32_t kDrawHead = uint32_t(sizeof(RecHdr) + sizeof(DrawRec) + 7) & ~7u;
struct CopyRec {
	uint32_t ctrl;
	void* copyDest;
	uint64_t gAcc[PERF_COUNT];  // the game thread's phases since the last copy
	uint64_t gWait[3];          // its waits for the render thread so far: queue full, frame ahead, arena reset
	uint64_t qBytes;            // bytes queued so far
	uint32_t qPeak, qExt;       // most bytes in the queue since the last copy; batches kept outside it so far
};

static uint8_t* s_ring;
static uint32_t s_ringCap, s_ringMask;
static std::atomic<uint32_t> s_head{0}, s_tail{0};  // bytes queued (game), bytes done (render)
static uint32_t s_wpos;                               // game: where the next record goes
static std::atomic<uint32_t> s_rDone{0};              // display copies the render side finished
static std::atomic<int> s_rSleeping{0}, s_gSleeping{0};
static bool s_rtWanted = true;
static int s_rtCore = -1;                             // the render thread's core; -1: none
static uint32_t s_gFrames;                            // display copies queued
static bool s_bNative;                                // the open batch's draws are native J3D ones (j3dBatchNative)
// Registers written since the last state record (game side).
static uint32_t s_dBp[8], s_dXm[4], s_dTlut[32];  // BP registers, 16-word XF memory blocks, 512-byte TLUT blocks
// The bits set in each mask, kept as they are set: stateBytes() is asked many times a frame, and a
// population count of all 44 words cost the GX thread ~7% of its time (the ARM11 has no popcount
// instruction).
static uint32_t s_dNbp, s_dNxm, s_dNt;
// The XF registers written since the last state record (bit r: register r), and their number: a
// record carries only these (it carried all 88 whenever one changed). The render side gives every
// register its last sent value again where native J3D changed it since (renderState), as copying
// all 88 did.
static uint32_t s_dXrMask[3], s_dNxr;
static bool s_dXr, s_dAny, s_dAll = true;
static const uint32_t kXmBlocks = XF_MEM_WORDS / 16, kTlutBlocks = TMEM_TLUT_SIZE / 512;
// Profiler. Game side: waits, queue use. Render side: busy and idle time, and the game side's
// numbers as the last copy record brought them.
static uint64_t s_gWait[3], s_qBytes;
static uint32_t s_qPeak, s_qExt;
static std::atomic<uint64_t> s_rBusy{0};  // (read by profile runs on the game thread: GX3DS_ThreadTicks)
static uint64_t s_rIdle, s_perfRBusy0, s_perfRIdle0;
static uint64_t s_gWaitSeen[3], s_perfGWait0[3], s_qBytesSeen, s_perfQBytes0;
static uint32_t s_qPeakWin, s_qExtSeen;

static bool renderThreaded() { return s_rtCore >= 0; }
static uint64_t gameWaitTicks() {
	uint64_t t = 0;
	for (int i = 0; i < 3; i++) t += s_gWaitSeen[i] - s_perfGWait0[i];
	return t;
}
static void renderThreadPerf(double frames) {
	const double perMs = 1.0 / frames / kTicksPerMs;
	double w[3];
	for (int i = 0; i < 3; i++) {
		w[i] = double(s_gWaitSeen[i] - s_perfGWait0[i]) * perMs;
		s_perfGWait0[i] = s_gWaitSeen[i];
	}
	const double kb = double(s_qBytesSeen - s_perfQBytes0) / frames / 1024.0;
	s_perfQBytes0 = s_qBytesSeen;
	if (s_rtCore >= 0) {
		fprintf(stderr,
		        "[perf]   render thread (core %d): busy %.1f, idle %.1f ms/frame | %s thread waited for it %.1f "
		        "ms/frame (queue full %.1f, a frame ahead %.1f, arena reset %.1f) | queue %.0f KB/frame, peak %u of %u "
		        "KB, %lu big batches outside it so far\n",
		        s_rtCore, double(s_rBusy - s_perfRBusy0) * perMs, double(s_rIdle - s_perfRIdle0) * perMs,
		        streamOn() ? "GX" : "game",
		        w[0] + w[1] + w[2], w[0], w[1], w[2], kb, unsigned(s_qPeakWin >> 10), unsigned(s_ringCap >> 10),
		        (unsigned long)s_qExtSeen);
	} else {
		fprintf(stderr, "[perf]   render thread: off (%s), records drawn on the game thread as queued | queue %.0f "
		        "KB/frame\n", s_rtWanted ? "could not start" : "Left held at launch", kb);
	}
	static uint32_t s_toHeap0;
	fprintf(stderr, "[perf]   %s thread queueing (in batch): registers %.2f, vertices %.2f, indices %.2f ms/frame | "
	        "%.1f batches/frame moved out of the queue\n", streamOn() ? "GX" : "game", s_perfAvgMs[PERF_QSTATE], s_perfAvgMs[PERF_QVERT],
	        s_perfAvgMs[PERF_QIDX], (s_batchToHeap - s_toHeap0) / frames);
	s_toHeap0 = s_batchToHeap;
	s_perfRBusy0 = s_rBusy;
	s_perfRIdle0 = s_rIdle;
	s_qPeakWin = 0;
}

// Front-half hooks (patch 0016).
void bpTouched(uint32_t reg) {
	uint32_t& w = s_dBp[(reg >> 5) & 7];
	const uint32_t bit = 1u << (reg & 31);
	s_dNbp += (w & bit) == 0;
	w |= bit;
	s_dAny = true;
}
void xfTouched(uint32_t addr) {
	if (addr < XF_MEM_WORDS) {
		const uint32_t b = addr >> 4;
		uint32_t& w = s_dXm[b >> 5];
		const uint32_t bit = 1u << (b & 31);
		s_dNxm += (w & bit) == 0;
		w |= bit;
	} else {
		const uint32_t r = addr - XF_REG_BASE;
		if (r < XF_REG_COUNT) {
			uint32_t& w = s_dXrMask[r >> 5];
			const uint32_t bit = 1u << (r & 31);
			s_dNxr += (w & bit) == 0;
			w |= bit;
		}
		s_dXr = true;
	}
	s_dAny = true;
}
void tlutTouched(uint32_t off, uint32_t bytes) {
	if (!bytes) return;
	for (uint32_t b = off >> 9, e = (off + bytes - 1) >> 9; b <= e && b < kTlutBlocks; b++) {
		uint32_t& w = s_dTlut[b >> 5];
		const uint32_t bit = 1u << (b & 31);
		s_dNt += (w & bit) == 0;
		w |= bit;
	}
	s_dAny = true;
}
void stateReset() { s_dAll = true; }

// Game thread: waits until ready(), sleeping on event 1 (the render thread signals it as it goes).
template <class F>
static void gameWaitFor(uint64_t& acc, F ready) {
	if (ready()) return;
	const uint64_t t0 = svcGetSystemTick();
	for (;;) {
		s_gSleeping.store(1);
		if (ready()) break;
		pica_event_wait(1);
		s_gSleeping.store(0);
		if (ready()) break;
	}
	s_gSleeping.store(0);
	const uint64_t dt = svcGetSystemTick() - t0;
	acc += dt;
	GX3DS_OursTicks += dt;  // not the front half's own time
	GX3DS_GxWaitTicks += dt;
}

static void renderDrain();
static void renderMain(void*);

static void rcInit() {
	if (s_ring) return;
	// Small enough to stay in the New 3DS's 2 MB L2 cache: the game thread writes each byte and the
	// render thread reads it moments later (an 8 MB ring went to memory and back every frame).
	for (uint32_t cap = 1u << 20; cap >= (1u << 20) && !s_ring; cap >>= 1) {  // (the first state record is ~0.5 MB)
		s_ring = static_cast<uint8_t*>(memalign(64, cap));
		s_ringCap = cap;
	}
	if (!s_ring) fatal("gx3ds: no memory for the render queue");
	s_ringMask = s_ringCap - 1;
	s_rtCore = s_rtWanted ? pica_thread_start(renderMain, nullptr) : -1;
	if (s_rtCore >= 0)
		fprintf(stderr, "[gx3ds] render thread on core %d, queue %u KB\n", s_rtCore, unsigned(s_ringCap >> 10));
	else
		fprintf(stderr, "[gx3ds] render thread %s: drawing on the game thread, queue %u KB\n",
		        s_rtWanted ? "could not start" : "off (Left held at launch)", unsigned(s_ringCap >> 10));
}

// Compact rings (GX3DS_SetCompactRings, bit 1; profile runs): when the render thread has caught up
// and the next record would start past kRewindAt, the queue starts again at its beginning (a wrap
// record), so it keeps reusing the same few hundred KB, which stay in the L2 cache the cores share,
// instead of cycling through the whole 1 MB (the game thread slows down sharply when another core
// sweeps the L2: prof_3ds.cpp's part 4). s_qSpan: the furthest a record has reached into the ring
// since GX3DS_RingStats last read it.
static const uint32_t kRewindAt = 128u << 10;
static std::atomic<bool> s_qCompact{false};
static uint32_t s_qSpan;
static void rcRewind() {
	const uint32_t off = s_wpos & s_ringMask;
	if (!s_qCompact.load(std::memory_order_relaxed) || off < kRewindAt || s_tail.load() != s_wpos) return;
	RecHdr* h = reinterpret_cast<RecHdr*>(s_ring + off);
	h->type = RC_WRAP;
	h->size = s_ringCap - off;
	s_wpos += s_ringCap - off;
	s_head.store(s_wpos);
}

// Room for a record of `size` bytes (a multiple of 8, at most the ring's size), contiguous.
static uint8_t* rcReserve(uint32_t size) {
	if (!s_vdata.inRing()) rcRewind();  // (a batch being written in the queue stays where it is)
	uint32_t off = s_wpos & s_ringMask;
	if (off + size > s_ringCap) {  // not before the end: a wrap record there, and on from the start
		const uint32_t pad = s_ringCap - off;
		gameWaitFor(s_gWait[0], [&] { return s_wpos + pad - s_tail.load() <= s_ringCap; });
		RecHdr* h = reinterpret_cast<RecHdr*>(s_ring + off);
		h->type = RC_WRAP;
		h->size = pad;
		s_wpos += pad;
		s_head.store(s_wpos);
		off = 0;
	}
	gameWaitFor(s_gWait[0], [&] { return s_wpos + size - s_tail.load() <= s_ringCap; });
	const uint32_t used = s_wpos + size - s_tail.load();
	if (used > s_qPeak) s_qPeak = used;
	if (off + size > s_qSpan) s_qSpan = off + size;
	return s_ring + off;
}

static void rcCommit(uint32_t size) {
	s_wpos += size;
	s_qBytes += size;
	s_head.store(s_wpos);
	if (s_rtCore < 0) renderDrain();
	else if (s_rSleeping.load()) pica_event_signal(0);
}

static inline uint32_t popcount(const uint32_t* m, int words) {
	uint32_t n = 0;
	for (int i = 0; i < words; i++) n += uint32_t(__builtin_popcount(m[i]));
	return n;
}

// The registers written since the last state record, as one. Layout after the header: flags (bit
// 0: XF registers follow, bit 1: XF memory blocks, bit 2: TLUT blocks), BP mask (8 words) and the
// registers it names, then if flagged the XF memory block mask (4) and the blocks, the TLUT block
// mask (32) and the blocks, and the XF register mask (3) and the registers it names. (The masks of no blocks are left out: TLUTs
// change a few times a frame, and copying and scanning their 32-word mask in every record cost the
// GX and render threads more than the registers.)
// Bytes of the registers written since the last state record (0: none).
static uint32_t stateBytes() {
	if (s_dAll) {
		memset(s_dBp, 0xFF, sizeof s_dBp);
		memset(s_dXm, 0, sizeof s_dXm);
		for (uint32_t b = 0; b < kXmBlocks; b++) s_dXm[b >> 5] |= 1u << (b & 31);
		memset(s_dTlut, 0xFF, sizeof s_dTlut);
		s_dNbp = popcount(s_dBp, 8);
		s_dNxm = popcount(s_dXm, 4);
		s_dNt = popcount(s_dTlut, 32);
		for (uint32_t r = 0; r < XF_REG_COUNT; r++) s_dXrMask[r >> 5] |= 1u << (r & 31);
		s_dNxr = XF_REG_COUNT;
		s_dXr = s_dAny = true;
		s_dAll = false;
	}
	if (!s_dAny) return 0;
	return 4 + 32 + s_dNbp * 4 + (s_dNxm ? 16 + s_dNxm * 64 : 0) + (s_dNt ? 128 + s_dNt * 512 : 0) +
	       (s_dXr ? 12 + s_dNxr * 4 : 0);
}

// Writes them (stateBytes() bytes) and starts over.
static void writeState(uint8_t* dst) {
	PerfScope ps(PERF_QSTATE);
	uint32_t* w = reinterpret_cast<uint32_t*>(dst);
	*w++ = (s_dXr ? 1 : 0) | (s_dNxm ? 2 : 0) | (s_dNt ? 4 : 0);
	memcpy(w, s_dBp, 32);
	w += 8;
	for (int k = 0; k < 8; k++)
		for (uint32_t m = s_dBp[k]; m; m &= m - 1) *w++ = g.bp[k * 32 + __builtin_ctz(m)];
	if (s_dNxm) {
		memcpy(w, s_dXm, 16);
		w += 4;
		for (int k = 0; k < 4; k++)
			for (uint32_t m = s_dXm[k]; m; m &= m - 1) {
				memcpy(w, &g.xfMem[(k * 32 + __builtin_ctz(m)) * 16], 64);
				w += 16;
			}
		memset(s_dXm, 0, sizeof s_dXm);
	}
	if (s_dNt) {
		memcpy(w, s_dTlut, 128);
		w += 32;
		for (int k = 0; k < 32; k++)
			for (uint32_t m = s_dTlut[k]; m; m &= m - 1) {
				memcpy(w, g.tlutMem + (k * 32 + __builtin_ctz(m)) * 512, 512);
				w += 128;
			}
		memset(s_dTlut, 0, sizeof s_dTlut);
	}
	if (s_dXr) {
		memcpy(w, s_dXrMask, 12);
		w += 3;
		for (int k = 0; k < 3; k++)
			for (uint32_t m = s_dXrMask[k]; m; m &= m - 1) *w++ = g.xfReg[k * 32 + __builtin_ctz(m)];
		memset(s_dXrMask, 0, sizeof s_dXrMask);
		s_dNxr = 0;
	}
	memset(s_dBp, 0, sizeof s_dBp);
	s_dNbp = s_dNxm = s_dNt = 0;
	s_dXr = s_dAny = false;
}

static void emitState() {
	GX_ZONE(GZ_STATE_REC);
	const uint32_t n = stateBytes();
	if (!n) return;
	const uint32_t size = (uint32_t(sizeof(RecHdr)) + n + 7) & ~7u;
	uint8_t* p = rcReserve(size);
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_STATE;
	h->size = size;
	writeState(p + sizeof(RecHdr));
	rcCommit(size);
}

static uint8_t* rcOpenBatch(size_t* cap) {
	rcInit();
	rcRewind();
	const uint32_t want = 128u << 10;  // room before the ring's end, else it starts at the beginning
	uint32_t off = s_wpos & s_ringMask;
	if (off + kDrawHead + want > s_ringCap) {
		const uint32_t pad = s_ringCap - off;
		gameWaitFor(s_gWait[0], [&] { return s_wpos + pad - s_tail.load() <= s_ringCap; });
		RecHdr* h = reinterpret_cast<RecHdr*>(s_ring + off);
		h->type = RC_WRAP;
		h->size = pad;
		s_wpos += pad;
		s_head.store(s_wpos);
		off = 0;
	}
	*cap = s_ringCap - off - kDrawHead;
	return s_ring + off + kDrawHead;
}

static void rcBatchRoom(size_t bytes) {
	const uint32_t need = kDrawHead + uint32_t(bytes);
	gameWaitFor(s_gWait[0], [&] { return s_wpos + need - s_tail.load() <= s_ringCap; });
	const uint32_t used = s_wpos + need - s_tail.load();
	if (used > s_qPeak) s_qPeak = used;
	const uint32_t end = (s_wpos & s_ringMask) + need;
	if (end > s_qSpan) s_qSpan = end;
}

static uint32_t s_xrSent[XF_REG_COUNT];  // render side: each XF register's last value sent
static uint32_t s_xrNative[3];            // ... the registers native J3D changed since (bit r)
static void renderState(const uint8_t* p) {
	const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
	const uint32_t flags = *w++;
	if ((flags & 5) || w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) s_rsGen++;
	uint32_t m[32];
	memcpy(m, w, 32);
	w += 8;
	for (int k = 0; k < 8; k++)
		for (uint32_t b = m[k]; b; b &= b - 1) rs.bp[k * 32 + __builtin_ctz(b)] = *w++;
	if (flags & 2) {
		memcpy(m, w, 16);
		w += 4;
		for (int k = 0; k < 4; k++)
			for (uint32_t b = m[k]; b; b &= b - 1) {
				const uint32_t w0 = (k * 32 + __builtin_ctz(b)) * 16;
				memcpy(&rs.xfMem[w0], w, 64);
				w += 16;
				xfMemChanged(w0, w0 + 15);
			}
	}
	if (flags & 4) {
		memcpy(m, w, 128);
		w += 32;
		for (int k = 0; k < 32; k++)
			for (uint32_t b = m[k]; b; b &= b - 1) {
				memcpy(rs.tlutMem + (k * 32 + __builtin_ctz(b)) * 512, w, 512);
				w += 128;
			}
	}
	if (flags & 1) {
		// The XF registers sent, then the last sent value again for each register native J3D
		// changed since: what copying all 88 from the GX thread's set did.
		uint32_t m3[3];
		memcpy(m3, w, 12);
		w += 3;
		for (int k = 0; k < 3; k++)
			for (uint32_t b = m3[k]; b; b &= b - 1) {
				const int r = k * 32 + __builtin_ctz(b);
				rs.xfReg[r] = s_xrSent[r] = *w++;
			}
		for (int k = 0; k < 3; k++)
			for (uint32_t b = s_xrNative[k]; b; b &= b - 1) {
				const int r = k * 32 + __builtin_ctz(b);
				rs.xfReg[r] = s_xrSent[r];
			}
		memset(s_xrNative, 0, sizeof s_xrNative);
	}
}

static void enqueueDraw() {
	const uint8_t* verts;
	uint32_t count;
	if (s_bArena) {
		// Drawn from the resident page where it is, its indices where the arena stored them (spans).
		count = s_bmax;
		verts = s_bArena;
		s_arenaDrawVerts += s_bmax - s_bmin;
		s_fmtVerts[s_bfmt & 0x1FFF] += s_bmax - s_bmin;
	} else {
		count = s_bcount;
		verts = s_vdata.data();
		s_fmtVerts[s_bfmt & 0x1FFF] += s_bcount;
	}
	const uint32_t nspans = s_bArena ? uint32_t(s_bSpans.size()) : 0;
	const uint32_t nidx = s_bArena ? s_bSpanIdx : uint32_t(s_idx.size());
	const size_t vbytes = s_bArena ? 0 : (size_t(count) * s_bstride + 3) & ~size_t(3);
	const size_t ibytes = s_bArena ? size_t(nspans) * sizeof(IdxSpan) : size_t(nidx) * 2;
	const size_t head = kDrawHead;
	uint8_t* p = nullptr;
	uint32_t stateOff = 0;
	size_t size;
	if (!s_bArena && s_vdata.inRing()) {
		// The vertices are already in the queue, after this record's header: the indices and the
		// registers follow them, if they fit before the ring's end.
		const size_t ib4 = (ibytes + 3) & ~size_t(3);
		const uint32_t sb = stateBytes();
		size = (head + vbytes + ib4 + sb + 7) & ~size_t(7);
		p = s_vdata.data() - head;
		if (p == s_ring + (s_wpos & s_ringMask) && size_t(p - s_ring) + size <= s_ringCap) {
			rcBatchRoom(size - head);
			if (sb) {
				stateOff = uint32_t(head + vbytes + ib4);
				writeState(p + stateOff);
			}
		} else {
			s_vdata.moveToHeap();
			verts = s_vdata.data();
			p = nullptr;
		}
	}
	if (!p) emitState();
	if (!p) size = (head + vbytes + ibytes + 7) & ~size_t(7);
	uint8_t* ext = nullptr;
	if (!p && size > s_ringCap / 4) {
		ext = static_cast<uint8_t*>(malloc(vbytes + ibytes));
		if (!ext) return;  // dropped
		size = head;
		s_qExt++;
	}
	const bool inPlace = p != nullptr;
	if (!inPlace) p = rcReserve(uint32_t(size));
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_DRAW;
	h->size = uint32_t(size);
	DrawRec* d = reinterpret_cast<DrawRec*>(p + sizeof(RecHdr));
	d->cls = s_bclass;
	d->fmt = s_bfmt;
	d->stride = s_bstride;
	d->count = count;
	d->nidx = nidx;
	d->arena = s_bArena ? verts : nullptr;
	d->ext = ext;
	d->L = vtxFmtLayout(s_bfmt);
	if (s_bNative) {  // as the native ops leave them (not yet in g)
		j3dEffKregTex(d->kreg, d->texImage);
	} else {
		memcpy(d->kreg, g.kreg, sizeof d->kreg);
		memcpy(d->texImage, g.texImage, sizeof d->texImage);
	}
	d->copyDest = g.copyDest;
	d->stateOff = stateOff;
	d->cut = s_bcut;
	d->amtx = s_bArena && (s_bfmt & VF_MTX);
	d->pnMax = s_bPnMax;
	memcpy(d->texMtx, s_bTexMtx, sizeof d->texMtx);
	d->nspans = nspans;
	d->bmin = s_bArena ? s_bmin : 0;
	d->native = s_bNative;
	uint8_t* data = ext ? ext : p + head;
	if (!s_bArena && !inPlace) {
		PerfScope pv(PERF_QVERT);
		memcpy(data, verts, size_t(count) * s_bstride);
	}
	{
		PerfScope pi(PERF_QIDX);
		if (s_bArena) {
			memcpy(data + vbytes, s_bSpans.data(), ibytes);
		} else {
			uint16_t* ix = reinterpret_cast<uint16_t*>(data + vbytes);
			const uint32_t* si = s_idx.data();
			for (uint32_t i = 0; i < nidx; i++) ix[i] = uint16_t(si[i]);
		}
	}
	rcCommit(uint32_t(size));
}

// Profile runs (prof_3ds.cpp, part 4; GX3DS_SetDropDraws): the render thread applies every record
// but draws nothing, so its draw work and the GPU's are gone while the GX thread works as ever.
static std::atomic<bool> s_dropDraws{false};
static void renderDraw(const RecHdr* h) {
	const DrawRec* d = reinterpret_cast<const DrawRec*>(h + 1);
	if (d->stateOff) renderState(reinterpret_cast<const uint8_t*>(h) + d->stateOff);
	if (memcmp(rs.kreg, d->kreg, sizeof rs.kreg) != 0 || memcmp(rs.texImage, d->texImage, sizeof rs.texImage) != 0 ||
	    rs.copyDest != d->copyDest) {
		memcpy(rs.kreg, d->kreg, sizeof rs.kreg);
		memcpy(rs.texImage, d->texImage, sizeof rs.texImage);
		rs.copyDest = d->copyDest;
		s_rsGen++;
	}
	const uint8_t* data = d->ext ? d->ext : reinterpret_cast<const uint8_t*>(h) + kDrawHead;
	const size_t vbytes = d->arena ? 0 : (size_t(d->count) * d->stride + 3) & ~size_t(3);
	r_class = PrimClass(d->cls);
	r_fmt = d->fmt;
	r_stride = d->stride;
	r_count = d->count;
	r_nidx = d->nidx;
	r_layout = d->L;
	r_arena = d->arena;
	r_verts = d->arena ? d->arena : data;
	r_amtx = d->amtx != 0;
	r_pnMax = d->pnMax;
	memcpy(r_texMtx, d->texMtx, sizeof r_texMtx);
	r_spans = d->nspans ? reinterpret_cast<const IdxSpan*>(data) : nullptr;
	r_nspans = d->nspans;
	r_bmin = d->bmin;
	r_idx = d->nspans ? nullptr : reinterpret_cast<const uint16_t*>(data + vbytes);
	if (!d->cut && !s_dropDraws.load(std::memory_order_relaxed)) drawBatch();  // (part 3 of a profile run: the target object's draws left out)
	free(d->ext);
}

static void renderCopy(const CopyRec* c);

// Native J3D (sms-pc-port patch 0035, gx_fifo.cpp; docs/native-j3d.md stage 1). GX thread: a
// material's image, the XF words a shape loads, and the native flag of the batches that follow,
// as records in order with everything else. Render thread: applied to its registers as the
// commands they stand for would have been.
void j3dBatchNative(bool native) {
	if (native == s_bNative) return;
	if (batchHasVertices()) flushBatch();
	s_bNative = native;
}
void j3dStateOut() {
	GX_ZONE(GZ_J3D_OUT);
	rcInit();
	if (!stateBytes()) return;
	if (batchHasVertices()) flushBatch();
	emitState();
}
void j3dEmitMaterial(const J3DImage* img) {
	rcInit();
	emitState();
	const uint32_t size = (uint32_t(sizeof(RecHdr)) + 8 + 7) & ~7u;
	uint8_t* p = rcReserve(size);
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_J3DMAT;
	h->size = size;
	memcpy(p + sizeof(RecHdr), &img, sizeof img);
	rcCommit(size);
}
void j3dEmitXF(const uint32_t* pairs, uint32_t n) {
	rcInit();
	const uint32_t size = (uint32_t(sizeof(RecHdr)) + 8 + n * 8 + 7) & ~7u;
	uint8_t* p = rcReserve(size);
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_J3DXF;
	h->size = size;
	uint32_t* w = reinterpret_cast<uint32_t*>(p + sizeof(RecHdr));
	w[0] = n;
	memcpy(w + 2, pairs, size_t(n) * 8);
	rcCommit(size);
}
static void renderXFWord(uint32_t a, uint32_t v) {
	if (a < XF_MEM_WORDS) {
		rs.xfMem[a] = v;
		xfMemChanged(a, a);
	} else if (a >= XF_REG_BASE && a < XF_REG_BASE + XF_REG_COUNT && rs.xfReg[a - XF_REG_BASE] != v) {
		const uint32_t r = a - XF_REG_BASE;
		rs.xfReg[r] = v;
		s_xrNative[r >> 5] |= 1u << (r & 31);
		s_rsGen++;
	}
}
static void renderJ3DMat(const uint8_t* body) {
	const J3DImage* img;
	memcpy(&img, body, sizeof img);
	s_rsGen++;
	for (const J3DImage::Bp& b : img->bp) {
		if (b.reg < 256) rs.bp[b.reg] = (rs.bp[b.reg] & ~b.mask) | (b.v & b.mask);
		else rs.kreg[b.reg - 256] = b.v;
	}
	for (const J3DImage::Xf& x : img->xf) renderXFWord(x.addr, x.v);
	for (int m = 0; m < 8; m++)
		if ((img->texMask >> m) & 1) rs.texImage[m] = img->texImage[m];
}
void j3dEmitSig(const uint32_t* w, uint32_t n) {
	rcInit();
	const uint32_t size = (uint32_t(sizeof(RecHdr)) + 8 + n * 4 + 7) & ~7u;
	uint8_t* p = rcReserve(size);
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_J3DSIG;
	h->size = size;
	uint32_t* d = reinterpret_cast<uint32_t*>(p + sizeof(RecHdr));
	d[0] = n;
	memcpy(d + 2, w, size_t(n) * 4);
	rcCommit(size);
}
// Verify mode: the render side's registers against what the GX thread says they should be (see
// gx_fifo.cpp, natVerifySig); each difference counted (s_j3dSigBad), the first ones logged.
static void renderJ3DSig(const uint8_t* body) {
	const uint32_t* w = reinterpret_cast<const uint32_t*>(body);
	const uint32_t* v = w + 2;
	s_j3dSigs++;
	static uint32_t logged;
	auto bad = [&](const char* what, uint32_t i, uint32_t want, uint32_t got) {
		s_j3dSigBad++;
		if (logged++ < 60)
			fprintf(stderr, "[j3d] verify (native draw check %lu): %s %03lx: render side %08lx, should be %08lx\n",
			        (unsigned long)s_j3dSigs, what, (unsigned long)i, (unsigned long)got, (unsigned long)want);
	};
	uint32_t k = 0;
	for (uint32_t i = 0; i < 256; i++, k++)
		if ((rs.bp[i] & 0xFFFFFF) != (v[k] & 0xFFFFFF)) bad("BP", i, v[k], rs.bp[i]);
	k += 16;  // konst colours and texture images: from the draw's record, as for every draw
	for (uint32_t i = 0; i < XF_REG_COUNT; i++, k++)
		if (rs.xfReg[i] != v[k]) bad("XF register", XF_REG_BASE + i, v[k], rs.xfReg[i]);
	for (uint32_t b = 0; b < XF_MEM_WORDS; b += 64, k++) {
		uint32_t h = 2166136261u;
		for (uint32_t a = b; a < b + 64 && a < XF_MEM_WORDS; a++) h = (h ^ rs.xfMem[a]) * 16777619u;
		if (h != v[k]) {
			bad("XF memory block at", b, v[k], h);
			if (logged <= 60)
				for (uint32_t a = b; a < b + 64 && a < XF_MEM_WORDS; a += 16)
					fprintf(stderr, "[j3d]   render side XF %03lx: %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx\n",
					        (unsigned long)a, (unsigned long)rs.xfMem[a], (unsigned long)rs.xfMem[a + 1], (unsigned long)rs.xfMem[a + 2],
					        (unsigned long)rs.xfMem[a + 3], (unsigned long)rs.xfMem[a + 4], (unsigned long)rs.xfMem[a + 5],
					        (unsigned long)rs.xfMem[a + 6], (unsigned long)rs.xfMem[a + 7], (unsigned long)rs.xfMem[a + 8],
					        (unsigned long)rs.xfMem[a + 9], (unsigned long)rs.xfMem[a + 10], (unsigned long)rs.xfMem[a + 11],
					        (unsigned long)rs.xfMem[a + 12], (unsigned long)rs.xfMem[a + 13], (unsigned long)rs.xfMem[a + 14],
					        (unsigned long)rs.xfMem[a + 15]);
		}
	}
}
static void renderJ3DXF(const uint8_t* body) {
	const uint32_t* w = reinterpret_cast<const uint32_t*>(body);
	const uint32_t n = w[0];
	for (uint32_t i = 0; i < n; i++) renderXFWord(w[2 + 2 * i], w[3 + 2 * i]);
}

// Draws every record queued so far.
static void renderDrain() {
	uint32_t t = s_tail.load(std::memory_order_relaxed);
	const uint32_t hd = s_head.load();
	while (t != hd) {
		const RecHdr* h = reinterpret_cast<const RecHdr*>(s_ring + (t & s_ringMask));
		const uint8_t* body = reinterpret_cast<const uint8_t*>(h + 1);
		switch (h->type) {
		case RC_STATE: renderState(body); break;
		case RC_DRAW: renderDraw(h); break;
		case RC_COPY: renderCopy(reinterpret_cast<const CopyRec*>(body)); break;
		case RC_J3DMAT: renderJ3DMat(body); break;
		case RC_J3DXF: renderJ3DXF(body); break;
		case RC_J3DSIG: renderJ3DSig(body); break;
		default: break;
		}
		t += h->size;
		s_tail.store(t);
		if (s_gSleeping.load()) pica_event_signal(1);
	}
}

static void renderMain(void*) {
	t_render = true;
	for (;;) {
		// Busy time is credited after each pass over the queue, not only when it empties: a render
		// thread that stays behind for seconds credited nothing in one profile window and all of it
		// in the next (0.0 ms, then 112.8).
		uint64_t t0 = svcGetSystemTick();
		while (s_head.load() != s_tail.load(std::memory_order_relaxed)) {
			renderDrain();
			const uint64_t t = svcGetSystemTick();
			s_rBusy += t - t0;
			t0 = t;
		}
		const uint64_t t1 = t0;
		SMS3DS_TimelineEvent(3, 0);
		// Alone on its core: poll for a while before sleeping, so the game thread rarely has to
		// wake it (a kernel call per record otherwise).
		if (s_rtCore == 2) {
			const uint64_t until = t1 + uint64_t(0.3 * kTicksPerMs);
			bool more = false;
			for (int i = 0; !more; i++) {
				more = s_head.load() != s_tail.load(std::memory_order_relaxed);
				if ((i & 255) == 255 && svcGetSystemTick() > until) break;
			}
			if (more) {
				s_rIdle += svcGetSystemTick() - t1;
				SMS3DS_TimelineEvent(4, 0);
				continue;
			}
		}
		s_rSleeping.store(1);
		if (s_head.load() == s_tail.load(std::memory_order_relaxed)) pica_event_wait(0);
		s_rSleeping.store(0);
		s_rIdle += svcGetSystemTick() - t1;
		SMS3DS_TimelineEvent(4, 0);
	}
}

// Pixel metrics. The goop counters (TPollutionCounterLayer::drawPollutionLayer) draw each goop map
// over its own size with an alpha test and read how many pixels passed; the game takes 4 off per
// polygon it drew, so a read is the pixels passed plus 4 a triangle (as on the GameCube, and in
// sms-pc-port's occlusion queries). The PICA has no counter: a draw of one quad textured with a
// one-byte map (I4, I8) is counted on the GX thread from the map's memory, through a table of
// which texel values pass (the TEV as GX computes it, then the alpha test). Its memory is the
// readback of the map's last copy, two frames behind. Other draws count their triangles only.
// With 0 the counters went to -8, each, which the event scripts take as a clean level at once
// (Sirena Beach's scrubbing episode ended as its timer started) and the unsigned checks as never
// clean (Noki Bay's goop-sealed rock).
static void pixMetricCount() {
	const uint32_t tris = uint32_t(s_idx.size() / 3);
	s_pmTris += tris;
	if (s_bArena || s_bclass != PRIM_TRIS || s_bcount != 4 || tris != 2) return;
	const uint32_t n = ((g.bp[BP_GENMODE] >> 10) & 15) + 1;
	if (n > 4 || ((g.bp[BP_GENMODE] >> 16) & 7) != 0) return;
	TevSymIn in;
	memset(&in, 0, sizeof in);
	in.numStages = int(n);
	int map = -1;
	for (uint32_t s = 0; s < n; s++) {
		in.ce[s] = g.bp[BP_TEV_COLOR_ENV + 2 * s] & 0xFFFFFF;
		in.ae[s] = g.bp[BP_TEV_COLOR_ENV + 2 * s + 1] & 0xFFFFFF;
		in.order[s] = (g.bp[BP_TREF + (s >> 1)] >> ((s & 1) * 12)) & 0x3FF;
		const uint32_t ks = g.bp[BP_TEV_KSEL + (s >> 1)];
		in.ksel[s] = (s & 1) ? ((ks >> 14) & 0x3FF) : ((ks >> 4) & 0x3FF);
		in.stagePair[s] = -1;
		if ((in.order[s] >> 6) & 1) {
			const int m = int(in.order[s] & 7);
			if (map >= 0 && m != map) return;  // one map only
			map = m;
			in.stagePair[s] = 0;
		}
	}
	if (map < 0) return;
	const uint32_t img0 = g.bp[bpTexReg(BP_TX_IMAGE0, map)];
	const uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1, fmt = (img0 >> 20) & 15;
	const uint8_t* mem = g.texImage[map];
	if (!mem || fmt > 1) return;
	for (int i = 0; i < 4; i++) in.swap[i] = uint8_t((g.bp[BP_TEV_KSEL + 2 * i] & 15) | (g.bp[BP_TEV_KSEL + 2 * i + 1] & 15) << 4);
	for (int r = 0; r < 4; r++) {
		const uint32_t ra = g.bp[BP_TEV_REG + 2 * r], bg = g.bp[BP_TEV_REG + 2 * r + 1];
		in.reg[r][0] = int16_t(sext11(ra & 0x7FF));
		in.reg[r][1] = int16_t(sext11((bg >> 12) & 0x7FF));
		in.reg[r][2] = int16_t(sext11(bg & 0x7FF));
		in.reg[r][3] = int16_t(sext11((ra >> 12) & 0x7FF));
		const uint32_t ka = g.kreg[2 * r], kb = g.kreg[2 * r + 1];
		in.konst[r][0] = uint8_t(ka & 0xFF);
		in.konst[r][1] = uint8_t((kb >> 12) & 0xFF);
		in.konst[r][2] = uint8_t(kb & 0xFF);
		in.konst[r][3] = uint8_t((ka >> 12) & 0xFF);
	}
	in.nPairs = 1;
	for (int c = 0; c < 2; c++) {
		const uint32_t mat = g.xfReg[XFR_MAT0 + c];
		in.rasConst[c][0] = int16_t(mat >> 24);
		in.rasConst[c][1] = int16_t((mat >> 16) & 0xFF);
		in.rasConst[c][2] = int16_t((mat >> 8) & 0xFF);
		in.rasConst[c][3] = int16_t(mat & 0xFF);
	}
	// The map's texel values (a histogram), then which of those pass the alpha test: worked out once
	// per setup and value (the counters' setups stay the same from frame to frame; 256 TEV
	// evaluations a map every frame doubled Noki Bay's GX thread).
	// (a word at a time: goop maps are mostly clean, and a zero word is four clean texels)
	static uint32_t hist[256];
	memset(hist, 0, sizeof hist);
	const uint32_t bytes = fmt == 1 ? w * h : w * h / 2;
	uint32_t zeroWords = 0, i = 0;
	if ((uintptr_t(mem) & 3) == 0) {
		const uint32_t* wp = reinterpret_cast<const uint32_t*>(mem);
		for (; i + 4 <= bytes; i += 4) {
			const uint32_t x = wp[i >> 2];
			if (!x) {
				zeroWords++;
				continue;
			}
			for (int k = 0; k < 4; k++) {
				const uint8_t b = uint8_t(x >> (8 * k));
				if (fmt == 1) hist[b]++;
				else hist[b >> 4]++, hist[b & 15]++;
			}
		}
	}
	for (; i < bytes; i++) {
		if (fmt == 1) hist[mem[i]]++;
		else hist[mem[i] >> 4]++, hist[mem[i] & 15]++;
	}
	hist[0] += zeroWords * (fmt == 1 ? 4 : 8);
	const uint32_t ac = g.bp[BP_ALPHACOMPARE];
	uint64_t key = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { key = (key ^ x) * 1099511628211ull; };
	mix(n | fmt << 8 | uint64_t(ac & 0xFFFFFF) << 16);
	for (uint32_t st = 0; st < n; st++) {
		mix(uint64_t(in.ce[st]) | uint64_t(in.ae[st]) << 24 | uint64_t(in.order[st] & 0x3FF) << 48);
		mix(in.ksel[st]);
	}
	for (int r = 0; r < 4; r++) {
		mix(uint64_t(uint16_t(in.reg[r][0])) | uint64_t(uint16_t(in.reg[r][1])) << 16 | uint64_t(uint16_t(in.reg[r][2])) << 32 |
		    uint64_t(uint16_t(in.reg[r][3])) << 48);
		mix(uint64_t(in.konst[r][0]) | uint64_t(in.konst[r][1]) << 8 | uint64_t(in.konst[r][2]) << 16 | uint64_t(in.konst[r][3]) << 24 |
		    uint64_t(in.swap[r]) << 32);
	}
	struct PassTable {
		uint64_t key;
		uint32_t used;
		uint8_t pass[256];  // 0: not worked out yet, 1: fails, 2: passes
	};
	static PassTable s_tables[32];
	static uint32_t s_tableClock;
	PassTable* t = nullptr;
	for (PassTable& x : s_tables)
		if (x.key == key && x.used) t = &x;
	if (!t) {  // the least recently used one
		t = &s_tables[0];
		for (PassTable& x : s_tables)
			if (x.used < t->used) t = &x;
		t->key = key;
		memset(t->pass, 0, sizeof t->pass);
	}
	t->used = ++s_tableClock;
	const uint32_t ref0 = ac & 0xFF, ref1 = (ac >> 8) & 0xFF, c0 = (ac >> 16) & 7, c1 = (ac >> 19) & 7, op = (ac >> 22) & 3;
	const int prim[4] = {in.rasConst[0][0], in.rasConst[0][1], in.rasConst[0][2], in.rasConst[0][3]};
	uint32_t count = 0;
	for (int v = 0; v < (fmt == 0 ? 16 : 256); v++) {
		if (!hist[v]) continue;
		if (!t->pass[v]) {
			const int tv = fmt == 0 ? v * 17 : v;
			const int tex[1][4] = {{tv, tv, tv, tv}};
			const uint32_t a = uint32_t(tevSymGxAlpha(in, tex, prim));
			const bool p0 = gxCompare(c0, a, ref0), p1 = gxCompare(c1, a, ref1);
			const bool p = op == 0 ? (p0 && p1) : op == 1 ? (p0 || p1) : op == 2 ? (p0 != p1) : (p0 == p1);
			t->pass[v] = p ? 2 : 1;
		}
		if (t->pass[v] == 2) count += hist[v];
	}
	s_pmCount += count;
	if (s_verbose) {  // each map's count about every 5 s
		static const uint8_t* s_seen[16];
		static uint32_t s_when[16];
		int i = 0;
		while (i < 16 && s_seen[i] && s_seen[i] != mem) i++;
		if (i < 16 && (!s_seen[i] || s_rFrames - s_when[i] >= 150)) {
			s_seen[i] = mem;
			s_when[i] = s_rFrames;
			fprintf(stderr, "[gx] pixel count of %p (%ux%u %s): %lu passed\n", (const void*)mem, unsigned(w), unsigned(h),
			        fmt ? "I8" : "I4", (unsigned long)count);
		}
	}
}

void flushBatch() {
	if (!s_ready) return;
	GX_ZONE(GZ_DRAW_REC);
	PerfScopeExcl ps(PERF_BATCH);
	if (batchHasVertices()) {
		if (s_pmActive) pixMetricCount();
		rcInit();
		enqueueDraw();
	}
	s_vdata.clear();
	s_idx.clear();
	s_bSpans.clear();
	s_bSpanIdx = 0;
	s_bcount = 0;
	s_bArena = nullptr;
}

void onStateChange() {
	if (batchHasVertices()) flushBatch();
}

PrimClass primClass(uint8_t op) { return op >= 0xB8 ? PRIM_POINTS : op >= 0xA8 ? PRIM_LINES : PRIM_TRIS; }

// A batch holds only cut or only uncut vertices.
static inline void cutCheck() {
	const uint8_t cut = SMS3DS_ProfCutActive ? uint8_t(SMS3DS_ProfCutNow() != 0) : 0;
	if (cut != s_bcut) {
		if (batchHasVertices()) flushBatch();
		s_bcut = cut;
	}
}

uint8_t* primitiveBegin(uint8_t op, uint32_t n, uint32_t fmt, uint32_t stride) {
	GX_ZONE(GZ_PRIM_BEGIN);
	PrimClass cls = primClass(op);
	cutCheck();
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt || s_bArena)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	size_t at = size_t(s_bcount) * stride;
	s_vdata.resize(at + size_t(n) * stride);
	s_decodeT0 = perfTick();  // the front half decodes the vertices until primitiveEnd (0: not timed)
	return s_vdata.data() + at;
}

uint32_t primitiveIndices(uint8_t op, uint32_t n, uint32_t base, uint32_t* w) {
	uint32_t* const w0 = w;
	auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
		w[0] = base + a;
		w[1] = base + b;
		w[2] = base + c;
		w += 3;
	};
	switch (op & 0xF8) {
	case 0x80:
	case 0x88:  // quads
		for (uint32_t i = 0; i + 3 < n; i += 4) {
			tri(i, i + 1, i + 2);
			tri(i, i + 2, i + 3);
		}
		break;
	case 0x90:  // triangles
		for (uint32_t i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2);
		break;
	case 0x98:  // strip
		for (uint32_t i = 2; i < n; i++) {
			if (i & 1) tri(i - 1, i - 2, i);
			else tri(i - 2, i - 1, i);
		}
		break;
	case 0xA0:  // fan
		for (uint32_t i = 2; i < n; i++) tri(0, i - 1, i);
		break;
	case 0xA8:  // lines
		for (uint32_t i = 0; i + 1 < n; i += 2) {
			*w++ = base + i;
			*w++ = base + i + 1;
		}
		break;
	case 0xB0:  // line strip
		for (uint32_t i = 1; i < n; i++) {
			*w++ = base + i - 1;
			*w++ = base + i;
		}
		break;
	default:  // points
		for (uint32_t i = 0; i < n; i++) *w++ = base + i;
		break;
	}
	return uint32_t(w - w0);
}

void primitiveEnd(uint8_t op, uint32_t n) {
	GX_ZONE(GZ_PRIM_END);
	if (s_decodeT0) {
		perfAdd(PERF_DECODE, svcGetSystemTick() - s_decodeT0);
		s_decodeT0 = 0;
	}
	if (!s_ready || n == 0) return;
	uint32_t base = s_bcount;
	s_bcount += n;
	GX3DS_GameVerts += n;
	if (SMS3DS_ProfOn) SMS3DS_ProfGxVerts(n);
	s_copyDirectVerts += n;
	s_vdata.resize(size_t(s_bcount) * s_bstride);
	size_t at = s_idx.size();
	s_idx.resize(at + size_t(n) * 3);  // no primitive makes more than 3 indices per vertex
	uint32_t k = primitiveIndices(op, n, base, s_idx.data() + at);
	s_idx.resize(at + k);
}

void appendDecoded(PrimClass cls, uint32_t fmt, uint32_t stride, const uint8_t* verts, uint32_t n, const uint32_t* idx,
                   uint32_t nidx) {
	PerfScopeExcl ps(PERF_APPEND);
	cutCheck();
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt || s_bArena)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	if (!s_ready || n == 0) return;
	s_vdata.resize(size_t(s_bcount) * stride);
	s_vdata.insert(s_vdata.end(), verts, verts + size_t(n) * stride);
	uint32_t base = s_bcount;
	s_bcount += n;
	GX3DS_GameVerts += n;
	if (SMS3DS_ProfOn) SMS3DS_ProfGxVerts(n);
	s_copyRunVerts += n;
	size_t at = s_idx.size();
	s_idx.resize(at + nidx);
	for (uint32_t i = 0; i < nidx; i++) s_idx[at + i] = idx[i] + base;
}

// ---- display-list arena: resident vertices ------------------------------------------------
// Cached display-list runs (gx_fifo.cpp, patch 0015: triangles without per-vertex matrices, the
// map and most scenery) are stored once in linear memory the GPU reads, and batches draw them
// where they are: no per-frame copy into the ring (append was ~25-30 ms a frame in Delfino Plaza)
// and no per-frame cache write-back. One page per vertex stride at a time, at most 65535
// vertices so 16-bit indices reach them all; runs on the same page still merge into one draw. A
// run too large for a shared page gets a page of its own.
// Their indices, already relative to the page, stay on the heap and are copied into the batch.
// When the pages may take no more linear memory, uploads stop and, between two frames (the GPU
// done with everything that drew from them), the pages are emptied and the runs stored again as
// they are drawn: at most once every few seconds, so a scene larger than the budget does not
// thrash. GX3DS_SetArena(0) (D-pad up at launch) keeps copying them into each batch instead.
struct ArenaPage {
	uint8_t* base;
	uint32_t stride, cap, used;
	int iblock = -1;  // its runs' current index block (s_ablocks): a page's runs' indices stay together
};
static std::vector<ArenaPage> s_apages;
static int s_apageFor[256];  // current page of each stride, or -1
static std::vector<uint32_t> s_aidx;
static uint32_t s_agen = 1;
static bool s_arenaOn = true, s_aresetDue, s_apagesInit;
static size_t s_abytes;
static uint32_t s_aresetFrame, s_aresets, s_afull;
static std::atomic<bool> s_newStage(false);  // GX3DS_NewStage: the main thread saw a new stage load
static const uint32_t kArenaPage = 256u << 10;
static const size_t kArenaMax = size_t(8) << 20;
static const uint32_t kArenaReserve = 3u << 20;  // linear memory left for textures and the rest
static const uint32_t kArenaResetGap = 300;      // frames between two resets at most
extern "C" void GX3DS_SetArena(int on) { s_arenaOn = on != 0; }

bool arenaEnabled() { return s_arenaOn; }
uint32_t arenaGeneration() { return s_agen; }

// Each stored run's indices are preceded by kRunInfo words: the matrices its vertices use when they
// carry matrix indices, worked out on its first draw from its CPU copy. [0]: bit 31 once worked out,
// low byte the highest position matrix index; [1], [2]: each coordinate's texture matrix index,
// 0xFF where it varies. The render side checks these instead of the page window's vertices.
// [3]: where its 16-bit indices are stored (arenaIdxAlloc), [4]: how many (with padding).
// [5]: the vertices stored (fewer than the run's when duplicates were merged), [6]: bit 0 when its
// triangles were reordered, [7...]: its vertex reuse (runReuse, kRunStat words).
static const uint32_t kRunInfo = 7 + kRunStat;

// Duplicate vertices (the same bytes) are stored once and their indices pointed at that one
// (s_mergeDup; circle pad right at launch: off).
extern "C" void GX3DS_SetMergeDuplicates(int on) { s_mergeDup = on != 0; }
// Their triangles reordered for the vertex cache (tipsify, 16 entries: about a third fewer vertices
// shaded in Delfino Plaza), when merged and the GX state draws them the same in any order
// (s_reorder; A held at launch: off).
extern "C" void GX3DS_SetReorder(int on) { s_reorder = on != 0; }
// The current GX state draws a run's triangles the same in any order: depth test and update on with
// an ordering compare (less or greater, with or without equal), and no blending, logic op or
// subtraction. (Exactly equal depths still go to the later triangle: coplanar overlaps in one run.)
static int orderWhy() {  // 0: free, else the first reason it is not (s_reordWhy)
	const uint32_t z = g.bp[BP_ZMODE], c = g.bp[BP_CMODE0];
	const uint32_t fn = (z >> 1) & 7;  // never, less, equal, lequal, greater, nequal, gequal, always
	if ((c & 3) || (c & (1u << 11))) return 1;
	if (!(z & 1)) return 2;
	if (!(z & 0x10)) return 3;
	if (!(fn == 1 || fn == 3 || fn == 4 || fn == 6)) return 4;
	return 0;
}
static bool orderFree() { return orderWhy() == 0; }

// Distinct vertices by content: canon[v] is the first vertex with v's bytes. Returns their count.
static uint32_t distinctVertices(const uint8_t* verts, uint32_t n, uint32_t stride, std::vector<uint32_t>& canon) {
	static std::vector<uint32_t> table;
	canon.resize(n);
	uint32_t size = 16;
	while (size < n * 2) size <<= 1;
	table.assign(size, ~0u);
	uint32_t distinct = 0;
	for (uint32_t v = 0; v < n; v++) {
		const uint8_t* p = verts + size_t(v) * stride;
		uint32_t h = 2166136261u;
		for (uint32_t b = 0; b < stride; b++) h = (h ^ p[b]) * 16777619u;
		for (uint32_t at = h & (size - 1);; at = (at + 1) & (size - 1)) {
			if (table[at] == ~0u) {
				table[at] = v;
				canon[v] = v;
				distinct++;
				break;
			}
			if (memcmp(verts + size_t(table[at]) * stride, p, stride) == 0) {
				canon[v] = table[at];
				break;
			}
		}
	}
	return distinct;
}

// Tipsify (Sander, Nehab and Barczak 2007): the triangles reordered for a FIFO vertex cache of k
// entries, each keeping its winding. For the [perf] estimate only (runReuse): reordering changes
// the drawing order, which blended triangles show.
static void tipsify(const uint32_t* idx, uint32_t nidx, uint32_t n, uint32_t k, std::vector<uint32_t>& out) {
	static std::vector<uint32_t> start, adj, live, ctime, dead;
	static std::vector<uint8_t> done;
	const uint32_t nt = nidx / 3;
	start.assign(n + 1, 0);
	for (uint32_t i = 0; i < nt * 3; i++) start[idx[i] + 1]++;
	for (uint32_t v = 0; v < n; v++) start[v + 1] += start[v];
	live.assign(n, 0);
	for (uint32_t v = 0; v < n; v++) live[v] = start[v + 1] - start[v];
	adj.resize(nt * 3);
	{
		static std::vector<uint32_t> fill;
		fill.assign(start.begin(), start.end() - 1);
		for (uint32_t t = 0; t < nt; t++)
			for (int j = 0; j < 3; j++) adj[fill[idx[t * 3 + j]]++] = t;
	}
	ctime.assign(n, 0);
	done.assign(nt, 0);
	dead.clear();
	out.clear();
	uint32_t s = k + 1, cursor = 0;
	static std::vector<uint32_t> cand;
	int64_t f = 0;
	while (f >= 0 && live[uint32_t(f)] == 0) f = ++cursor < n ? int64_t(cursor) : -1;
	while (f >= 0) {
		cand.clear();
		for (uint32_t a = start[uint32_t(f)]; a < start[uint32_t(f) + 1]; a++) {
			const uint32_t t = adj[a];
			if (done[t]) continue;
			done[t] = 1;
			for (int j = 0; j < 3; j++) {
				const uint32_t v = idx[t * 3 + j];
				out.push_back(v);
				dead.push_back(v);
				cand.push_back(v);
				live[v]--;
				if (s - ctime[v] > k) ctime[v] = s++;
			}
		}
		int64_t best = -1, bp = -1;
		for (uint32_t v : cand) {
			if (!live[v]) continue;
			int64_t pr = 0;
			if (int64_t(s - ctime[v]) + 2 * int64_t(live[v]) <= int64_t(k)) pr = s - ctime[v];
			if (pr > bp) bp = pr, best = v;
		}
		if (best < 0) {
			while (!dead.empty() && best < 0) {
				const uint32_t d = dead.back();
				dead.pop_back();
				if (live[d]) best = d;
			}
			while (best < 0 && cursor < n) {
				if (live[cursor]) best = cursor;
				else cursor++;
			}
		}
		f = best;
	}
}

// A run's vertex reuse, worked out when it is stored: its triangles, those that continue the one
// before as a strip does (a native strip would draw them with one index each), its distinct
// vertices (the same bytes: the same vertex), and the vertices a FIFO post-transform cache of 8,
// 16 and 32 entries would shade, for the indices as stored and with duplicates merged (the GPU
// shades a vertex again whenever its index has left the cache, and duplicates never hit it).
static void runReuse(const uint8_t* verts, uint32_t n, uint32_t stride, const uint32_t* idx, uint32_t nidx,
                     uint32_t* st) {
	static std::vector<uint32_t> canon, ins, merged, order;
	const uint32_t distinct = distinctVertices(verts, n, stride, canon);
	static const uint32_t kSizes[3] = {8, 16, 32};
	ins.resize(n);
	for (int merged = 0; merged < 2; merged++)
		for (int k = 0; k < 3; k++) {
			std::fill(ins.begin(), ins.end(), ~0u);
			uint32_t shaded = 0;  // misses; ins[v]: the miss count when v went in
			for (uint32_t i = 0; i < nidx; i++) {
				const uint32_t v = merged ? canon[idx[i]] : idx[i];
				if (ins[v] == ~0u || shaded - ins[v] >= kSizes[k]) ins[v] = shaded++;
			}
			st[RS_C8 + merged * 3 + k] = shaded;
		}
	uint32_t strip = 0;
	for (uint32_t t = 3; t + 2 < nidx; t += 3) {
		const uint32_t* a = idx + t - 3;
		const uint32_t* b = idx + t;
		uint32_t m1 = std::max(a[0], std::max(a[1], a[2]));
		uint32_t m2 = a[0] + a[1] + a[2] - m1 - std::min(a[0], std::min(a[1], a[2]));
		int has = 0;
		uint32_t third = 0;
		for (int j = 0; j < 3; j++) {
			if (b[j] == m1 || b[j] == m2) has++;
			else third = b[j];
		}
		if (has == 2 && third == m1 + 1) strip++;
	}
	st[RS_TRIS] = nidx / 3;
	st[RS_STRIP] = strip;
	st[RS_DISTINCT] = distinct;
	// duplicates merged and the triangles reordered (tipsify), with the GPU's cache of about 16
	merged.resize(nidx);
	for (uint32_t i = 0; i < nidx; i++) merged[i] = canon[idx[i]];
	tipsify(merged.data(), nidx, n, 16, order);
	std::fill(ins.begin(), ins.end(), ~0u);
	uint32_t shaded = 0;
	for (uint32_t v : order)
		if (ins[v] == ~0u || shaded - ins[v] >= 16) ins[v] = shaded++;
	st[RS_T16] = order.size() == nidx - nidx % 3 ? shaded : 0;
}

// The runs' indices, 16-bit and relative to their page, in GPU-readable blocks: written once when
// the run is stored, drawn from there (IdxSpan). Each page has blocks of its own, so the runs a
// batch draws from one page, stored one after another, make one span (every span is a draw of its
// own, which on the PICA leaves drawing mode and clears the vertex cache: with the pages' runs
// interleaved, Delfino Plaza's GPU time went up 47 -> 55 ms). Each run's start stays 4-byte aligned
// (as the ring's indices are): an odd count is padded with one degenerate triangle. Blocks count
// towards the arena's budget and go with its reset.
struct IdxBlock {
	uint16_t* base;
	uint32_t cap, used;  // indices
};
static std::vector<IdxBlock> s_ablocks;
static const uint32_t kIdxBlock = 128u << 10;  // indices (256 KB)
static uint16_t* arenaIdxAlloc(ArenaPage& pg, uint32_t n) {
	if (pg.iblock >= 0) {
		IdxBlock& b = s_ablocks[size_t(pg.iblock)];
		if (b.used + n <= b.cap) {
			uint16_t* p = b.base + b.used;
			b.used += n;
			return p;
		}
	}
	// about three indices a vertex the page holds, at most kIdxBlock (a run larger gets its own)
	const uint32_t cap = std::max(n, std::min(kIdxBlock, (pg.cap * 3u + 1023u) & ~1023u));
	const uint32_t bytes = (cap * 2u + 4095u) & ~4095u;
	if (s_abytes + bytes > kArenaMax || pica_linear_free() <= kArenaReserve + bytes) return nullptr;
	uint16_t* p = static_cast<uint16_t*>(pica_linear_alloc(bytes));
	if (!p) return nullptr;
	s_abytes += bytes;
	s_ablocks.push_back(IdxBlock{p, bytes / 2, n});
	pg.iblock = int(s_ablocks.size() - 1);
	return p;
}
static const uint32_t* arenaRunMtx(const ArenaRef& ref, const uint8_t* verts, uint32_t n, uint32_t stride,
                                   uint32_t mtxOff) {
	uint32_t* info = &s_aidx[ref.iOff - kRunInfo];
	if (!(info[0] >> 31)) {
		uint8_t pn = 0, tex[8];
		memcpy(tex, verts + mtxOff + 1, 8);
		for (uint32_t i = 0; i < n; i++) {
			const uint8_t* m = verts + size_t(i) * stride + mtxOff;
			if (m[0] > pn) pn = m[0];
			for (int c = 0; c < 8; c++)
				if (m[1 + c] != tex[c]) tex[c] = 0xFF;
		}
		info[0] = 0x80000000u | pn;
		memcpy(&info[1], tex, 8);
	}
	return info;
}

bool arenaUpload(const uint8_t* verts, uint32_t n, uint32_t stride, const uint32_t* idx, uint32_t nidx, ArenaRef& out) {
	if (!s_ready || !s_arenaOn || n == 0 || nidx == 0 || stride == 0 || stride >= 256) return false;
	if (!s_apagesInit) {
		for (int& p : s_apageFor) p = -1;
		s_apagesInit = true;
	}
	// A page: shared by the runs of one stride, or, for a run larger than that, its own and sized to
	// it (Delfino Plaza's map pieces; 16-bit indices still bound it to 65535 vertices).
	const uint32_t cap = std::min<uint32_t>(65535u, kArenaPage / stride);
	const bool own = n > cap;
	if (n > 65535u) {
		s_aTooBigVerts += n;
		return false;
	}
	int& cur = s_apageFor[stride];
	int pi = cur;
	if (own || pi < 0 || s_apages[pi].used + n > s_apages[pi].cap) {
		const uint32_t bytes = own ? (n * stride + 4095u) & ~4095u : kArenaPage;
		void* b = nullptr;
		if (s_abytes + bytes <= kArenaMax && pica_linear_free() > kArenaReserve + bytes) b = pica_linear_alloc(bytes);
		if (!b) {
			s_afull++;
			s_aNoRoomVerts += n;
			s_aresetDue = true;  // between frames, if the last reset was long enough ago
			return false;
		}
		s_apages.push_back(ArenaPage{static_cast<uint8_t*>(b), stride, own ? n : cap, 0});
		pi = int(s_apages.size() - 1);
		if (!own) cur = pi;
		s_abytes += bytes;
	}
	ArenaPage& pg = s_apages[size_t(pi)];
	const uint32_t nstore = (nidx & 1) ? nidx + 3 : nidx;
	// (n and the page were chosen for the run's own vertex count: merging only stores fewer)
	uint16_t* ix = arenaIdxAlloc(pg, nstore);
	if (!ix) {
		s_afull++;
		s_aNoRoomVerts += n;
		s_aresetDue = true;
		return false;
	}
	const uint32_t vb = pg.used;
	uint8_t* dst = pg.base + size_t(vb) * stride;
	uint32_t nstored = n, flags = 0;
	if (s_mergeDup) {
		static std::vector<uint32_t> canon, slot, merged, order;
		distinctVertices(verts, n, stride, canon);
		slot.resize(n);
		nstored = 0;
		for (uint32_t v = 0; v < n; v++)
			if (canon[v] == v) {
				memcpy(dst + size_t(nstored) * stride, verts + size_t(v) * stride, stride);
				slot[v] = nstored++;
			}
		merged.resize(nidx);
		for (uint32_t i = 0; i < nidx; i++) merged[i] = slot[canon[idx[i]]];
		const uint32_t ntri = nidx - nidx % 3;
		const int why = orderWhy();
		s_reordWhy[why] += nidx;
		if (s_reorder && ntri >= 6 && why == 0) {
			tipsify(merged.data(), ntri, nstored, 16, order);
			if (order.size() == ntri) {
				memcpy(merged.data(), order.data(), size_t(ntri) * 4);
				flags |= 1;
			}
		}
		for (uint32_t i = 0; i < nidx; i++) ix[i] = uint16_t(merged[i] + vb);
	} else {
		for (uint32_t i = 0; i < nidx; i++) ix[i] = uint16_t(idx[i] + vb);
		memcpy(dst, verts, size_t(n) * stride);
	}
	for (uint32_t i = nidx; i < nstore; i++) ix[i] = ix[nidx - 1];
	pica_flush(ix, nstore * 2);
	pica_flush(dst, nstored * stride);
	pg.used += nstored;
	const size_t io = s_aidx.size() + kRunInfo;
	s_aidx.resize(io);
	s_aidx[io - kRunInfo] = 0;  // its matrices: not worked out yet (arenaRunMtx)
	s_aidx[io - kRunInfo + 3] = uint32_t(uintptr_t(ix));
	s_aidx[io - kRunInfo + 4] = nstore;
	s_aidx[io - kRunInfo + 5] = nstored;
	s_aidx[io - kRunInfo + 6] = flags;
	runReuse(verts, n, stride, idx, nidx, &s_aidx[io - kRunInfo + 7]);
	{
		// the vertices a 16-entry FIFO cache shades for the indices as stored (merged, reordered)
		static std::vector<uint32_t> ins;
		ins.assign(nstored, ~0u);
		uint32_t shaded = 0;
		for (uint32_t i = 0; i < nidx; i++) {
			const uint32_t v = uint32_t(ix[i]) - vb;
			if (ins[v] == ~0u || shaded - ins[v] >= 16) ins[v] = shaded++;
		}
		s_aidx[io - kRunInfo + 7 + RS_S16] = shaded;
	}
	out.gen = s_agen;
	out.baseVertex = int32_t((uint32_t(pi) << 16) | vb);
	out.iOff = io;
	s_arenaUploadBytes += uint64_t(nstored) * stride;
	return true;
}

void appendArena(PrimClass cls, uint32_t fmt, uint32_t stride, const ArenaRef& ref, uint32_t n, uint32_t nidx,
                 const uint8_t* verts) {
	PerfScopeExcl ps(PERF_APPEND);
	GX3DS_GameVerts += n;
	if (SMS3DS_ProfOn) SMS3DS_ProfGxVerts(n);
	const ArenaPage& pg = s_apages[size_t(uint32_t(ref.baseVertex) >> 16)];
	const uint32_t vb = uint32_t(ref.baseVertex) & 0xFFFF;
	cutCheck();
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt || s_bArena != pg.base)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	if (!s_ready || n == 0 || nidx == 0) return;
	if (fmt & VF_MTX) {
		const uint32_t* info = arenaRunMtx(ref, verts, n, stride, vtxFmtLayout(fmt).mtx);
		const uint8_t pn = uint8_t(info[0]);
		uint8_t tex[8];
		memcpy(tex, &info[1], 8);
		if (!batchHasVertices()) {
			s_bPnMax = pn;
			memcpy(s_bTexMtx, tex, 8);
		} else {
			if (pn > s_bPnMax) s_bPnMax = pn;
			for (int c = 0; c < 8; c++)
				if (s_bTexMtx[c] != tex[c]) s_bTexMtx[c] = 0xFF;
		}
	}
	const uint32_t* info = &s_aidx[ref.iOff - kRunInfo];
	const uint32_t nstored = info[5];
	if (!batchHasVertices()) {
		s_bArena = pg.base;
		s_bmin = vb;
		s_bmax = vb + nstored;
	} else {
		s_bmin = std::min(s_bmin, vb);
		s_bmax = std::max(s_bmax, vb + nstored);
	}
	const uint16_t* ix = reinterpret_cast<const uint16_t*>(uintptr_t(info[3]));
	const uint32_t ns = info[4];
	s_reuse[0] += nidx;
	s_reuse[1] += nstored;
	for (int j = 0; j < kRunStat; j++) s_reuse[2 + j] += info[7 + j];
	if (info[6] & 1) {
		s_reordDraws++;
		if (!orderFree()) s_reordUnsafe++;
	}
	if (!s_bSpans.empty() && s_bSpans.back().idx + s_bSpans.back().n == ix) s_bSpans.back().n += ns;
	else s_bSpans.push_back(IdxSpan{ix, ns});
	s_bSpanIdx += ns;
}

// The batch reads its indices from s_aidx's copy, never from the front half's runs.
bool batchUsesArena() { return false; }

static size_t arenaPageCount() { return s_apages.size(); }
static size_t arenaBytes() { return s_abytes; }
static uint32_t arenaResets() { return s_aresets; }
static uint32_t arenaFullCount() { return s_afull; }

// After a frame went to the GPU: empty the pages once uploads have stopped for lack of room (or
// at once for a new stage: force). Game side, after a display copy was queued: the render thread
// and the GPU are let finish first.
static void arenaFrameEnd(bool force) {
	if (!s_aresetDue) return;
	s_aresetDue = false;
	if (!force && g_displayFrames - s_aresetFrame < kArenaResetGap && s_aresets) return;
	if (s_rtCore >= 0) gameWaitFor(s_gWait[2], [] { return s_tail.load() == s_wpos; });
	pica_gpu_idle();  // every queued frame done with the pages
	for (const ArenaPage& p : s_apages) pica_linear_free_block(p.base);
	s_apages.clear();
	for (const IdxBlock& b : s_ablocks) pica_linear_free_block(b.base);
	s_ablocks.clear();
	for (int& p : s_apageFor) p = -1;
	s_aidx.clear();
	s_aidx.shrink_to_fit();
	s_abytes = 0;
	s_agen++;
	s_aresets++;
	s_aresetFrame = g_displayFrames;
}

// ---- GX thread: streamed GX ---------------------------------------------------------------------
// The game thread's GX calls only write command bytes (patch 0017: the API sends everything as
// FIFO bytes and keeps its own copy of the registers, as the SDK's __GXData; the write-gather pipe
// fills 64 KB chunks). This thread, on core 1, parses them: the command parser, the display-list
// cache and this file's front half (batches, the render queue) run here, a frame at most behind
// the game, which then only waits when it gets further ahead (J3D double-buffers what the GPU reads
// and expects it read within a frame). Draw-sync tokens go back to the game thread, whose callback
// runs game code: an interrupt source there runs it at its next check point. Off with D-pad right
// held at launch (GX3DS_SetGxThread), and with the render thread off.
// 3DS port: 8 KB chunks (32, the most s_sFree's mask holds): the GX thread starts on a frame's
// commands as soon as the first chunk fills, and 64 KB ones left it idle longer at each frame's
// start, on the frame's critical path in Azahar (41.5 -> 38.6 ms a frame with 16 KB ones).
static const uint32_t kStreamChunk = 8u << 10, kStreamChunks = 32;
static uint8_t* s_sChunk[kStreamChunks];
static uint32_t s_sLen[kStreamChunks];
// Which buffer each chunk fills (by chunk number), and the buffers free. Taken in turn, or with
// compact rings (GX3DS_SetCompactRings, bit 0; profile runs) the lowest free one: the few in use
// are then the ones the GX thread has just read, still in the L2 cache, instead of all 16 (1 MB)
// in turn. The GX thread frees a chunk's buffer before it marks the chunk done. s_sUsed: buffers
// taken since GX3DS_RingStats last read it.
static uint8_t s_sBufOf[kStreamChunks];
static std::atomic<uint32_t> s_sFree{kStreamChunks >= 32 ? ~0u : (1u << (kStreamChunks & 31)) - 1};  // (no shift by 32)
static std::atomic<bool> s_sCompact{false};
static uint32_t s_sNext, s_sUsed;
static std::atomic<uint32_t> s_sHead{0}, s_sTail{0};  // chunks handed over; done (released after parsing)
static std::atomic<uint32_t> s_sFramesDone{0};        // display copies the GX thread has run
static uint32_t s_sFramesSent;                         // ... and the game thread has sent
static bool s_gxWanted = true, s_streamOn;
static std::atomic<uint64_t> s_sWait[4];  // game thread waiting: a free chunk, the frame's end, a drain, a list rewrite
static std::atomic<uint64_t> s_sBusy{0}, s_sIdle{0};  // GX thread: parsing, waiting for chunks
// 3DS port: each thread signals the other's event only while that one waits on it (GXPC_GxIdle for the
// GX thread, s_gameWaits for the game thread): a signal is a kernel call, which costs a lot in Azahar,
// and there were two a chunk. Each sets its flag, then (a full fence between) tests its condition
// again before it waits; the other publishes its change, then (a fence) tests the flag: one of the
// two sees the other's write, so no wake-up is lost (the events are one-shot: a spare signal only
// ends one wait early, and every wait is in a loop).
static std::atomic<int> s_gameWaits{0};
// Whether the GX thread polls before it sleeps (settings.txt gx_poll; on by default, off in builds
// made with SMS3DS_GX_POLL_DEFAULT=0). On the New 3DS the GX thread shares core 1 with the system,
// which caps the application's time there: polling may spend that share for nothing.
#ifndef SMS3DS_GX_POLL_DEFAULT
#define SMS3DS_GX_POLL_DEFAULT 1
#endif
static bool s_gxPoll = SMS3DS_GX_POLL_DEFAULT;
static bool s_gamePoll = true;  // the game thread's waits poll first too (GX3DS_SetGamePoll; the profiler)
enum { EV_GX = 2, EV_GAME = 3 };

// Game thread: waits until ready() (the GX thread signals EV_GAME as it releases chunks and frames).
template <class F>
static void streamWait(std::atomic<uint64_t>& acc, F ready) {
	if (ready()) return;
	const uint64_t t0 = svcGetSystemTick();
	SMS3DS_SampWaiting = int(&acc - s_sWait) + 1;
	// Polled for up to 2 ms before it sleeps, a pause between checks (pica_poll_pause): Azahar runs the
	// cores in turn, a slice at a time, and the game thread saw the GX thread's frame done only at its
	// slice's end, 1.2-1.4 ms after the signal at each frame's end, sleeping or not.
	for (uint32_t i = 1; s_gamePoll; i++) {
		if (ready()) break;
		pica_poll_pause();
		if ((i & 15) == 0 && svcGetSystemTick() - t0 > uint64_t(2.0 * kTicksPerMs)) break;
	}
	for (;;) {
		s_gameWaits.store(1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (ready()) break;
		pica_event_wait(EV_GAME);
	}
	s_gameWaits.store(0, std::memory_order_relaxed);
	SMS3DS_SampWaiting = 0;
	acc += svcGetSystemTick() - t0;
}

// The write-gather pipe's hand-over (gx_fifo.cpp): publishes the full chunk and returns the next,
// once the GX thread is done with what it held last.
// Bytes the game thread has written to the stream (handed over, plus the current buffer's), and
// the writes that took the slow path (GXPC_StreamPuts, patch 0017): a profile run reports them
// per object (prof_3ds.cpp).
static uint64_t s_sBytesSent;
extern "C" uint32_t GXPC_StreamPuts, GXPC_PipeLen;
uint32_t GXPC_StreamPuts;
extern "C" uint64_t GX3DS_StreamWritten(void) { return s_sBytesSent + GXPC_PipeLen; }

static uint8_t* streamSwap(uint8_t* buf, uint32_t len, uint32_t* cap) {
	*cap = kStreamChunk;
	if (buf) {
		s_sBytesSent += len;
		s_sLen[s_sHead.load(std::memory_order_relaxed) % kStreamChunks] = len;
		s_sHead.fetch_add(1, std::memory_order_release);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (GXPC_GxIdle) pica_event_signal(EV_GX);  // (only while the GX thread waits: see s_gameWaits)
	}
	const uint32_t h = s_sHead.load(std::memory_order_relaxed);
	streamWait(s_sWait[0], [h] { return h - s_sTail.load(std::memory_order_acquire) < kStreamChunks; });
	const uint32_t free = s_sFree.load(std::memory_order_acquire);  // (at least one: a chunk is free)
	uint32_t i;
	if (s_sCompact.load(std::memory_order_relaxed)) {
		i = uint32_t(__builtin_ctz(free));
	} else {
		i = s_sNext;
		while (!((free >> i) & 1)) i = (i + 1) % kStreamChunks;
		s_sNext = (i + 1) % kStreamChunks;
	}
	s_sFree.fetch_and(~(1u << i), std::memory_order_relaxed);
	s_sBufOf[h % kStreamChunks] = uint8_t(i);
	s_sUsed |= 1u << i;
	return s_sChunk[i];
}

// Display lists the game rewrites in place. J3D rebuilds a material's list (GD) every time a model
// is entered for drawing, and a list shared by every instance of a model (J3DMaterial's
// newSharedDisplayList) swaps its two buffers once per instance: with an even number on screen the
// same buffer is rewritten every frame, while the GX thread may not yet have read last frame's call
// of it (a crash on Delfino Airstrip: a nested call out of a half-rewritten list). The GameCube
// is safe through the frame's GXDrawDone (THPPlayerDrawDone, before the copy); here each call
// through the stream stamps the 32-byte blocks of MEM1 it covers with the chunk that carries it,
// and before the game starts writing a list (GDInitGDLObj) it waits while the GX thread has not
// finished that chunk. Those waits cost the game thread 4-5 ms a frame on Delfino Plaza, so a list
// written through GDInitGDLObj in the last kDlCopyFrames frames is sent as a copy
// (kStreamOpCallDLCopy, up to kDlCopyMax bytes): the GX thread reads the copy and the game
// rewrites the list without waiting. A list written once (at setup) goes by address again after
// that, as copies are hashed on every call while a list by address is trusted to its write stamps
// (copying every list ever written doubled the bytes hashed on Delfino Plaza). Calls by address
// are stamped as before.
static uint32_t* s_dlMark;  // per 32-byte block of MEM1: 1 + the chunk of its last call (0: none)
static uint8_t* s_dlRw;     // per 32-byte block of MEM1: rwStamp of its last GDInitGDLObj (0: never)
static uint32_t s_dlBlocks, s_dlWaits, s_dlCopies, s_dlCopyBytes;
// 3DS port: the same by 4 KB page, so that the lists in pages no list was ever written in (the
// models' shapes, hundreds of blocks a call) cost a store a page, not one a block. A page's block
// marks are kept only once a list in it has been written (s_dlPageRw): until then its page mark
// stands for each of its blocks (conservatively: a write waits for any call in the page), and its
// first write starts its blocks at the page mark. In a page never written every s_dlRw is 0, so
// streamCopyList's answer there is known without its blocks.
static uint32_t* s_dlPageMark;  // per page: 1 + the chunk of the last call in it (0: none)
static uint8_t* s_dlPageRw;     // per page: 1 once a list in it was written (GDInitGDLObj)
static const uint32_t kDlPageShift = 7;  // 128 blocks a page
static const uint32_t kDlCopyMax = 16u << 10, kDlCopyFrames = 4;
static bool s_dlCopyOn = true;  // off (SELECT held at launch): every rewrite waits, as before copies
static uint8_t rwStamp(uint32_t frame) { return uint8_t(frame % 255u + 1u); }

static bool mem1Blocks(const void* p, uint32_t n, uint32_t& b0, uint32_t& b1) {
	const uintptr_t a = reinterpret_cast<uintptr_t>(p), base = reinterpret_cast<uintptr_t>(port_mem1_base);
	if (!s_dlMark || !n || a < base || a - base >= port_mem1_size) return false;
	b0 = uint32_t(a - base) >> 5;
	b1 = std::min<uint32_t>(uint32_t(a - base + n - 1) >> 5, s_dlBlocks - 1);
	return true;
}
// Game thread, after the call's command is in the stream: the chunk being filled carries its end.
static void streamNoteDL(const void* list, uint32_t n) {
	uint32_t b0, b1;
	if (!mem1Blocks(list, n, b0, b1)) return;
	const uint32_t mark = s_sHead.load(std::memory_order_relaxed) + 1;
	for (uint32_t pg = b0 >> kDlPageShift; pg <= b1 >> kDlPageShift; pg++) {
		s_dlPageMark[pg] = mark;
		if (!s_dlPageRw[pg]) continue;
		uint32_t b = std::max(b0, pg << kDlPageShift);
		const uint32_t e = std::min(b1, ((pg + 1) << kDlPageShift) - 1);
		for (; b + 3 <= e; b += 4) {  // (four a step)
			s_dlMark[b] = mark;
			s_dlMark[b + 1] = mark;
			s_dlMark[b + 2] = mark;
			s_dlMark[b + 3] = mark;
		}
		for (; b <= e; b++) s_dlMark[b] = mark;
	}
}
// Game thread, before it writes [p, p + n).
static void streamBeforeWrite(const void* p, uint32_t n) {
	uint32_t b0, b1;
	if (!s_streamOn || !mem1Blocks(p, n, b0, b1)) return;
	const uint32_t tail = s_sTail.load(std::memory_order_acquire);
	int32_t ahead = 0;  // the latest call's chunk + 1, past the GX thread's
	for (uint32_t pg = b0 >> kDlPageShift; pg <= b1 >> kDlPageShift; pg++) {
		if (!s_dlPageRw[pg]) {
			ahead = std::max(ahead, int32_t(s_dlPageMark[pg] - tail));
			continue;
		}
		const uint32_t e = std::min(b1, ((pg + 1) << kDlPageShift) - 1);
		for (uint32_t b = std::max(b0, pg << kDlPageShift); b <= e; b++) ahead = std::max(ahead, int32_t(s_dlMark[b] - tail));
	}
	if (ahead <= 0) return;  // read already (or never called)
	const uint32_t mark = tail + uint32_t(ahead);
	if (mark - 1 == s_sHead.load(std::memory_order_relaxed)) GXPC_StreamFlush();  // still being filled
	s_dlWaits++;
	streamWait(s_sWait[3], [mark] { return int32_t(mark - s_sTail.load(std::memory_order_acquire)) <= 0; });
}
// Game thread, as GDInitGDLObj starts on [p, p + n): calls of it go as copies for a while.
static void streamNoteRewrite(const void* p, uint32_t n) {
	uint32_t b0, b1;
	if (!s_streamOn || !mem1Blocks(p, n, b0, b1)) return;
	for (uint32_t pg = b0 >> kDlPageShift; pg <= b1 >> kDlPageShift; pg++) {
		if (s_dlPageRw[pg]) continue;
		s_dlPageRw[pg] = 1;  // its blocks kept from here on, from its page mark
		const uint32_t e = std::min(s_dlBlocks, (pg + 1) << kDlPageShift);
		for (uint32_t b = pg << kDlPageShift; b < e; b++) s_dlMark[b] = s_dlPageMark[pg];
	}
	memset(s_dlRw + b0, rwStamp(s_sFramesSent), b1 - b0 + 1);
}
// Game thread, GXCallDisplayList: true when the call should carry the list's bytes.
static bool streamCopyList(const void* list, uint32_t n) {
	uint32_t b0, b1;
	if (!s_dlCopyOn || n > kDlCopyMax || !mem1Blocks(list, n, b0, b1)) return false;
	const int now = rwStamp(s_sFramesSent);
	const auto recent = [now](uint32_t b) {
		if (!s_dlRw[b]) return false;
		int age = now - int(s_dlRw[b]);
		if (age < 0) age += 255;
		return uint32_t(age) <= kDlCopyFrames;
	};
	// Pages no list was ever written in: none of their blocks is recent.
	uint32_t pg = b0 >> kDlPageShift;
	while (pg <= b1 >> kDlPageShift && !s_dlPageRw[pg]) pg++;
	if (pg > b1 >> kDlPageShift) return false;
	// Blocks never rewritten (almost all) four at a time: a shape's list spans hundreds of blocks.
	uint32_t b = b0;
	bool hit = false;
	for (; b <= b1 && (b & 3) && !hit; b++) hit = recent(b);
	for (; b + 3 <= b1 && !hit; b += 4) {
		uint32_t w;
		memcpy(&w, s_dlRw + b, 4);
		if (w) hit = recent(b) || recent(b + 1) || recent(b + 2) || recent(b + 3);
	}
	for (; b <= b1 && !hit; b++) hit = recent(b);
	if (hit) {
		s_dlCopies++;
		s_dlCopyBytes += n;
	}
	return hit;
}

// Draw-sync tokens, GX thread -> game thread.
static const uint32_t kTokens = 256;
static uint16_t s_tok[kTokens];
static uint32_t s_tokMetric[kTokens];  // the pixel count as each token passed (pixMetricRead)
static std::atomic<uint32_t> s_tokHead{0}, s_tokTail{0};
static uint32_t s_tokDropped;
static void streamToken(uint16_t t) {
	const uint32_t h = s_tokHead.load(std::memory_order_relaxed);
	if (h - s_tokTail.load(std::memory_order_acquire) >= kTokens) {
		if (!s_tokDropped++) fprintf(stderr, "[gx3ds] draw-sync tokens not taken by the game: dropped\n");
		return;
	}
	s_tok[h % kTokens] = t;
	s_tokMetric[h % kTokens] = s_pmCount + 4 * s_pmTris;
	s_pmActive = false;  // a count runs from its clear to the token its read waits for
	s_tokHead.store(h + 1, std::memory_order_release);
	port_irq_kick();
}
// Game thread, in interrupt context (an emulated-OS interrupt source): the callbacks.
static void streamTokenPoll() {
	for (;;) {
		const uint32_t t = s_tokTail.load(std::memory_order_relaxed);
		if (t == s_tokHead.load(std::memory_order_acquire)) return;
		const uint16_t token = s_tok[t % kTokens];
		s_pmTokenValue = s_tokMetric[t % kTokens];
		s_tokTail.store(t + 1, std::memory_order_release);
		if (!drawSyncCallback) continue;
		GXPC_DrawSyncCalls++;
		const uint64_t t0 = svcGetSystemTick();
		s_pmInToken = true;
		drawSyncCallback(token);
		s_pmInToken = false;
		GXPC_DrawSyncTicks += svcGetSystemTick() - t0;
	}
}

static void gxMain(void*) {
	std::vector<uint8_t> carry;  // a command the last chunk cut, and its whole length
	uint32_t carryNeed = 0;
	uint64_t woke = svcGetSystemTick();  // (busy time is counted up to each wait: the clock read only there)
	for (;;) {
		if (s_sTail.load(std::memory_order_relaxed) == s_sHead.load(std::memory_order_acquire)) {
			const uint64_t t0 = svcGetSystemTick();
			s_sBusy += t0 - woke;
			SMS3DS_TimelineEvent(1, 0);
			GXPC_GxIdle = 1;  // (the game thread hands over what it has, and signals: streamOp, streamSwap)
			GX_ZONE_SET(GZ_IDLE);
			std::atomic_thread_fence(std::memory_order_seq_cst);
			// polled for up to 2 ms first, as the game thread's waits (streamWait), unless s_gxPoll is off
			for (uint32_t i = 1; s_gxPoll && s_sTail.load(std::memory_order_relaxed) == s_sHead.load(std::memory_order_acquire); i++) {
				pica_poll_pause();
				if ((i & 15) == 0 && svcGetSystemTick() - t0 > uint64_t(2.0 * kTicksPerMs)) break;
			}
			while (s_sTail.load(std::memory_order_relaxed) == s_sHead.load(std::memory_order_acquire)) pica_event_wait(EV_GX);
			GXPC_GxIdle = 0;
			GX_ZONE_SET(GZ_CARRY);
			SMS3DS_TimelineEvent(2, 0);
			woke = svcGetSystemTick();
			s_sIdle += woke - t0;
			GX3DS_GxWaitTicks += woke - t0;
		}
		const uint32_t t = s_sTail.load(std::memory_order_relaxed);
		const uint32_t buf = s_sBufOf[t % kStreamChunks];
		const uint8_t* p = s_sChunk[buf];
		uint32_t n = s_sLen[t % kStreamChunks];
		while (!carry.empty() && n) {
			const uint32_t k = std::min(n, carryNeed - uint32_t(carry.size()));
			carry.insert(carry.end(), p, p + k);
			p += k;
			n -= k;
			if (carry.size() < carryNeed) break;
			uint32_t need = 0;
			const uint32_t used = GXPC_StreamParse(carry.data(), uint32_t(carry.size()), &need);
			carry.erase(carry.begin(), carry.begin() + used);
			carryNeed = need;
		}
		if (carry.empty() && n) {
			uint32_t need = 0;
			const uint32_t used = GXPC_StreamParse(p, n, &need);
			if (used < n) {
				carry.assign(p + used, p + n);
				carryNeed = need;
			}
		}
		s_sFree.fetch_or(1u << buf, std::memory_order_release);  // (before the chunk is marked done)
		s_sTail.store(t + 1, std::memory_order_release);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (s_gameWaits.load(std::memory_order_relaxed)) pica_event_signal(EV_GAME);  // (only while it waits)
	}
}

// From GXPC_InitAuto (GXInit, game thread), before the game's first GX command.
static void streamStart() {
	if (!s_gxWanted || !s_rtWanted) return;
	for (uint32_t i = 0; i < kStreamChunks; i++) {
		s_sChunk[i] = static_cast<uint8_t*>(memalign(64, kStreamChunk));
		if (!s_sChunk[i]) {
			fprintf(stderr, "[gx3ds] no memory for the GX thread's buffers: GX on the game thread\n");
			return;
		}
	}
	s_dlBlocks = port_mem1_size >> 5;
	s_dlMark = static_cast<uint32_t*>(calloc(s_dlBlocks, sizeof(uint32_t)));
	s_dlRw = static_cast<uint8_t*>(calloc(s_dlBlocks, 1));
	const uint32_t pages = (s_dlBlocks + (1u << kDlPageShift) - 1) >> kDlPageShift;
	s_dlPageMark = static_cast<uint32_t*>(calloc(pages, sizeof(uint32_t)));
	s_dlPageRw = static_cast<uint8_t*>(calloc(pages, 1));
	if (!s_dlMark || !s_dlRw || !s_dlPageMark || !s_dlPageRw) {
		fprintf(stderr, "[gx3ds] no memory for the GX thread's display-list marks: GX on the game thread\n");
		return;
	}
	s_sFramesSent = g_displayFrames;
	s_sFramesDone.store(g_displayFrames);
	port_irq_add_source(streamTokenPoll);
	if (pica_gx_thread_start(gxMain, nullptr) < 0) {
		fprintf(stderr, "[gx3ds] GX thread could not start: GX on the game thread\n");
		return;
	}
	s_streamOn = true;
	GXPC_SetStream(streamSwap, streamToken);
	fprintf(stderr, "[gx3ds] GX thread on core 1: the game thread only writes commands (%u KB of buffers)\n",
	        unsigned(kStreamChunk * kStreamChunks >> 10));
}

static bool streamOn() { return s_streamOn; }
extern "C" uint64_t GXPC_OpTicks[10];
extern uint32_t GXPC_WrAll[2][3], GXPC_WrSame[2][3], GXPC_BpSame[256], GXPC_CpSame[256];  // patch 0022
extern uint32_t GXPC_WrFilter[2][3];
extern "C" uint32_t GXPC_FilterDropped[3];  // patch 0023: writes the game thread left out
extern "C" uint32_t GXPC_OpTimed[10], GXPC_OpAll[10];
// The profiler window's GX-thread numbers (render thread, at the window's end): the game thread's
// wait for it, then, with print, the line (which starts the next window).
static void streamPerf(uint64_t& gameWait, double perMs, bool print) {
	static uint64_t w0[4], busy0, idle0;
	static uint32_t dlWaits0, dlCopies0, dlCopyBytes0;
	uint64_t w[4];
	for (int i = 0; i < 4; i++) w[i] = s_sWait[i].load() - w0[i];
	gameWait = w[0] + w[1] + w[2] + w[3];
	if (!print) return;
	const uint64_t busy = s_sBusy.load(), idle = s_sIdle.load();
	const double perFrame = perMs * kTicksPerMs;  // 1 / frames
	fprintf(stderr,
	        "[perf]   GX thread (core 1): busy %.1f, idle %.1f ms/frame | game thread waited for it %.1f ms/frame "
	        "(buffers full %.1f, frame end %.1f, GXInit %.1f, rewriting a list it had yet to read %.1f, %u times) "
	        "| %u chunks queued | lists sent as copies %.1f per frame, %.1f KB\n",
	        double(busy - busy0) * perMs, double(idle - idle0) * perMs, double(gameWait) * perMs, double(w[0]) * perMs,
	        double(w[1]) * perMs, double(w[2]) * perMs, double(w[3]) * perMs, unsigned(s_dlWaits - dlWaits0),
	        unsigned(s_sHead.load() - s_sTail.load()), (s_dlCopies - dlCopies0) * perFrame,
	        (s_dlCopyBytes - dlCopyBytes0) * perFrame / 1024.0);
	if (GX3DS_ZoneSampling) {  // the zone sampler's shares since the last line
		static const char* const kZone[gx::GZ_COUNT] = {
		    "parse", "idle", "BP", "CP", "XF", "indexed XF", "list check", "list state", "list runs", "list uncached",
		    "list record", "vertices", "state record", "draw record", "J3D material", "J3D shape", "J3D hand-over",
		    "state block", "stream op", "copy", "chunk start", "vertex setup", "primitive begin", "primitive end",
		    "J3D matrices", "J3D runs", "J3D state out"};
		uint32_t h[gx::GZ_COUNT], all = 0;
		for (int z = 0; z < gx::GZ_COUNT; z++) {
			h[z] = GX3DS_ZoneHist[z];
			GX3DS_ZoneHist[z] = 0;
			all += h[z];
		}
		if (all) {
			char line[700];
			int at = 0;
			for (int z = 0; z < gx::GZ_COUNT; z++)
				if (h[z]) at += snprintf(line + at, sizeof line - at, "%s %s %.1f%%", at ? " |" : "", kZone[z], 100.0 * h[z] / all);
			fprintf(stderr, "[perf]   GX thread by zone (%u samples):%s\n", all, line);
		}
	}
	{
		// Its top-level commands by kind (patch 0020): count a frame and estimated ms a frame, from
		// the one in 16 timed.
		static const char* const kKind[10] = {"BP", "CP", "XF", "indexed XF", "list call", "primitives",
		                                      "stream list call", "stream list-copy call", "flush", "other stream"};
		static uint64_t t0[10];
		static uint32_t n0[10], a0[10];
		char line[640];
		int at = 0;
		double sum = 0;
		for (int k = 0; k < 10; k++) {
			const uint64_t t = GXPC_OpTicks[k] - t0[k];
			const uint32_t timed = GXPC_OpTimed[k] - n0[k], all = GXPC_OpAll[k] - a0[k];
			t0[k] += t;
			n0[k] += timed;
			a0[k] += all;
			if (!all) continue;
			const double ms = timed ? double(t) / timed * all * perMs : 0.0;
			sum += ms;
			at += snprintf(line + at, sizeof line - at, "%s %s %.0f/%.1f ms", at ? " |" : "", kKind[k], all * perFrame,
			               ms);
		}
		if (sum > 0)  // timed only in profile runs (GXPC_Detail, patch 0027)
			fprintf(stderr, "[perf]   GX thread by command (count/frame, ms/frame estimated from 1 in 16 timed; %.1f "
			        "ms in all):%s\n", sum, line);
	}
	{
		// Register writes that changed nothing (patch 0022), and the registers most direct ones hit.
		static uint32_t all0[2][3], same0[2][3], bp0[256], cp0[256];
		uint32_t all[2][3], same[2][3];
		for (int k = 0; k < 2; k++)
			for (int t = 0; t < 3; t++) {
				all[k][t] = GXPC_WrAll[k][t] - all0[k][t];
				same[k][t] = GXPC_WrSame[k][t] - same0[k][t];
				all0[k][t] = GXPC_WrAll[k][t];
				same0[k][t] = GXPC_WrSame[k][t];
			}
		auto top = [&](const uint32_t* now, uint32_t* before, int nTop, char* out, size_t cap) {
			uint32_t d[256];
			for (int r = 0; r < 256; r++) d[r] = now[r] - before[r], before[r] = now[r];
			int at = 0;
			for (int i = 0; i < nTop; i++) {
				int best = -1;
				for (int r = 0; r < 256; r++)
					if (d[r] && (best < 0 || d[r] > d[best])) best = r;
				if (best < 0) break;
				at += snprintf(out + at, cap - at, " %02x:%.0f", best, d[best] * perFrame);
				d[best] = 0;
			}
			if (!at) out[0] = 0;
		};
		{
			static uint32_t f0[2][3];
			uint32_t f[2][3];
			for (int k = 0; k < 2; k++)
				for (int t = 0; t < 3; t++) f[k][t] = GXPC_WrFilter[k][t] - f0[k][t], f0[k][t] = GXPC_WrFilter[k][t];
			static uint32_t d0[3];
			uint32_t d[3];
			for (int t = 0; t < 3; t++) d[t] = GXPC_FilterDropped[t] - d0[t], d0[t] = GXPC_FilterDropped[t];
			if (all[0][0] + all[0][1] + all[0][2] == 0)  // the GX thread's counters run in profile runs only
				fprintf(stderr, "[perf]   the game thread's write filter left out BP %.0f, CP %.0f, XF words %.0f\n",
				        d[0] * perFrame, d[1] * perFrame, d[2] * perFrame);
			else
				fprintf(stderr,
				        "[perf]   the game thread's write filter left out BP %.0f, CP %.0f, XF words %.0f | of the rest, a "
				        "filter could still drop: forgetting everything at each list call BP %.0f, CP %.0f, XF words %.0f "
				        "| forgetting only what each list writes BP %.0f, CP %.0f, XF words %.0f\n",
				        d[0] * perFrame, d[1] * perFrame, d[2] * perFrame, f[0][0] * perFrame, f[0][1] * perFrame,
				        f[0][2] * perFrame, f[1][0] * perFrame, f[1][1] * perFrame, f[1][2] * perFrame);
			static uint32_t sb0[3];
			uint32_t sb[3];
			for (int t = 0; t < 3; t++) sb[t] = GXPC_StateBlockCalls[t] - sb0[t], sb0[t] = GXPC_StateBlockCalls[t];
			if (sb[0] + sb[1] + sb[2])
				fprintf(stderr, "[perf]   GX state resets per frame: %.1f written as blocks, %.1f checked against the game's calls, %.1f as the calls\n",
				        sb[0] * perFrame, sb[1] * perFrame, sb[2] * perFrame);
		}
		char bpTop[160], cpTop[120];
		top(GXPC_BpSame, bp0, 8, bpTop, sizeof bpTop);
		top(GXPC_CpSame, cp0, 6, cpTop, sizeof cpTop);
		if (all[0][0] + all[0][1] + all[0][2] + all[1][0] + all[1][1] + all[1][2])
		fprintf(stderr,
		        "[perf]   writes that changed nothing, per frame: direct BP %.0f of %.0f, CP %.0f of %.0f, XF words %.0f "
		        "of %.0f | in cached lists BP %.0f of %.0f, CP %.0f of %.0f, XF words %.0f of %.0f | direct BP by register"
		        "%s | CP%s\n",
		        same[0][0] * perFrame, all[0][0] * perFrame, same[0][1] * perFrame, all[0][1] * perFrame,
		        same[0][2] * perFrame, all[0][2] * perFrame, same[1][0] * perFrame, all[1][0] * perFrame,
		        same[1][1] * perFrame, all[1][1] * perFrame, same[1][2] * perFrame, all[1][2] * perFrame, bpTop, cpTop);
	}
	for (int i = 0; i < 4; i++) w0[i] += w[i];
	dlWaits0 = s_dlWaits;
	dlCopies0 = s_dlCopies;
	dlCopyBytes0 = s_dlCopyBytes;
	busy0 = busy;
	idle0 = idle;
}

// ---- frame / copies ---------------------------------------------------------------------------
// Called when a frame is complete (a display copy). Set by the 3DS side to present it.
typedef void (*FrameHook)(const Stats& frame, uint32_t frameNumber);
static FrameHook s_frameHook;

// Game side: the copy goes into the queue after the batch and the registers it needs. A display
// copy ends the game's frame: the arena is emptied here when due, and the game waits while the
// render thread is still a frame behind.
void executeCopy(uint32_t ctrl) {
	if (!s_ready) return;
	GX_ZONE(GZ_COPY);
	flushBatch();
	rcInit();
	emitState();
	const uint32_t size = uint32_t(sizeof(RecHdr) + sizeof(CopyRec) + 7) & ~7u;
	uint8_t* p = rcReserve(size);
	RecHdr* h = reinterpret_cast<RecHdr*>(p);
	h->type = RC_COPY;
	h->size = size;
	CopyRec* c = reinterpret_cast<CopyRec*>(p + sizeof(RecHdr));
	c->ctrl = ctrl;
	c->copyDest = g.copyDest;
	memcpy(c->gAcc, s_gAcc, sizeof c->gAcc);
	memset(s_gAcc, 0, sizeof s_gAcc);
	memcpy(c->gWait, s_gWait, sizeof c->gWait);
	c->qBytes = s_qBytes + size;
	c->qPeak = s_qPeak;
	c->qExt = s_qExt;
	s_qPeak = 0;
	rcCommit(size);
	if ((ctrl >> 14) & 1) {
		g_displayFrames++;
		s_gFrames++;
		if (s_streamOn) {  // the game thread may start its next frame
			s_sFramesDone.store(g_displayFrames, std::memory_order_release);
			pica_event_signal(EV_GAME);
		}
		// A new stage (GX3DS_NewStage): the display-list cache and the resident vertex pages start
		// over, here between two frames (kept, the stages left behind filled the heap).
		bool fresh = false;
		// Out of heap (main.cpp's out-of-memory handler gave its reserve back): the same, and the reserve
		// taken again.
		const bool relief = SMS3DS_MemReserveSpent && SMS3DS_MemReserveSpent();
		if (s_newStage.load(std::memory_order_relaxed) || relief) {
			const uint32_t n = GXPC_DlEntryCount(), kb = uint32_t(GXPC_DlCacheBytes() >> 10);
			if (GXPC_DlCacheClearNow()) {
				s_newStage.store(false);
				s_aresetDue = fresh = true;
				fprintf(stderr, "[gx] %s: the display-list cache (%lu lists, %lu KB) and the resident vertex pages (%lu KB) emptied\n",
				        relief ? "out of heap" : "new stage", (unsigned long)n, (unsigned long)kb, (unsigned long)(s_abytes >> 10));
				if (relief && SMS3DS_MemReserveRestore) SMS3DS_MemReserveRestore();
			}
		}
		arenaFrameEnd(fresh);
		if (s_rtCore >= 0) gameWaitFor(s_gWait[1], [] { return s_gFrames - s_rDone.load() <= 1; });
	}
}

// Render side.
static void renderCopy(const CopyRec* cr) {
	const uint32_t ctrl = cr->ctrl;
	rs.copyDest = cr->copyDest;
	s_rsGen++;  // (the EFB copies, which textures can be found among, change below)
	for (int i = 0; i < PERF_COUNT; i++) s_perfAcc[i] += cr->gAcc[i];
	memcpy(s_gWaitSeen, cr->gWait, sizeof s_gWaitSeen);
	s_qBytesSeen = cr->qBytes;
	s_qPeakWin = std::max(s_qPeakWin, cr->qPeak);
	s_qExtSeen = cr->qExt;
	if (s_tracing) {
		const uint32_t tl = rs.bp[BP_COPY_SRC_TL], wh = rs.bp[BP_COPY_SRC_WH];
		fprintf(tf(), "[trace] copy %s rect %u,%u %ux%u fmt %u clear %u (AR %04x GB %04x Z %06x) dest %p\n",
		        (ctrl >> 14) & 1 ? "DISPLAY" : "texture", tl & 0x3FF, (tl >> 10) & 0x3FF, (wh & 0x3FF) + 1,
		        ((wh >> 10) & 0x3FF) + 1, ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7), (ctrl >> 11) & 1,
		        rs.bp[BP_CLEAR_AR] & 0xFFFF, rs.bp[BP_CLEAR_GB] & 0xFFFF, rs.bp[BP_CLEAR_Z] & 0xFFFFFF, rs.copyDest);
	}
	if ((ctrl >> 14) & 1) {  // a display copy ends the traced frame, or starts the armed one
		if (s_tracing) {
			s_tracing = false;
			fprintf(tf(), "[trace] end\n");
			if (s_traceFile) {
				fclose(s_traceFile);
				s_traceFile = nullptr;
				fprintf(stderr, "[trace] frame %u written to sdmc:/sms/trace-%u.txt\n", s_rFrames, s_traceNo);
			}
		} else if (s_traceArm) {
			s_traceArm = false;
			s_tracing = true;
			s_traceDraw = 0;
			s_traceDumped.clear();
			char path[48];
			snprintf(path, sizeof path, "sdmc:/sms/trace-%u.txt", ++s_traceNo);
			s_traceFile = fopen(path, "w");
			if (!s_traceFile) fprintf(stderr, "[trace] cannot create %s: the trace goes to this log\n", path);
			fprintf(tf(), "[trace] begin frame %u\n", s_rFrames + 1);
		}
	}
	s_stats.copies++;
	bool disp = (ctrl >> 14) & 1;  // copy to the external frame buffer: the frame is done
	bool clear = (ctrl >> 11) & 1;
	bool leftOut = false;  // a colour copy, and its clear, left out (GX3DS_SetSkipColourCopies)
	if (!disp) {
		s_texCopies++;
		s_afterTexCopy = true;
		const void* d = rs.copyDest;
		// BP 0x49: top-left (x, y), 0x4A: size - 1 (w, h), in EFB pixels.
		const uint32_t tl = rs.bp[BP_COPY_SRC_TL], wh = rs.bp[BP_COPY_SRC_WH];
		const float rx = float(tl & 0x3FF), ry = float((tl >> 10) & 0x3FF);
		const float rw = float((wh & 0x3FF) + 1), rh = float(((wh >> 10) & 0x3FF) + 1);
		const uint32_t cfmt = ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7);
		const bool half = (ctrl >> 9) & 1;
		// One-byte formats are maps the game draws back and copies again (the goop): texel for texel.
		const bool exact = cfmt == 1 || (cfmt >= 7 && cfmt <= 10);
		leftOut = s_skipColourCopies && !exact;
		uint16_t tw, th;
		float win[4];
		const float rowScale = efbRowScale();
		pica_copy_tex_layout(rx, ry, rw, rh, half, exact, rowScale, &tw, &th, win);
		const int rgb565 = cfmt == 4;  // RGB565: no alpha to keep
		EfbCopy* c = d ? findEfbCopy(d) : nullptr;
		// Another size or format than its last copy: a new texture. Not for a copy that is not a map
		// while its texture has the format and at least half the size: that holds the copy's window
		// as well (finer or coarser), and making the texture again each time the size changed, with
		// VRAM short, lost the copy on some frames and not others (issue #1: the water's and FLUDD's
		// screen copy flickered). A coarser one tries for its own size every 2 s.
		if (c && !pica_copy_tex_fits(c->tex, tw, th, rgb565)) {
			if (exact || !pica_copy_tex_at_least(c->tex, tw / 2, th / 2, rgb565)) {
				pica_tex_free(c->tex);
				s_efbCopies.erase(s_efbCopies.begin() + (c - s_efbCopies.data()));
				c = nullptr;
			} else if (!pica_copy_tex_at_least(c->tex, tw, th, rgb565) && s_rFrames - c->triedFrame >= 60) {
				c->triedFrame = s_rFrames;
				if (PicaTex* t = pica_copy_tex_try(tw, th, rgb565)) {
					pica_tex_free(c->tex);
					c->tex = t;
					c->reduced = false;
				}
			}
		}
		if (d && !c) {
			// The least recently sampled copy other than a map (exact) first, else any (-1: none).
			auto victim = [&](bool mapsToo) {
				int v = -1;
				for (size_t i = 0; i < s_efbCopies.size(); i++) {
					const uint32_t f = s_efbCopies[i].fmt;
					const bool map = f == 1 || (f >= 7 && f <= 10);
					if ((!map || mapsToo) && (v < 0 || s_efbCopies[i].usedFrame < s_efbCopies[size_t(v)].usedFrame)) v = int(i);
				}
				return v;
			};
			if (s_efbCopies.size() >= kMaxEfbCopies) {  // replace the least recently sampled
				int v = victim(false);
				if (v < 0) v = victim(true);
				pica_tex_free(s_efbCopies[size_t(v)].tex);
				s_efbCopies.erase(s_efbCopies.begin() + v);
			}
			PicaTex* t = pica_copy_tex_create(tw, th, rgb565);
			// No VRAM for a map's copy (a goop map, drawn back and copied every frame): the other
			// copies give way, least recently sampled first (they are made again when copied next).
			while (!t && exact) {
				const int v = victim(false);
				if (v < 0) break;
				pica_tex_free(s_efbCopies[size_t(v)].tex);
				s_efbCopies.erase(s_efbCopies.begin() + v);
				t = pica_copy_tex_create(tw, th, rgb565);
				if (t) logCopyOnce(d, "no VRAM for a map's copy: other copies gave way", tw, th);
			}
			// No VRAM for it at its size (the copy arena full, the scene's textures in VRAM): at half the
			// size, which samples the same (the texture holds the copy's window, whatever its size), only
			// coarser. Without one, draws read the destination's memory, which nothing wrote: on a
			// console whatever the heap held (Sirena Beach's Phantamanta tints the screen through its
			// 640x448 alpha copy, the fourth copy of the frame: manta-coloured noise everywhere). Not a
			// map: its copy must hold its texels as they are (halved, a goop map read back and drawn
			// again shrank, and its goop moved: Pianta Village, the Bowser fight).
			bool reduced = false;
			if (!t && !exact && tw >= 16 && th >= 16) {
				t = pica_copy_tex_create(tw / 2, th / 2, rgb565);
				reduced = t != nullptr;
				if (reduced) logCopyOnce(d, "no VRAM at its size: a texture of half the size", tw, th);
			}
			if (t) {
				s_efbCopies.push_back(EfbCopy{d, t, 0, 0, 0, 0, {0, 0, 1, 1}, s_rFrames, 0, 0, reduced, s_rFrames});
				c = &s_efbCopies.back();
			} else if (!exact) {  // a map keeps its memory as the last readback left it
				lostCopy(d, cfmt, half, rw, rh, tw, th);
			}
		}
		if (c) {
			c->x = rx;
			c->y = ry;
			c->w = rw;
			c->h = rh;
			memcpy(c->win, win, sizeof win);
			c->clock = writeClock();
			// Alpha reads as 1 when the EFB format has none (anything but RGBA6_Z24, as sms-pc-port's
			// copy) or the copy's texture format has none (RGB565: a GX texture of it decodes alpha 1).
			const uint32_t pix = rs.bp[BP_PE_CONTROL] & 7;
			c->fmt = cfmt;
			// A frame a GPU test may have touched (profile runs): a map's copy is left out, so the
			// map stays as before the test (the goop map is drawn from its last copy, then copied).
			const bool keep = exact && pica_gpu_test_seen();
			// A Z copy as Z24X8 (Bowser's hot tub: the water's height field is drawn as depth and
			// copied, and the CPU reads it to shape the water and float what is on it): from the
			// depth buffer, read back to memory.
			const bool zcopy = pix == 3 && cfmt == 6 && !leftOut;
			if (zcopy) {
				pica_efb_depth_copy_to(c->tex, c->win, rowScale);
				const int dw = half ? int(c->w) / 2 : int(c->w), dh = half ? int(c->h) / 2 : int(c->h);
				pica_efb_readback(c->tex, const_cast<void*>(d), dw, dh, (dw + 3) / 4, 6, c->x, c->y, c->w, c->h);
			} else if (!keep && !leftOut) {
				PerfScope ps(PERF_COPY);
				// One-channel formats read back with that channel in all four (R8 and I8 as I8),
				// RA4 / RA8 with red as the intensity.
				const int rep = cfmt == 0 || cfmt == 1 || cfmt == 8 ? 0 : cfmt == 7 ? 3 : cfmt == 9 ? 1 : cfmt == 10 ? 2
				                : cfmt == 2 || cfmt == 3 ? 5 : -1;
				pica_efb_copy_to_ch(c->tex, pix != 1 || cfmt == 4, rep, c->win, rowScale);
			}
			// One-byte formats (I8, R8, G8, B8) go back to memory too: the CPU may read them (the goop
			// map, TPollutionLayer::isPolluted). Not A8: the game's two A8 copies (Sirena Beach's
			// Phantamanta shadow, 640x448 every frame, and the bath water) only feed the GPU, and
			// converting the manta's took the render thread ~5 ms a frame on a console.
			if (!keep && !zcopy && (cfmt == 1 || (cfmt >= 8 && cfmt <= 10))) {
				const bool intensity = ((ctrl >> 15) & 3) == 3;
				const int dw = half ? int(c->w) / 2 : int(c->w), dh = half ? int(c->h) / 2 : int(c->h);
				const int chan = cfmt == 7 ? 3 : cfmt == 9 ? 1 : cfmt == 10 ? 2 : cfmt == 1 && intensity ? 4 : 0;
				// Tiles a row from the copy's width (BP_COPY_STRIDE holds bytes a texel row in sms-pc-port's
				// front half, not the hardware's 32-byte tile units).
				pica_efb_readback(c->tex, const_cast<void*>(d), dw, dh, (dw + 7) / 8, chan, c->x, c->y, c->w, c->h);
			}
			logCopyKind(c, ctrl, pix);
		}
		if (s_tracing)
			fprintf(tf(), "[trace]   copy to %p: %s (%ux%u texture, %u copies kept)\n", d,
			        c ? (pica_tex_where(c->tex), "kept as a copy texture") : d ? "NO COPY TEXTURE (draws read the memory)" : "no destination",
			        unsigned(tw), unsigned(th), unsigned(s_efbCopies.size()));
	}
	if (clear || disp) {
		// BP 0x4F: A<<8|R, 0x50: G<<8|B, 0x51: 24-bit Z.
		uint32_t ar = rs.bp[BP_CLEAR_AR], gb = rs.bp[BP_CLEAR_GB];
		uint32_t rgba = ((ar & 0xFF) << 24) | (((gb >> 8) & 0xFF) << 16) | ((gb & 0xFF) << 8) | ((ar >> 8) & 0xFF);
		if (disp) {
			{
				PerfScope ps(PERF_PRESENT);
				pica_present();
			}
			perfFrameEnd();
		}
		// The display copy's clear applies to the next frame.
		if (clear && disp) {
			pica_clear(rgba, rs.bp[BP_CLEAR_Z] & 0xFFFFFF);
		} else if (clear && !leftOut) {  // a texture copy clears its source rectangle only
			const uint32_t tl = rs.bp[BP_COPY_SRC_TL], wh = rs.bp[BP_COPY_SRC_WH];
			const float rows = efbRowScale();
			pica_clear_rect(float(tl & 0x3FF), float((tl >> 10) & 0x3FF) / rows, float((wh & 0x3FF) + 1),
			                float(((wh >> 10) & 0x3FF) + 1) / rows, rgba, rs.bp[BP_CLEAR_Z] & 0xFFFFFF);
		}
	}
	if (disp) {
		s_rFrames++;
		s_rDone.store(s_rFrames);
		s_stats.frames++;
		s_stats.batches += s_frame.batches;
		s_stats.vertices += s_frame.vertices;
		s_stats.triangles += s_frame.triangles;
		s_stats.lines += s_frame.lines;
		s_stats.points += s_frame.points;
		if (s_frameHook) s_frameHook(s_frame, s_rFrames);
		memset(&s_frame, 0, sizeof s_frame);
		s_lastBefore = s_drawsBeforeCopy;
		s_lastAfter = s_drawsAfterCopy;
		s_lastCopies = s_texCopies;
		s_drawsBeforeCopy = s_drawsAfterCopy = s_texCopies = 0;
		s_afterTexCopy = false;
	}
}

void markXfMemDirty() {}
void textureInvalidateAll() {}
void textureInvalidateRange(const void*, uint32_t) {}
void textureCpuWrote(const void*, uint32_t) {}
unsigned efbCopyLookup(const void*, int*, int*) { return 0; }
// GXPeekARGB: the EFB is not read back here. Its one caller is Mario's occlusion probe
// (TMario::drawSyncCallback), which takes alpha 0x10 at his screen position for "in view" and
// anything else for "hidden": with 0 he was hidden every frame, so the silhouette mode stayed on and
// drew every nearby actor's and item's "?" marker through the ground (TQuestionManager, depth
// GREATER). Reported in view until the EFB can be peeked (no silhouette behind walls meanwhile).
uint32_t peekColor(int, int) { return 0x10000000u; }
uint32_t peekZ(int, int) { return 0; }
// Pixel metrics (GXClearPixMetric, GXReadPixMetric): pixMetricCount below; the read is what that
// counted (with each draw-sync token: pixMetricToken).
void pixMetricClear() {
	s_pmCount = s_pmTris = 0;
	s_pmActive = true;
}
uint32_t pixMetricRead() {
	if (s_pmInToken) return s_pmTokenValue;
	s_pmActive = false;
	return s_pmCount + 4 * s_pmTris;
}
bool rendererReady() { return s_ready; }
float efbScale() { return 1.0f; }

}  // namespace gx

// ---- C interface -------------------------------------------------------------------------------
extern "C" {

// GXInit calls this when the renderer is not ready; the 3DS side normally calls it first.
int GXPC_InitAuto(int) {
	if (!gx::s_ready) {
		gx::resetState();
		gx::s_ready = true;
		gx::streamStart();
	}
	return 1;
}

// Streamed GX, game thread (gx_api.cpp): a display copy was sent; waits until the GX thread has
// finished the frame. The GX thread reads the game's memory as it parses (vertex arrays, matrices
// for indexed loads, display lists), and the game refills some of it in place each frame (the
// water's, for one), as the console allows: its CPU waits for the GPU at each frame's end. Letting
// the game simulate the next frame while the GX thread was still a frame behind made those draws
// read the next frame's data (flicker and ghost copies in Delfino Plaza, worse once the write
// filter sped both threads up). It costs Delfino Plaza about 1 ms a frame.
void GXPC_StreamFrameEnd(void) {
	gx::GXPC_StreamFlush();
	const uint32_t sent = ++gx::s_sFramesSent;
	gx::SMS3DS_TimelineEvent(5, 0);
	gx::streamWait(gx::s_sWait[1], [sent] { return int32_t(sent - gx::s_sFramesDone.load(std::memory_order_acquire)) <= 0; });
	gx::SMS3DS_TimelineEvent(6, 0);
}
// ... a display list was called through the stream by address (GXCallDisplayList).
void GXPC_StreamNoteDL(const void* list, uint32_t nbytes) { gx::streamNoteDL(list, nbytes); }
// ... GXCallDisplayList: 1 when the call should carry a copy of the list (one the game rewrites).
int GXPC_StreamCopyDL(const void* list, uint32_t nbytes) { return gx::streamCopyList(list, nbytes); }
// The game is about to write a display list (GD; linked with --wrap=GDInitGDLObj): waits while
// the GX thread has yet to read a call of it by address; its calls go as copies from then on.
void __real_GDInitGDLObj(void* dl, void* start, uint32_t length);
void __wrap_GDInitGDLObj(void* dl, void* start, uint32_t length) {
	gx::streamBeforeWrite(start, length);
	gx::streamNoteRewrite(start, length);
	__real_GDInitGDLObj(dl, start, length);
}
// ... waits until the GX thread has run every command sent (GXInit: the registers start over).
void GXPC_StreamDrain(void) {
	if (!gx::s_streamOn) return;
	gx::GXPC_StreamFlush();
	const uint32_t h = gx::s_sHead.load();
	gx::streamWait(gx::s_sWait[2], [h] { return gx::s_sTail.load(std::memory_order_acquire) == h; });
}
// Off before the first frame: GX runs on the game thread (no GX thread).
void GX3DS_SetGxThread(int on) { gx::s_gxWanted = on != 0; }
void GX3DS_SetPerfTimers(int on) { gx::s_perfTimers = on != 0; }  // settings.txt perf_timers
void GX3DS_SetGxPoll(int on) { gx::s_gxPoll = on != 0; }  // settings.txt gx_poll
void GX3DS_SetGamePoll(int on) { gx::s_gamePoll = on != 0; }
void GX3DS_SetSkipColourCopies(int on) { gx::s_skipColourCopies = on != 0; }
void GX3DS_SetLightMode(int mode) { gx::s_lightMode = mode; }
void GX3DS_NewStage(void) { gx::s_newStage.store(true); }
void GX3DS_SetSkipTev(unsigned id, unsigned to) {
	gx::s_skipTev = id;
	gx::s_skipTevTo = to > id ? to : id;
}
// The game thread's waits for the GX thread so far (ticks): a free chunk, the frame's end, a drain,
// a list rewrite (the profiler's game-thread work without phase timers).
uint64_t GX3DS_GameWaitTicks(void) {
	uint64_t t = 0;
	for (int i = 0; i < 4; i++) t += gx::s_sWait[i].load(std::memory_order_relaxed);
	return t;
}
int GX3DS_GxPoll(void) { return gx::s_gxPoll; }
void GX3DS_SetDrawStateCheck(int on) {  // settings.txt draw_state_check
	gx::s_fpCheck = on != 0;
	gx::GXPC_FastListCheck = on != 0;
}
void GX3DS_SetListCopies(int on) { gx::s_dlCopyOn = on != 0; }

// 0: material, 1: debug colours, 2: opaque (material colours, no blending or alpha test).
void GX3DS_TraceNextFrame(void) {
	gx::s_traceArm = true;
	fprintf(stderr, "[trace] armed: the next frame goes to sdmc:/sms/trace-%u.txt\n", gx::s_traceNo + 1);
}
void GX3DS_TraceState(int out[2]) {
	out[0] = gx::s_traceArm || gx::s_tracing;
	out[1] = int(gx::s_traceNo) - (gx::s_tracing ? 1 : 0);
}

void GX3DS_SetGpuTransform(int on) { gx::s_gpuTransform = on != 0; }
// Off before the first frame: the renderer draws on the game thread (no render thread).
void GX3DS_SetRenderThread(int on) { gx::s_rtWanted = on != 0; }
void GX3DS_SetDropDraws(int on) { gx::s_dropDraws.store(on != 0); }
// Compact rings (profile runs): bit 0 the stream's buffers, lowest free first; bit 1 the render
// queue started again at its beginning when the render thread has caught up.
void GX3DS_SetCompactRings(int mask) {
	gx::s_sCompact.store((mask & 1) != 0);
	gx::s_qCompact.store((mask & 2) != 0);
}
// Since the last call: the stream buffers taken (how many), the furthest into the render queue a
// record reached (bytes).
void GX3DS_RingStats(uint32_t out[2]) {
	out[0] = uint32_t(__builtin_popcount(gx::s_sUsed));
	out[1] = gx::s_qSpan;
	gx::s_sUsed = 0;
	gx::s_qSpan = 0;
}
// Native J3D's verify mode: differences the render thread has found so far.
uint32_t GX3DS_J3DSigBad(void) { return uint32_t(gx::s_j3dSigBad); }
// The GX thread's and the render thread's busy time so far, in ticks (profile runs).
void GX3DS_ThreadTicks(uint64_t out[2]) {
	out[0] = gx::s_sBusy.load();
	out[1] = gx::s_rBusy;
}
void GX3DS_SetVerboseLog(int on) { gx::s_verbose = on != 0; }
int GX3DS_GetGpuTransform(void) { return gx::s_gpuTransform ? 1 : 0; }
// Batches drawn on the GPU and the CPU path so far.
void GX3DS_GetPathStats(uint32_t out[2]) {
	out[0] = gx::s_gpuBatches;
	out[1] = gx::s_cpuBatches;
}

// os_core.cpp's cache operations (DCFlushRange, DCStoreRange, ...) report CPU writes here (a weak
// reference sms-pc-port's GL renderer defines): stamp the pages, so the texture cache rehashes
// only what the game rewrote.
void GXPC_InvalidateRange(const void* p, uint32_t n) { gx::memoryWritten(p, n); }

// Averages over the last profiler window, in ms: tex, tev, xform, submit, copy, present, frame,
// other; then batches and vertices per frame.
void GX3DS_GetPerf(float out[11]) {
	// tex, tev, xform, submit, copy, present, frame, other, batches, verts, decode
	for (int i = 0; i < 6; i++) out[i] = gx::s_perfAvgMs[i];
	out[6] = gx::s_perfAvgMs[gx::PERF_COUNT];
	out[7] = gx::s_perfAvgMs[gx::PERF_COUNT + 1];
	out[8] = gx::s_perfAvgBatches;
	out[9] = gx::s_perfAvgVerts;
	out[10] = gx::s_perfAvgMs[gx::PERF_DECODE];
}

void GX3DS_SetDebugColors(int mode) {
	gx::s_debugColors = mode == 1;
	gx::s_opaque = mode == 2;
}
int GX3DS_GetDebugColors(void) { return gx::s_debugColors ? 1 : gx::s_opaque ? 2 : 0; }

// Last frame: triangle draws before its first EFB-to-texture copy, after it, and the copies.
// Texture uploads so far, cache bytes, draws skipped for sampling an EFB copy.
void GX3DS_GetTexStats(uint32_t out[3]) {
	out[0] = gx::s_texUploads;
	out[1] = gx::s_texBytes;
	out[2] = gx::s_copyTexDraws;
}

void GX3DS_GetCopyStats(uint32_t out[3]) {
	out[0] = gx::s_lastBefore;
	out[1] = gx::s_lastAfter;
	out[2] = gx::s_lastCopies;
}

void GX3DS_SetFrameHook(void (*hook)(const void* frameStats, uint32_t frameNumber)) {
	gx::s_frameHook = reinterpret_cast<gx::FrameHook>(hook);
}

// frames, batches, vertices, triangles, lines, points, copies (all frames so far)
void GX3DS_GetStats(uint32_t out[7]) {
	out[0] = gx::s_stats.frames;
	out[1] = gx::s_stats.batches;
	out[2] = gx::s_stats.vertices;
	out[3] = gx::s_stats.triangles;
	out[4] = gx::s_stats.lines;
	out[5] = gx::s_stats.points;
	out[6] = gx::s_stats.copies;
}

}  // extern "C"
