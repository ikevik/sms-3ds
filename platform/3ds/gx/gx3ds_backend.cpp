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
static uint64_t s_arenaDrawVerts, s_arenaUploadBytes; // resident vertices drawn, stored (display-list arena)
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

static inline bool batchHasVertices() { return !s_idx.empty(); }
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
static const uint16_t* r_idx;      // r_nidx indices
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
extern "C" uint64_t svcGetSystemTick(void);
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
static uint8_t s_bcut;  // the open batch's vertices have their pixels cut
// Running total of every phase, for the front half's display-list timers (patch 0007) to leave out.
extern "C" uint64_t GX3DS_OursTicks;
uint64_t GX3DS_OursTicks;
extern "C" uint64_t GXPC_DlTicks[4];
extern "C" uint32_t GXPC_DlCalls, GXPC_DlHits, GXPC_DlVerts, GXPC_DlRecords, GXPC_DlNew, GXPC_DlUncached;
extern "C" uint64_t GXPC_DlHashed;
extern "C" uint32_t GXPC_CullRuns, GXPC_CullVerts;  // patch 0014
extern "C" int GXPC_CullOff;
static uint32_t s_perfCullRuns0, s_perfCullVerts0;
extern "C" size_t GXPC_DlCacheBytes(void);
extern "C" uint32_t GXPC_DlEntryCount(void);
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
struct PerfScope {
	PerfPhase ph;
	uint64_t t0;
	explicit PerfScope(PerfPhase p) : ph(p), t0(svcGetSystemTick()) {}
	~PerfScope() { perfAdd(ph, svcGetSystemTick() - t0); }
};
// A phase without the phases nested in it.
struct PerfScopeExcl {
	PerfPhase ph;
	uint64_t t0, o0;
	explicit PerfScopeExcl(PerfPhase p) : ph(p), t0(svcGetSystemTick()), o0(GX3DS_OursTicks) {}
	~PerfScopeExcl() { perfAdd(ph, (svcGetSystemTick() - t0) - (GX3DS_OursTicks - o0)); }
};

// Why batches fell back to the CPU path: batches and vertices per reason over the profiler window,
// logged with the [perf] lines.
enum GpuFallback {
	FB_TRACE, FB_NOCLIP, FB_SIZE, FB_PNMTX, FB_TWOCOLOURS, FB_TEXGEN_TYPE, FB_TEXGEN_PROJ, FB_TEXGEN_SRC,
	FB_TEXMTX_VARIES, FB_POST_NORMALISE, FB_POST_PROJ, FB_LAYOUT, FB_COUNT
};
static const char* const kFallbackName[FB_COUNT] = {
	"trace", "clip-off", "size", "pos-matrix>27", "two-colours", "texgen-type", "texgen-proj", "texgen-src",
	"texmtx-varies", "post-normalise", "post-proj", "layout"};
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
static void arenaFrameEnd();
static int debugColorMode();  // SELECT's mode: 0 material, 1 debug colours, 2 opaque
static bool renderThreaded();    // the render thread runs (else records are drawn as they are queued)
static uint64_t gameWaitTicks(); // the profiler window's game-thread wait for the render thread
static void renderThreadPerf(double frames);

static uint64_t s_perfEnd0[PICA_END_COUNT];
// Streamed GX (the GX thread, below): its numbers for the profiler window.
static bool streamOn();
static void streamPerf(uint64_t& gameWait, double perMs, bool print);

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
	        "colours %s\n", pica_overlap() ? "frame end, frames overlapped" : "C3D_FrameBegin", waitMs, gpuMs, splits, (unsigned long)pica_lost_completions(), pica_gpu_test() ? "pixels cut to 8x8" : "off",
	        debugColorMode() == 1 ? "debug (no textures)" : debugColorMode() == 2 ? "opaque (no blending)" : "material");
	fprintf(stderr,
	        "[perf] frame %.1f ms (%.1f fps) | tex %.2f tev %.2f xform %.2f submit %.2f copy %.2f present %.2f | "
	        "decode %.1f batch %.2f append %.2f other %.1f | %.0f batches %.0f verts/frame\n",
	        s_perfAvgMs[PERF_COUNT], 1000.0 / s_perfAvgMs[PERF_COUNT], s_perfAvgMs[PERF_TEX], s_perfAvgMs[PERF_TEV],
	        s_perfAvgMs[PERF_XFORM], s_perfAvgMs[PERF_SUBMIT], s_perfAvgMs[PERF_COPY], s_perfAvgMs[PERF_PRESENT],
	        s_perfAvgMs[PERF_DECODE], s_perfAvgMs[PERF_BATCH], s_perfAvgMs[PERF_APPEND], s_perfAvgMs[PERF_COUNT + 1],
	        s_perfAvgBatches, s_perfAvgVerts);
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
		fprintf(stderr, "[perf]   off-screen culling: %.1f display-list draws (%.0f verts) skipped per frame%s\n",
		        (GXPC_CullRuns - s_perfCullRuns0) / frames, (GXPC_CullVerts - s_perfCullVerts0) / frames,
		        GXPC_CullOff ? " (off: L held at launch)" : "");
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
		s_perfDsCalls0 = GXPC_DrawSyncCalls;
		s_perfDsTicks0 = GXPC_DrawSyncTicks;
		s_perfSwitches0 = port_os_switches();
		uint32_t ringPeak = pica_ring_peak();
		uint32_t dropRing, dropCmd;
		pica_drop_reasons(&dropRing, &dropCmd);
		fprintf(stderr, "[perf]   vertex ring: peak %u KB of %u KB per frame | GPU commands: peak %u KB of %u KB | "
		        "%lu draws dropped so far (ring full %lu, command buffer full %lu) | linear free %u KB\n",
		        unsigned(ringPeak >> 10), unsigned(pica_ring_capacity() >> 10), unsigned(pica_cmd_peak() >> 10),
		        unsigned(pica_cmd_capacity() >> 10), (unsigned long)pica_dropped(),
		        (unsigned long)dropRing, (unsigned long)dropCmd, unsigned(pica_linear_free() >> 10));
		{
			uint32_t vKB, vN, lKB, lN, vFree, aUsed, aPeak, aSize;
			pica_tex_stats(&vKB, &vN, &lKB, &lN, &vFree);
			pica_copy_arena(&aUsed, &aPeak, &aSize);
			fprintf(stderr, "[perf]   textures: %u with a VRAM copy (%u KB), %u in FCRAM only (%u KB) | VRAM free %u KB | "
			        "%lu copies into VRAM, %lu evicted, %lu animated kept in FCRAM so far | EFB copies %u KB of %u (peak %u)\n",
			        unsigned(vN), unsigned(vKB), unsigned(lN), unsigned(lKB), unsigned(vFree), (unsigned long)pica_vram_copies(),
			        (unsigned long)pica_vram_evicted(), (unsigned long)pica_vram_animated(), unsigned(aUsed), unsigned(aSize),
			        unsigned(aPeak));
		}
		renderThreadPerf(frames);
		fprintf(stderr, "[perf]   resident vertices: %.0f drawn from %u pages (%u KB) per frame, %.0f KB stored per "
		        "frame | %lu resets, %lu times full%s\n", double(s_arenaDrawVerts - s_perfArenaDraw0) / frames,
		        unsigned(arenaPageCount()), unsigned(arenaBytes() >> 10),
		        double(s_arenaUploadBytes - s_perfArenaUp0) / frames / 1024.0, (unsigned long)arenaResets(),
		        (unsigned long)arenaFullCount(), arenaEnabled() ? "" : " (off: Up held at launch)");
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
		int top = int(tl & 0x7FF) - 342, left = int((tl >> 12) & 0x7FF) - 342;
		int bottom = int(br & 0x7FF) - 342 + 1, right = int((br >> 12) & 0x7FF) - 342 + 1;
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
};
static std::unordered_map<TexKey, TexEntry, TexKeyHash> s_texCache;
static uint32_t s_texBytes;
static std::vector<uint8_t> s_decodeBuf, s_scaleBuf;
static uint32_t s_texUploads, s_copyTexDraws;
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
};
static std::vector<EfbCopy> s_efbCopies;
static const size_t kMaxEfbCopies = 6;  // pica_draw.c's copy_reserve counts on this many

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
	        (ctrl >> 11) & 1, rs.bp[BP_CLEAR_AR] & 0xFFFF, rs.bp[BP_CLEAR_GB] & 0xFFFF, pix, pix == 3 ? " (Z copy: not supported, colour copied)" : "");
}

// GX coordinates (s, t) of a copy to its texture's: (s * as + bs, t * at + bt).
static inline void copyMap(const EfbCopy* c, float& as, float& bs, float& at, float& bt) {
	as = c->w / c->win[2];
	bs = (c->x - c->win[0]) / c->win[2];
	at = -c->h / c->win[3];
	bt = 1.f - (c->y - c->win[1]) / c->win[3];
}

static EfbCopy* findEfbCopy(const void* p) {
	for (EfbCopy& c : s_efbCopies)
		if (c.dest == p) return &c;
	return nullptr;
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

static void uploadI8(const uint8_t* src, uint32_t w, uint32_t h, PicaTex* t, uint32_t pw, uint32_t ph) {
	uint16_t* dst = static_cast<uint16_t*>(pica_tex_data(t));
	if (!dst) return;
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
	pica_tex_written(t);
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
		// as the goop map of the next level): the memory, not the old snapshot.
		const uint32_t fmt0 = (img0 >> 20) & 15;
		if (trustWriteStamps() && unwrittenSince(ptr, 0, 0) && !unwrittenSince(ptr, texLevelBytes(fmt0, w, h), c->clock)) {
			pica_tex_free(c->tex);
			s_efbCopies.erase(s_efbCopies.begin() + (c - s_efbCopies.data()));
		} else {
			c->usedFrame = s_rFrames;
			s_copyTexDraws++;
			*copy = c;
			return c->tex;
		}
	}
	// Mip levels: a mipmapped texture (min filter with mipmaps, max LOD of 1 or more) up to the
	// PICA's 8x8, the levels the GPU may pick. Only uploaded with them when level 0 is blank (the
	// sea's distant waves in sea.bmd, seen only through the smaller levels): every level costs
	// linear memory, which runs short in the bigger stages.
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
	if (e.tex && e.checkedFrame == s_rFrames) return e.tex;
	const uint32_t bytes = texLevelBytes(fmt, w, h);
	// As sms-pc-port's cache: when no write stamp (GXPC_InvalidateRange below) touched the data
	// since it was last hashed, trust it, but still hash every kRecheckFrames frames for writes
	// nothing reported. TLUTs live in TMEM, rewritten by register loads: always hashed (small).
	const uint64_t th = tlut ? hashBytes(tlut, tlutBytes) : 0;
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
	if (!rewritten && e.tex && dh == e.dataHash && th == e.tlutHash) return e.tex;

	const uint32_t pw = pow2Side(w), ph = pow2Side(h);
	const bool la8 = fmt == 1;  // I8: decoded straight into an LA8 texture (I, I), as GX samples it
	int levels = 1;
	if (mipLevels > 1) {
		bool blank = true;
		for (uint32_t i = 0; i < bytes && blank; i++) blank = ptr[i] == 0;
		if (blank) levels = mipLevels;
	}
	// Single-level textures are decoded first, to pick the texel format from what they hold.
	const uint8_t* src = nullptr;
	int tf = PICA_TF_RGBA8;
	if (!la8) {
		s_decodeBuf.resize(size_t(w) * h * 4);
		decodeTexture(ptr, fmt, w, h, tlut, key.tlutFmt, s_decodeBuf.data());
		src = s_decodeBuf.data();
		if (levels == 1) tf = smallTexFormat(src, size_t(w) * h, fmt, key.tlutFmt);
	} else {
		tf = PICA_TF_LA8;
	}
	if (e.tex && (e.levels != levels || (levels == 1 && pica_tex_format(e.tex) != tf))) {
		s_texBytes -= pica_tex_bytes(e.tex);
		pica_tex_free(e.tex);
		e.tex = nullptr;
	}
	if (!e.tex) {
		e.levels = levels;
		e.tex = levels > 1 ? pica_tex_create_mip(pw, ph, levels, 0.f)
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
		const uint8_t* lp = ptr;
		for (int l = 1; l < levels; l++) {
			lp += texLevelBytes(fmt, w >> (l - 1), h >> (l - 1));
			s_decodeBuf.resize(size_t(w >> l) * (h >> l) * 4);
			decodeTexture(lp, fmt, w >> l, h >> l, tlut, key.tlutFmt, s_decodeBuf.data());
			pica_tex_upload_level(e.tex, l, s_decodeBuf.data());
		}
	} else {
		pica_tex_upload(e.tex, src);
	}
	e.dataHash = dh;
	e.tlutHash = th;
	s_texUploads++;
	evictTextures();
	return e.tex;
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
};

static void readTev(TevSetup& t) {
	memset(&t, 0, sizeof t);
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

		// A channel whose result nothing reads is not computed.
		const bool dead[2] = {!(liveAfter[s] >> (2 * ((ce >> 22) & 3) + CH_RGB) & 1),
		                      !(liveAfter[s] >> (2 * ((ae >> 22) & 3) + CH_A) & 1)};
		ChanPlan plan[2];
		memset(plan, 0, sizeof plan);
		if (!dead[CH_RGB])
			plan[CH_RGB] = planChannel(colorIn((ce >> 12) & 15), colorIn((ce >> 8) & 15), colorIn((ce >> 4) & 15),
		                           colorIn(ce & 15), 3, (ce >> 16) & 3, (ce >> 18) & 1, (ce >> 20) & 3, (ce >> 19) & 1);
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
			                         alphaIn((ae >> 4) & 7), 1, (ae >> 16) & 3, (ae >> 18) & 1, (ae >> 20) & 3, (ae >> 19) & 1);
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

static void alphaTest(PicaState& st) {
	const uint32_t r = rs.bp[BP_ALPHACOMPARE];
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
};
static std::unordered_map<uint64_t, TevCacheEntry> s_tevCache;
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
static bool s_xfDirty = true;
static bool s_gpuTransform = true;  // GX3DS_SetGpuTransform: off draws everything on the CPU path
static float s_posRows[PICA_GX_MTX_ROWS][4], s_nrmRows[PICA_GX_MTX_ROWS][4];
static uint32_t s_gpuBatches, s_cpuBatches;


static bool drawGpu(const VtxFmtLayout& L, uint32_t n, uint32_t nidx, const TevSetup& tev, bool debug,
                    PicaTex* const utex[PICA_MAX_TEX], const EfbCopy* const ucopy[PICA_MAX_TEX],
                    const int ucoord[PICA_MAX_TEX], const PicaState& st) {
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
	const bool lit = (((cctrl >> 1) & 1) || debug) && hasNrm;

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
	for (int un = 0; un < PICA_MAX_TEX; un++) {
		if (!utex[un]) continue;
		const int coord = ucoord[un];
		const uint32_t tg = rs.xfReg[XFR_TEXGEN + coord];
		const uint32_t proj = (tg >> 1) & 1, form = (tg >> 2) & 1, type = (tg >> 4) & 7, row = (tg >> 7) & 31;
		if (proj && type == 0) return gpuFail(FB_TEXGEN_PROJ, n);
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
		float R[2][4];
		if (type != 0) {
			// Emboss and colour texgens: the CPU path passes the source's (s, t) through (texCoord), so
			// does this, with no texture or post matrix.
			const float id[2][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}};
			memcpy(R, id, sizeof R);
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
		for (int r = 0; r < 2; r++) memcpy(R[r], m + 4 * r, 16);
		if (row >= 5 || form == 0)  // the source's third component is 1 (AB11; texture attributes)
			for (int r = 0; r < 2; r++) R[r][3] += R[r][2], R[r][2] = 0.f;
		// (t0, t1, t2 = 1) through the post matrix
		if (rs.xfReg[XFR_DUALTEX] & 1) {
			const uint32_t post = rs.xfReg[XFR_POSTTEX + coord];
			if ((post >> 8) & 1) return gpuFail(FB_POST_NORMALISE, n);  // normalised before the post matrix
			const float* pm = reinterpret_cast<const float*>(&rs.xfMem[0x500 + (post & 63) * 4]);
			if (pm[8] != 0.f || pm[9] != 0.f) return gpuFail(FB_POST_PROJ, n);  // varying third component: divided per vertex
			float c = pm[10] + pm[11];
			float Q[2][4];
			for (int r = 0; r < 2; r++)
				for (int k = 0; k < 4; k++)
					Q[r][k] = pm[4 * r] * R[0][k] + pm[4 * r + 1] * R[1][k] + (k == 3 ? pm[4 * r + 2] + pm[4 * r + 3] : 0.f);
			if (c != 0.f && c != 1.f)
				for (int r = 0; r < 2; r++)
					for (int k = 0; k < 4; k++) Q[r][k] /= c;
			memcpy(R, Q, sizeof R);
		}
		}
	efbRect:
		if (const EfbCopy* cp = ucopy[un]) {  // through the copy's source rectangle into its texture's window
			float as, bs, at, bt;
			copyMap(cp, as, bs, at, bt);
			for (int k = 0; k < 4; k++) R[0][k] *= as, R[1][k] *= at;
			R[0][3] += bs;
			R[1][3] += bt;
		}
		memcpy(u.T[2 * un], R[0], 16);
		memcpy(u.T[2 * un + 1], R[1], 16);
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
	u.LD[0] = 0.3f, u.LD[1] = 0.6f, u.LD[2] = 0.74f, u.LD[3] = lit ? 1.f : 0.f;

	// View space -> EFB pixels and GameCube depth (homogeneous), as the CPU loop computes them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = rs.xfReg[XFR_PROJ + 6] & 1;
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1), vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = xff(XFR_VIEWPORT + 4) - 342.0f, vcz = xff(XFR_VIEWPORT + 5);
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

	// Matrix rows, refreshed when XF memory changed (renderState).
	const bool dirty = s_xfDirty;
	if (dirty) {
		for (int r = 0; r < PICA_GX_MTX_ROWS; r++) {
			memcpy(s_posRows[r], &rs.xfMem[4 * r], 16);
			memcpy(s_nrmRows[r], &rs.xfMem[256 + 3 * r], 12);
			s_nrmRows[r][3] = 0.f;
		}
		s_xfDirty = false;
	}

	void* vb;
	uint16_t* ib;
	{
		PerfScope ps(PERF_XFORM);
		const bool inPlace = r_arena != nullptr;  // decoded into the ring and claimed: drawn where it is
		if (!pica_alloc_raw(inPlace ? 0 : n * r_stride, nidx, &vb, &ib)) return true;  // ring full: dropped
		if (inPlace) vb = const_cast<uint8_t*>(r_verts);
		else memcpy(vb, r_verts, size_t(n) * r_stride);
		memcpy(ib, r_idx, size_t(nidx) * 2);
	}
	PerfScope ps(PERF_SUBMIT);
	if (!pica_draw_gx(vb, n, &lay, ib, nidx, &u, dirty, s_posRows, s_nrmRows, &st)) {
		s_xfDirty = s_xfDirty || dirty;
		return gpuFail(FB_LAYOUT, n);
	}
	s_gpuBatches++;
	return true;
}

static void drawTriangles() {
	if (((rs.bp[BP_GENMODE] >> 14) & 3) == 3) return;  // every face culled
	if (s_afterTexCopy) s_drawsAfterCopy++;
	else s_drawsBeforeCopy++;
	const VtxFmtLayout& L = r_layout;
	uint32_t n = r_count, nidx = uint32_t(r_nidx);

	s_perfBatches++;
	s_perfVerts += n;
	// The textures the TEV stages sample, per (map, coordinate) pair, and the combiner stages,
	// which give PICA texture units to the pairs they use (none in debug-colour mode).
	const bool debug = s_debugColors;
	uint64_t perfTev0 = svcGetSystemTick();
	TevSetup tev;
	readTev(tev);
	perfAdd(PERF_TEV, svcGetSystemTick() - perfTev0);
	PicaTex* ptex[kMaxPairs] = {};
	const EfbCopy* pcopy[kMaxPairs] = {};
	bool pwhite[kMaxPairs];
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
	perfTev0 = svcGetSystemTick();

	PicaState st;
	memset(&st, 0, sizeof st);  // the PICA side compares states bytewise
	currentPicaState(st);
	alphaTest(st);
	bool translated = true;
	unsigned dumpId = 0;
	const uint64_t tevKey = tevCacheKey(pwhite, tev.nPairs, st, debug, ptex);
	auto tevHit = s_tevCache.find(tevKey);
	const bool cached = tevHit != s_tevCache.end();
	if (cached) {
		tev = tevHit->second.tev;
		st = tevHit->second.st;
		translated = tevHit->second.translated;
		dumpId = tevHit->second.dumpId;
	}
	if (cached) {
	} else if (debug) {
		tev.nUnits = 0;
	} else if (isThpSetup(tev) && ptex[tev.stagePair[0]] && ptex[tev.stagePair[1]] && ptex[tev.stagePair[2]]) {
		translateThp(tev, st);
	} else if (isStampSetup(tev) && ptex[tev.stagePair[0]] && ptex[tev.stagePair[2]]) {
		translateStamp(tev, st);
	} else {
		translated = translateTevFallback(tev, pwhite, st);
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
		if (s_tevCache.size() >= kTevCacheMax) s_tevCache.clear();
		TevCacheEntry& e = s_tevCache[tevKey];
		e.tev = tev;
		e.st = st;
		e.translated = translated;
		e.dumpId = 0;
	}
	static const uint8_t kWrap[4] = {0, 2, 3, 2};  // clamp, repeat, mirror -> GPU_TEXTURE_WRAP_PARAM
	PicaTex* utex[PICA_MAX_TEX] = {};
	const EfbCopy* ucopy[PICA_MAX_TEX] = {};
	int ucoord[PICA_MAX_TEX] = {};
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
		char extra[200];
		snprintf(extra, sizeof extra,
		         "clip %s cmode0 %06x zmode %06x alphacmp %06x chans %u ctrl %04x/%04x %04x/%04x mat %08x %08x vtx clr %08x %08x (fmt %x)",
		         rs.xfReg[0x05] ? "off" : "on", rs.bp[BP_CMODE0] & 0xFFFFFF, rs.bp[BP_ZMODE] & 0xFFFFFF, rs.bp[BP_ALPHACOMPARE] & 0xFFFFFF,
		         rs.xfReg[XFR_NUMCHANS] & 3, rs.xfReg[XFR_COLOR0CTRL] & 0x7FFF, rs.xfReg[XFR_ALPHA0CTRL] & 0x7FFF,
		         rs.xfReg[XFR_COLOR0CTRL + 1] & 0x7FFF, rs.xfReg[XFR_ALPHA0CTRL + 1] & 0x7FFF, rs.xfReg[XFR_MAT0],
		         rs.xfReg[XFR_MAT0 + 1], __builtin_bswap32(c0), __builtin_bswap32(c1), r_fmt);
		dumpId = tevDump(tev, st, translated, ptex, pcopy, extra);
		if (!cached || !s_tevCache[tevKey].dumpId) s_tevCache[tevKey].dumpId = dumpId;
	}
	if (s_opaque) {
		st.blend = 0;
		st.alpha_test = st.alpha_test2 = 0;
		st.alpha_func = 1;  // GPU_ALWAYS
	}

	perfAdd(PERF_TEV, svcGetSystemTick() - perfTev0);
	if (s_gpuTransform && drawGpu(L, n, nidx, tev, debug, utex, ucopy, ucoord, st)) return;
	s_cpuBatches++;
	// A projective texgen on unit 0 (the sea's lookup of the screen copy, which the sea's colour
	// comes from): its (s q, t q, q) go to the GPU, which divides per pixel (texture unit 0 in
	// projection mode). Divided here per vertex, the coordinates bend across a large polygon and
	// it shows the wrong part of the screen. q travels in unit 2's coordinate, so unit 2 must be free.
	if (utex[0] && !utex[2]) {
		const uint32_t tg0 = rs.xfReg[XFR_TEXGEN + ucoord[0]];
		st.proj0 = ((tg0 >> 1) & 1) && ((tg0 >> 4) & 7) == 0;
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
	const uint64_t perfXform0 = svcGetSystemTick();

	// Projection (GXSetProjection) and viewport (GXSetViewport), as the GX shader reads them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = rs.xfReg[XFR_PROJ + 6] & 1;
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1), vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = xff(XFR_VIEWPORT + 4) - 342.0f, vcz = xff(XFR_VIEWPORT + 5);
	const float kInvZ = 1.0f / 16777215.0f;
	const bool noClip = rs.xfReg[0x05] != 0;

	// The colour channels the TEV stages read (0 or 1) for the rgb and the alpha: material
	// register or vertex colour; lighting reduced to one fixed light.
	const int ch = debug ? 0 : tev.rasRgbChan, cha = debug ? 0 : tev.rasAChan;
	const uint32_t cctrl = rs.xfReg[XFR_COLOR0CTRL + ch], actrl = rs.xfReg[XFR_ALPHA0CTRL + cha], mat = rs.xfReg[XFR_MAT0 + ch],
	               amat = rs.xfReg[XFR_MAT0 + cha];
	const bool hasClr = (r_fmt & (VF_CLR0 << ch)) != 0, hasAClr = (r_fmt & (VF_CLR0 << cha)) != 0,
	           hasNrm = (r_fmt & VF_NRM) != 0, hasMtx = (r_fmt & VF_MTX) != 0;
	const bool clrFromVtx = (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasAClr;
	const bool lit = ((cctrl >> 1) & 1) && hasNrm;
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
		float shade = 1.0f;
		if (lit || (debug && hasNrm)) {
			float nrm[3];
			memcpy(nrm, v + L.nrm, 12);
			const float* nm = xfMtx(256 + pm * 3);  // normal matrices: 3x3, word 256 + 3 * index
			float nx = nm[0] * nrm[0] + nm[1] * nrm[1] + nm[2] * nrm[2];
			float ny = nm[3] * nrm[0] + nm[4] * nrm[1] + nm[5] * nrm[2];
			float nz = nm[6] * nrm[0] + nm[7] * nrm[1] + nm[8] * nrm[2];
			float len = sqrtf(nx * nx + ny * ny + nz * nz);
			float d = len > 0.f ? (0.3f * nx + 0.6f * ny + 0.74f * nz) / len : 1.0f;  // light from up-front
			shade = 0.35f + 0.65f * (d > 0.f ? d : 0.f);
		}
		o.rgba[0] = clamp255(src[0] * shade);
		o.rgba[1] = clamp255(src[1] * shade);
		o.rgba[2] = clamp255(src[2] * shade);
		o.rgba[3] = alphaFromVtx ? v[L.clr[cha] + 3] : matRGBA[3];
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
	perfAdd(PERF_XFORM, svcGetSystemTick() - perfXform0);
	{
		PerfScope ps(PERF_SUBMIT);
		if (submit) pica_draw_triangles(out, oidx, nidx, &st);
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
enum RecType : uint32_t { RC_WRAP = 1, RC_STATE, RC_DRAW, RC_COPY };
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
// Registers written since the last state record (game side).
static uint32_t s_dBp[8], s_dXm[4], s_dTlut[32];  // BP registers, 16-word XF memory blocks, 512-byte TLUT blocks
static bool s_dXr, s_dAny, s_dAll = true;
static const uint32_t kXmBlocks = XF_MEM_WORDS / 16, kTlutBlocks = TMEM_TLUT_SIZE / 512;
// Profiler. Game side: waits, queue use. Render side: busy and idle time, and the game side's
// numbers as the last copy record brought them.
static uint64_t s_gWait[3], s_qBytes;
static uint32_t s_qPeak, s_qExt;
static uint64_t s_rBusy, s_rIdle, s_perfRBusy0, s_perfRIdle0;
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
	s_dBp[(reg >> 5) & 7] |= 1u << (reg & 31);
	s_dAny = true;
}
void xfTouched(uint32_t addr) {
	if (addr < XF_MEM_WORDS) {
		const uint32_t b = addr >> 4;
		s_dXm[b >> 5] |= 1u << (b & 31);
	} else {
		s_dXr = true;
	}
	s_dAny = true;
}
void tlutTouched(uint32_t off, uint32_t bytes) {
	if (!bytes) return;
	for (uint32_t b = off >> 9, e = (off + bytes - 1) >> 9; b <= e && b < kTlutBlocks; b++)
		s_dTlut[b >> 5] |= 1u << (b & 31);
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

// Room for a record of `size` bytes (a multiple of 8, at most the ring's size), contiguous.
static uint8_t* rcReserve(uint32_t size) {
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
// 0: XF registers follow), BP mask (8 words) and the registers it names, XF memory block mask (4)
// and the blocks, TLUT block mask (32) and the blocks, then the XF registers.
// Bytes of the registers written since the last state record (0: none).
static uint32_t stateBytes() {
	if (s_dAll) {
		memset(s_dBp, 0xFF, sizeof s_dBp);
		memset(s_dXm, 0, sizeof s_dXm);
		for (uint32_t b = 0; b < kXmBlocks; b++) s_dXm[b >> 5] |= 1u << (b & 31);
		memset(s_dTlut, 0xFF, sizeof s_dTlut);
		s_dXr = s_dAny = true;
		s_dAll = false;
	}
	if (!s_dAny) return 0;
	const uint32_t nbp = popcount(s_dBp, 8), nxm = popcount(s_dXm, 4), nt = popcount(s_dTlut, 32);
	return 4 + 32 + nbp * 4 + 16 + nxm * 64 + 128 + nt * 512 + (s_dXr ? XF_REG_COUNT * 4 : 0);
}

// Writes them (stateBytes() bytes) and starts over.
static void writeState(uint8_t* dst) {
	PerfScope ps(PERF_QSTATE);
	uint32_t* w = reinterpret_cast<uint32_t*>(dst);
	*w++ = s_dXr ? 1 : 0;
	memcpy(w, s_dBp, 32);
	w += 8;
	for (int k = 0; k < 8; k++)
		for (uint32_t m = s_dBp[k]; m; m &= m - 1) *w++ = g.bp[k * 32 + __builtin_ctz(m)];
	memcpy(w, s_dXm, 16);
	w += 4;
	for (int k = 0; k < 4; k++)
		for (uint32_t m = s_dXm[k]; m; m &= m - 1) {
			memcpy(w, &g.xfMem[(k * 32 + __builtin_ctz(m)) * 16], 64);
			w += 16;
		}
	memcpy(w, s_dTlut, 128);
	w += 32;
	for (int k = 0; k < 32; k++)
		for (uint32_t m = s_dTlut[k]; m; m &= m - 1) {
			memcpy(w, g.tlutMem + (k * 32 + __builtin_ctz(m)) * 512, 512);
			w += 128;
		}
	if (s_dXr) memcpy(w, g.xfReg, XF_REG_COUNT * 4);
	memset(s_dBp, 0, sizeof s_dBp);
	memset(s_dXm, 0, sizeof s_dXm);
	memset(s_dTlut, 0, sizeof s_dTlut);
	s_dXr = s_dAny = false;
}

static void emitState() {
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
}

static void renderState(const uint8_t* p) {
	const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
	const bool xr = *w++ & 1;
	uint32_t m[32];
	memcpy(m, w, 32);
	w += 8;
	for (int k = 0; k < 8; k++)
		for (uint32_t b = m[k]; b; b &= b - 1) rs.bp[k * 32 + __builtin_ctz(b)] = *w++;
	memcpy(m, w, 16);
	w += 4;
	for (int k = 0; k < 4; k++)
		for (uint32_t b = m[k]; b; b &= b - 1) {
			memcpy(&rs.xfMem[(k * 32 + __builtin_ctz(b)) * 16], w, 64);
			w += 16;
			s_xfDirty = true;
		}
	memcpy(m, w, 128);
	w += 32;
	for (int k = 0; k < 32; k++)
		for (uint32_t b = m[k]; b; b &= b - 1) {
			memcpy(rs.tlutMem + (k * 32 + __builtin_ctz(b)) * 512, w, 512);
			w += 128;
		}
	if (xr) memcpy(rs.xfReg, w, XF_REG_COUNT * 4);
}

static void enqueueDraw() {
	const uint8_t* verts;
	uint32_t count;
	if (s_bArena) {
		// Drawn from its window of the resident page: indices from the window's start (rebased below).
		count = s_bmax - s_bmin;
		verts = s_bArena + size_t(s_bmin) * s_bstride;
		s_arenaDrawVerts += count;
	} else {
		count = s_bcount;
		verts = s_vdata.data();
	}
	const uint32_t nidx = uint32_t(s_idx.size());
	const size_t vbytes = s_bArena ? 0 : (size_t(count) * s_bstride + 3) & ~size_t(3);
	const size_t ibytes = size_t(nidx) * 2;
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
	memcpy(d->kreg, g.kreg, sizeof d->kreg);
	memcpy(d->texImage, g.texImage, sizeof d->texImage);
	d->copyDest = g.copyDest;
	d->stateOff = stateOff;
	d->cut = s_bcut;
	d->amtx = s_bArena && (s_bfmt & VF_MTX);
	d->pnMax = s_bPnMax;
	memcpy(d->texMtx, s_bTexMtx, sizeof d->texMtx);
	uint8_t* data = ext ? ext : p + head;
	if (!s_bArena && !inPlace) {
		PerfScope pv(PERF_QVERT);
		memcpy(data, verts, size_t(count) * s_bstride);
	}
	{
		PerfScope pi(PERF_QIDX);
		uint16_t* ix = reinterpret_cast<uint16_t*>(data + vbytes);
		const uint32_t* si = s_idx.data();
		const uint32_t base = s_bArena ? s_bmin : 0;
		for (uint32_t i = 0; i < nidx; i++) ix[i] = uint16_t(si[i] - base);
	}
	rcCommit(uint32_t(size));
}

static void renderDraw(const RecHdr* h) {
	const DrawRec* d = reinterpret_cast<const DrawRec*>(h + 1);
	if (d->stateOff) renderState(reinterpret_cast<const uint8_t*>(h) + d->stateOff);
	memcpy(rs.kreg, d->kreg, sizeof rs.kreg);
	memcpy(rs.texImage, d->texImage, sizeof rs.texImage);
	rs.copyDest = d->copyDest;
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
	r_idx = reinterpret_cast<const uint16_t*>(data + vbytes);
	if (d->cut) pica_set_draw_cut(1);
	drawBatch();
	if (d->cut) pica_set_draw_cut(0);
	free(d->ext);
}

static void renderCopy(const CopyRec* c);

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
		const uint64_t t0 = svcGetSystemTick();
		while (s_head.load() != s_tail.load(std::memory_order_relaxed)) renderDrain();
		const uint64_t t1 = svcGetSystemTick();
		s_rBusy += t1 - t0;
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
				continue;
			}
		}
		s_rSleeping.store(1);
		if (s_head.load() == s_tail.load(std::memory_order_relaxed)) pica_event_wait(0);
		s_rSleeping.store(0);
		s_rIdle += svcGetSystemTick() - t1;
	}
}

void flushBatch() {
	if (!s_ready) return;
	PerfScopeExcl ps(PERF_BATCH);
	if (batchHasVertices()) {
		rcInit();
		enqueueDraw();
	}
	s_vdata.clear();
	s_idx.clear();
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
	PrimClass cls = primClass(op);
	cutCheck();
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt || s_bArena)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	size_t at = size_t(s_bcount) * stride;
	s_vdata.resize(at + size_t(n) * stride);
	s_decodeT0 = svcGetSystemTick();  // the front half decodes the vertices until primitiveEnd
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
	if (s_decodeT0) {
		perfAdd(PERF_DECODE, svcGetSystemTick() - s_decodeT0);
		s_decodeT0 = 0;
	}
	if (!s_ready || n == 0) return;
	uint32_t base = s_bcount;
	s_bcount += n;
	GX3DS_GameVerts += n;
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
};
static std::vector<ArenaPage> s_apages;
static int s_apageFor[256];  // current page of each stride, or -1
static std::vector<uint32_t> s_aidx;
static uint32_t s_agen = 1;
static bool s_arenaOn = true, s_aresetDue, s_apagesInit;
static size_t s_abytes;
static uint32_t s_aresetFrame, s_aresets, s_afull;
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
static const uint32_t kRunInfo = 3;
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
	const uint32_t vb = pg.used;
	uint8_t* dst = pg.base + size_t(vb) * stride;
	memcpy(dst, verts, size_t(n) * stride);
	pica_flush(dst, n * stride);
	pg.used += n;
	const size_t io = s_aidx.size() + kRunInfo;
	s_aidx.resize(io + nidx);
	s_aidx[io - kRunInfo] = 0;  // its matrices: not worked out yet (arenaRunMtx)
	for (uint32_t i = 0; i < nidx; i++) s_aidx[io + i] = idx[i] + vb;
	out.gen = s_agen;
	out.baseVertex = int32_t((uint32_t(pi) << 16) | vb);
	out.iOff = io;
	s_arenaUploadBytes += uint64_t(n) * stride;
	return true;
}

void appendArena(PrimClass cls, uint32_t fmt, uint32_t stride, const ArenaRef& ref, uint32_t n, uint32_t nidx,
                 const uint8_t* verts) {
	PerfScopeExcl ps(PERF_APPEND);
	GX3DS_GameVerts += n;
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
	if (!batchHasVertices()) {
		s_bArena = pg.base;
		s_bmin = vb;
		s_bmax = vb + n;
	} else {
		s_bmin = std::min(s_bmin, vb);
		s_bmax = std::max(s_bmax, vb + n);
	}
	const size_t at = s_idx.size();
	s_idx.resize(at + nidx);
	memcpy(&s_idx[at], &s_aidx[ref.iOff], size_t(nidx) * 4);
}

// The batch reads its indices from s_aidx's copy, never from the front half's runs.
bool batchUsesArena() { return false; }

static size_t arenaPageCount() { return s_apages.size(); }
static size_t arenaBytes() { return s_abytes; }
static uint32_t arenaResets() { return s_aresets; }
static uint32_t arenaFullCount() { return s_afull; }

// After a frame went to the GPU: empty the pages once uploads have stopped for lack of room.
// Game side, after a display copy was queued: the render thread and the GPU are let finish first.
static void arenaFrameEnd() {
	if (!s_aresetDue) return;
	s_aresetDue = false;
	if (g_displayFrames - s_aresetFrame < kArenaResetGap && s_aresets) return;
	if (s_rtCore >= 0) gameWaitFor(s_gWait[2], [] { return s_tail.load() == s_wpos; });
	pica_gpu_idle();  // every queued frame done with the pages
	for (const ArenaPage& p : s_apages) pica_linear_free_block(p.base);
	s_apages.clear();
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
static const uint32_t kStreamChunk = 64u << 10, kStreamChunks = 16;
static uint8_t* s_sChunk[kStreamChunks];
static uint32_t s_sLen[kStreamChunks];
static std::atomic<uint32_t> s_sHead{0}, s_sTail{0};  // chunks handed over; done (released after parsing)
static std::atomic<uint32_t> s_sFramesDone{0};        // display copies the GX thread has run
static uint32_t s_sFramesSent;                         // ... and the game thread has sent
static bool s_gxWanted = true, s_streamOn;
static std::atomic<uint64_t> s_sWait[4];  // game thread waiting: a free chunk, a frame ahead, a drain, a list rewrite
static std::atomic<uint64_t> s_sBusy{0}, s_sIdle{0};  // GX thread: parsing, waiting for chunks
enum { EV_GX = 2, EV_GAME = 3 };

// Game thread: waits until ready() (the GX thread signals EV_GAME as it releases chunks and frames).
template <class F>
static void streamWait(std::atomic<uint64_t>& acc, F ready) {
	if (ready()) return;
	const uint64_t t0 = svcGetSystemTick();
	while (!ready()) pica_event_wait(EV_GAME);
	acc += svcGetSystemTick() - t0;
}

// The write-gather pipe's hand-over (gx_fifo.cpp): publishes the full chunk and returns the next,
// once the GX thread is done with what it held last.
static uint8_t* streamSwap(uint8_t* buf, uint32_t len, uint32_t* cap) {
	*cap = kStreamChunk;
	if (buf) {
		s_sLen[s_sHead.load(std::memory_order_relaxed) % kStreamChunks] = len;
		s_sHead.fetch_add(1, std::memory_order_release);
		pica_event_signal(EV_GX);
	}
	const uint32_t h = s_sHead.load(std::memory_order_relaxed);
	streamWait(s_sWait[0], [h] { return h - s_sTail.load(std::memory_order_acquire) < kStreamChunks; });
	return s_sChunk[h % kStreamChunks];
}

// Display lists the game rewrites in place. J3D rebuilds a material's list (GD) every time a model
// is entered for drawing, and a list shared by every instance of a model (J3DMaterial's
// newSharedDisplayList) swaps its two buffers once per instance: with an even number on screen the
// same buffer is rewritten every frame, while the GX thread may not yet have read last frame's call
// of it (a crash on Delfino Airstrip: a nested call out of a half-rewritten list). The GameCube
// is safe through the frame's GXDrawDone (THPPlayerDrawDone, before the copy); here each call
// through the stream stamps the 32-byte blocks of MEM1 it covers with the chunk that carries it,
// and before the game starts writing a list (GDInitGDLObj) it waits while the GX thread has not
// finished that chunk.
static uint32_t* s_dlMark;  // per 32-byte block of MEM1: 1 + the chunk of its last call (0: none)
static uint32_t s_dlBlocks, s_dlWaits;

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
	for (uint32_t b = b0; b <= b1; b++) s_dlMark[b] = mark;
}
// Game thread, before it writes [p, p + n).
static void streamBeforeWrite(const void* p, uint32_t n) {
	uint32_t b0, b1;
	if (!s_streamOn || !mem1Blocks(p, n, b0, b1)) return;
	const uint32_t tail = s_sTail.load(std::memory_order_acquire);
	int32_t ahead = 0;  // the latest call's chunk + 1, past the GX thread's
	for (uint32_t b = b0; b <= b1; b++) ahead = std::max(ahead, int32_t(s_dlMark[b] - tail));
	if (ahead <= 0) return;  // read already (or never called)
	const uint32_t mark = tail + uint32_t(ahead);
	if (mark - 1 == s_sHead.load(std::memory_order_relaxed)) GXPC_StreamFlush();  // still being filled
	s_dlWaits++;
	streamWait(s_sWait[3], [mark] { return int32_t(mark - s_sTail.load(std::memory_order_acquire)) <= 0; });
}

// Draw-sync tokens, GX thread -> game thread.
static const uint32_t kTokens = 256;
static uint16_t s_tok[kTokens];
static std::atomic<uint32_t> s_tokHead{0}, s_tokTail{0};
static uint32_t s_tokDropped;
static void streamToken(uint16_t t) {
	const uint32_t h = s_tokHead.load(std::memory_order_relaxed);
	if (h - s_tokTail.load(std::memory_order_acquire) >= kTokens) {
		if (!s_tokDropped++) fprintf(stderr, "[gx3ds] draw-sync tokens not taken by the game: dropped\n");
		return;
	}
	s_tok[h % kTokens] = t;
	s_tokHead.store(h + 1, std::memory_order_release);
	port_irq_kick();
}
// Game thread, in interrupt context (an emulated-OS interrupt source): the callbacks.
static void streamTokenPoll() {
	for (;;) {
		const uint32_t t = s_tokTail.load(std::memory_order_relaxed);
		if (t == s_tokHead.load(std::memory_order_acquire)) return;
		const uint16_t token = s_tok[t % kTokens];
		s_tokTail.store(t + 1, std::memory_order_release);
		if (!drawSyncCallback) continue;
		GXPC_DrawSyncCalls++;
		const uint64_t t0 = svcGetSystemTick();
		drawSyncCallback(token);
		GXPC_DrawSyncTicks += svcGetSystemTick() - t0;
	}
}

static void gxMain(void*) {
	std::vector<uint8_t> carry;  // a command the last chunk cut, and its whole length
	uint32_t carryNeed = 0;
	for (;;) {
		const uint64_t t0 = svcGetSystemTick();
		while (s_sTail.load(std::memory_order_relaxed) == s_sHead.load(std::memory_order_acquire)) pica_event_wait(EV_GX);
		const uint64_t t1 = svcGetSystemTick();
		s_sIdle += t1 - t0;
		const uint32_t t = s_sTail.load(std::memory_order_relaxed);
		const uint8_t* p = s_sChunk[t % kStreamChunks];
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
		s_sTail.store(t + 1, std::memory_order_release);
		pica_event_signal(EV_GAME);
		s_sBusy += svcGetSystemTick() - t1;
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
	if (!s_dlMark) {
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
// The profiler window's GX-thread numbers (render thread, at the window's end): the game thread's
// wait for it, then, with print, the line (which starts the next window).
static void streamPerf(uint64_t& gameWait, double perMs, bool print) {
	static uint64_t w0[4], busy0, idle0;
	static uint32_t dlWaits0;
	uint64_t w[4];
	for (int i = 0; i < 4; i++) w[i] = s_sWait[i].load() - w0[i];
	gameWait = w[0] + w[1] + w[2] + w[3];
	if (!print) return;
	const uint64_t busy = s_sBusy.load(), idle = s_sIdle.load();
	fprintf(stderr,
	        "[perf]   GX thread (core 1): busy %.1f, idle %.1f ms/frame | game thread waited for it %.1f ms/frame "
	        "(buffers full %.1f, a frame ahead %.1f, GXInit %.1f, rewriting a list it had yet to read %.1f, %u times) "
	        "| %u chunks queued\n",
	        double(busy - busy0) * perMs, double(idle - idle0) * perMs, double(gameWait) * perMs, double(w[0]) * perMs,
	        double(w[1]) * perMs, double(w[2]) * perMs, double(w[3]) * perMs, unsigned(s_dlWaits - dlWaits0),
	        unsigned(s_sHead.load() - s_sTail.load()));
	for (int i = 0; i < 4; i++) w0[i] += w[i];
	dlWaits0 = s_dlWaits;
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
		arenaFrameEnd();
		if (s_rtCore >= 0) gameWaitFor(s_gWait[1], [] { return s_gFrames - s_rDone.load() <= 1; });
	}
}

// Render side.
static void renderCopy(const CopyRec* cr) {
	const uint32_t ctrl = cr->ctrl;
	rs.copyDest = cr->copyDest;
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
		uint16_t tw, th;
		float win[4];
		pica_copy_tex_layout(rx, ry, rw, rh, half, exact, &tw, &th, win);
		const int rgb565 = cfmt == 4;  // RGB565: no alpha to keep
		EfbCopy* c = d ? findEfbCopy(d) : nullptr;
		if (c && !pica_copy_tex_fits(c->tex, tw, th, rgb565)) {  // another size or format than its last copy
			pica_tex_free(c->tex);
			s_efbCopies.erase(s_efbCopies.begin() + (c - s_efbCopies.data()));
			c = nullptr;
		}
		if (d && !c) {
			if (s_efbCopies.size() >= kMaxEfbCopies) {  // replace the least recently sampled
				size_t oldest = 0;
				for (size_t i = 1; i < s_efbCopies.size(); i++)
					if (s_efbCopies[i].usedFrame < s_efbCopies[oldest].usedFrame) oldest = i;
				pica_tex_free(s_efbCopies[oldest].tex);
				s_efbCopies.erase(s_efbCopies.begin() + oldest);
			}
			PicaTex* t = pica_copy_tex_create(tw, th, rgb565);
			if (t) {
				s_efbCopies.push_back(EfbCopy{d, t, 0, 0, 0, 0, {0, 0, 1, 1}, s_rFrames, 0, 0});
				c = &s_efbCopies.back();
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
			{
				PerfScope ps(PERF_COPY);
				// One-channel formats read back with that channel in all four (R8 and I8 as I8),
				// RA4 / RA8 with red as the intensity.
				const int rep = cfmt == 0 || cfmt == 1 || cfmt == 8 ? 0 : cfmt == 7 ? 3 : cfmt == 9 ? 1 : cfmt == 10 ? 2
				                : cfmt == 2 || cfmt == 3 ? 5 : -1;
				pica_efb_copy_to_ch(c->tex, pix != 1 || cfmt == 4, rep, c->win);
			}
			// One-byte formats (I8, A8, R8, G8, B8) go back to memory too: the CPU may read them
			// (the goop map, TPollutionLayer::isPolluted).
			if (cfmt == 1 || (cfmt >= 7 && cfmt <= 10)) {
				const bool intensity = ((ctrl >> 15) & 3) == 3;
				const int dw = half ? int(c->w) / 2 : int(c->w), dh = half ? int(c->h) / 2 : int(c->h);
				const int chan = cfmt == 7 ? 3 : cfmt == 9 ? 1 : cfmt == 10 ? 2 : cfmt == 1 && intensity ? 4 : 0;
				// Tiles a row from the copy's width (BP_COPY_STRIDE holds bytes a texel row in sms-pc-port's
				// front half, not the hardware's 32-byte tile units).
				pica_efb_readback(c->tex, const_cast<void*>(d), dw, dh, (dw + 7) / 8, chan, c->x, c->y, c->w, c->h);
			}
			logCopyKind(c, ctrl, pix);
		}
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
		} else if (clear) {  // a texture copy clears its source rectangle only
			const uint32_t tl = rs.bp[BP_COPY_SRC_TL], wh = rs.bp[BP_COPY_SRC_WH];
			pica_clear_rect(float(tl & 0x3FF), float((tl >> 10) & 0x3FF), float((wh & 0x3FF) + 1),
			                float(((wh >> 10) & 0x3FF) + 1), rgba, rs.bp[BP_CLEAR_Z] & 0xFFFFFF);
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
uint32_t peekColor(int, int) { return 0; }
uint32_t peekZ(int, int) { return 0; }
void pixMetricClear() {}
uint32_t pixMetricRead() { return 0; }
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

// Streamed GX, game thread (gx_api.cpp): a display copy was sent; waits while the GX thread is
// more than a frame behind.
void GXPC_StreamFrameEnd(void) {
	gx::GXPC_StreamFlush();
	const uint32_t sent = ++gx::s_sFramesSent;
	gx::streamWait(gx::s_sWait[1], [sent] { return sent - gx::s_sFramesDone.load(std::memory_order_acquire) <= 1; });
}
// ... a display list was called through the stream (GXCallDisplayList).
void GXPC_StreamNoteDL(const void* list, uint32_t nbytes) { gx::streamNoteDL(list, nbytes); }
// The game is about to write a display list (GD; linked with --wrap=GDInitGDLObj): waits while
// the GX thread has yet to read a call of it.
void __real_GDInitGDLObj(void* dl, void* start, uint32_t length);
void __wrap_GDInitGDLObj(void* dl, void* start, uint32_t length) {
	gx::streamBeforeWrite(start, length);
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
