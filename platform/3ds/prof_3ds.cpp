// Profile run (touch the bottom right of the bottom screen): where a frame's time goes.
//
// Part 1, CPU: for kFrames frames the game thread's time is split by frame phase (decomp patch
// 0009 marks them in TApplication::gameLoop and TMarDirector::direct), and every object's
// perform (JDrama::TViewObj::testPerform) is timed less the objects it performs in turn, then
// summed by class (vtable) and phase. scripts/prof_names.py turns the vtables into class names.
// Part 2, GPU: for kFrames more frames every game draw is scissored to 8x8 pixels
// (pica_set_gpu_test), so the GPU still transforms and sets up every triangle but draws almost
// no pixels. GPU time in part 1 less part 2 is the pixel work.
// Part 3, GPU by object: the objects that sent vertices in part 1 (most vertices first, up to
// kTargets) and the draws made outside any object, one at a time, are left undrawn while
// everything else draws normally (the GX backend batches them apart; SMS3DS_ProfCutNow). The
// GPU time saved is what that object costs the GPU in all: its vertices, triangles and pixels.
// Normal GPU time is measured again every kPerBase objects, so a drifting view skews only the
// objects near it.
// Objects are told to the GX thread in the stream (kStreamOpProfObj, patch 0028): it runs behind
// the game thread, so the object being performed now is not the one whose draws it is parsing.
// Vertices are counted there (SMS3DS_ProfGxVerts) and part 3's cut decided there.
// The GX thread's own time is credited there too (SMS3DS_ProfGxTicks), by object and by where the
// commands came from: the J3D draw code says which of its parts is sending (decomp patch 3ds-0019,
// SMS3DS_GxOrigin, kStreamOpOrigin: native J3D stage 0, docs/native-j3d.md).
// Part 4, the game thread (kGameSteps; also on its own: the bottom right of the bottom screen, low
// down): its time by phase for kFrames frames in each step, with only the phases timed (object
// timing costs the game thread some 10 ms a frame, far more than its estimate). The steps (a table,
// kGameSteps): native J3D off (decomp patch 3ds-0020: the static draw buffers' J3D draws as one op
// each), on through stage 1's path only (GXPC_J3DPackages 0), on with compiled shapes (sms-port
// patch 0037), off again (how far the view drifted), and on with compiled shapes in verify mode
// (the render thread compares its registers before each native draw with what the old path would
// have sent it: correctness, not timing). Each reports the game thread's work by phase and the GX
// and render threads' busy time. Earlier runs' steps, kept as settings in the table:
// objects timed too cost 10.1 ms a frame; inexact maths (sms_math_fast) saved nothing; of the game
// thread's 41.8 ms, the GPU's drawing slowed it by 3.2, the render thread's by 1.2, the GX
// thread's work and the stream's buffers by 5.0, its GX calls cost 3.3; alone, a thread on core 2
// rewriting 64 KB slowed it by 12.1 ms, 1 MB by 19.1-19.4, 8 MB by 23.0 (kHogTiny, kHogSmall:
// other cores' L2 traffic and space cost it far more than memory bandwidth); the whole-linear-heap
// flush at each frame's end (pica_set_heap_flush) 0.6. GPU tests other than 0 leave the EFB
// copies' readbacks out, and the goop map's copies (pica_gpu_test_seen).
// Each part skips kSettle frames first: the GPU runs up to two frames behind the game thread.
// Sampling, in builds made with SAMPLE=1 (scripts/build_game_lib.sh, build_platform_lib.sh,
// build_gx_lib.sh): the game's, the platform layer's and the GX API's functions are compiled with
// -finstrument-functions, and every call on the game thread pushes the function on a shadow stack
// (__cyg_profile_func_enter/exit below). Through part 1 a thread on core 2 reads its top every
// kSampleNs: the function the game thread is in, by phase, without timing each call. Other builds
// never call the hooks and leave it out.
#include <3ds.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
extern uint32_t GX3DS_GameVerts;
// GX command stream (gx3ds_backend.cpp, patch 0017): bytes written so far, slow-path writes
uint64_t GX3DS_StreamWritten(void);
extern uint32_t GXPC_StreamPuts;
extern uint8_t* GXPC_PipeBuf;
extern uint32_t GXPC_PipeLen, GXPC_PipeFast;
void GXPC_Write8(uint8_t v);
void GXPC_Write32(uint32_t v);
double pica_gpu_ms_sum(void);
void pica_set_gpu_test(int mode);
extern int sms_math_fast;          // mtx.cpp: inexact maths (part 4)
void GXPC_SetMute(int on);         // gx_api.cpp: the GX writes dropped (part 4)
void GX3DS_SetDropDraws(int on);   // gx3ds_backend.cpp: the render thread draws nothing (part 4)
void pica_set_heap_flush(int on);  // pica_draw.c: the whole-linear-heap flush at frame end (part 4)
void GXPC_SetNativeJ3DMode(int mode);  // gx_api.cpp: native J3D's mode, its buffers kept (part 4)
int GXPC_NativeJ3DMode(void);
extern int GXPC_J3DPackages;       // gx_fifo.cpp: native J3D's compiled shapes (part 4)
void GX3DS_ThreadTicks(uint64_t out[2]);  // gx3ds_backend.cpp: the GX and render threads' busy ticks
uint32_t GX3DS_J3DSigBad(void);    // ... native J3D's verify mode: differences found so far
extern uint32_t GXPC_J3DStats[16], GXPC_J3DPkgStats[4];  // gx_fifo.cpp: native J3D's counters
int SMS3DS_ProfOn;
void SMS3DS_ProfPhase(int phase);
void SMS3DS_ProfObjBegin(const void* vtable, const char* name);
void SMS3DS_ProfObjEnd(void);
void SMS3DS_ProfFrame(void);
void SMS3DS_ProfRequest(void);
void SMS3DS_ProfRequestGame(void);
int SMS3DS_ProfState(void);
volatile int SMS3DS_ProfCutActive;
int SMS3DS_ProfCutNow(void);
extern int GXPC_Detail;  // patch 0027: the GX thread's detailed counters, for the run
extern uint32_t GXPC_ProfObj;  // patch 0028: the object of the draws the GX thread is parsing
void GXPC_StreamOp(uint8_t op, const void* args, uint32_t n);
void SMS3DS_ProfGxVerts(uint32_t n);
void SMS3DS_ProfGxTicks(uint32_t obj, int origin, uint32_t ticks);
int SMS3DS_GxOrigin(int origin);
void GXPC_StreamOrigin(uint8_t origin);
extern uint32_t GXPC_J3DExits;
// The GameCube's MTX/VEC routines and MSL's maths (patch 0024): calls per routine, and a run of
// one routine on sample data.
extern unsigned sms_mtx_calls[];
extern const char* sms_mtx_names[];
extern int sms_mtx_n;
void sms_mtx_bench(int i, int n);
extern unsigned sms_msl_calls[];
extern const char* const sms_msl_names[];
extern const int sms_msl_n;
void sms_msl_bench(int k, int n);
// The game thread waiting for the GX thread (1 + which wait), set by the GX backend.
extern volatile int SMS3DS_SampWaiting;
void SMS3DS_SampReport(const char* why);
int SMS3DS_ZoneBegin(int zone);  // decomp patch 3ds-0016: timers around a few drawing routines
void SMS3DS_ZoneEnd(int zone, int token);
__attribute__((no_instrument_function)) void __cyg_profile_func_enter(void* fn, void* site);
__attribute__((no_instrument_function)) void __cyg_profile_func_exit(void* fn, void* site);
}

namespace {

const int kFrames = 60, kSettle = 6;
const double kTicksPerMs = SYSCLOCK_ARM11 / 1000.0;

enum { PHASES = 16 };
const char* const kPhaseName[PHASES] = {
    "start rendering", "pads",          "director other",  "hit check",      "movement list",
    "list unk30",      "director move", "calc-anim list",  "last-tick list", "pre-draw lists",
    "GX list",         "silhouette",    "GX post list",    "change state",   "fader, sound",
    "end rendering",
};

// Where the GX thread's commands came from (kStreamOpOrigin): 0 other, then J3D's parts.
enum { kOrigins = 6 };
const char* const kOriginName[kOrigins] = {"other", "J3D material lists", "J3D shape packets", "J3D vertex formats",
                                           "J3D arrays and matrices", "J3D draw lists"};
struct Entry {
	const void* vt;
	const char* name;  // draw buffers (one class, many jobs) are told apart by name
	uint64_t ticks, bytes;
	uint32_t calls, verts, puts;
	int phase;
	uint32_t gx[kOrigins];  // the GX thread's time on its draws, by origin (ticks)
};
const int kTable = 2048;  // power of two
Entry s_table[kTable];
int s_tableUsed;

struct Open {
	Entry* e;
	uint64_t start, child, b0, childBytes;
	uint32_t v0, childVerts, p0, childPuts;
};
Open s_stack[64];
int s_depth;

uint64_t s_phaseTicks[PHASES], s_phaseLast, s_phaseBytes[PHASES], s_phaseB0;
uint32_t s_phasePuts[PHASES], s_phaseP0;
// What one write to the GX stream costs the game thread, measured at the start of the CPU part
// (calibrate): a slow-path call (GXPC_Write8, as the register writers make two or three of) and
// a byte through the inline path (GXPC_Put8, the vertex writers).
double s_putUs, s_byteUs;
// What one call of each maths routine costs on sample data (calibrate), and the call counts at
// the start of the window.
enum { kMathMax = 64 };
double s_mathUs[kMathMax];
uint32_t s_math0[kMathMax];
int s_nMath;
const char* mathName(int i) { return i < sms_mtx_n ? sms_mtx_names[i] : sms_msl_names[i - sms_mtx_n]; }
uint32_t mathCalls(int i) { return i < sms_mtx_n ? sms_mtx_calls[i] : sms_msl_calls[i - sms_mtx_n]; }
uint32_t s_phaseEntries[PHASES];
int s_phase;
uint64_t s_calls;

// Drawing routines timed in part 1 (decomp patch 3ds-0016), nested ones included: calls, ticks,
// GX stream bytes. The open ones are a stack (they nest, on the game thread).
enum { kZones = 9 };
const char* const kZoneName[kZones] = {"ReInitializeGX", "SMS_DrawInit", "J3DSys::drawInit", "TLightCommon::setLight",
                                       "TLightMario::setLight", "J3DDrawBuffer::draw", "JPAEmitterManager::draw",
                                       "TMapObjWave::draw", "ReInitializeGX + SMS_DrawInit"};
uint64_t s_zoneTicks[kZones], s_zoneBytes[kZones];
uint32_t s_zoneCalls[kZones];
struct ZoneOpen {
	uint64_t t0, b0;
};
ZoneOpen s_zoneStack[32];
int s_zoneDepth;

// The GX thread's time credited outside any object, and in all, by origin (ticks); the origin the
// game thread last told it, and the GX thread's J3D exits at the start of the window.
uint64_t s_gxOutside[kOrigins], s_gxAll[kOrigins];
int s_gxOrigin;
uint32_t s_j3dExits0;

// Sampling (see the top). The shadow stack is the game thread's alone: the hooks compare the
// calling thread's TLS pointer with the game thread's, taken at its first frame (gameLoop, which
// is never on the stack itself; what it was called from is left off too).
// Each group on its own cache lines: every thread's hooks read the first, the game thread's write
// the second.
const uint32_t kShadow = 256;
struct alignas(32) {
	void* volatile tls;
	volatile bool hooked;  // a hook ran: a SAMPLE=1 build
} s_game;
struct alignas(32) {
	volatile uint32_t depth;
	uint32_t calls;     // hook pairs on the game thread
	uint32_t deepest;   // the most entries it has held
	void* volatile fn[kShadow];  // a ring: entry d at fn[d % kShadow], so the innermost are kept
} s_shadow;
double s_hookUs;          // what a pair costs (calibrate)

// state: 0 idle, 1 requested, 2 CPU part (settling, then measuring), 3 GPU part (pixels cut), 7 GPU
// part (pixels cut, the picture shrunk 8x: nothing clipped at the screen's sides), 5 GPU part
// (positions collapsed), 6 GPU part (no draws), 8-12 GPU part (GPU tests 5, 6, 0, 7, 8: see
// s_gpuMinShaderMs), 4 GPU by object, 13 the game thread (kGameSteps), 14 the last muted frames'
// copies passing through the GPU
volatile int s_request;  // 1 the whole run, 2 part 4 alone
int s_state, s_frame;
uint64_t s_winStart;
double s_gpu0, s_cpuFrameMs, s_cpuGpuMs, s_gpuCutMs, s_gpuCollapsedMs, s_cutFrame, s_collapsedFrame, s_gpuShrunkMs,
    s_shrunkFrame, s_gpuNoDrawMs, s_noDrawFrame;
// Part 2's further steps (pica_draw.c GPU tests 5-8): pixels cut with a minimal vertex shader, and
// with only positions fetched; drawing normally again, then with textures outside VRAM swapped for
// one small texture, and every texture.
double s_gpuMinShaderMs, s_gpuPosOnlyMs, s_gpuNormal2Ms, s_gpuTexMainMs, s_gpuTexAllMs;

Entry* lookup(const void* vt, int phase, const char* name) {
	// JDrama::TDrawBufObj (every "DrawBuf ..." object) draws a whole buffer of models: one entry
	// per buffer, by name. Other classes: one entry per class.
	const bool byName = name && strncmp(name, "DrawBuf", 7) == 0;
	const char* key = byName ? name : nullptr;
	uint32_t h = (uint32_t(uintptr_t(vt)) >> 2) * 2654435761u + uint32_t(phase) * 40503u + uint32_t(uintptr_t(key));
	for (int n = 0; n < kTable; n++) {
		Entry* e = &s_table[(h + n) & (kTable - 1)];
		if (e->vt == vt && e->phase == phase && (!byName || e->name == key)) return e;
		if (!e->vt) {
			if (s_tableUsed >= kTable * 3 / 4) return nullptr;
			s_tableUsed++;
			e->vt = vt;
			e->phase = phase;
			if (byName) e->name = key;
			return e;
		}
	}
	return nullptr;
}

void resetCpu() {
	memset(s_table, 0, sizeof s_table);
	s_tableUsed = 0;
	memset(s_phaseTicks, 0, sizeof s_phaseTicks);
	memset(s_phaseEntries, 0, sizeof s_phaseEntries);
	memset(s_phaseBytes, 0, sizeof s_phaseBytes);
	memset(s_phasePuts, 0, sizeof s_phasePuts);
	s_phaseB0 = GX3DS_StreamWritten();
	s_phaseP0 = GXPC_StreamPuts;
	s_calls = 0;
	s_depth = 0;
	memset(s_zoneTicks, 0, sizeof s_zoneTicks);
	memset(s_zoneBytes, 0, sizeof s_zoneBytes);
	memset(s_zoneCalls, 0, sizeof s_zoneCalls);
	s_zoneDepth = 0;
	memset(s_gxOutside, 0, sizeof s_gxOutside);
	memset(s_gxAll, 0, sizeof s_gxAll);
	s_gxOrigin = 0;
	GXPC_StreamOrigin(0);
	s_j3dExits0 = GXPC_J3DExits;
	for (int i = 0; i < s_nMath; i++) s_math0[i] = mathCalls(i);
}

// Tells the GX thread the object whose draws follow (kStreamOpProfObj, patch 0028).
void markObj(const Entry* e) {
	const uint32_t v = uint32_t(uintptr_t(e));
	const uint8_t a[4] = {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
	GXPC_StreamOp(0xCB, a, 4);
}

// Writes 4096 NOPs (GX opcode 0x00) to the stream each way and times them. Called with the stream
// at a command boundary (the top of a frame): the GX thread skips NOPs.
void calibrate() {
	const int n = 4096;
	uint64_t t0 = svcGetSystemTick();
	for (int i = 0; i < n; i++) GXPC_Write8(0);
	const uint64_t t1 = svcGetSystemTick();
	for (int i = 0; i < n; i++) {
		if (GXPC_PipeLen + 1 <= GXPC_PipeFast) GXPC_PipeBuf[GXPC_PipeLen++] = 0;
		else GXPC_Write8(0);
	}
	const uint64_t t2 = svcGetSystemTick();
	s_putUs = double(t1 - t0) / n / kTicksPerMs * 1000.0;
	s_byteUs = double(t2 - t1) / n / kTicksPerMs * 1000.0;
	// Each maths routine, 1000 runs (the call counts these add are before the window).
	s_nMath = sms_mtx_n + sms_msl_n < kMathMax ? sms_mtx_n + sms_msl_n : kMathMax;
	for (int i = 0; i < s_nMath; i++) {
		s_mathUs[i] = 0;
		if (!mathName(i)) continue;
		const uint64_t a = svcGetSystemTick();
		if (i < sms_mtx_n) sms_mtx_bench(i, 1000);
		else sms_msl_bench(i - sms_mtx_n, 1000);
		s_mathUs[i] = double(svcGetSystemTick() - a) / 1000.0 / kTicksPerMs * 1000.0;
	}
	// A pair of the sampling hooks on the game thread (SAMPLE=1 builds).
	if (s_game.hooked) {
		const uint64_t a = svcGetSystemTick();
		for (int i = 0; i < 1000; i++) {
			__cyg_profile_func_enter((void*)&calibrate, nullptr);
			__cyg_profile_func_exit((void*)&calibrate, nullptr);
		}
		s_hookUs = double(svcGetSystemTick() - a) / 1000.0 / kTicksPerMs * 1000.0;
	}
}

// Maths routines by estimated time: calls in the window times the cost of a call on sample data.
void reportMath(double frames) {
	int order[kMathMax], n = 0;
	double ms[kMathMax], calls[kMathMax], all = 0, allCalls = 0;
	for (int i = 0; i < s_nMath; i++) {
		if (!mathName(i)) continue;
		calls[i] = double(uint32_t(mathCalls(i) - s_math0[i])) / frames;
		ms[i] = calls[i] * s_mathUs[i] / 1000.0;
		all += ms[i];
		allCalls += calls[i];
		order[n++] = i;
	}
	for (int a = 1; a < n; a++)  // by ms, largest first
		for (int b = a; b > 0 && ms[order[b]] > ms[order[b - 1]]; b--) {
			const int t = order[b];
			order[b] = order[b - 1];
			order[b - 1] = t;
		}
	fprintf(stderr,
	        "[prof]   maths routines (MTX/VEC and MSL, exact Gekko arithmetic): %.0f calls/frame, about %.2f "
	        "ms/frame (calls x the cost of one call on sample data):\n",
	        allCalls, all);
	char line[512];
	int at = 0, shown = 0;
	for (int k = 0; k < n; k++) {
		const int i = order[k];
		if (calls[i] < 0.5) continue;
		at += snprintf(line + at, sizeof line - at, "%s %s %.0f x %.2f us = %.2f ms", at ? " |" : "", mathName(i),
		               calls[i], s_mathUs[i], ms[i]);
		if (++shown % 4 == 0) {
			fprintf(stderr, "[prof]   %s\n", line);
			at = 0;
		}
	}
	if (at) fprintf(stderr, "[prof]   %s\n", line);
}

// What the profiler's own two tick reads cost per object, measured: subtracted from the totals.
double tickCostMs() {
	const uint64_t t0 = svcGetSystemTick();
	for (int i = 0; i < 1000; i++) (void)svcGetSystemTick();
	return double(svcGetSystemTick() - t0) / 1000.0 / kTicksPerMs;
}

void reportCpu(double frames, double frameMs) {
	const double per = 1.0 / frames / kTicksPerMs;
	double phaseSum = 0;
	for (int i = 0; i < PHASES; i++) phaseSum += s_phaseTicks[i] * per;
	const double ticks = s_phaseEntries[3] / frames;
	const double overheadMs = 2.0 * tickCostMs() * double(s_calls) / frames;
	fprintf(stderr,
	        "[prof] CPU, %.0f frames: %.1f ms/frame (%.1f fps); game thread in marked phases %.1f ms/frame, "
	        "%.1f simulation ticks/frame, %.0f object performs/frame (the profiler's own timing adds about %.1f ms/frame)\n",
	        frames, frameMs, 1000.0 / frameMs, phaseSum, ticks, double(s_calls) / frames, overheadMs);
	char line[512];
	int at = 0;
	for (int i = 0; i < PHASES; i++) {
		const double ms = s_phaseTicks[i] * per;
		if (ms < 0.05) continue;
		at += snprintf(line + at, sizeof line - at, "%s %s %.1f", at ? " |" : "", kPhaseName[i], ms);
		if (at > 400) {
			fprintf(stderr, "[prof]   phases (ms/frame):%s\n", line);
			at = 0;
		}
	}
	if (at) fprintf(stderr, "[prof]   phases (ms/frame):%s\n", line);
	// GX stream writes by phase, and what they cost at the calibrated rates.
	fprintf(stderr, "[prof]   GX stream: a slow-path write %.3f us, an inline byte %.4f us (measured now)\n", s_putUs,
	        s_byteUs);
	at = 0;
	for (int i = 0; i < PHASES; i++) {
		const double kb = s_phaseBytes[i] / frames / 1024.0, puts = s_phasePuts[i] / frames;
		if (kb < 0.5 && puts < 50) continue;
		at += snprintf(line + at, sizeof line - at, "%s %s %.0f KB %.0f slow (%.1f ms)", at ? " |" : "", kPhaseName[i],
		               kb, puts, (puts * s_putUs + kb * 1024.0 * s_byteUs) / 1000.0);
		if (at > 400) {
			fprintf(stderr, "[prof]   GX stream writes by phase, per frame:%s\n", line);
			at = 0;
		}
	}
	if (at) fprintf(stderr, "[prof]   GX stream writes by phase, per frame:%s\n", line);
	// Top classes by exclusive time.
	static Entry* order[kTable];
	int n = 0;
	for (int i = 0; i < kTable; i++)
		if (s_table[i].vt) order[n++] = &s_table[i];
	qsort(order, n, sizeof order[0], [](const void* a, const void* b) {
		const uint64_t x = (*(Entry* const*)a)->ticks, y = (*(Entry* const*)b)->ticks;
		return x < y ? 1 : x > y ? -1 : 0;
	});
	double listed = 0, all = 0;
	for (int i = 0; i < n; i++) all += order[i]->ticks * per;
	fprintf(stderr,
	        "[prof]   objects (exclusive ms/frame, performs/frame, vertices sent/frame, GX stream KB and slow-path "
	        "writes/frame and their estimated ms, phase, vtable, a name); %.1f ms/frame in all %d\n",
	        all, n);
	for (int i = 0; i < n && i < 40; i++) {
		const Entry* e = order[i];
		const double ms = e->ticks * per;
		if (ms < 0.1) break;
		listed += ms;
		char name[40];
		snprintf(name, sizeof name, "%s", e->name ? e->name : "");
		for (char* c = name; *c; c++)
			if ((unsigned char)*c < 0x20 || (unsigned char)*c >= 0x7f) *c = '?';  // Shift-JIS names
		const double kb = e->bytes / frames / 1024.0, puts = e->puts / frames;
		fprintf(stderr, "[prof]   %6.2f ms %6.1f %7.0fv %5.1f KB %6.0f slow (%4.2f ms) %-15s vt %08lx \"%s\"\n", ms,
		        e->calls / frames, e->verts / frames, kb, puts, (puts * s_putUs + kb * 1024.0 * s_byteUs) / 1000.0,
		        kPhaseName[e->phase], (unsigned long)uintptr_t(e->vt), name);
	}
	fprintf(stderr, "[prof]   (listed %.1f of %.1f ms/frame)\n", listed, all);
	// The same entries by the vertices they sent: what the GPU's vertex work and the renderer's
	// per-vertex costs come from.
	qsort(order, n, sizeof order[0], [](const void* a, const void* b) {
		const uint32_t x = (*(Entry* const*)a)->verts, y = (*(Entry* const*)b)->verts;
		return x < y ? 1 : x > y ? -1 : 0;
	});
	double allV = 0;
	for (int i = 0; i < n; i++) allV += order[i]->verts / frames;
	fprintf(stderr, "[prof]   by vertices sent (%.0f/frame in all):\n", allV);
	for (int i = 0; i < n && i < 20 && order[i]->verts; i++) {
		const Entry* e = order[i];
		char name[40];
		snprintf(name, sizeof name, "%s", e->name ? e->name : "");
		for (char* c = name; *c; c++)
			if ((unsigned char)*c < 0x20 || (unsigned char)*c >= 0x7f) *c = '?';
		fprintf(stderr, "[prof]   %7.0fv %6.2f ms %-15s vt %08lx \"%s\"\n", e->verts / frames, e->ticks * per,
		        kPhaseName[e->phase], (unsigned long)uintptr_t(e->vt), name);
	}
	// The GX thread's time by origin, and the objects that cost it most (native J3D stage 0).
	{
		double all = 0, j3d = 0;
		int at = 0;
		for (int o = 0; o < kOrigins; o++) {
			const double ms = s_gxAll[o] * per;
			all += ms;
			if (o) j3d += ms;
			at += snprintf(line + at, sizeof line - at, "%s %s %.2f", at ? " |" : "", kOriginName[o], ms);
		}
		fprintf(stderr,
		        "[prof]   GX thread by where its commands came from (ms/frame, its waits left out):%s | J3D %.2f of "
		        "%.2f | J3D followed by a command from elsewhere %.1f times/frame\n",
		        line, j3d, all, (GXPC_J3DExits - s_j3dExits0) / frames);
		const auto gxSum = [](const Entry* e) {
			uint64_t t = 0;
			for (int o = 0; o < kOrigins; o++) t += e->gx[o];
			return t;
		};
		qsort(order, n, sizeof order[0], [](const void* a, const void* b) {
			uint64_t x = 0, y = 0;
			for (int o = 0; o < kOrigins; o++) x += (*(Entry* const*)a)->gx[o], y += (*(Entry* const*)b)->gx[o];
			return x < y ? 1 : x > y ? -1 : 0;
		});
		fprintf(stderr,
		        "[prof]   by GX-thread time (ms/frame: in all = J3D material lists + shape packets + vertex formats + "
		        "arrays and matrices + draw lists + other; phase, vtable, a name):\n");
		double outside = 0;
		for (int o = 0; o < kOrigins; o++) outside += s_gxOutside[o] * per;
		fprintf(stderr, "[prof]   %6.2f = %5.2f + %5.2f + %5.2f + %5.2f + %5.2f + %5.2f  (outside any object)\n", outside,
		        s_gxOutside[1] * per, s_gxOutside[2] * per, s_gxOutside[3] * per, s_gxOutside[4] * per,
		        s_gxOutside[5] * per, s_gxOutside[0] * per);
		for (int i = 0; i < n && i < 30; i++) {
			const Entry* e = order[i];
			const double ms = gxSum(e) * per;
			if (ms < 0.05) break;
			char name[40];
			snprintf(name, sizeof name, "%s", e->name ? e->name : "");
			for (char* c = name; *c; c++)
				if ((unsigned char)*c < 0x20 || (unsigned char)*c >= 0x7f) *c = '?';
			fprintf(stderr, "[prof]   %6.2f = %5.2f + %5.2f + %5.2f + %5.2f + %5.2f + %5.2f  %-15s vt %08lx \"%s\"\n", ms,
			        e->gx[1] * per, e->gx[2] * per, e->gx[3] * per, e->gx[4] * per, e->gx[5] * per, e->gx[0] * per,
			        kPhaseName[e->phase], (unsigned long)uintptr_t(e->vt), name);
		}
	}
	// The timed drawing routines (each figure includes the routines it calls, timed or not).
	{
		const double tick = tickCostMs();
		int at = 0;
		for (int z = 0; z < kZones; z++) {
			if (!s_zoneCalls[z]) continue;
			const double calls = s_zoneCalls[z] / frames;
			const double ms = s_zoneTicks[z] * per - calls * tick;  // less one tick read a call
			at += snprintf(line + at, sizeof line - at, "%s %s %.1f x %.3f = %.2f ms, %.1f KB", at ? " |" : "", kZoneName[z],
			               calls, calls > 0 ? ms / calls : 0.0, ms, s_zoneBytes[z] / frames / 1024.0);
			if (at > 380) {
				fprintf(stderr, "[prof]   drawing routines (calls/frame x ms each = ms/frame, GX stream KB/frame):%s\n", line);
				at = 0;
			}
		}
		if (at) fprintf(stderr, "[prof]   drawing routines (calls/frame x ms each = ms/frame, GX stream KB/frame):%s\n", line);
	}
	reportMath(frames);
}

// Part 3: steps alternate a normal-drawing measurement with kPerBase targets; nullptr in
// s_targets is "outside any object". Each target is printed as soon as it is measured, against
// the normal GPU time measured just before it (the view may drift over the part).
const int kTargets = 20, kPerBase = 4, kCutSettle = 3, kCutFrames = 8;
Entry* s_targets[kTargets + 1];
double s_targetMs[kTargets + 1], s_targetBase[kTargets + 1];
uint32_t s_targetVerts[kTargets + 1];  // from part 1 (the entries keep counting in part 3)
int s_nTargets, s_step, s_next;  // s_next: the next target; s_step -1 measures normal drawing
Entry* s_cutEntry;
bool s_cutOutside;
double s_base, s_cutFrameMs;

// Part 4's memory hog: a thread on core 2 (idle in the steps that use it) that writes one word of
// every 32-byte cache line of its buffer, over and over, until stopped.
const uint32_t kHogTiny = 64u << 10, kHogSmall = 1u << 20;
struct {
	Thread th;
	uint32_t* buf;
	uint32_t words;
	volatile bool run;
	volatile uint32_t passes;
	uint64_t t0;
} s_hog;
void hogMain(void*) {
	uint32_t* const b = s_hog.buf;
	const uint32_t n = s_hog.words;
	while (s_hog.run) {
		for (uint32_t i = 0; i < n; i += 8) b[i]++;
		s_hog.passes++;
	}
}
void hogStop() {
	if (!s_hog.th) return;
	s_hog.run = false;
	threadJoin(s_hog.th, U64_MAX);
	threadFree(s_hog.th);
	s_hog.th = nullptr;
	free(s_hog.buf);
	s_hog.buf = nullptr;
}
// Starts the hog on bytes of memory; false when there is no memory or no thread for it.
bool hogStart(uint32_t bytes) {
	hogStop();
	s_hog.buf = static_cast<uint32_t*>(memalign(32, bytes));
	if (!s_hog.buf) return false;
	memset(s_hog.buf, 0, bytes);
	s_hog.words = bytes / 4;
	s_hog.run = true;
	s_hog.passes = 0;
	s_hog.t0 = svcGetSystemTick();
	s_hog.th = threadCreate(hogMain, nullptr, 4u << 10, 0x3F, 2, false);
	if (!s_hog.th) {
		free(s_hog.buf);
		s_hog.buf = nullptr;
		return false;
	}
	return true;
}
// MB/s of its buffer the hog has gone through since it started (each pass reads and writes every line).
double hogRate() {
	if (!s_hog.th) return 0;
	const double ms = double(svcGetSystemTick() - s_hog.t0) / kTicksPerMs;
	return ms > 0 ? double(s_hog.passes) * (s_hog.words * 4.0) / (1 << 20) / (ms / 1000.0) : 0;
}

// Part 4's steps: SMS3DS_ProfOn (1 objects timed too, 2 phases only), sms_math_fast, the GPU test,
// the GX mute level (GXPC_SetMute), the render thread's draws dropped, the whole-heap flush, the
// hog's buffer (0: none), native J3D (-1 as the settings set it, 0 off, 1 on), compiled shapes.
struct GameStep {
	const char* name;
	int profOn, math, gpuTest, mute, dropDraws, heapFlush;
	uint32_t hog;
	int native, packages;
};
enum { kGsOff, kGsStage1, kGsCompiled, kGsOffAgain, kGsVerify, kGameStepN };
const GameStep kGameSteps[kGameStepN] = {
    {"native J3D off", 2, 0, 0, 0, 0, 1, 0, 0, 1},
    {"native J3D on, stage 1's path", 2, 0, 0, 0, 0, 1, 0, 1, 0},
    {"native J3D on, compiled shapes", 2, 0, 0, 0, 0, 1, 0, 1, 1},
    {"native J3D off again", 2, 0, 0, 0, 0, 1, 0, 0, 1},
    {"native J3D on, compiled shapes, verified (not for timing)", 2, 0, 0, 0, 0, 1, 0, 3, 1},
};
// The verify step's counters at its start: native draws verified, shape ops drawn from compiled
// shapes, differences.
uint32_t s_verify0[3];
// Native J3D's settings before part 4 (put back after it).
int s_nativeSaved = -1, s_packagesSaved = 1;
// Each step's phases (ms/frame), frame and GPU time, and the hog's rate (MB/s, 0: none or not
// started); the step being measured.
double s_gamePh[kGameStepN][PHASES], s_gameFrame[kGameStepN], s_gameGpu[kGameStepN], s_gameHog[kGameStepN];
double s_gameGx[kGameStepN], s_gameRender[kGameStepN];  // the GX and render threads' busy ms/frame
uint64_t s_thread0[2];
int s_gameStep;
// Puts step k's settings in place; past the last, normal play's (the GPU test stays until the
// tail of part 4 ends).
void setGameStep(int k) {
	const GameStep* g = k < kGameStepN ? &kGameSteps[k] : nullptr;
	sms_math_fast = g ? g->math : 0;
	GXPC_SetMute(g ? g->mute : 0);
	GX3DS_SetDropDraws(g ? g->dropDraws : 0);
	pica_set_heap_flush(g ? g->heapFlush : 1);
	GXPC_SetNativeJ3DMode(g && g->native >= 0 ? g->native : s_nativeSaved);
	GXPC_J3DPackages = g ? g->packages : s_packagesSaved;
	if (g) pica_set_gpu_test(g->gpuTest);
	if (g && g->hog) {
		if (!hogStart(g->hog)) fprintf(stderr, "[prof] game thread: no memory or thread for a %lu KB hog\n", (unsigned long)(g->hog >> 10));
	} else {
		hogStop();
	}
}
// Part 4: a step's phases (end rendering apart: it holds the frame's waits for the GX thread and
// the screen), and at the end what each step saved, by phase.
double gameWork(int k) {
	double sum = 0;
	for (int i = 0; i < PHASES; i++)
		if (i != 15) sum += s_gamePh[k][i];
	return sum;
}
void reportGameStep(int k) {
	char line[768];
	int at = 0;
	for (int i = 0; i < PHASES; i++)
		if (i != 15 && s_gamePh[k][i] >= 0.05)
			at += snprintf(line + at, sizeof line - at, "%s %s %.1f", at ? " |" : "", kPhaseName[i], s_gamePh[k][i]);
	char hog[48] = "";
	if (kGameSteps[k].hog) snprintf(hog, sizeof hog, "; core 2 went through %.0f MB/s", s_gameHog[k]);
	fprintf(stderr, "[prof] game thread, %s: frame %.1f ms, GPU %.1f, GX thread %.1f, render thread %.1f%s; its work %.1f "
	        "ms/frame (end rendering, waits included, %.1f more):%s\n", kGameSteps[k].name, s_gameFrame[k], s_gameGpu[k],
	        s_gameGx[k], s_gameRender[k], hog, gameWork(k), s_gamePh[k][15], line);
}
void reportGameSaved() {
	for (int k = 1; k < kGameStepN; k++) {
		char line[768];
		int at = 0;
		for (int i = 0; i < PHASES; i++) {
			const double d = s_gamePh[0][i] - s_gamePh[k][i];
			if (i != 15 && (d >= 0.1 || d <= -0.1))
				at += snprintf(line + at, sizeof line - at, "%s %s %.1f", at ? " |" : "", kPhaseName[i], d);
		}
		fprintf(stderr, "[prof] game thread, %s saves %.1f ms/frame of its work (frame %.1f -> %.1f ms) by phase:%s\n",
		        kGameSteps[k].name, gameWork(0) - gameWork(k), s_gameFrame[0], s_gameFrame[k], at ? line : " none");
	}
	fprintf(stderr,
	        "[prof] native J3D, ms/frame: game thread's work %.1f off, %.1f stage 1, %.1f compiled shapes (%.1f off "
	        "again) | GX thread %.1f, %.1f, %.1f (%.1f) | render thread %.1f, %.1f, %.1f (%.1f) | frame %.1f, %.1f, "
	        "%.1f (%.1f) | GPU %.1f, %.1f, %.1f (%.1f)\n",
	        gameWork(kGsOff), gameWork(kGsStage1), gameWork(kGsCompiled), gameWork(kGsOffAgain), s_gameGx[kGsOff],
	        s_gameGx[kGsStage1], s_gameGx[kGsCompiled], s_gameGx[kGsOffAgain], s_gameRender[kGsOff],
	        s_gameRender[kGsStage1], s_gameRender[kGsCompiled], s_gameRender[kGsOffAgain], s_gameFrame[kGsOff],
	        s_gameFrame[kGsStage1], s_gameFrame[kGsCompiled], s_gameFrame[kGsOffAgain], s_gameGpu[kGsOff],
	        s_gameGpu[kGsStage1], s_gameGpu[kGsCompiled], s_gameGpu[kGsOffAgain]);
}
void startGame() {
	GXPC_Detail = 0;  // as in normal play
	s_nativeSaved = GXPC_NativeJ3DMode();
	s_packagesSaved = GXPC_J3DPackages;
	s_gameStep = 0;
	setGameStep(0);
	s_state = 13;
	s_frame = 0;
}

void pickTargets() {
	static Entry* order[kTable];
	int n = 0;
	for (int i = 0; i < kTable; i++)
		if (s_table[i].vt && s_table[i].verts) order[n++] = &s_table[i];
	qsort(order, n, sizeof order[0], [](const void* a, const void* b) {
		const uint32_t x = (*(Entry* const*)a)->verts, y = (*(Entry* const*)b)->verts;
		return x < y ? 1 : x > y ? -1 : 0;
	});
	s_nTargets = 0;
	s_targetVerts[s_nTargets] = 0;
	s_targets[s_nTargets++] = nullptr;
	for (int i = 0; i < n && s_nTargets < kTargets + 1; i++) {
		s_targetVerts[s_nTargets] = order[i]->verts;
		s_targets[s_nTargets++] = order[i];
	}
}

void setCut(int t) {
	s_cutEntry = t >= 0 ? s_targets[t] : nullptr;
	s_cutOutside = t >= 0 && !s_targets[t];
	SMS3DS_ProfCutActive = t >= 0;
}

void targetName(int t, char* name, size_t n) {
	const Entry* e = s_targets[t];
	snprintf(name, n, "%s", e ? (e->name ? e->name : "") : "(draws outside any object)");
	for (char* c = name; *c; c++)
		if ((unsigned char)*c < 0x20 || (unsigned char)*c >= 0x7f) *c = '?';
}

void printTarget(const char* tag, int t) {
	const Entry* e = s_targets[t];
	char name[40];
	targetName(t, name, sizeof name);
	fprintf(stderr, "[prof] %s %6.2f ms saved (%.1f of %.1f) %7.0fv %-15s vt %08lx \"%s\"\n", tag,
	        s_targetBase[t] - s_targetMs[t], s_targetMs[t], s_targetBase[t], s_targetVerts[t] / double(kFrames),
	        e ? kPhaseName[e->phase] : "", (unsigned long)uintptr_t(e ? e->vt : nullptr), name);
}

void reportCut() {
	fprintf(stderr, "[prof] GPU by object, done; most GPU ms/frame saved first (frame %.1f ms drawing normally):\n",
	        s_cutFrameMs);
	int order[kTargets + 1];
	for (int i = 0; i < s_nTargets; i++) order[i] = i;
	for (int i = 1; i < s_nTargets; i++)
		for (int j = i; j > 0; j--) {
			const double a = s_targetBase[order[j]] - s_targetMs[order[j]];
			const double b = s_targetBase[order[j - 1]] - s_targetMs[order[j - 1]];
			if (a <= b) break;
			const int t = order[j];
			order[j] = order[j - 1];
			order[j - 1] = t;
		}
	for (int i = 0; i < s_nTargets; i++) printTarget(" ", order[i]);
}


const uint64_t kSampleNs = 100000;
struct Samp {
	const void* fn;      // the innermost instrumented function; in s_pairTab, with the one that called it
	const void* caller;
	uint32_t self, incl;  // samples with it innermost, and with it anywhere on the stack
	int phase;
};
const int kSampTab = 4096;  // power of two
Samp s_fnTab[kSampTab], s_pairTab[kSampTab];
uint32_t s_sampN, s_sampLost, s_sampPhase[PHASES], s_sampOut[PHASES], s_sampWait[PHASES][4], s_hookCalls0;
uint32_t s_topDepthMin, s_topDepthMax;  // the shadow stack's depth at the frame's top: 0 when balanced
uint64_t s_sampT0, s_sampT1;
volatile bool s_sampling;
Thread s_sampThread;

Samp* sampFind(Samp* t, const void* fn, const void* caller, int phase) {
	const uint32_t h = (uint32_t(uintptr_t(fn)) >> 2) * 2654435761u ^ (uint32_t(uintptr_t(caller)) >> 2) * 40503u ^
	                   uint32_t(phase) * 0x9E3779B1u;
	for (int n = 0; n < 64; n++) {
		Samp* e = &t[(h + n) & (kSampTab - 1)];
		if (!e->fn) {
			e->fn = fn;
			e->caller = caller;
			e->phase = phase;
			return e;
		}
		if (e->fn == fn && e->caller == caller && e->phase == phase) return e;
	}
	s_sampLost++;
	return nullptr;
}

// Sampler thread: one look at the game thread.
void sampleOnce() {
	const int ph = s_phase;
	if (ph < 0 || ph >= PHASES) return;
	s_sampN++;
	s_sampPhase[ph]++;
	const int w = SMS3DS_SampWaiting;
	if (w > 0 && w <= 4) {
		s_sampWait[ph][w - 1]++;
		return;
	}
	const uint32_t depth = s_shadow.depth;
	const uint32_t d = depth > kShadow ? kShadow : depth;
	if (!d) {
		s_sampOut[ph]++;
		return;
	}
	const void* st[kShadow];  // outermost first
	for (uint32_t i = 0; i < d; i++) st[i] = s_shadow.fn[(depth - d + i) % kShadow];
	if (Samp* e = sampFind(s_fnTab, st[d - 1], nullptr, ph)) e->self++;
	if (ph == 10 || ph == 12)
		if (Samp* e = sampFind(s_pairTab, st[d - 1], d > 1 ? st[d - 2] : nullptr, ph)) e->self++;
	for (uint32_t i = 0; i < d; i++) {
		bool seen = false;
		for (uint32_t j = 0; j < i && !seen; j++) seen = st[j] == st[i];
		if (!seen)
			if (Samp* e = sampFind(s_fnTab, st[i], nullptr, ph)) e->incl++;
	}
}

void samplerMain(void*) {
	while (s_sampling) {
		svcSleepThread(kSampleNs);
		if (s_sampling) sampleOnce();
	}
}

// Game thread, at the measuring window's start and end.
void sampStart() {
	if (!s_game.hooked) return;
	memset(s_fnTab, 0, sizeof s_fnTab);
	memset(s_pairTab, 0, sizeof s_pairTab);
	memset(s_sampPhase, 0, sizeof s_sampPhase);
	memset(s_sampOut, 0, sizeof s_sampOut);
	memset(s_sampWait, 0, sizeof s_sampWait);
	s_sampN = s_sampLost = 0;
	s_topDepthMin = ~0u;
	s_topDepthMax = 0;
	s_hookCalls0 = s_shadow.calls;
	s_sampT0 = svcGetSystemTick();
	s_sampling = true;
	// Above the render thread on core 2 (the New 3DS's), else on core 1 above the GX thread.
	s_sampThread = threadCreate(samplerMain, nullptr, 16u << 10, 0x18, 2, false);
	if (!s_sampThread) s_sampThread = threadCreate(samplerMain, nullptr, 16u << 10, 0x18, 1, false);
	if (!s_sampThread) s_sampling = false;
}
void sampStop() {
	if (!s_sampThread) return;
	s_sampling = false;
	threadJoin(s_sampThread, U64_MAX);
	threadFree(s_sampThread);
	s_sampThread = nullptr;
	s_sampT1 = svcGetSystemTick();
}

struct Agg {
	const void* fn;
	const void* caller;
	double ms;
};
Agg s_agg[kSampTab];
// Sums s_agg's entries of the same function (and caller), largest first; returns the count.
int aggMerge(int n) {
	qsort(s_agg, n, sizeof s_agg[0], [](const void* a, const void* b) {
		const Agg *x = (const Agg*)a, *y = (const Agg*)b;
		if (x->fn != y->fn) return x->fn < y->fn ? -1 : 1;
		return x->caller < y->caller ? -1 : x->caller > y->caller ? 1 : 0;
	});
	int m = 0;
	for (int i = 0; i < n; i++) {
		if (m && s_agg[m - 1].fn == s_agg[i].fn && s_agg[m - 1].caller == s_agg[i].caller) s_agg[m - 1].ms += s_agg[i].ms;
		else s_agg[m++] = s_agg[i];
	}
	qsort(s_agg, m, sizeof s_agg[0], [](const void* a, const void* b) {
		const double x = ((const Agg*)a)->ms, y = ((const Agg*)b)->ms;
		return x < y ? 1 : x > y ? -1 : 0;
	});
	return m;
}
// Fills s_agg from a table: phases in mask, self or inclusive samples, in ms/frame.
int aggFrom(const Samp* t, uint32_t mask, bool incl, const double* msPer) {
	int n = 0;
	for (int i = 0; i < kSampTab; i++) {
		const Samp& e = t[i];
		if (!e.fn || !(mask & (1u << e.phase))) continue;
		const uint32_t c = incl ? e.incl : e.self;
		if (c) s_agg[n++] = {e.fn, e.caller, c * msPer[e.phase]};
	}
	return aggMerge(n);
}
void printAgg(int n, int max, double total) {
	for (int i = 0; i < n && i < max; i++) {
		if (s_agg[i].ms < 0.03) break;
		if (s_agg[i].caller)
			fprintf(stderr, "[prof]   %6.2f ms %4.1f%% fn %08lx <- fn %08lx\n", s_agg[i].ms, 100.0 * s_agg[i].ms / total,
			        (unsigned long)uintptr_t(s_agg[i].fn), (unsigned long)uintptr_t(s_agg[i].caller));
		else
			fprintf(stderr, "[prof]   %6.2f ms %4.1f%% fn %08lx\n", s_agg[i].ms, 100.0 * s_agg[i].ms / total,
			        (unsigned long)uintptr_t(s_agg[i].fn));
	}
}

void reportSamples(double frames) {
	if (!s_game.hooked) return;
	if (!s_sampN) {
		fprintf(stderr, "[prof] sampling: no samples (the sampler thread did not start)\n");
		return;
	}
	const double secs = double(s_sampT1 - s_sampT0) / SYSCLOCK_ARM11;
	const double hooks = double(s_shadow.calls - s_hookCalls0) / frames;
	fprintf(stderr,
	        "[prof] game thread sampled %u times (%.1f a ms; a sample is the innermost instrumented function, "
	        "maths routines included; %u lost to a full table); the hooks: %.0f calls/frame costing about %.1f "
	        "ms/frame, spread over the functions below; shadow stack at the frame's top %u..%u (0 when balanced)\n",
	        s_sampN, s_sampN / (secs * 1000.0), s_sampLost, hooks, hooks * s_hookUs / 1000.0, s_topDepthMin,
	        s_topDepthMax);
	double msPer[PHASES], all = 0;
	for (int p = 0; p < PHASES; p++) {
		const double ms = s_phaseTicks[p] / frames / kTicksPerMs;
		msPer[p] = s_sampPhase[p] ? ms / s_sampPhase[p] : 0;
		all += ms;
	}
	const int kDraw[2] = {10, 12};
	for (int k = 0; k < 2; k++) {
		const int p = kDraw[k];
		const double ms = s_sampPhase[p] * msPer[p];
		const uint32_t* w = s_sampWait[p];
		fprintf(stderr,
		        "[prof]   %s, %.1f ms/frame (%u samples): waiting on the GX thread %.2f ms (a free chunk %.2f, a list "
		        "rewrite %.2f, other %.2f), outside any instrumented function %.2f ms; by function:\n",
		        kPhaseName[p], ms, s_sampPhase[p], (w[0] + w[1] + w[2] + w[3]) * msPer[p], w[0] * msPer[p],
		        w[3] * msPer[p], (w[1] + w[2]) * msPer[p], s_sampOut[p] * msPer[p]);
		printAgg(aggFrom(s_fnTab, 1u << p, false, msPer), 30, ms);
	}
	const uint32_t draw = (1u << 10) | (1u << 12);
	const double drawMs = (s_sampPhase[10] * msPer[10] + s_sampPhase[12] * msPer[12]);
	fprintf(stderr, "[prof]   GX list and GX post together, %.1f ms/frame, by function and all it calls:\n", drawMs);
	printAgg(aggFrom(s_fnTab, draw, true, msPer), 40, drawMs);
	fprintf(stderr, "[prof]   GX list and GX post together, by function and its caller:\n");
	printAgg(aggFrom(s_pairTab, draw, false, msPer), 40, drawMs);
	fprintf(stderr, "[prof]   all phases, %.1f ms/frame, by function:\n", all);
	printAgg(aggFrom(s_fnTab, 0xFFFFu, false, msPer), 50, all);
}

}  // namespace

// The -finstrument-functions hooks (SAMPLE=1 builds): every instrumented function's entry and
// exit, on every thread.
void __cyg_profile_func_enter(void* fn, void*) {
	if (getThreadLocalStorage() != s_game.tls) {
		if (!s_game.hooked) s_game.hooked = true;
		return;
	}
	const uint32_t d = s_shadow.depth;
	s_shadow.fn[d % kShadow] = fn;
	s_shadow.depth = d + 1;
	s_shadow.calls++;
	if (d >= s_shadow.deepest) s_shadow.deepest = d + 1;
}
void __cyg_profile_func_exit(void*, void*) {
	if (getThreadLocalStorage() != s_game.tls) return;
	const uint32_t d = s_shadow.depth;
	if (d) s_shadow.depth = d - 1;  // (a function entered before the game thread was known)
}
volatile int SMS3DS_SampWaiting;

// SAMPLE=1 builds: where the game thread is, from the shadow stack (for a stall report, from
// another thread: the entries can change as they are read).
void SMS3DS_SampReport(const char* why) {
	if (!s_game.hooked) return;
	const uint32_t depth = s_shadow.depth, n = depth < 32 ? depth : 32;
	fprintf(stderr, "[sample] %s: the game thread's call stack is %lu deep (at most %lu so far), innermost first:\n", why,
	        (unsigned long)depth, (unsigned long)s_shadow.deepest);
	for (uint32_t i = 0; i < n; i++)
		fprintf(stderr, "[sample]   fn %08lx\n", (unsigned long)uintptr_t(s_shadow.fn[(depth - 1 - i) % kShadow]));
}

// Called by the GX backend (GX thread) before it adds vertices, while SMS3DS_ProfCutActive:
// non-zero when the draws' object (the stream's last kStreamOpProfObj) is part 3's target.
int SMS3DS_ProfCutNow(void) {
	const Entry* cur = reinterpret_cast<const Entry*>(uintptr_t(GXPC_ProfObj));
	return s_cutOutside ? cur == nullptr : (s_cutEntry && cur == s_cutEntry);
}

// GX thread: n vertices added to a batch, credited to the draws' object while profiling.
void SMS3DS_ProfGxVerts(uint32_t n) {
	if (SMS3DS_ProfOn != 1) return;
	if (Entry* e = reinterpret_cast<Entry*>(uintptr_t(GXPC_ProfObj))) e->verts += n;
}

// GX thread: its time since the last credit, on commands of object obj (an Entry, or none) from
// origin.
void SMS3DS_ProfGxTicks(uint32_t obj, int origin, uint32_t ticks) {
	if (!SMS3DS_ProfOn || origin < 0 || origin >= kOrigins) return;
	if (Entry* e = reinterpret_cast<Entry*>(uintptr_t(obj))) e->gx[origin] += ticks;
	else s_gxOutside[origin] += ticks;
	s_gxAll[origin] += ticks;
}

// Game thread (decomp patch 3ds-0019, SMS3DSGxOrigin): the GX commands that follow come from
// origin. Returns the one before, for the caller to put back.
int SMS3DS_GxOrigin(int origin) {
	const int prev = s_gxOrigin;
	if (origin != prev) {
		s_gxOrigin = origin;
		GXPC_StreamOrigin(uint8_t(origin));
	}
	return prev;
}

int SMS3DS_ZoneBegin(int zone) {
	if (SMS3DS_ProfOn != 1 || zone < 0 || zone >= kZones || s_zoneDepth >= int(sizeof s_zoneStack / sizeof s_zoneStack[0])) return -1;
	s_zoneStack[s_zoneDepth] = {svcGetSystemTick(), GX3DS_StreamWritten()};
	return s_zoneDepth++;
}
void SMS3DS_ZoneEnd(int zone, int token) {
	const uint64_t now = svcGetSystemTick();
	if (token != s_zoneDepth - 1) {  // unbalanced (a run started inside one): drop the stack
		s_zoneDepth = 0;
		return;
	}
	s_zoneDepth--;
	s_zoneTicks[zone] += now - s_zoneStack[token].t0;
	s_zoneBytes[zone] += GX3DS_StreamWritten() - s_zoneStack[token].b0;
	s_zoneCalls[zone]++;
}

void SMS3DS_ProfRequest(void) {
	if (s_state == 0) s_request = 1;
}
void SMS3DS_ProfRequestGame(void) {
	if (s_state == 0) s_request = 2;
}

// 0 idle, 1 CPU part, 2 GPU part, 3 GPU by object, 4 the game thread's steps (for the status screen)
int SMS3DS_ProfState(void) {
	return s_state >= 13 || s_request == 2 ? 4 : s_state == 4 ? 3 : (s_state == 3 || (s_state >= 5 && s_state <= 12)) ? 2 : s_state == 2 || s_request ? 1 : 0;
}

void SMS3DS_ProfPhase(int phase) {
	const uint64_t now = svcGetSystemTick();
	s_phaseTicks[s_phase] += now - s_phaseLast;
	s_phaseLast = now;
	const uint64_t b = GX3DS_StreamWritten();
	const uint32_t p = GXPC_StreamPuts;
	s_phaseBytes[s_phase] += b - s_phaseB0;
	s_phasePuts[s_phase] += p - s_phaseP0;
	s_phaseB0 = b;
	s_phaseP0 = p;
	if (phase >= 0 && phase < PHASES) {
		s_phase = phase;
		s_phaseEntries[phase]++;
	}
}

void SMS3DS_ProfObjBegin(const void* vtable, const char* name) {
	if (SMS3DS_ProfOn != 1) return;  // phases only
	const uint64_t now = svcGetSystemTick();
	s_calls++;
	if (s_depth >= int(sizeof s_stack / sizeof s_stack[0])) {
		s_depth++;
		return;
	}
	Entry* e = lookup(vtable, s_phase, name);
	if (e && !e->name) e->name = name;
	s_stack[s_depth++] = {e, now, 0, GX3DS_StreamWritten(), 0, GX3DS_GameVerts, 0, GXPC_StreamPuts, 0};
	markObj(e);
}

void SMS3DS_ProfObjEnd(void) {
	if (SMS3DS_ProfOn != 1) return;
	const uint64_t now = svcGetSystemTick();
	if (s_depth <= 0) return;
	if (--s_depth >= int(sizeof s_stack / sizeof s_stack[0])) return;
	const Open& o = s_stack[s_depth];
	const uint64_t dur = now - o.start;
	const uint64_t bytes = GX3DS_StreamWritten() - o.b0;
	const uint32_t puts = GXPC_StreamPuts - o.p0;
	if (o.e) {
		o.e->ticks += dur > o.child ? dur - o.child : 0;
		o.e->bytes += bytes > o.childBytes ? bytes - o.childBytes : 0;
		o.e->puts += puts > o.childPuts ? puts - o.childPuts : 0;
		o.e->calls++;
	}
	if (s_depth > 0) {
		s_stack[s_depth - 1].child += dur;
		s_stack[s_depth - 1].childBytes += bytes;
		s_stack[s_depth - 1].childPuts += puts;
	}
	markObj(s_depth > 0 ? s_stack[s_depth - 1].e : nullptr);  // the draws that follow are the parent's
}

// Called by the game thread at the top of every frame (gameLoop), with no object perform open.
void SMS3DS_ProfFrame(void) {
	const uint64_t now = svcGetSystemTick();
	if (!s_game.tls) s_game.tls = getThreadLocalStorage();  // the sampling hooks record this thread
	if (SMS3DS_ProfOn) SMS3DS_ProfPhase(-1);
	if (s_sampling) {
		const uint32_t d = s_shadow.depth;
		if (d < s_topDepthMin) s_topDepthMin = d;
		if (d > s_topDepthMax) s_topDepthMax = d;
	}
	switch (s_state) {
	case 0:
		if (!s_request) return;
		if (s_request == 2) {
			s_request = 0;
			fprintf(stderr, "[prof] game-thread run: %d frames each ", kFrames);
			for (int k = 0; k < kGameStepN; k++) fprintf(stderr, "%s%s", k ? ", " : "", kGameSteps[k].name);
			fprintf(stderr, " (stand still until \"[prof] done\")\n");
			startGame();
			return;
		}
		s_request = 0;
		s_state = 2;
		s_frame = 0;
		GXPC_Detail = 1;  // the GX thread's per-write counters and command timing, for this run
		fprintf(stderr,
		        "[prof] profile run: %d frames timed, then %d each with pixels cut to 8x8, that and the picture shrunk 8x, "
		        "positions collapsed, no draws, pixels cut with a minimal vertex shader, that with only positions "
		        "fetched, drawing normally, textures outside VRAM swapped for a small one, every texture swapped, "
		        "then objects' pixels cut one at a time, then the game thread drawing normally, with inexact maths "
		        "and with its GX writes muted (stand still)\n",
		        kFrames, kFrames);
		return;
	case 2:  // CPU part
		if (++s_frame == kSettle) {
			calibrate();
			resetCpu();
			s_winStart = s_phaseLast = now;
			s_phase = 0;
			s_gpu0 = pica_gpu_ms_sum();
			SMS3DS_ProfOn = 1;
			sampStart();
		} else if (s_frame == kSettle + kFrames) {
			SMS3DS_ProfOn = 0;
			sampStop();
			s_cpuFrameMs = double(now - s_winStart) / kFrames / kTicksPerMs;
			s_cpuGpuMs = (pica_gpu_ms_sum() - s_gpu0) / kFrames;
			reportCpu(kFrames, s_cpuFrameMs);
			reportSamples(kFrames);
			pica_set_gpu_test(1);
			s_state = 3;
			s_frame = 0;
		}
		return;
	case 3:  // GPU part, pixels cut to 8x8
	case 7:  // pixels cut, the picture shrunk 8x about the centre
	case 5:  // positions collapsed
	case 6:  // no draws
	case 8:  // pixels cut, minimal vertex shader
	case 9:  // that, only positions fetched
	case 10: // drawing normally again
	case 11: // textures outside VRAM swapped for a small one
	case 12: // every texture swapped
		if (++s_frame == kSettle) {
			s_winStart = now;
			s_gpu0 = pica_gpu_ms_sum();
		} else if (s_frame == kSettle + kFrames) {
			const double frameMs = double(now - s_winStart) / kFrames / kTicksPerMs;
			const double gpuMs = (pica_gpu_ms_sum() - s_gpu0) / kFrames;
			if (s_state == 3) {
				s_gpuCutMs = gpuMs;
				s_cutFrame = frameMs;
				pica_set_gpu_test(4);
				s_state = 7;
				s_frame = 0;
				return;
			}
			if (s_state == 7) {
				s_gpuShrunkMs = gpuMs;
				s_shrunkFrame = frameMs;
				pica_set_gpu_test(2);
				s_state = 5;
				s_frame = 0;
				return;
			}
			if (s_state == 5) {
				s_gpuCollapsedMs = gpuMs;
				s_collapsedFrame = frameMs;
				pica_set_gpu_test(3);
				s_state = 6;
				s_frame = 0;
				return;
			}
			// the further steps, each to the next
			static const int kNext[13][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {8, 5}, {0, 0},
			                                 {9, 6}, {10, 0}, {11, 7}, {12, 8}, {0, 0}};
			double* const kSlot[13] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &s_gpuNoDrawMs, nullptr,
			                           &s_gpuMinShaderMs, &s_gpuPosOnlyMs, &s_gpuNormal2Ms, &s_gpuTexMainMs, &s_gpuTexAllMs};
			*kSlot[s_state] = gpuMs;
			if (s_state == 6) s_noDrawFrame = frameMs;
			if (s_state != 12) {
				pica_set_gpu_test(kNext[s_state][1]);
				s_state = kNext[s_state][0];
				s_frame = 0;
				return;
			}
			pica_set_gpu_test(0);
			const double noDrawMs = s_gpuNoDrawMs, noDrawFrame = s_noDrawFrame;
			const double gpuCut = s_gpuCutMs;
			fprintf(stderr,
			        "[prof] GPU: %.1f ms/frame drawing normally (frame %.1f ms), %.1f with pixels cut to 8x8 (frame %.1f "
			        "ms), %.1f with that and the picture shrunk 8x (frame %.1f), %.1f with positions collapsed (frame "
			        "%.1f), %.1f with no draws (frame %.1f)\n",
			        s_cpuGpuMs, s_cpuFrameMs, gpuCut, s_cutFrame, s_gpuShrunkMs, s_shrunkFrame, s_gpuCollapsedMs,
			        s_collapsedFrame, noDrawMs, noDrawFrame);
			fprintf(stderr,
			        "[prof] GPU split: pixels %.1f ms | triangles running past the screen's sides (the rasteriser "
			        "walks them; clipping is what limits it) %.1f ms | triangle setup and near-plane clipping %.1f ms | "
			        "vertex fetch and shading %.1f ms | the rest (state, commands, copies, the present) %.1f ms\n",
			        s_cpuGpuMs - gpuCut, gpuCut - s_gpuShrunkMs, s_gpuShrunkMs - s_gpuCollapsedMs,
			        s_gpuCollapsedMs - noDrawMs, noDrawMs);
			fprintf(stderr,
			        "[prof] GPU, what lighter work would save: pixels cut and a minimal vertex shader (no lighting, no "
			        "texture coordinates) %.1f ms/frame, %.1f less than pixels cut alone | and only positions fetched "
			        "%.1f, %.1f less again | drawing normally again %.1f; textures outside VRAM swapped for one small "
			        "texture %.1f, %.1f less; every texture swapped %.1f, %.1f less\n",
			        s_gpuMinShaderMs, gpuCut - s_gpuMinShaderMs, s_gpuPosOnlyMs, s_gpuMinShaderMs - s_gpuPosOnlyMs,
			        s_gpuNormal2Ms, s_gpuTexMainMs, s_gpuNormal2Ms - s_gpuTexMainMs, s_gpuTexAllMs,
			        s_gpuNormal2Ms - s_gpuTexAllMs);
			// Part 3 needs the objects' open/close calls (the entries stay as part 1 left them).
			pickTargets();
			fprintf(stderr,
			        "[prof] GPU by object: %d objects' draws left out one at a time (the GPU time each costs in all), each line printed as it is "
			        "measured (keep standing still until \"[prof] done\")\n",
			        s_nTargets);
			s_depth = 0;
			SMS3DS_ProfOn = 1;
			s_step = -1;
			s_next = 0;
			setCut(-1);
			s_state = 4;
			s_frame = 0;
		}
		return;
	case 4:  // GPU by object
		if (++s_frame == kCutSettle) {
			s_winStart = now;
			s_gpu0 = pica_gpu_ms_sum();
		} else if (s_frame == kCutSettle + kCutFrames) {
			const double gpu = (pica_gpu_ms_sum() - s_gpu0) / kCutFrames;
			if (s_step < 0) {
				s_base = gpu;
				s_cutFrameMs = double(now - s_winStart) / kCutFrames / kTicksPerMs;
				fprintf(stderr, "[prof]   drawing normally: GPU %.1f ms/frame, frame %.1f ms\n", gpu, s_cutFrameMs);
			} else {
				s_targetMs[s_step] = gpu;
				s_targetBase[s_step] = s_base;
				printTarget(" ", s_step);
			}
			if (s_step >= 0 && s_next >= s_nTargets) {
				setCut(-1);
				SMS3DS_ProfOn = 0;
				reportCut();
				GXPC_Detail = 0;  // part 4 as in normal play
				fprintf(stderr, "[prof] game thread: %d frames each step (keep standing still until \"[prof] done\")\n",
				        kFrames);
				startGame();
				return;
			}
			// After every kPerBase targets, normal drawing again.
			s_step = (s_step >= 0 && s_next % kPerBase == 0) ? -1 : s_next++;
			setCut(s_step);
			s_frame = 0;
		}
		return;
	case 13:  // the game thread, step s_gameStep
		if (++s_frame == kSettle) {
			resetCpu();
			s_winStart = s_phaseLast = now;
			s_phase = 0;
			s_gpu0 = pica_gpu_ms_sum();
			GX3DS_ThreadTicks(s_thread0);
			s_verify0[0] = GXPC_J3DStats[15];  // NJ_VERIFIED
			s_verify0[1] = GXPC_J3DPkgStats[0];
			s_verify0[2] = GX3DS_J3DSigBad();
			SMS3DS_ProfOn = kGameSteps[s_gameStep].profOn;
		} else if (s_frame == kSettle + kFrames) {
			SMS3DS_ProfOn = 0;
			const int k = s_gameStep;
			for (int i = 0; i < PHASES; i++) s_gamePh[k][i] = s_phaseTicks[i] / double(kFrames) / kTicksPerMs;
			s_gameFrame[k] = double(now - s_winStart) / kFrames / kTicksPerMs;
			s_gameGpu[k] = (pica_gpu_ms_sum() - s_gpu0) / kFrames;
			s_gameHog[k] = hogRate();
			uint64_t th[2];
			GX3DS_ThreadTicks(th);
			s_gameGx[k] = double(th[0] - s_thread0[0]) / kFrames / kTicksPerMs;
			s_gameRender[k] = double(th[1] - s_thread0[1]) / kFrames / kTicksPerMs;
			reportGameStep(k);
			if (k == kGsVerify)
				fprintf(stderr,
				        "[prof] compiled shapes verified: %lu native draws checked by the render thread, %lu shape ops "
				        "drawn from compiled shapes, %lu differences (the render thread logs each as [j3d] lines)\n",
				        (unsigned long)(GXPC_J3DStats[15] - s_verify0[0]), (unsigned long)(GXPC_J3DPkgStats[0] - s_verify0[1]),
				        (unsigned long)(GX3DS_J3DSigBad() - s_verify0[2]));
			setGameStep(++s_gameStep);
			if (s_gameStep == kGameStepN) s_state = 14;
			s_frame = 0;
		}
		return;
	case 14:  // the last muted frames reach the GPU behind the game thread
		if (++s_frame == kSettle) {
			pica_set_gpu_test(0);
			reportGameSaved();
			fprintf(stderr, "[prof] done\n");
			s_state = 0;
		}
		return;
	}
}
