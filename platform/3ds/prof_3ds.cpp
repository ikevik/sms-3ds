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
// kTargets) and the draws made outside any object, one at a time, have their pixels cut while
// everything else draws normally (the GX backend batches them apart; SMS3DS_ProfCutNow). The
// GPU time saved is roughly what that object's pixels cost. Normal GPU time is measured again
// every kPerBase objects, so a drifting view skews only the objects near it.
// Each part skips kSettle frames first: the GPU runs up to two frames behind the game thread.
#include <3ds.h>
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
int SMS3DS_ProfOn;
void SMS3DS_ProfPhase(int phase);
void SMS3DS_ProfObjBegin(const void* vtable, const char* name);
void SMS3DS_ProfObjEnd(void);
void SMS3DS_ProfFrame(void);
void SMS3DS_ProfRequest(void);
int SMS3DS_ProfState(void);
volatile int SMS3DS_ProfCutActive;
int SMS3DS_ProfCutNow(void);
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

struct Entry {
	const void* vt;
	const char* name;  // draw buffers (one class, many jobs) are told apart by name
	uint64_t ticks, bytes;
	uint32_t calls, verts, puts;
	int phase;
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
uint32_t s_phaseEntries[PHASES];
int s_phase;
uint64_t s_calls;

// state: 0 idle, 1 requested, 2 CPU part (settling, then measuring), 3 GPU part, 4 GPU by object
volatile int s_request;
int s_state, s_frame;
uint64_t s_winStart;
double s_gpu0, s_cpuFrameMs, s_cpuGpuMs;

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

}  // namespace

// Called by the GX backend before it adds vertices, while SMS3DS_ProfCutActive: non-zero when
// the object being performed (the innermost open one) is part 3's current target.
int SMS3DS_ProfCutNow(void) {
	if (s_depth == 0) return s_cutOutside;
	if (s_depth > int(sizeof s_stack / sizeof s_stack[0])) return 0;
	const Entry* e = s_stack[s_depth - 1].e;
	return s_cutEntry && e == s_cutEntry;
}

void SMS3DS_ProfRequest(void) {
	if (s_state == 0) s_request = 1;
}

// 0 idle, 1 CPU part, 2 GPU part (for the status screen)
int SMS3DS_ProfState(void) { return s_state == 4 ? 3 : s_state == 3 ? 2 : s_state == 2 ? 1 : s_request ? 1 : 0; }

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
	const uint64_t now = svcGetSystemTick();
	s_calls++;
	if (s_depth >= int(sizeof s_stack / sizeof s_stack[0])) {
		s_depth++;
		return;
	}
	Entry* e = lookup(vtable, s_phase, name);
	if (e && !e->name) e->name = name;
	s_stack[s_depth++] = {e, now, 0, GX3DS_StreamWritten(), 0, GX3DS_GameVerts, 0, GXPC_StreamPuts, 0};
}

void SMS3DS_ProfObjEnd(void) {
	const uint64_t now = svcGetSystemTick();
	if (s_depth <= 0) return;
	if (--s_depth >= int(sizeof s_stack / sizeof s_stack[0])) return;
	const Open& o = s_stack[s_depth];
	const uint64_t dur = now - o.start;
	const uint32_t verts = GX3DS_GameVerts - o.v0;
	const uint64_t bytes = GX3DS_StreamWritten() - o.b0;
	const uint32_t puts = GXPC_StreamPuts - o.p0;
	if (o.e) {
		o.e->ticks += dur > o.child ? dur - o.child : 0;
		o.e->verts += verts > o.childVerts ? verts - o.childVerts : 0;
		o.e->bytes += bytes > o.childBytes ? bytes - o.childBytes : 0;
		o.e->puts += puts > o.childPuts ? puts - o.childPuts : 0;
		o.e->calls++;
	}
	if (s_depth > 0) {
		s_stack[s_depth - 1].child += dur;
		s_stack[s_depth - 1].childVerts += verts;
		s_stack[s_depth - 1].childBytes += bytes;
		s_stack[s_depth - 1].childPuts += puts;
	}
}

// Called by the game thread at the top of every frame (gameLoop), with no object perform open.
void SMS3DS_ProfFrame(void) {
	const uint64_t now = svcGetSystemTick();
	if (SMS3DS_ProfOn) SMS3DS_ProfPhase(-1);
	switch (s_state) {
	case 0:
		if (!s_request) return;
		s_request = 0;
		s_state = 2;
		s_frame = 0;
		fprintf(stderr,
		        "[prof] profile run: %d frames timed, then %d with pixels cut to 8x8, then objects' pixels cut one at a "
		        "time (stand still)\n",
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
		} else if (s_frame == kSettle + kFrames) {
			SMS3DS_ProfOn = 0;
			s_cpuFrameMs = double(now - s_winStart) / kFrames / kTicksPerMs;
			s_cpuGpuMs = (pica_gpu_ms_sum() - s_gpu0) / kFrames;
			reportCpu(kFrames, s_cpuFrameMs);
			pica_set_gpu_test(1);
			s_state = 3;
			s_frame = 0;
		}
		return;
	case 3:  // GPU part, pixels cut to 8x8
		if (++s_frame == kSettle) {
			s_winStart = now;
			s_gpu0 = pica_gpu_ms_sum();
		} else if (s_frame == kSettle + kFrames) {
			const double frameMs = double(now - s_winStart) / kFrames / kTicksPerMs;
			const double gpuCut = (pica_gpu_ms_sum() - s_gpu0) / kFrames;
			pica_set_gpu_test(0);
			fprintf(stderr,
			        "[prof] GPU: %.1f ms/frame drawing normally (frame %.1f ms), %.1f with pixels cut to 8x8 (frame %.1f "
			        "ms): pixels about %.1f ms, vertices and triangle setup about %.1f ms\n",
			        s_cpuGpuMs, s_cpuFrameMs, gpuCut, frameMs, s_cpuGpuMs - gpuCut, gpuCut);
			// Part 3 needs the objects' open/close calls (the entries stay as part 1 left them).
			pickTargets();
			fprintf(stderr,
			        "[prof] GPU by object: %d objects' pixels cut one at a time, each line printed as it is "
			        "measured (keep standing still until \"done\")\n",
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
				s_state = 0;
				return;
			}
			// After every kPerBase targets, normal drawing again.
			s_step = (s_step >= 0 && s_next % kPerBase == 0) ? -1 : s_next++;
			setCut(s_step);
			s_frame = 0;
		}
		return;
	}
}
