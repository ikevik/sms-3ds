// Super Mario Sunshine 3DS - stage 2 boot attempt (MODE=game).
//
// Runs the real game (decomp + sms-pc-port platform layer) on a big-stack thread and shows
// progress on the bottom screen. Graphics are counted, not drawn yet (gx3ds_backend.cpp).
// Everything the port logs (port_log -> stderr) goes to sdmc:/sms/log.txt.
#include <3ds.h>
#include "pad_3ds.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
void port_init(int argc, char** argv);
void GX3DS_GetStats(uint32_t out[7]);
uint32_t pica_dropped(void);
int pica_cycle_view(void);
int pica_cycle_gpu_test(void);
int pica_view(void);
int pica_view_count(void);
void GX3DS_SetDebugColors(int on);
void GX3DS_TraceNextFrame(void);
void GX3DS_TraceState(int out[2]);  // armed or tracing, traces written
int GX3DS_GetDebugColors(void);
void GX3DS_GetCopyStats(uint32_t out[3]);
void GX3DS_GetPerf(float out[11]);
void GX3DS_SetGpuTransform(int on);
const char* pica_where(void);
const char* pica_queue_state(void);
uint32_t pica_frame_draws(void);
int pica_selftest(void);
void pica_vertex_bench(void);
void pica_set_gpu_check(int on);
void pica_set_split(int on);
void pica_set_vram_textures(int on);
void pica_set_vram_check(int on);
void pica_set_state_skip(int on);
void pica_set_readback_dumps(int on);
void pica_set_overlap(int on);
void pica_set_big_ring(int on);
extern "C" void SMS3DS_ProfRequest(void);
extern "C" void SMS3DS_ProfRequestGame(void);  // part 4 alone: the game thread's steps
extern "C" int SMS3DS_ProfState(void);
extern "C" void SMS3DS_SampReport(const char* why);  // prof_3ds.cpp, SAMPLE=1 builds
extern "C" int port_os_stack_report(char* out, int n);  // patch 0029
void GX3DS_SetArena(int on);
void GX3DS_SetRenderThread(int on);
void GX3DS_SetGxThread(int on);
void GX3DS_SetListCopies(int on);
void GX3DS_SetResidentIndicesInPlace(int on);
void GX3DS_SetMergeDuplicates(int on);
void GX3DS_SetReorder(int on);
void GXPC_SetWriteFilter(int on);
extern int GXPC_CullOff;  // patch 0014: skip off-screen display-list draws
// Demo builds (make DEMO=slide, DEMO=test11): the stage and scenario the game boots into and
// returns to (decomp patch 0008); stage -1 in the normal build.
#ifndef SMS_DEMO_STAGE
#define SMS_DEMO_STAGE -1
#endif
#ifndef SMS_DEMO_SCENARIO
#define SMS_DEMO_SCENARIO 0
#endif
extern "C" int SMS3DS_DemoStage, SMS3DS_DemoScenario;
int SMS3DS_DemoStage = SMS_DEMO_STAGE, SMS3DS_DemoScenario = SMS_DEMO_SCENARIO;
// Unattended runs, driven from a computer (as in Azahar): sdmc:/sms/autorun.txt may give
// boot_stage (<stage>-<scenario>: booted into and returned to, as a demo build's), skip_movies
// (on: the movies skipped, as settings.txt's skip_movies), profile_auto (seconds of free play
// before the low touch's run starts by itself; 0 off), profile_auto_quit (on: quit once that run
// is done) and quit_after (seconds of free play before quitting; 0 off). Free play: the stage's
// director in its play state, drawing a scene. It may also give perf_timers and draw_state_check,
// as settings.txt does, and gx_detail (on: the GX thread's commands timed, as in profile runs).
static int s_profAuto, s_profAutoQuit, s_quitAfter, s_timelineAt;
extern "C" int SMS3DS_TimelineOn, SMS3DS_ProfOn;  // prof_3ds.cpp: the frame timeline (autorun.txt timeline)
extern "C" void SMS3DS_TimelineReport(void);
// autorun.txt game_thread_core (profiling in Azahar, whose debugger stub reads fresh registers only
// for the threads of its later cores): the game thread made on that core with threadCreate (as
// libctru makes pthreads: priority 0x3F) instead of pthread_create's core 0. -1: as normal.
static int s_gameCore = -1;
extern "C" int pica_gx_core;  // pica_draw.c: the GX thread's core (autorun.txt gx_thread_core, profiling)
extern "C" unsigned SMS3DS_ProfRunsDone;  // prof_3ds.cpp: profile runs finished so far
void pica_request_stop(void);
void GX3DS_SetVerboseLog(int on);
extern volatile uint64_t GX3DS_MainBusyTicks;  // gx3ds_backend.cpp: the [perf] "status screen" share
int GX3DS_GetGpuTransform(void);
void GX3DS_GetPathStats(uint32_t out[2]);
void GX3DS_GetTexStats(uint32_t out[3]);
}
void SMS_main(void);  // the decomp's main(), renamed by port_compat.h (C++ linkage)

#define DISC_PATH "sdmc:/sms/GMSE01.iso"
#define SMS_DIR   "sdmc:/sms"
#define LOG_PATH  SMS_DIR "/log.txt"

static volatile int g_phase;  // 0 not started, 1 port_init, 2 SMS_main, 3 SMS_main returned

// --- scene probe ----------------------------------------------------------------------
// Reads the game's TApplication (System/Application.hpp) by offset, since this file is not
// built with the decomp's headers. Offsets checked with static_assert against the ARM build:
// mDirector 0x04, mAppState 0x08, mCurrArea 0x0E, mNextArea 0x12, mMovie 0x18; a
// TGameSequence is {u8 stage, u8 scenario, u16 flag}.
extern unsigned char gpApplication[];
// THPPlayer (libs/THPPlayer): state 0xA4, internalState 0xA5, playFlag 0xA6, curVideoNumber
// 0xF0, curAudioNumber 0xF4 (offsets from the ARM build). state 3 = played, 5 = stopped/error.
extern "C" unsigned char ActivePlayer[];
extern "C" uint32_t THPPlayerGetTotalFrame(void);
// TMarDirector (System/MarDirector.hpp): mState u8 at 0x64, TCardLoad* at 0x70; TCardLoad
// (GC2D/CardLoad.hpp): ints at 0x10, 0x14, 0x18. Checked with static_assert on the ARM build.
extern unsigned char _ZTV12TMarDirector[];
// The emulated OS scheduler (sms-pc-port platform/os/os_thread.cpp) and OSThread fields
// (dolphin/os/OSThread.h; offsets checked on the ARM build): state u16 0x2C8, suspend s32
// 0x2CC, priority s32 0x2D0, queue (thread queue it waits on) 0x2DC, mutex (waited on) 0x2F0.
extern "C" int port_os_thread_list(void** out, int max);
extern "C" void* __gCurrentThread;

// Writes every emulated thread to the log: resolve the addresses with nm on sms3ds.elf
// (global OSThreads, message queues and mutexes have symbols).
static void flush_log(void);

static void dump_os_threads(const char* why)
{
	void* list[48];
	int n = port_os_thread_list(list, 48);
	fprintf(stderr, "[threads] %s: %d emulated threads, current %08lx\n", why, n,
	        (unsigned long)(uintptr_t)__gCurrentThread);
	for (int i = 0; i < n && i < 48; i++) {
		const unsigned char* t = (const unsigned char*)list[i];
		uint16_t state;
		int32_t suspend, prio;
		uint32_t queue, mutex;
		memcpy(&state, t + 0x2C8, 2);
		memcpy(&suspend, t + 0x2CC, 4);
		memcpy(&prio, t + 0x2D0, 4);
		memcpy(&queue, t + 0x2DC, 4);
		memcpy(&mutex, t + 0x2F0, 4);
		const char* st = state == 1 ? "ready" : state == 2 ? "running" : state == 4 ? "waiting" : state == 8 ? "exited" : "?";
		fprintf(stderr, "[threads]   %08lx prio %2ld %-7s suspend %ld queue %08lx mutex %08lx\n",
		        (unsigned long)(uintptr_t)t, (long)prio, st, (long)suspend, (unsigned long)queue,
		        (unsigned long)mutex);
	}
}

static const char* app_state_name(unsigned s)
{
	static const char* const names[] = {"WAIT", "DEFAULT", "BOOT", "NLOGO", "DONE",
	                                    "GAMEPLAY", "MOVIE", "QUIT", "TITLE", "MENU"};
	return s < sizeof names / sizeof names[0] ? names[s] : "?";
}

struct SceneState {
	unsigned state, curr_stage, curr_scen, next_stage, next_scen, movie;
	uintptr_t director, director_vtbl;
	unsigned thp_state, thp_internal, thp_play;  // compared for changes; frame counters are not
	int mar_state;                               // TMarDirector::mState, -1 for other directors
	uintptr_t cardload;
	int cl10, cl14, cl18;
};

static void read_thp_frames(int32_t* video, int32_t* audio)
{
	memcpy(video, ActivePlayer + 0xF0, sizeof *video);
	memcpy(audio, ActivePlayer + 0xF4, sizeof *audio);
}

// The probe reads the game's objects from the main thread while the game changes them (a stage
// load frees and builds its directors), so it follows a pointer only into mapped memory: the
// regular heap (MEM1 and the game's heaps) or the program image. A stale TCardLoad pointer
// (0x7a060) followed while Pianta Village loaded faulted the main thread (crash dump 37).
extern "C" u32 __ctru_heap, __ctru_heap_size;
extern "C" char __start__[], __end__[];
static bool probe_readable(uintptr_t p, size_t n)
{
	const uintptr_t heap = __ctru_heap, image = (uintptr_t)__start__;
	return p + n >= p && ((p >= heap && p + n <= heap + __ctru_heap_size) || (p >= image && p + n <= (uintptr_t)__end__));
}

static SceneState read_scene()
{
	const unsigned char* a = gpApplication;
	SceneState s;
	uintptr_t dir;
	memcpy(&dir, a + 0x04, sizeof dir);
	s.director      = dir;
	s.director_vtbl = dir && probe_readable(dir, 4) ? *(const uintptr_t*)dir : 0;  // resolve with nm to get the class
	s.state         = a[0x08];
	s.curr_stage    = a[0x0E];
	s.curr_scen     = a[0x0F];
	s.next_stage    = a[0x12];
	s.next_scen     = a[0x13];
	memcpy(&s.movie, a + 0x18, sizeof s.movie);
	s.thp_state    = ActivePlayer[0xA4];
	s.thp_internal = ActivePlayer[0xA5];
	s.thp_play     = ActivePlayer[0xA6];
	s.mar_state = -1;
	s.cardload = 0;
	s.cl10 = s.cl14 = s.cl18 = 0;
	if (dir && s.director_vtbl == (uintptr_t)(_ZTV12TMarDirector + 8) && probe_readable(dir, 0x74)) {
		const unsigned char* d = (const unsigned char*)dir;
		s.mar_state = d[0x64];
		memcpy(&s.cardload, d + 0x70, sizeof s.cardload);
		if (s.cardload && probe_readable(s.cardload, 0x1C)) {
			memcpy(&s.cl10, (const unsigned char*)s.cardload + 0x10, 4);
			memcpy(&s.cl14, (const unsigned char*)s.cardload + 0x14, 4);
			memcpy(&s.cl18, (const unsigned char*)s.cardload + 0x18, 4);
		}
	}
	return s;
}

// Logs once a second, and at once whenever the state, area, movie or director changes.
// frames and batches are GX3DS_GetStats totals; draws/frame is over the time since the last line.
static void log_scene(uint32_t frames, uint32_t batches)
{
	static uint32_t last_frames, last_batches;
	static SceneState last;
	static bool have_last;
	static u64 last_ms;
	SceneState s = read_scene();
	u64 now      = osGetTime();
	bool changed = !have_last || memcmp(&s, &last, sizeof s) != 0;
	if (!changed && now - last_ms < 1000)
		return;
	fprintf(stderr,
	        "[scene] f=%lu %sstate=%u(%s) area=%u-%u next=%u-%u movie=%u director=%08lx vtbl=%08lx\n",
	        (unsigned long)frames, changed ? "CHANGED " : "", s.state, app_state_name(s.state),
	        s.curr_stage, s.curr_scen, s.next_stage, s.next_scen, s.movie, (unsigned long)s.director,
	        (unsigned long)s.director_vtbl);
	// A stage whose setup never finishes sits in TMarDirector state 0: dump the threads after
	// 8 seconds of that, then every 30.
	static int state0_secs;
	if (s.mar_state == 0) {
		state0_secs++;
		if (state0_secs == 8 || (state0_secs > 8 && state0_secs % 30 == 8)) {
			SMS3DS_SampReport("stage setup not finished after 8 s");
			flush_log();
			dump_os_threads("stage setup not finished after 8 s");
		}
	} else {
		state0_secs = 0;
	}
	uint32_t df = frames - last_frames, db = batches - last_batches;
	last_frames = frames;
	last_batches = batches;
	if (s.mar_state >= 0)
		fprintf(stderr, "[mar] state=%d cardload=%08lx cl=%d,%d,%d draws/frame=%lu\n", s.mar_state,
		        (unsigned long)s.cardload, s.cl10, s.cl14, s.cl18, (unsigned long)(df ? db / df : 0));
	if (s.thp_state || s.thp_internal || s.thp_play) {
		int32_t video, audio;
		read_thp_frames(&video, &audio);
		fprintf(stderr, "[thp] state=%u internal=%u play=%u video=%ld/%lu audio=%ld\n", s.thp_state,
		        s.thp_internal, s.thp_play, (long)video, (unsigned long)THPPlayerGetTotalFrame(),
		        (long)audio);
	}
	last      = s;
	have_last = true;
	last_ms   = now;
}

// No new frame for 5 s: says where the renderer is on the game thread and dumps the emulated
// threads, once, then every 30 s while it lasts. Batches still rising means the game is drawing
// but never presents; C3D_FrameBegin means the GPU never finished the last frame.
static void check_stall(uint32_t frames, uint32_t batches)
{
	static uint32_t last_frames = ~0u;
	static u64 since;
	static int reports;
	// Milliseconds from the system tick: osGetTime follows the clock, which steps back now and
	// then, and a step back read as a stall of billions of seconds (17 Oct log, frame 4588).
	u64 now = svcGetSystemTick() / (SYSCLOCK_ARM11 / 1000);
	if (frames != last_frames) {
		if (reports) fprintf(stderr, "[stall] frames moving again at %lu\n", (unsigned long)frames);
		last_frames = frames;
		since = now;
		reports = 0;
		return;
	}
	u64 secs = (now - since) / 1000;
	if (secs < 5 || (reports && secs < 5 + 30u * (u64)reports)) return;
	reports++;
	fprintf(stderr, "[stall] no frame for %lu s at frame %lu: renderer %s, %lu draws this frame, %lu batches total, %s transform\n",
	        (unsigned long)secs, (unsigned long)frames, pica_where(), (unsigned long)pica_frame_draws(),
	        (unsigned long)batches, GX3DS_GetGpuTransform() ? "GPU" : "CPU");
	fprintf(stderr, "[stall]   %s\n", pica_queue_state());
	SMS3DS_SampReport("no frame for 5 s");  // sampling builds: the game thread's calls
	{
		char st[1024];
		if (port_os_stack_report(st, sizeof st)) fprintf(stderr, "[stall]   emulated threads' stacks (KB):%s\n", st);
	}
	flush_log();
	dump_os_threads("no frame for 5 s");
	flush_log();
	printf("\x1b[29;0Hstalled: %.30s", pica_where());
}

// Commits the log to the SD card. Written data only becomes part of the file when it is flushed
// or closed, so a console turned off without this leaves gaps and stale bytes in log.txt.
static void flush_log(void)
{
	fflush(stderr);
	fsync(fileno(stderr));
}

// sdmc:/sms/settings.txt as first written: the settings that do something on the 3DS, at its
// defaults. sms-pc-port reads the file at start (port_runtime.cpp, load_settings); any other
// sms-pc-port setting can be added by name.
static const char kDefaultSettings[] =
    "# Super Mario Sunshine (3DS) settings: one \"name = value\" a line, read at start.\n"
    "# Lines starting with # are ignored. Delete this file to get the defaults back.\n"
    "\n"
    "# Gameplay frame rate: 30 (the game's own) or 60. At 60 the game's logic runs twice as\n"
    "# often, which the 3DS cannot keep up with.\n"
    "frame_rate = 30\n"
    "\n"
    "# The heat-wave shimmer in sunny places (a screen copy and a full-screen pass each frame):\n"
    "# on or off. Off on the 3DS for speed.\n"
    "heat_haze = off\n"
    "\n"
    "# Skip the game's movies: on or off.\n"
    "skip_movies = off\n"
    "\n"
    "# Game audio (docs/audio-3ds.md): on (music and sound effects, mixed by the 3DS's DSP), off (no\n"
    "# sound: a little faster), software (sms-pc-port's software mixer instead, to compare) or silent\n"
    "# (the game's sound driver runs and nothing plays, to measure). on and software need\n"
    "# sdmc:/3ds/dspfirm.cdc (made once by the DSP1 homebrew); without it the game runs without sound.\n"
    "audio = on\n"
    "# JAudio updates each voice every N subframes of 2.5 ms: 2 (the default: about half the work, the\n"
    "# same sound to the ear, since the 3DS's DSP takes a voice's state every 4.9 ms anyway) or 1 (the\n"
    "# game's own).\n"
    "audio_update_interval = 2\n"
    "# With audio not off: every N seconds turn to the next of on, software and silent, and log what\n"
    "# each costs ([audio] lines). 0: stay.\n"
    "# audio_cycle = 10\n"
    "\n"
    "# Delfino Plaza's Shine shadow: how many of its 5 shells to draw (0: all 5). Each costs\n"
    "# about 1 ms a frame; they only soften the shadow's edge over a pixel or two.\n"
    "shine_shadow_shells = 1\n"
    "\n"
    "# Small mipmapped textures drawn with their smaller levels, as the GameCube does (on), or\n"
    "# at full size everywhere (off). On is faster and shimmers less in the distance.\n"
    "mipmaps = on\n"
    "\n"
    "# The game's GX state resets written as precomputed blocks (on), with the same result as its\n"
    "# own calls (off), which cost several ms a frame. Off is for comparing the two.\n"
    "gx_state_blocks = on\n"
    "\n"
    "# Each GX state reset sent to the GX thread as one command with its final values (on), or as\n"
    "# its separate register writes (off). The same registers either way.\n"
    "gx_state_block_ops = on\n"
    "\n"
    "# That command compiled once per block, its values in the order the API keeps them (on), or\n"
    "# worked out entry by entry each time (off). The same registers either way.\n"
    "gx_state_block_compiled = on\n"
    "\n"
    "# The whole linear heap flushed from the CPU caches at each frame's end, as citro3d does (on),\n"
    "# or only the buffers flushed by their own ranges (off: an experiment; the GPU hung at the\n"
    "# first scene change with it off).\n"
    "heap_flush = on\n"
    "\n"
    "# The reset the lights call loads its 8 null textures once (on), or twice as the original (off):\n"
    "# the second set rewrites every register the first set wrote, with the same values.\n"
    "reset_textures_once = on\n"
    "\n"
    "# The state-only display lists the GX thread has cached (materials, vertex formats) replayed as\n"
    "# register words, skipping writes that change nothing (on), or command by command (off). The\n"
    "# same result either way; off is for comparing the two.\n"
    "gx_fast_state_lists = on\n"
    "\n"
    "# J3D models drawn natively (docs/native-j3d.md): their GX calls sent as one op per material and\n"
    "# shape, drawn from what the GX thread compiled (on), the calls as before (off), or both, the\n"
    "# ops checked against what the calls leave (check), or on with the render thread's registers\n"
    "# checked before each native draw (verify), in the draw buffers named. On saves Delfino Plaza's\n"
    "# frame about 2 ms, with the same pixels (verify mode: 0 differences).\n"
    "# native_j3d = on\n"
    "# native_j3d_buffers = MapOpa,StaticMapObj_SunOpa\n"
    "# Native shapes drawn from compiled blocks once they have drawn natively (on), or found and\n"
    "# checked in the display-list cache on every draw (off). The same draws either way.\n"
    "# native_j3d_packages = on\n"
    "\n"
    "# The sea's vertices worked out once per column and row (on), or vertex by vertex as the\n"
    "# original (off). The same vertices, bit for bit; on saves most of its exact sines.\n"
    "sea_precomputed = on\n"
    "\n"
    "# The news ticker's glyphs wholly outside its window left out (on), or every glyph of the\n"
    "# message drawn each frame as the original (off). The same pixels: the window's scissor\n"
    "# leaves nothing of a glyph outside it.\n"
    "ticker_clipped = on\n"
    "\n"
    "# The event watcher's lookups of objects by name answered from a cache (on), or each one a\n"
    "# full search of the scene's objects as the original (off). The same answers: the cache\n"
    "# drops them whenever the scene's objects change. verify: each answer checked against a full\n"
    "# search (slower).\n"
    "name_cache = on\n"
    "\n"
    "# The [perf] lines' breakdown of the renderer's time by phase (tex, tev, xform, submit, decode,\n"
    "# batch, append), timed (on) or not (off). It reads the clock thousands of times a frame: about\n"
    "# 2 ms a frame on a New 3DS and far more in Azahar. The frame and the threads' times are always\n"
    "# timed.\n"
    "# perf_timers = off\n"
    "\n"
    "# What touching the bottom right of the bottom screen low down runs: the game thread's steps\n"
    "# (steps) or the game thread by object (objects). Higher up it starts the whole profile run.\n"
    "profile_low_touch = steps\n";

// Small mipmapped textures with their mip levels (GX3DS_SetMips; settings.txt mipmaps).
#ifndef SMS3DS_MIPS_DEFAULT
#define SMS3DS_MIPS_DEFAULT 1
#endif
extern "C" void GX3DS_SetMips(int on);
// The game's GX state resets as precomputed blocks (sms-pc-port patch 0031; settings.txt gx_state_blocks).
#ifndef SMS3DS_STATE_BLOCKS_DEFAULT
#define SMS3DS_STATE_BLOCKS_DEFAULT 1
#endif
extern "C" void GXPC_SetStateBlocks(int on);
// Cached state-only display lists replayed as register words (sms-pc-port patch 0033; settings.txt
// gx_fast_state_lists).
#ifndef SMS3DS_FAST_STATE_LISTS_DEFAULT
#define SMS3DS_FAST_STATE_LISTS_DEFAULT 1
#endif
extern "C" void GXPC_SetFastStateLists(int on);
// J3D drawn natively (decomp patch 3ds-0020, sms-pc-port patch 0035; settings.txt native_j3d,
// native_j3d_buffers): 0 off, 1 on, 2 check, 3 verify (on, the render side's registers checked).
#ifndef SMS3DS_NATIVE_J3D_DEFAULT
#define SMS3DS_NATIVE_J3D_DEFAULT 1
#endif
// The draw buffers (less "DrawBuf ", '_' for a space): the static ones TDrawBufObj draws.
#define SMS3DS_NATIVE_J3D_BUFFERS \
	"MapOpa,MapXlu,StaticMapObj_SunOpa,StaticMapObj_ShadowOpa,StaticMapObj_SunXlu,StaticMapObj_ShadowXlu," \
	"Mirror_Opa,Mirror_Xlu,Sky_Opa,Sky_Xlu,AfterIndirect_Xlu,ChrOpa,ChrXlu"
extern "C" void GXPC_SetNativeJ3D(int mode, const char* buffers);
extern "C" int GXPC_J3DPackages;  // patch 0037: native shapes from compiled blocks (settings.txt native_j3d_packages)
extern "C" int SMS3DS_WaveMode;    // decomp patch 3ds-0022: the sea precomputed (settings.txt sea_precomputed)
extern "C" void GXPC_SetStateBlockOps(int mode);  // patch 0038: state resets as one command (gx_state_block_ops)
extern "C" int SMS3DS_ResetTexOnce;  // decomp patch 3ds-0024: the lights' reset loads its textures once
extern "C" int SMS3DS_TelopClip;     // decomp patch 3ds-0026: the ticker's glyphs outside its window left out
extern "C" int SMS3DS_NameCache;     // decomp patch 3ds-0029: the event watcher's name lookups cached
extern "C" void GX3DS_SetPerfTimers(int on);  // gx3ds_backend.cpp: the [perf] breakdown by phase timed
extern "C" void GX3DS_SetDrawStateCheck(int on);  // ... draws taking the last draw's setup checked
extern "C" int GXPC_Detail;  // patch 0027: the GX thread's command timing, as in profile runs
extern "C" void GXPC_SetStateBlockCompiled(int on);  // patch 0039 (gx_state_block_compiled)
extern "C" void pica_set_heap_flush(int on);  // pica_draw.c (heap_flush)
extern "C" void SMS3DS_ProfSetLowTouch(int part);  // prof_3ds.cpp (settings.txt profile_low_touch)
// Sound on the 3DS's DSP (platform/3ds/audio_3ds.cpp; settings.txt audio, audio_cycle): 1 on,
// 0 off, 2 software mixer, 3 silent.
#ifndef SMS3DS_AUDIO_DEFAULT
#define SMS3DS_AUDIO_DEFAULT 1
#endif
// Subframes between JAudio's updates of each voice (settings.txt audio_update_interval; the game's own
// is 1). 2 halves most of that work in Delfino Plaza (0.5-0.6 ms a frame) and sounded the same.
#ifndef SMS3DS_AUDIO_UPDATE_INTERVAL_DEFAULT
#define SMS3DS_AUDIO_UPDATE_INTERVAL_DEFAULT 2
#endif
extern "C" int SMS3DS_AudioInit(int mode, int cycleSeconds);
extern "C" void SMS3DS_AudioSetUpdateInterval(int subframes);
extern "C" void SMS3DS_AudioExit(void);

// Delfino Plaza's Shine shadow volume: shells drawn (decomp patch 3ds-0012; 0: all of them).
#ifndef SMS3DS_SHINE_SHELLS_DEFAULT
#define SMS3DS_SHINE_SHELLS_DEFAULT 1
#endif
extern "C" {
int SMS3DS_ShineShells = SMS3DS_SHINE_SHELLS_DEFAULT;
}

// The value settings.txt gives `name` (a line "name = value"), into out; false when it gives none.
static bool setting_value(const char* path, const char* name, char* out, size_t outSize)
{
	FILE* f = fopen(path, "r");
	if (!f) return false;
	char line[256];
	const size_t n = strlen(name);
	bool found = false;
	while (!found && fgets(line, sizeof line, f)) {
		const char* p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (strncmp(p, name, n) != 0) continue;
		p += n;
		while (*p == ' ' || *p == '\t') p++;
		if (*p++ != '=') continue;
		while (*p == ' ' || *p == '\t') p++;
		found = *p && *p != '\n' && *p != '\r' && *p != '#';
		if (found) {
			size_t k = 0;
			while (p[k] && p[k] != '\n' && p[k] != '\r' && p[k] != ' ' && p[k] != '\t' && p[k] != '#' && k + 1 < outSize) {
				out[k] = p[k];
				k++;
			}
			out[k] = 0;
		}
	}
	fclose(f);
	return found;
}

// Whether settings.txt gives `name` a value (a line "name = value").
static bool setting_given(const char* path, const char* name)
{
	FILE* f = fopen(path, "r");
	if (!f) return false;
	char line[256];
	const size_t n = strlen(name);
	bool found = false;
	while (!found && fgets(line, sizeof line, f)) {
		const char* p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (strncmp(p, name, n) != 0) continue;
		p += n;
		while (*p == ' ' || *p == '\t') p++;
		if (*p++ != '=') continue;
		while (*p == ' ' || *p == '\t') p++;
		found = *p && *p != '\n' && *p != '\r' && *p != '#';
	}
	fclose(f);
	return found;
}

static void* game_thread(void*)
{
	static char a0[] = "sms3ds";
	static char a1[] = DISC_PATH;
	char* argv[] = {a0, a1, NULL};
	// Everything the game keeps lives under sdmc:/sms/, next to the disc image: the memory card
	// (sms-pc-port's card.cpp: SMS_SAVE_DIR), settings.txt (port_runtime.cpp: SMS_SETTINGS), and,
	// through the working directory, the relative lookups (mods/, mods/languages/,
	// mods/hd-cutscenes/). Without this they went to a PC-style ./.local/share/sms-port path.
	mkdir(SMS_DIR, 0777);
	chdir(SMS_DIR "/");
	setenv("SMS_SAVE_DIR", SMS_DIR "/card-a", 0);
	static const char kSettingsPath[] = SMS_DIR "/settings.txt";
	setenv("SMS_SETTINGS", kSettingsPath, 0);
	// settings.txt with the 3DS's defaults when there is none, so they can be seen and changed.
	if (FILE* f = fopen(kSettingsPath, "r")) {
		fclose(f);
	} else if ((f = fopen(kSettingsPath, "w"))) {
		fputs(kDefaultSettings, f);
		fclose(f);
		fprintf(stderr, "[3ds] wrote %s with the 3DS defaults\n", kSettingsPath);
	}
	// Gameplay at the game's own 30 fps unless settings.txt says otherwise. sms-pc-port defaults
	// to 60, which runs the game's logic twice as often: on a New 3DS that doubled the CPU work
	// per second of play and halved the speed of a game that already cannot keep up. (An
	// environment variable wins over settings.txt, so it is only set when the file has none.)
	if (!setting_given(kSettingsPath, "frame_rate")) setenv("SMS_FRAME_RATE", "30", 0);
	// No heat-wave shimmer unless settings.txt asks for it: a screen copy and a full-screen
	// distortion pass every frame in the sunny stages (Delfino Airstrip and Plaza among them).
	if (!setting_given(kSettingsPath, "heat_haze")) setenv("SMS_HEAT_HAZE", "0", 0);
	// Sound on the 3DS's DSP unless settings.txt says otherwise (audio = off, software or silent).
	// With audio = off, or when the DSP cannot be used (no sdmc:/3ds/dspfirm.cdc), MSound gets an
	// empty sound configuration and the AI DMA stays idle (sms-pc-port's noaudio.cpp,
	// SMS_NO_AUDIO): the game runs silently, as before there was a 3DS DSP backend. (sms-pc-port
	// also reads `audio` itself, as SMS_AUDIO: anything but off, or 0 set here for silent, leaves
	// its output on.)
	{
		char v[16];
		int mode = SMS3DS_AUDIO_DEFAULT, cycle = 0;
		// (autorun.txt's audio, for an unattended run, before settings.txt's)
		if (setting_value(SMS_DIR "/autorun.txt", "audio", v, sizeof v) || setting_value(kSettingsPath, "audio", v, sizeof v))
			mode = strcmp(v, "off") == 0 ? 0 : strcmp(v, "on") == 0 ? 1 : strcmp(v, "software") == 0 ? 2
			     : strcmp(v, "silent") == 0 ? 3 : SMS3DS_AUDIO_DEFAULT;
		if (setting_value(kSettingsPath, "audio_cycle", v, sizeof v)) cycle = atoi(v);
		if (!mode || !SMS3DS_AudioInit(mode, cycle)) {
			mode = 0;
			setenv("SMS_NO_AUDIO", "1", 0);
		} else if (mode == 3 && cycle <= 0) {
			setenv("SMS_AUDIO", "0", 0);  // no output: ai.cpp paces the AI DMA from the host clock
		}
		if (mode)
			SMS3DS_AudioSetUpdateInterval(setting_value(kSettingsPath, "audio_update_interval", v, sizeof v)
			                                  ? atoi(v) : SMS3DS_AUDIO_UPDATE_INTERVAL_DEFAULT);
		static const char* const kAudio[] = {"off", "on (the 3DS DSP)", "software mixer", "silent (no voice plays)"};
		fprintf(stderr, "[3ds] audio: %s (audio%s)\n", kAudio[mode], cycle > 0 && mode ? ", audio_cycle" : "");
	}
	{
		char v[16];
		if (setting_value(kSettingsPath, "shine_shadow_shells", v, sizeof v)) SMS3DS_ShineShells = atoi(v);
		fprintf(stderr, "[3ds] Delfino Plaza's Shine shadow: %d shell%s (shine_shadow_shells; 0: all 5)\n",
		        SMS3DS_ShineShells, SMS3DS_ShineShells == 1 ? "" : "s");
		int mips = SMS3DS_MIPS_DEFAULT;
		if (setting_value(kSettingsPath, "mipmaps", v, sizeof v)) mips = strcmp(v, "off") != 0;
		GX3DS_SetMips(mips);
		fprintf(stderr, "[3ds] small mipmapped textures: %s (mipmaps)\n", mips ? "with their mip levels" : "level 0 only");
		int blocks = SMS3DS_STATE_BLOCKS_DEFAULT;
		if (setting_value(kSettingsPath, "gx_state_blocks", v, sizeof v)) blocks = strcmp(v, "off") != 0;
		GXPC_SetStateBlocks(blocks);
		fprintf(stderr, "[3ds] GX state resets: %s (gx_state_blocks)\n", blocks ? "as precomputed blocks" : "as the game's calls");
		int fastLists = SMS3DS_FAST_STATE_LISTS_DEFAULT;
		if (setting_value(kSettingsPath, "gx_fast_state_lists", v, sizeof v)) fastLists = strcmp(v, "off") != 0;
		GXPC_SetFastStateLists(fastLists);
		fprintf(stderr, "[3ds] cached state-only display lists: %s (gx_fast_state_lists)\n",
		        fastLists ? "replayed as register words" : "replayed command by command");
		int nativeJ3D = SMS3DS_NATIVE_J3D_DEFAULT;
		if (setting_value(kSettingsPath, "native_j3d", v, sizeof v))
			nativeJ3D = strcmp(v, "on") == 0 ? 1 : strcmp(v, "check") == 0 ? 2 : strcmp(v, "verify") == 0 ? 3 : 0;
		char bufs[256] = SMS3DS_NATIVE_J3D_BUFFERS;
		setting_value(kSettingsPath, "native_j3d_buffers", bufs, sizeof bufs);
		GXPC_SetNativeJ3D(nativeJ3D, bufs);
		if (setting_value(kSettingsPath, "native_j3d_packages", v, sizeof v)) GXPC_J3DPackages = strcmp(v, "off") != 0;
		if (setting_value(kSettingsPath, "sea_precomputed", v, sizeof v)) SMS3DS_WaveMode = strcmp(v, "off") != 0;
		if (setting_value(kSettingsPath, "gx_state_block_ops", v, sizeof v)) GXPC_SetStateBlockOps(strcmp(v, "off") != 0);
		if (setting_value(kSettingsPath, "reset_textures_once", v, sizeof v)) SMS3DS_ResetTexOnce = strcmp(v, "off") != 0;
		if (setting_value(kSettingsPath, "gx_state_block_compiled", v, sizeof v)) GXPC_SetStateBlockCompiled(strcmp(v, "off") != 0);
		if (setting_value(kSettingsPath, "heap_flush", v, sizeof v)) pica_set_heap_flush(strcmp(v, "off") != 0);
		if (setting_value(kSettingsPath, "ticker_clipped", v, sizeof v)) SMS3DS_TelopClip = strcmp(v, "off") != 0;
		if (setting_value(kSettingsPath, "name_cache", v, sizeof v))
			SMS3DS_NameCache = strcmp(v, "off") == 0 ? 0 : strcmp(v, "verify") == 0 ? 2 : 1;
		if (setting_value(kSettingsPath, "perf_timers", v, sizeof v)) GX3DS_SetPerfTimers(strcmp(v, "on") == 0);
		if (setting_value(kSettingsPath, "draw_state_check", v, sizeof v)) GX3DS_SetDrawStateCheck(strcmp(v, "on") == 0);
		static const char kAutoPath[] = SMS_DIR "/autorun.txt";
		if (setting_value(kAutoPath, "boot_stage", v, sizeof v)) {
			int stage = -1, scenario = 0;
			if (sscanf(v, "%d-%d", &stage, &scenario) >= 1) {
				SMS3DS_DemoStage = stage;
				SMS3DS_DemoScenario = scenario;
			}
		}
		if (setting_value(kAutoPath, "skip_movies", v, sizeof v) && strcmp(v, "on") == 0)
			setenv("SMS_SKIP_MOVIES", "1", 1);
		if (setting_value(kAutoPath, "quit_after", v, sizeof v)) s_quitAfter = atoi(v);
		if (setting_value(kAutoPath, "timeline", v, sizeof v)) s_timelineAt = atoi(v);
		if (setting_value(kAutoPath, "perf_timers", v, sizeof v)) GX3DS_SetPerfTimers(strcmp(v, "on") == 0);
		if (setting_value(kAutoPath, "draw_state_check", v, sizeof v)) GX3DS_SetDrawStateCheck(strcmp(v, "on") == 0);
		if (setting_value(kAutoPath, "gx_detail", v, sizeof v) && strcmp(v, "on") == 0) GXPC_Detail = 1;
		if (setting_value(kAutoPath, "profile_auto", v, sizeof v)) s_profAuto = atoi(v);
		if (setting_value(kAutoPath, "profile_auto_quit", v, sizeof v)) s_profAutoQuit = strcmp(v, "on") == 0;
		if (SMS3DS_DemoStage >= 0 || s_profAuto > 0 || s_quitAfter > 0)
			fprintf(stderr,
			        "[3ds] autorun.txt: boot stage %d-%d, movies %s, the low touch's run after %d s of free play%s, "
			        "quit after %d s of it\n",
			        SMS3DS_DemoStage, SMS3DS_DemoScenario, getenv("SMS_SKIP_MOVIES") ? "skipped" : "played", s_profAuto,
			        s_profAutoQuit ? " (then quit)" : "", s_quitAfter);
		fprintf(stderr, "[3ds] the sea: %s (sea_precomputed)\n", SMS3DS_WaveMode ? "worked out once per column and row" : "vertex by vertex");
		if (setting_value(kSettingsPath, "profile_low_touch", v, sizeof v))
			SMS3DS_ProfSetLowTouch(strcmp(v, "objects") == 0 ? 5 : 4);
		fprintf(stderr, "[3ds] J3D drawn natively: %s, in %s, %s (native_j3d, native_j3d_buffers, native_j3d_packages)\n",
		        nativeJ3D == 1 ? "on" : nativeJ3D == 2 ? "check" : nativeJ3D == 3 ? "verify" : "off", bufs,
		        GXPC_J3DPackages ? "compiled shapes" : "no compiled shapes");
	}
	g_phase = 1;
	port_init(2, argv);
	g_phase = 2;
	fprintf(stderr, "[3ds] entering SMS_main\n");
	SMS_main();
	fprintf(stderr, "[3ds] SMS_main returned\n");
	g_phase = 3;
	return NULL;
}

// The bottom screen's status text. Drawing it takes tens of milliseconds (the console draws
// every glyph with the CPU): on the game's core, at the main thread's priority, each redraw held
// the game thread that long, twice a second. It runs on core 1 (status_thread) when it can.
// The guard after MEM1 (patch 0030): logged when its changed words change.
extern "C" int port_mem1_guard_check(u32* first, u32* last, u32* count, u32 words[8]);
static void check_mem1_guard(uint32_t frame)
{
	static u32 lastCount;
	u32 first, last, count, w[8];
	if (!port_mem1_guard_check(&first, &last, &count, w) || count == lastCount)
		return;
	lastCount = count;
	fprintf(stderr, "[heap] written past MEM1's end (frame %lu): %lu words changed, from byte %lu to byte %lu past the end; at the first: %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx\n",
	        (unsigned long)frame, (unsigned long)count, (unsigned long)first, (unsigned long)last, (unsigned long)w[0], (unsigned long)w[1],
	        (unsigned long)w[2], (unsigned long)w[3], (unsigned long)w[4], (unsigned long)w[5], (unsigned long)w[6], (unsigned long)w[7]);
	flush_log();
}
#if defined(SMS3DS_HEAPCHECK)
extern "C" void SMS3DS_HeapCheckPoll(uint32_t frame);  // heapcheck_3ds.cpp, HEAPCHECK=1 builds
#endif
static void draw_status(const uint32_t* s) {
	check_mem1_guard(s[0]);
#if defined(SMS3DS_HEAPCHECK)
	SMS3DS_HeapCheckPoll(s[0]);
#endif
	static const char* const phase[] = {"waiting", "port_init", "SMS_main (running)", "SMS_main returned"};
	printf("\x1b[8;0Hphase: %-22s\nframes %6lu  batches %7lu\nverts %8lu  tris %8lu\n", phase[g_phase],
	       (unsigned long)s[0], (unsigned long)s[1], (unsigned long)s[2], (unsigned long)s[3]);
	if (g_phase == 2) {
		SceneState sc = read_scene();
		printf("\nscene: %-9s area %2u-%-2u next %2u-%-2u\nmovie %u  director %08lx\n",
		       app_state_name(sc.state), sc.curr_stage, sc.curr_scen, sc.next_stage, sc.next_scen,
		       sc.movie, (unsigned long)sc.director_vtbl);
	}
	printf("\x1b[16;0Hdropped draws %lu\ncolours: %-14s\n", (unsigned long)pica_dropped(),
	       GX3DS_GetDebugColors() == 1 ? "debug (per draw)" : GX3DS_GetDebugColors() == 2 ? "opaque" : "material");
	uint32_t cs[3];
	GX3DS_GetCopyStats(cs);
	printf("draws: %3lu before copy, %3lu after\ntexture copies/frame: %lu  \n", (unsigned long)cs[0],
	       (unsigned long)cs[1], (unsigned long)cs[2]);
	uint32_t ts[3];
	GX3DS_GetTexStats(ts);
	{
		int tst[2];
		GX3DS_TraceState(tst);
		printf("\x1b[23;0Htrace: %-8s %d written (sms/trace-N.txt)  ", tst[0] ? "ARMED" : "idle", tst[1]);
		static const char* const profState[] = {"idle: bottom right, low=game", "timing the CPU...", "GPU, pixels cut...", "GPU, object by object...", "game thread runs..."};
		printf("\x1b[24;0Hprofile: %-28s", profState[SMS3DS_ProfState()]);
	}
	printf("\x1b[22;0Htop shows: %s %d of %d  (touch top: next)\n", pica_view() ? "EFB copy" : "frame",
	       pica_view(), pica_view_count());
	printf("\x1b[20;0Htextures: %lu uploads, %lu KB\nEFB-copy texture draws: %lu   \n", (unsigned long)ts[0],
	       (unsigned long)(ts[1] >> 10), (unsigned long)ts[2]);
	float pf[11];
	GX3DS_GetPerf(pf);
	// Renderer profile (ms per frame, averaged over 60 frames): see gx3ds_backend.cpp.
	printf("\x1b[24;0Hframe %5.1fms %4.1ffps dec%5.1f oth%5.1f\n"
	       "tex %4.1f tev %4.1f xform %4.1f sub %4.1f\n"
	       "copy %4.1f present %4.1f  %4.0f draws  \n",
	       pf[6], pf[6] > 0 ? 1000.0f / pf[6] : 0.0f, pf[10], pf[7], pf[0], pf[1], pf[2], pf[3], pf[4], pf[5], pf[8]);
	uint32_t ps2[2];
	GX3DS_GetPathStats(ps2);
	printf("transform: %s  gpu %lu cpu %lu   \n", GX3DS_GetGpuTransform() ? "GPU" : "CPU", (unsigned long)ps2[0],
	       (unsigned long)ps2[1]);
	// No gfxSwapBuffers: citro3d swaps the top screen, and the console's bottom screen is
	// single-buffered, so writing back the CPU cache is enough.
	gfxFlushBuffers();
}

static Thread s_statusThread;
static volatile bool s_statusRun = true;
static void status_thread(void*) {
	while (s_statusRun) {
		// Twice a second, in 50 ms steps, so a quit does not wait half a second for this thread.
		for (int i = 0; i < 10 && s_statusRun; i++) svcSleepThread(50000000LL);
		if (!s_statusRun) break;
		uint32_t s[7];
		GX3DS_GetStats(s);
		draw_status(s);
		static unsigned n;
		if (++n % 2 == 0) flush_log();  // the log's SD write, once a second, off the game's core too
	}
}

// Quitting: libctru's last step (__libctru_exit, weak) unmaps the linear heap and the app heap,
// then ends the process. The port's other threads are still alive then (the GX and render
// threads, the VI retrace timer, the game thread parked at PADRead), their stacks and data in
// those heaps, and one that woke in between faulted on its own stack (crash dumps 35 and 36: the
// render thread, then the retrace timer). When nothing runs in this process afterwards (no loader
// return address), the heaps are left to svcExitProcess, which stops every thread before it frees
// the process's memory; a loader that returns into the process gets them back first, as there.
extern "C" {
extern u32 __ctru_heap, __ctru_heap_size, __ctru_linear_heap, __ctru_linear_heap_size;
extern void (*__system_retAddr)(void);
void envDestroyHandles(void);
void __sync_fini(void);
void __libctru_exit(int rc)
{
	(void)rc;
	if (__system_retAddr) {
		u32 tmp = 0;
		svcControlMemory(&tmp, __ctru_linear_heap, 0, __ctru_linear_heap_size, MEMOP_FREE, (MemPerm)0);
		svcControlMemory(&tmp, __ctru_heap, 0, __ctru_heap_size, MEMOP_FREE, (MemPerm)0);
	}
	envDestroyHandles();
	__sync_fini();
	if (__system_retAddr) __system_retAddr();
	svcExitProcess();
}
}

int main()
{
	osSetSpeedupEnable(true);
	gfxInitDefault();
	// The top screen belongs to the renderer (platform/3ds/pica, citro3d on the game thread);
	// this status text lives on the bottom screen.
	consoleInit(GFX_BOTTOM, NULL);

	// Everything the port prints goes to the SD card (sdmc:/sms/log.txt).
	// The previous run's log is kept as log-prev.txt (FAT's rename does not replace a file).
	remove(SMS_DIR "/log-prev.txt");
	rename(LOG_PATH, SMS_DIR "/log-prev.txt");
	freopen(LOG_PATH, "w", stderr);
	// A 64 KB buffer, written out once a second by the main loop (flush_log). Unbuffered, every
	// line was its own SD write: slow on a New 3DS, and garbled when threads wrote at once. A
	// crash loses at most the last second (Luma's crash dump has the rest).
	static char logBuf[64 << 10];
	setvbuf(stderr, logBuf, _IOFBF, sizeof logBuf);

	bool isNew = false;
	APT_CheckNew3DS(&isNew);
	printf("SMS 3DS stage 2 (%s 3DS) %s %s\n", isNew ? "New" : "Old", __DATE__, __TIME__);
	fprintf(stderr, "[3ds] build %s %s\n", __DATE__, __TIME__);
	printf("heap %lu MB, linear free %lu MB\n", (unsigned long)(envGetHeapSize() >> 20),
	       (unsigned long)(linearSpaceFree() >> 20));
	printf("app mem free %lu MB\n", (unsigned long)(osGetMemRegionFree(MEMREGION_APPLICATION) >> 20));
	fprintf(stderr, "[3ds] %s 3DS, heap %lu MB, linear free %lu MB\n", isNew ? "New" : "Old",
	        (unsigned long)(envGetHeapSize() >> 20), (unsigned long)(linearSpaceFree() >> 20));
	fprintf(stderr, "[3ds] at quit: %s\n",
	        __system_retAddr ? "the loader returns into this process (the heaps handed back first)"
	                         : "the process ends (its heaps left to svcExitProcess)");
	printf("log: %s\nSTART+SELECT = exit, SELECT = colours\ntouch: top=view  bottom: L=trace M=GPU R=pixel test\n", LOG_PATH);

	// Holding B at launch starts on the CPU vertex path (as the BR touch toggle), to tell a GPU
	// lock-up in the GPU transform path from other hangs on hardware.
	// The first HID reads after launch can come back empty: watch the buttons for half a second.
	u32 launchHeld = 0;
	for (int i = 0; i < 30; i++) {
		hidScanInput();
		launchHeld |= hidKeysHeld();
		gspWaitForVBlank();
	}
	// Holding Y at launch waits for the GPU after every draw and logs the one it never finishes.
	if (launchHeld & KEY_Y) {
		pica_set_gpu_check(1);
		printf("Y held: GPU check after every draw (slow)\n");
		fprintf(stderr, "[3ds] Y held at launch: GPU check after every draw\n");
	}
	// Holding X at launch logs each new TEV setup, unsupported TEV pattern and EFB copy kind, and
	// writes the goop map readbacks to the SD card.
	if (launchHeld & KEY_X) {
		GX3DS_SetVerboseLog(1);
		pica_set_readback_dumps(1);
		printf("X held: detailed renderer log\n");
		fprintf(stderr, "[3ds] X held at launch: [tevdump]/[tev]/[copy] lines and goop map files on\n");
	}
	// Holding R at launch sends frames to the GPU in parts as they are made (experimental: GSP
	// stopped reporting GPU completions with it, at level loads).
	if (launchHeld & KEY_R) {
		pica_set_split(1);
		printf("R held: mid-frame GPU submission (experimental)\n");
		fprintf(stderr, "[3ds] R held at launch: mid-frame GPU submission on\n");
	} else {
		fprintf(stderr, "[3ds] mid-frame GPU submission off (hold R at launch to try it)\n");
	}
	// Holding L at launch draws everything, off-screen map pieces included (to compare).
	if (launchHeld & KEY_L) {
		GXPC_CullOff = 1;
		printf("L held: off-screen culling off\n");
		fprintf(stderr, "[3ds] L held at launch: off-screen culling off\n");
	}
	// Holding D-pad up at launch copies cached geometry into every frame again (resident
	// vertices off, to compare).
	if (launchHeld & KEY_DUP) {
		GX3DS_SetArena(0);
		pica_set_big_ring(1);  // every vertex copied each frame
		printf("Up held: resident vertices off\n");
		fprintf(stderr, "[3ds] Up held at launch: cached geometry copied every frame (no resident vertices)\n");
	}
	// Holding D-pad left at launch draws on the game thread, as before the render thread (to compare).
	if (launchHeld & KEY_DLEFT) {
		GX3DS_SetRenderThread(0);
		printf("Left held: no render thread\n");
		fprintf(stderr, "[3ds] Left held at launch: renderer on the game thread (no render thread)\n");
	}
	// Holding D-pad right at launch keeps GX command parsing on the game thread (no GX thread).
	if (launchHeld & KEY_DRIGHT) {
		GX3DS_SetGxThread(0);
		printf("Right held: GX on the game thread\n");
		fprintf(stderr, "[3ds] Right held at launch: GX thread off\n");
	}
	// Holding ZL at launch keeps every texture in FCRAM, as before VRAM textures (to compare).
	if (launchHeld & KEY_ZL) {
		pica_set_vram_textures(0);
		printf("ZL held: textures in FCRAM only\n");
		fprintf(stderr, "[3ds] ZL held at launch: textures in FCRAM only (no VRAM textures)\n");
	}
	// Holding ZR at launch checks VRAM texture copies against their originals ([vram] lines).
	if (launchHeld & KEY_ZR) {
		pica_set_vram_check(1);
		printf("ZR held: VRAM copy self-check\n");
		fprintf(stderr, "[3ds] ZR held at launch: VRAM texture copies checked against their originals\n");
	}
	// Holding START at launch sends every GPU setting again whenever a draw's state differs at
	// all, as before unchanged settings were skipped piece by piece (to compare).
	if (launchHeld & KEY_START) {
		pica_set_state_skip(0);
		printf("START held: GPU settings sent in full\n");
		fprintf(stderr, "[3ds] START held at launch: unchanged GPU settings sent again (no skipping)\n");
	}
	// Holding SELECT at launch sends every display-list call by address, as before copies of the
	// lists the game rewrites: each rewrite waits for the GX thread instead (to compare).
	if (launchHeld & KEY_SELECT) {
		GX3DS_SetListCopies(0);
		printf("SELECT held: display lists by address\n");
		fprintf(stderr, "[3ds] SELECT held at launch: display-list copies off (rewrites wait for the GX thread)\n");
	}
	// Circle pad left at launch: every GX register write is sent, as before the write filter
	// (gx_api.cpp, patch 0023) left out those that change nothing (to compare).
	if (launchHeld & KEY_CPAD_LEFT) {
		GXPC_SetWriteFilter(0);
		printf("Circle pad left: GX write filter off\n");
		fprintf(stderr, "[3ds] circle pad left at launch: GX write filter off (every register write sent)\n");
	}
	// Circle pad down at launch: the write filter's check mode (patch 0026): every write is sent, and
	// each one the filter would have left out is checked against the GX thread's register.
	if (launchHeld & KEY_CPAD_DOWN) {
		GXPC_SetWriteFilter(2);
		printf("Circle pad down: GX write filter check\n");
		fprintf(stderr, "[3ds] circle pad down at launch: GX write filter check mode (nothing left out, each skip checked)\n");
	}
	// Circle pad up at launch: resident display-list runs' indices are copied into the GPU ring each
	// frame, rebased to the batch (the render thread's copy, as before), not drawn where stored.
	if (launchHeld & KEY_CPAD_UP) {
		GX3DS_SetResidentIndicesInPlace(0);
		printf("Circle pad up: resident indices copied\n");
		fprintf(stderr, "[3ds] circle pad up at launch: resident indices copied into the ring, rebased (not in place)\n");
	}
	// Circle pad right at launch: resident display-list runs keep their duplicate vertices (each
	// stored as recorded), as before they were merged (to compare).
	if (launchHeld & KEY_CPAD_RIGHT) {
		GX3DS_SetMergeDuplicates(0);
		printf("Circle pad right: duplicate vertices kept\n");
		fprintf(stderr, "[3ds] circle pad right at launch: resident runs keep their duplicate vertices (not merged)\n");
	}
	// A held through launch: resident runs keep their triangles in the order recorded (merged runs
	// are otherwise reordered for the vertex cache where order makes no difference).
	if (launchHeld & KEY_A) {
		GX3DS_SetReorder(0);
		printf("A held: triangle order kept\n");
		fprintf(stderr, "[3ds] A held at launch: resident runs keep their triangle order (not reordered)\n");
	}
	if (launchHeld & KEY_B) {
		GX3DS_SetGpuTransform(0);
		pica_set_big_ring(1);  // CPU-transformed vertices are larger
		printf("B held: CPU vertex path\n");
		fprintf(stderr, "[3ds] B held at launch: starting on the CPU vertex path\n");
	}

	{
		// What one profiler clock read costs (svcGetSystemTick is a kernel call): the [perf]
		// timers make several per draw and two per decoded primitive.
		const u64 t0 = svcGetSystemTick();
		for (int i = 0; i < 10000; i++) (void)svcGetSystemTick();
		const u64 t1 = svcGetSystemTick();
		fprintf(stderr, "[perf] one svcGetSystemTick: %.0f ns\n", double(t1 - t0) / 10000.0 / 0.268111856);
	}
	printf("GPU self-test... ");
	const bool gpuOk = pica_selftest() != 0;
	printf(gpuOk ? "ok\n" : "HUNG: see log; game not started\n");
	// Touching the screen at launch times what a vertex costs the GPU under different vertex
	// formats and memory ([vbench] lines, about a second).
	if (gpuOk && (launchHeld & KEY_TOUCH)) {
		printf("GPU vertex benchmark... ");
		pica_vertex_bench();
		printf("done\n");
	}
	// Overlapped frames (after the self-test, which runs one frame at a time): the CPU builds
	// the next frame while the GPU draws the last. D-pad down at launch keeps them apart.
	if (launchHeld & KEY_DDOWN) {
		printf("Down held: frames not overlapped\n");
		fprintf(stderr, "[3ds] Down held at launch: CPU and GPU frames not overlapped\n");
	} else if (!(launchHeld & (KEY_R | KEY_Y))) {
		pica_set_overlap(1);
	}
	flush_log();

	{
		char v[16];
		if (setting_value(SMS_DIR "/autorun.txt", "game_thread_core", v, sizeof v)) s_gameCore = atoi(v);
		if (setting_value(SMS_DIR "/autorun.txt", "gx_thread_core", v, sizeof v)) pica_gx_core = atoi(v);
	}
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, 4u << 20);  // the game's call chains are deep
	pthread_t th;
	if (!gpuOk) {
	} else if (s_gameCore >= 0) {
		if (!threadCreate([](void* a) { game_thread(a); }, NULL, 4u << 20, 0x3F, s_gameCore, true)) {
			printf("cannot start the game thread\n");
			fprintf(stderr, "[3ds] threadCreate on core %d failed\n", s_gameCore);
		} else {
			fprintf(stderr, "[3ds] the game thread on core %d (autorun.txt game_thread_core)\n", s_gameCore);
		}
		// A hung GPU would only stop the game at its first frame again.
	} else if (pthread_create(&th, &attr, game_thread, NULL) != 0) {
		printf("cannot start the game thread\n");
		fprintf(stderr, "[3ds] pthread_create failed\n");
	}
	pthread_attr_destroy(&attr);
	// Status text on core 1 (the system core lends apps up to 80% when asked), below everything
	// else there; on the game's core only when core 1 is not available.
	if (R_SUCCEEDED(APT_SetAppCpuTimeLimit(80)))
		s_statusThread = threadCreate(status_thread, NULL, 32u << 10, 0x3F, 1, false);
	fprintf(stderr, "[3ds] status screen %s\n", s_statusThread ? "redrawn on core 1" : "redrawn on the game's core");

	while (aptMainLoop()) {
		// Edges are detected on the held state (the game's PADRead once scanned HID too, consuming
		// presses before hidKeysDown() saw them). START alone is the game's Start button, so quitting
		// needs START+SELECT.
		hidScanInput();
		pad3ds_poll();  // the game's PADRead takes this reading (pad_3ds.c)
		static u32 prevHeld = launchHeld;  // buttons held through launch are not presses
		u32 held = hidKeysHeld(), pressed = held & ~prevHeld, released = prevHeld & ~held;
		prevHeld = held;
		if ((held & (KEY_START | KEY_SELECT)) == (KEY_START | KEY_SELECT))
			break;
		// Unattended runs (autorun.txt), checked once a second. Free play: the stage's director in its
		// play state (TMarDirector state 4) drawing a scene (100 or more draw batches a frame; an
		// arrival movie draws a few quads). The low touch's run starts after profile_auto seconds of
		// it, and the app quits once that run is done or after quit_after seconds of it.
		if ((s_profAuto > 0 || s_profAutoQuit || s_quitAfter > 0) && g_phase == 2) {
			static unsigned autoTick, autoSecs, playSecs, autoFrames0, autoBatches0;
			static bool autoStarted;
			if (++autoTick % 60 == 0) {
				const SceneState sc = read_scene();
				uint32_t st[7];
				GX3DS_GetStats(st);
				const uint32_t df = st[0] - autoFrames0, db = st[1] - autoBatches0;
				autoFrames0 = st[0];
				autoBatches0 = st[1];
				const bool freePlay = sc.state == 5 && sc.mar_state == 4 && df && db >= 100 * df;  // (THP plays as world textures)
				if (freePlay) playSecs++;
				// The frame timeline (autorun.txt timeline): phases timed for a sixth of a second
				// after that many seconds of free play, then its report.
				static int timelineState;
				if (s_timelineAt > 0 && timelineState == 0 && playSecs >= (unsigned)s_timelineAt && !SMS3DS_ProfOn) {
					SMS3DS_ProfOn = 2;
					SMS3DS_TimelineOn = 1;
					timelineState = 1;
				} else if (timelineState == 1) {
					SMS3DS_TimelineOn = 0;
					SMS3DS_ProfOn = 0;
					SMS3DS_TimelineReport();
					timelineState = 2;
				}
				if (s_quitAfter > 0 && playSecs >= (unsigned)s_quitAfter) {
					fprintf(stderr, "[3ds] autorun: %u s of free play, quitting\n", playSecs);
					break;
				}
				if (s_profAuto > 0 && !autoStarted) {
					autoSecs = freePlay ? autoSecs + 1 : 0;
					if (autoSecs >= (unsigned)s_profAuto) {
						autoStarted = true;
						fprintf(stderr, "[3ds] autorun: the low touch's run starts after %u s of gameplay\n", autoSecs);
						SMS3DS_ProfRequestGame();
					}
				} else if (s_profAutoQuit && SMS3DS_ProfRunsDone) {  // also after a run started by touch
					fprintf(stderr, "[3ds] autorun: the run is done, quitting\n");
					break;
				}
			}
		}
		// SELECT on its own cycles the colours (material, debug, opaque), on its release, so the
		// first half of a START+SELECT quit pressed SELECT first does not switch them.
		static bool selectAlone;
		if (pressed & KEY_SELECT) selectAlone = true;
		if (held & KEY_START) selectAlone = false;
		if ((released & KEY_SELECT) && selectAlone) {
			selectAlone = false;
			GX3DS_SetDebugColors((GX3DS_GetDebugColors() + 1) % 3);
		}
		// Debug input stays on the touch screen, which the game does not use (its buttons all
		// reach the game through pad_3ds.c): the top half cycles the top screen through the last
		// frame's EFB copies, the bottom half logs the next frame's draws to sdmc:/sms/log.txt.
		if (pressed & KEY_TOUCH) {
			touchPosition tp;
			hidTouchRead(&tp);
			fprintf(stderr, "[touch] %d,%d: %s\n", tp.px, tp.py,
			        tp.py < 120 ? "top half (view)" : tp.px < 107 ? "bottom left (trace)" :
			        tp.px < 213 ? "bottom middle (GPU transform)" :
			        tp.py < 180 ? "bottom right (profile run)" : "bottom right, low (game thread by object)");
			if (tp.py < 120)
				pica_cycle_view();
			else if (tp.px < 107)
				GX3DS_TraceNextFrame();
			else if (tp.px < 213)
				GX3DS_SetGpuTransform(!GX3DS_GetGpuTransform());
			else if (tp.py < 180)
				SMS3DS_ProfRequest();  // [prof] lines: CPU by phase and class, then GPU pixels vs vertices
			else
				SMS3DS_ProfRequestGame();  // [prof] lines: the game thread by object (prof_3ds.cpp, part 5)
		}
		uint32_t s[7];
		GX3DS_GetStats(s);
		if (g_phase == 2) {
			log_scene(s[0], s[1]);
			check_stall(s[0], s[1]);
		}
		// The status text is redrawn twice a second, not every vblank: this thread shares the
		// game thread's core, and redrawing it 60 times a second took a large share of the
		// frame on a New 3DS. Only the redraw is timed, for the [perf] "status screen" figure.
		static unsigned statusTick;
		if (s_statusThread || ++statusTick % 30 != 0) {
			gspWaitForVBlank();
			continue;
		}
		const u64 busy0 = svcGetSystemTick();
		draw_status(s);
		GX3DS_MainBusyTicks += svcGetSystemTick() - busy0;
		static u64 lastFlush;
		if (osGetTime() - lastFlush >= 1000) {
			lastFlush = osGetTime();
			flush_log();
		}
		gspWaitForVBlank();
	}

	if (s_statusThread) {
		s_statusRun = false;
		threadJoin(s_statusThread, U64_MAX);
		threadFree(s_statusThread);
	}
	fprintf(stderr, "[3ds] exiting\n");
	flush_log();
	// The game thread may be mid-frame: stop it submitting GPU work and let what is in flight
	// finish, so gfxExit does not free the screens under a running display transfer (a quit
	// that sometimes did not return cleanly).
	// Park the game thread at its next PADRead (once a frame; a slow frame is ~200 ms), so it is not
	// running game or GX code while the services and the heap go away.
	pad3ds_quit = 1;
	for (int i = 0; i < 40 && !pad3ds_parked; i++) svcSleepThread(10000000LL);  // at most 400 ms
	pica_request_stop();
	SMS3DS_AudioExit();  // the DSP stops playing what it was given
	svcSleepThread(100000000LL);
	// _exit, not a return from main: exit's atexit handlers and static destructors can wait on
	// the game thread (its locks, its emulated OS), and a stuck game thread then kept the
	// console from quitting. _exit still runs libctru's service cleanup.
	gfxExit();
	_exit(0);
}
