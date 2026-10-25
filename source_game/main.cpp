// Super Mario Sunshine 3DS - stage 2 boot attempt (MODE=game).
//
// Runs the real game (decomp + sms-pc-port platform layer) on a big-stack thread and shows
// progress on the bottom screen. Graphics are counted, not drawn yet (gx3ds_backend.cpp).
// Everything the port logs (port_log -> stderr) goes to sdmc:/sms/log.txt.
#include <3ds.h>
#include "pad_3ds.h"
#include <malloc.h>
#include <new>
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
// as settings.txt does, gx_detail (on: the GX thread's commands timed, as in profile runs),
// gx_zones (on: the GX thread's zones sampled, in a GX_ZONES=1 build), status_screen (off: the
// bottom screen not redrawn) and profile_steps (gpu: the profile run's windows are the GPU's
// ceilings; reverts: parts of this build put back as before, one window each; cache: the frame's
// cache work three ways; fixed: the GPU's fixed work by part; trades: quality trade-offs' ceilings;
// lite: the lite setting off and on in turn; liteplus: more trims on top of it). It may also give
// lite (on), as settings.txt does.
static int s_profAuto, s_profAutoQuit, s_quitAfter, s_timelineAt;
// lite = plus: Bianco Hills' draw distance (lite_stage_draw_distance). In Azahar at its spawn view, of
// ~90K vertices a frame 20000 leaves out ~31K (mostly behind the hills nearby), 15000 ~43K, 25000 ~21K;
// 10000 and less showed in wide views.
static const int kPlusBiancoFar = 20000;
// autorun.txt buttons = <seconds>:<keys>[:<hold seconds>],... (keys A B X Y L R ZL ZR START SELECT
// UP DOWN LEFT RIGHT, the Circle Pad pushed all the way SUP SDOWN SLEFT SRIGHT, the C-stick CUP CDOWN
// CLEFT CRIGHT, joined by '+'): those held for 0.25 s (or the hold given) from that many seconds
// after launch (pad_3ds.c, pad3ds_script). shots = <seconds>,...: the frame drawn then saved as sdmc:/sms/shot-<tenths of a
// second>.ppm (pica_draw.c).
struct ScriptPress {
	float at, hold;
	u32 keys;
};
static ScriptPress s_press[64];
static int s_nPress;
static float s_shotAt[64];
static int s_nShot, s_shotNext;
static float s_newStageAt[16];  // autorun.txt new_stage_at = <seconds>,...: the renderer told a new stage loads (testing)
static int s_nNewStage, s_newStageNext;
static float s_memTestAt = -1.0f;  // autorun.txt mem_test = <seconds>: the heap run out then (the reserve's path, testing)
static float s_cpuPathAt = -1.0f;  // autorun.txt cpu_path_at = <seconds>: the CPU vertex path from then on (comparing the paths)
static float s_traceAt = -1.0f;  // autorun.txt trace_at = <seconds>: that frame's GX commands traced (sdmc:/sms/trace-N.txt)
static float s_goopDumpAt = -1.0f;  // autorun.txt goop_dump_at = <seconds>: each map's next readback written (sdmc:/sms/goop-<address>.pgm)
extern "C" void pica_request_goop_dumps(void);  // pica_draw.c
static u64 s_launchMs;
extern "C" void pica_request_shot(int id);  // pica_draw.c
static u32 script_keys(const char* s)
{
	static const struct {
		const char* name;
		u32 key;
	} kKeys[] = {{"A", KEY_A},         {"B", KEY_B},          {"X", KEY_X},         {"Y", KEY_Y},
	             {"L", KEY_L},         {"R", KEY_R},          {"ZL", KEY_ZL},       {"ZR", KEY_ZR},
	             {"START", KEY_START}, {"SELECT", KEY_SELECT},
	             {"UP", KEY_DUP},      {"DOWN", KEY_DDOWN},   {"LEFT", KEY_DLEFT},  {"RIGHT", KEY_DRIGHT},
	             {"SUP", KEY_CPAD_UP}, {"SDOWN", KEY_CPAD_DOWN}, {"SLEFT", KEY_CPAD_LEFT}, {"SRIGHT", KEY_CPAD_RIGHT},
	             {"CUP", KEY_CSTICK_UP}, {"CDOWN", KEY_CSTICK_DOWN}, {"CLEFT", KEY_CSTICK_LEFT}, {"CRIGHT", KEY_CSTICK_RIGHT}};
	u32 keys = 0;
	while (*s) {
		size_t n = strcspn(s, "+");
		for (const auto& k : kKeys)
			if (strlen(k.name) == n && strncmp(s, k.name, n) == 0) keys |= k.key;
		s += n;
		if (*s == '+') s++;
	}
	return keys;
}
static bool s_statusDraw = true;  // autorun.txt status_screen = off: the bottom screen not redrawn
// settings.txt (or autorun.txt) debug_input = on: the touch screen's, SELECT's and the launch holds'
// debug and test modes (main). Off by default, so that a player's touch or held button changes nothing.
static bool s_debugInput;
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
void GX3DS_SetSkipTev(unsigned id, unsigned to);  // gx3ds_backend.cpp: autorun.txt skip_tev (debugging)
void GX3DS_NewStage(void);  // gx3ds_backend.cpp: a new stage loads (the renderer's caches of the last go)
void SMS3DS_LiteStage(int stage);  // prof_3ds.cpp: the stage that loads (its draw distance with lite on)
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
// autorun.txt flags_on = <flag>,...: those game flags kept set (testing: Delfino Plaza's gates
// open without a save that has them, flags 0x10385 to 0x10387; TModelGate::loadAfter).
class TFlagManager {
public:
	static TFlagManager* smInstance;
	void setBool(bool value, u32 flag);
};
static u32 s_flagsOn[16];
static int s_nFlagsOn;

static void log_scene(uint32_t frames, uint32_t batches)
{
	if (TFlagManager* fm = TFlagManager::smInstance)
		for (int i = 0; i < s_nFlagsOn; i++) fm->setBool(true, s_flagsOn[i]);
	static uint32_t last_frames, last_batches;
	static SceneState last;
	static bool have_last;
	static u64 last_ms;
	SceneState s = read_scene();
	u64 now      = osGetTime();
	bool changed = !have_last || memcmp(&s, &last, sizeof s) != 0;
	// Another stage or scenario: a new stage loads, and the display lists cached for the last one
	// go (the stages left behind filled the heap: crash dump 41).
	if (have_last && (s.curr_stage != last.curr_stage || s.curr_scen != last.curr_scen)) GX3DS_NewStage();
	if (!have_last || s.curr_stage != last.curr_stage) SMS3DS_LiteStage(s.curr_stage);  // (its draw distance)
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
    "# Where JAudio's sequencer and voice updates run: game (JAudio's own audio thread, taking turns\n"
    "# with the game thread on its core) or core2 (a thread on core 2, the game's sound code kept apart\n"
    "# from it by a lock, as the GameCube kept them apart by interrupts).\n"
    "# audio_thread = game\n"
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
    "# Materials the renderer's stage-by-stage translation gets wrong (NPCs' colours, the Shine\n"
    "# Gate's graffiti, the hot Chain Chomp) compiled through their algebra and checked against the\n"
    "# GameCube's arithmetic (on), or drawn as translated (off).\n"
    "tev_compiler = on\n"
    "\n"
    "# Draws with projective texgens (the sea's and Bowser's hot tub's lookups of the screen) on the\n"
    "# GPU path through a second shader (on), or on the CPU path (off). Not working yet: its first\n"
    "# try locked a New 3DS's GPU up at the title screen, and the title's sky still comes out black.\n"
    "# gpu_projective = off\n"
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
    "# The port's debug and test controls (on): touching the screen (the top half shows the frame's\n"
    "# copies on the top screen; the bottom half traces a frame, switches between the GPU and CPU\n"
    "# vertex paths or starts a profile run), L + SELECT (cycles the colours) and buttons held at\n"
    "# launch (test modes, named in the log). Off: the port's only controls are SELECT (R's press)\n"
    "# and START + SELECT (quit).\n"
    "debug_input = off\n"
    "\n"
    "# The R button as a soft press (soft: FLUDD sprays while Mario moves) or a hard one (hard: Mario\n"
    "# stops to aim, as the GameCube's R pushed all the way). SELECT switches between them in play.\n"
    "# ZR is always the soft press.\n"
    "r_press = hard\n"
    "\n"
    "# What touching the bottom right of the bottom screen low down runs (debug_input = on): the game\n"
    "# thread's steps (steps) or the game thread by object (objects). Higher up it starts the whole\n"
    "# profile run.\n"
    "profile_low_touch = steps\n";

// Small mipmapped textures with their mip levels (GX3DS_SetMips; settings.txt mipmaps).
#ifndef SMS3DS_MIPS_DEFAULT
#define SMS3DS_MIPS_DEFAULT 1
#endif
#ifndef SMS3DS_ETC1_DEFAULT  // etc1_textures when settings.txt does not say (Makefile ETC1=1: on)
#define SMS3DS_ETC1_DEFAULT 0
#endif
extern "C" void GX3DS_SetMips(int on);
extern "C" void GX3DS_SetEtc1(int on);  // gx3ds_backend.cpp: settings.txt etc1_textures
extern "C" void GX3DS_SetTevCompiler(int on);  // gx3ds_backend.cpp: settings.txt tev_compiler
extern "C" int pica_set_projective_shader(int on);  // pica_draw.c: settings.txt gpu_projective
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
extern "C" int SMS3DS_RankCheck;     // decomp patch 3ds-0031: the scene sound's reused distances checked
extern "C" void GX3DS_StartZoneSampler(void);  // gx3ds_backend.cpp: autorun.txt gx_zones
extern "C" void SMS3DS_ProfGpuSteps(void);      // prof_3ds.cpp: autorun.txt profile_steps = gpu
extern "C" void SMS3DS_ProfRevertSteps(void);   // ... profile_steps = reverts
extern "C" void SMS3DS_ProfCacheSteps(void);    // ... profile_steps = cache
extern "C" void SMS3DS_ProfFixedSteps(void);    // ... profile_steps = fixed
extern "C" void SMS3DS_ProfTradeSteps(void);    // ... profile_steps = trades
extern "C" void SMS3DS_ProfLiteSteps(void);     // ... profile_steps = lite
extern "C" void SMS3DS_SetLite(int on, int drawDist, int shadowDist);  // prof_3ds.cpp: settings.txt lite
extern "C" void SMS3DS_SetStageFar(int stage, int dist);  // ... lite_stage_draw_distance
extern "C" void SMS3DS_SetLiteParts(int particles, int seaStep, int actorDist, int shineOff);
extern "C" void SMS3DS_SetLiteMantaShadows(int n);  // prof_3ds.cpp: settings.txt lite_manta_shadows
extern "C" void SMS3DS_ProfLitePlusSteps(void);  // ... profile_steps = liteplus
extern "C" void SMS3DS_ProfLightingSteps(void);  // ... profile_steps = lighting
extern "C" void GX3DS_SetGxPoll(int on);        // gx3ds_backend.cpp: settings.txt gx_poll
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
extern "C" void SMS3DS_AudioSetThread(int core2);
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
		// (autorun.txt's audio_thread, for an unattended run, before settings.txt's)
		if (mode && (setting_value(SMS_DIR "/autorun.txt", "audio_thread", v, sizeof v) ||
		             setting_value(kSettingsPath, "audio_thread", v, sizeof v)))
			SMS3DS_AudioSetThread(strcmp(v, "core2") == 0);
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
		if (setting_value(kSettingsPath, "gx_poll", v, sizeof v)) GX3DS_SetGxPoll(strcmp(v, "off") != 0);
		// lite = on: console-only quality trades for frame rate (prof_3ds.cpp, SMS3DS_SetLite): shadows near
		// the camera only (lite_shadow_distance, default 2000; 0: Mario's and Yoshi's only), one particle
		// in lite_particles (default 2),
		// nothing drawn past lite_draw_distance (default 4000; 0: no limit), no mirror; and if given, NPCs and objects
		// clipped past lite_actor_distance, the sea's grid every lite_sea_step, lite_shine_shadow = off.
		// lite = plus: more of those, as measured on a New 3DS in Delfino Plaza (~36 -> ~33 ms a frame on
		// top of lite): Mario's and Yoshi's shadows only, one particle in 4, the sea's grid every 2, the
		// Shine's shadow not drawn, and no draw distance limit, NPCs and objects clipped as the game does
		// (shorter ones looked unfinished, and moving through Plaza they saved nothing measurable). Each
		// lite_* key still overrides (lite_draw_distance = 0: no limit). autorun.txt's lite before
		// settings.txt's.
		if (setting_value(SMS_DIR "/autorun.txt", "lite", v, sizeof v) || setting_value(kSettingsPath, "lite", v, sizeof v)) {
			const bool plus = strcmp(v, "plus") == 0, on = plus || strcmp(v, "on") == 0;
			const int dd = setting_value(kSettingsPath, "lite_draw_distance", v, sizeof v) ? atoi(v) : plus ? 0 : -1;
			const int sd = setting_value(kSettingsPath, "lite_shadow_distance", v, sizeof v) ? atoi(v) : plus ? 0 : -1;
			const int pn = setting_value(kSettingsPath, "lite_particles", v, sizeof v) ? atoi(v) : plus ? 4 : -1;
			const int ss = setting_value(kSettingsPath, "lite_sea_step", v, sizeof v) ? atoi(v) : plus ? 2 : -1;
			const int ad = setting_value(kSettingsPath, "lite_actor_distance", v, sizeof v) ? atoi(v) : -1;
			const int so = setting_value(kSettingsPath, "lite_shine_shadow", v, sizeof v) ? strcmp(v, "off") == 0 : plus ? 1 : -1;
			SMS3DS_SetLiteParts(pn, ss, ad, so);
			// lite_manta_shadows = N: the Phantamanta's shadow volumes (Sirena Beach) for only the N mantas
			// nearest Mario (default 8; 0: all of them, as the game).
			SMS3DS_SetLiteMantaShadows(setting_value(kSettingsPath, "lite_manta_shadows", v, sizeof v) ? atoi(v) : -1);
			// lite_stage_draw_distance = <stage>:<distance>,...: a stage's own draw distance in place of
			// lite_draw_distance (0: no limit; stages as the log's [scene] area= gives them: 0 Delfino
			// Airstrip, 1 Delfino Plaza, 2 Bianco Hills, 3 Ricco Harbor). lite = plus gives Bianco Hills
			// kPlusBiancoFar: it lagged on a New 3DS (43 ms a frame, the GPU the limit). autorun.txt's first.
			if (plus) SMS3DS_SetStageFar(2, kPlusBiancoFar);
			char list[128];
			if (setting_value(SMS_DIR "/autorun.txt", "lite_stage_draw_distance", list, sizeof list) ||
			    setting_value(kSettingsPath, "lite_stage_draw_distance", list, sizeof list))
				for (char* p = strtok(list, ","); p; p = strtok(nullptr, ",")) {
					int stage, dist;
					if (sscanf(p, " %d : %d", &stage, &dist) == 2) SMS3DS_SetStageFar(stage, dist);
				}
			SMS3DS_SetLite(on, dd, sd);
		}
		if (setting_value(kSettingsPath, "draw_state_check", v, sizeof v)) {
			GX3DS_SetDrawStateCheck(strcmp(v, "on") == 0);
			SMS3DS_RankCheck = strcmp(v, "on") == 0;
		}
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
		// etc1_textures = on: CMPR textures as ETC1 (ETC1A4 with alpha): less memory, more of them in VRAM,
		// some loss of quality. Off by default (on in Makefile ETC1=1 builds); off turns it off there.
		{
			int etc1 = SMS3DS_ETC1_DEFAULT;
			if (setting_value(kAutoPath, "etc1_textures", v, sizeof v) || setting_value(kSettingsPath, "etc1_textures", v, sizeof v))
				etc1 = strcmp(v, "on") == 0;
			GX3DS_SetEtc1(etc1);
			fprintf(stderr, "[3ds] CMPR textures: %s (etc1_textures)\n", etc1 ? "as ETC1" : "as decoded (RGB565 or RGBA5551)");
		}
		// tev_compiler = off: materials drawn as the stage-by-stage translation gives them, never
		// compiled through their algebra (gx3ds_tevsym.cpp; on by default).
		{
			const bool tevc = !((setting_value(kAutoPath, "tev_compiler", v, sizeof v) ||
			                     setting_value(kSettingsPath, "tev_compiler", v, sizeof v)) &&
			                    strcmp(v, "off") == 0);
			GX3DS_SetTevCompiler(tevc);
			fprintf(stderr, "[3ds] materials the translation gets wrong: %s (tev_compiler)\n",
			        tevc ? "compiled through their algebra" : "drawn as translated");
		}
		// gpu_projective = on: draws with projective texgens (the screen-copy lookups of the sea and of
		// Bowser's hot tub's water) on the GPU path, through shaders/gx_proj.v.pica. Not working yet:
		// the first try of it locked a New 3DS's GPU up at the title screen, and in Azahar the title's
		// sky still comes out black. Off: the CPU path.
		if ((setting_value(kAutoPath, "gpu_projective", v, sizeof v) || setting_value(kSettingsPath, "gpu_projective", v, sizeof v)) &&
		    strcmp(v, "on") == 0) {
			const int ok = pica_set_projective_shader(1);
			fprintf(stderr, "[3ds] projective texgens: %s (gpu_projective)\n", ok ? "on the GPU path (gx_proj shader)" : "shader not loaded: CPU path");
		}
		// autorun.txt vram_textures = off: every texture drawn from FCRAM, as holding ZL at launch (testing).
		if (setting_value(kAutoPath, "vram_textures", v, sizeof v) && strcmp(v, "off") == 0) {
			pica_set_vram_textures(0);
			fprintf(stderr, "[3ds] autorun.txt: textures in FCRAM only (no VRAM textures)\n");
		}
		if (setting_value(kAutoPath, "quit_after", v, sizeof v)) s_quitAfter = atoi(v);
		// renderer_log = on: the renderer's once-per-kind [tevdump], [tev] and [copy] lines (as holding
		// X at launch, without the goop map files).
		if (setting_value(kAutoPath, "renderer_log", v, sizeof v) && strcmp(v, "on") == 0) GX3DS_SetVerboseLog(1);
		// cpu_path = on: every draw on the CPU vertex path, as B held at launch (comparing the paths)
		if (setting_value(kAutoPath, "cpu_path", v, sizeof v) && strcmp(v, "on") == 0) {
			GX3DS_SetGpuTransform(0);
			pica_set_big_ring(1);
		}
		// skip_tev = N or N-M: draws of the renderer log's TEV setups #N (to #M) left out (with
		// renderer_log; debugging)
		if (setting_value(kAutoPath, "skip_tev", v, sizeof v)) {
			unsigned a = 0, b = 0;
			if (sscanf(v, "%u-%u", &a, &b) < 2) b = a;
			GX3DS_SetSkipTev(a, b);
		}
		{
			char list[256];
			if (setting_value(kAutoPath, "buttons", list, sizeof list))
				for (char* p = strtok(list, ","); p && s_nPress < 64; p = strtok(nullptr, ",")) {
					char* c = strchr(p, ':');
					if (!c) continue;
					*c = 0;
					char* h = strchr(c + 1, ':');
					if (h) *h = 0;
					s_press[s_nPress++] = ScriptPress{float(atof(p)), h ? float(atof(h + 1)) : 0.25f, script_keys(c + 1)};
				}
			if (setting_value(kAutoPath, "shots", list, sizeof list))
				for (char* p = strtok(list, ","); p && s_nShot < 64; p = strtok(nullptr, ",")) s_shotAt[s_nShot++] = float(atof(p));
			if (setting_value(kAutoPath, "trace_at", list, sizeof list)) s_traceAt = float(atof(list));
			if (setting_value(kAutoPath, "goop_dump_at", list, sizeof list)) s_goopDumpAt = float(atof(list));
			if (setting_value(kAutoPath, "cpu_path_at", list, sizeof list)) s_cpuPathAt = float(atof(list));
			if (setting_value(kAutoPath, "mem_test", list, sizeof list)) s_memTestAt = float(atof(list));
			if (setting_value(kAutoPath, "new_stage_at", list, sizeof list))
				for (char* p = strtok(list, ","); p && s_nNewStage < 16; p = strtok(nullptr, ","))
					s_newStageAt[s_nNewStage++] = float(atof(p));
			if (setting_value(kAutoPath, "flags_on", list, sizeof list))
				for (char* p = strtok(list, ","); p && s_nFlagsOn < 16; p = strtok(nullptr, ","))
					s_flagsOn[s_nFlagsOn++] = (u32)strtoul(p, nullptr, 0);
			if (s_nPress || s_nShot)
				fprintf(stderr, "[3ds] autorun.txt: %d scripted button presses, %d screenshots\n", s_nPress, s_nShot);
		}
		if (setting_value(kAutoPath, "timeline", v, sizeof v)) s_timelineAt = atoi(v);
		if (setting_value(kAutoPath, "perf_timers", v, sizeof v)) GX3DS_SetPerfTimers(strcmp(v, "on") == 0);
		if (setting_value(kAutoPath, "draw_state_check", v, sizeof v)) {
			GX3DS_SetDrawStateCheck(strcmp(v, "on") == 0);
			SMS3DS_RankCheck = strcmp(v, "on") == 0;
		}
		if (setting_value(kAutoPath, "gx_detail", v, sizeof v) && strcmp(v, "on") == 0) GXPC_Detail = 1;
		if (setting_value(kAutoPath, "status_screen", v, sizeof v) && strcmp(v, "off") == 0) s_statusDraw = false;
		// gx_zones = on: the GX thread's zones sampled (a GX library built with GX_ZONES=1).
		if (setting_value(kAutoPath, "gx_zones", v, sizeof v) && strcmp(v, "on") == 0) GX3DS_StartZoneSampler();
		if (setting_value(kAutoPath, "profile_auto", v, sizeof v)) s_profAuto = atoi(v);
		// profile_steps = gpu: the profile run's windows are the GPU's ceilings (prof_3ds.cpp, kGpuSteps);
		// reverts: parts of this build put back as before, one window each (kRvSteps); cache: the frame's
		// cache work three ways (kCacheSteps); fixed: the GPU's fixed work by part (kFixedSteps).
		if (setting_value(kAutoPath, "profile_steps", v, sizeof v)) {
			if (strcmp(v, "gpu") == 0) SMS3DS_ProfGpuSteps();
			if (strcmp(v, "reverts") == 0) SMS3DS_ProfRevertSteps();
			if (strcmp(v, "cache") == 0) SMS3DS_ProfCacheSteps();
			if (strcmp(v, "fixed") == 0) SMS3DS_ProfFixedSteps();
			if (strcmp(v, "trades") == 0) SMS3DS_ProfTradeSteps();
			if (strcmp(v, "lite") == 0) SMS3DS_ProfLiteSteps();
			if (strcmp(v, "liteplus") == 0) SMS3DS_ProfLitePlusSteps();
			if (strcmp(v, "lighting") == 0) SMS3DS_ProfLightingSteps();
		}
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
		const int ps = SMS3DS_ProfState();
		printf("\x1b[24;0Hprofile: %-28s", ps == 0 && !s_debugInput ? "idle" : profState[ps]);
	}
	printf("\x1b[22;0Htop shows: %s %d of %d%s\n", pica_view() ? "EFB copy" : "frame", pica_view(),
	       pica_view_count(), s_debugInput ? "  (touch top: next)" : "");
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
	printf("\x1b[28;0HR: %s press (SELECT switches)", pad3ds_soft_r ? "soft" : "hard");
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
		if (sms3ds_suspended) continue;  // the HOME Menu is up: no drawing, no SD writes (core 1 is the system's)
		uint32_t s[7];
		GX3DS_GetStats(s);
		if (s_statusDraw) draw_status(s);
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
// A fatal error on another thread (abort(): OSPanic, the GX layer's fatal(), a thread that could
// not be made; or exit()) ends the process through libctru's __appExit with the main loop still
// running, and libctru's own version lost the error: abort() does not flush stdio, so its message
// was still in the log's buffer when the SD card was closed, and hidExit unmapped HID's shared
// memory under the main loop, whose next hidScanInput faulted (crash dump 41, on the way back to
// Delfino Plaza after Bianco Hills' first episode). Here the log is written first, with the
// thread and the code addresses on its stack, and HID stays mapped (the process's end frees it).
static u32 s_mainThreadId;
extern "C" char __start__[], __exidx_start[];  // 3dsx.ld: code and read-only data
extern "C" void __appExit(void)
{
	u32 id = 0;
	svcGetThreadId(&id, CUR_THREAD_HANDLE);
	if (id != s_mainThreadId) {
		fprintf(stderr, "[3ds] the process is ending from thread %lu, not the main thread (%lu): a fatal error there\n",
		        (unsigned long)id, (unsigned long)s_mainThreadId);
		// Words on this thread's stack that point into code or read-only data, nearest first: its
		// callers among them (build/elf-archive's ELF, addr2line).
		const u32* sp = (const u32*)__builtin_frame_address(0);
		const u32 lo = (u32)(uintptr_t)__start__, hi = (u32)(uintptr_t)__exidx_start;
		fprintf(stderr, "[3ds]   code addresses on its stack:");
		for (int i = 0, shown = 0; i < 256 && shown < 40; i++)
			if (sp[i] >= lo && sp[i] < hi && !(sp[i] & 3)) fprintf(stderr, " %08lx", (unsigned long)sp[i]), shown++;
		fputc('\n', stderr);
		dump_os_threads("at the fatal error");
	}
	// aptExit before the SD card is closed, so that the log says whether it returned (after a close
	// from the HOME Menu the process did not end, somewhere past graphics).
	fprintf(stderr, "[3ds] quit: aptExit\n");
	flush_log();
	aptExit();
	fprintf(stderr, "[3ds] quit: aptExit returned; the SD card and the service manager next\n");
	flush_log();
	archiveUnmountAll();
	fsExit();
	srvExit();
}

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

// Out of heap (an operator new with no room). After two hours of play a console's heap was taken to
// 85 of its 86 MB, 71 MB in use and the rest in pieces, and recording Pianta Village's display lists
// found no block for a buffer: std::bad_alloc ended the game. A reserve is held from the start: the
// first failure gives it back (the allocation is tried again, in its room), and the GX thread then
// empties its caches between two frames, as for a new stage, and takes the reserve again
// (SMS3DS_MemReserveSpent, SMS3DS_MemReserveRestore). Only with the reserve spent does the
// allocation fail as before (the process ends: the [3ds] exiting lines name the thread and its stack).
static const size_t kMemReserve = 4u << 20;
static void* s_memReserve;
static volatile int s_memReserveSpent;
static void out_of_memory()
{
	const struct mallinfo mi = mallinfo();
	if (s_memReserve) {
		free(s_memReserve);
		s_memReserve = nullptr;
		s_memReserveSpent = 1;
		fprintf(stderr, "[mem] out of memory (%lu KB in use, arena %lu of %lu KB): the %lu KB reserve given back, the caches "
		        "to be emptied\n", (unsigned long)mi.uordblks >> 10, (unsigned long)mi.arena >> 10,
		        (unsigned long)__ctru_heap_size >> 10, (unsigned long)(kMemReserve >> 10));
		return;  // (operator new tries again)
	}
	fprintf(stderr, "[mem] out of memory: %lu KB in use, arena %lu of %lu KB (the rest free in pieces too small), no reserve left\n",
	        (unsigned long)mi.uordblks >> 10, (unsigned long)mi.arena >> 10, (unsigned long)__ctru_heap_size >> 10);
	fflush(stderr);
	std::set_new_handler(nullptr);
}
// The GX thread (gx3ds_backend.cpp), between two frames: whether the reserve went, and taking it again
// once its caches are emptied.
extern "C" int SMS3DS_MemReserveSpent(void) { return s_memReserveSpent; }
extern "C" void SMS3DS_MemReserveRestore(void)
{
	if (s_memReserve) return;
	s_memReserve = malloc(kMemReserve);
	const struct mallinfo mi = mallinfo();
	fprintf(stderr, "[mem] caches emptied: %lu KB in use, arena %lu KB; the reserve %s\n", (unsigned long)mi.uordblks >> 10,
	        (unsigned long)mi.arena >> 10, s_memReserve ? "taken again" : "not taken again (no room): the next failure ends the game");
	if (s_memReserve) s_memReserveSpent = 0;
}

int main()
{
	std::set_new_handler(out_of_memory);
	s_memReserve = malloc(kMemReserve);
	s_launchMs = osGetTime();
	svcGetThreadId(&s_mainThreadId, CUR_THREAD_HANDLE);  // (__appExit: who ended the process)
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
	{
		char v[8];
		s_debugInput = (setting_value(SMS_DIR "/autorun.txt", "debug_input", v, sizeof v) ||
		                setting_value(SMS_DIR "/settings.txt", "debug_input", v, sizeof v)) &&
		               strcmp(v, "on") == 0;
	}
	fprintf(stderr, "[3ds] debug input: %s (debug_input)\n",
	        s_debugInput ? "on (the touch screen, L+SELECT, buttons held at launch)" : "off");
	{
		char v[8];
		pad3ds_soft_r = setting_value(SMS_DIR "/settings.txt", "r_press", v, sizeof v) && strcmp(v, "soft") == 0;
	}
	fprintf(stderr, "[3ds] R: %s press (r_press; SELECT switches)\n", pad3ds_soft_r ? "soft" : "hard");
	printf("log: %s\nSELECT: R soft/hard, START+SELECT: exit\n", LOG_PATH);
	if (s_debugInput) printf("L+SELECT = colours\ntouch: top=view  bottom: L=trace M=GPU R=profile\n");

	// Holding B at launch starts on the CPU vertex path (as the BR touch toggle), to tell a GPU
	// lock-up in the GPU transform path from other hangs on hardware.
	// The first HID reads after launch can come back empty: watch the buttons for half a second.
	// The test modes below are debug_input's; without it a button held through launch only isn't
	// taken as a press.
	u32 launchHeld = 0;
	for (int i = 0; i < 30; i++) {
		hidScanInput();
		launchHeld |= hidKeysHeld();
		gspWaitForVBlank();
	}
	const u32 testHeld = s_debugInput ? launchHeld : 0;
	// Holding Y at launch waits for the GPU after every draw and logs the one it never finishes.
	if (testHeld & KEY_Y) {
		pica_set_gpu_check(1);
		printf("Y held: GPU check after every draw (slow)\n");
		fprintf(stderr, "[3ds] Y held at launch: GPU check after every draw\n");
	}
	// Holding X at launch logs each new TEV setup, unsupported TEV pattern and EFB copy kind, and
	// writes the goop map readbacks to the SD card.
	if (testHeld & KEY_X) {
		GX3DS_SetVerboseLog(1);
		pica_set_readback_dumps(1);
		printf("X held: detailed renderer log\n");
		fprintf(stderr, "[3ds] X held at launch: [tevdump]/[tev]/[copy] lines and goop map files on\n");
	}
	// Holding R at launch sends frames to the GPU in parts as they are made (experimental: GSP
	// stopped reporting GPU completions with it, at level loads).
	if (testHeld & KEY_R) {
		pica_set_split(1);
		printf("R held: mid-frame GPU submission (experimental)\n");
		fprintf(stderr, "[3ds] R held at launch: mid-frame GPU submission on\n");
	} else {
		fprintf(stderr, "[3ds] mid-frame GPU submission off (hold R at launch to try it)\n");
	}
	// Holding L at launch draws everything, off-screen map pieces included (to compare).
	if (testHeld & KEY_L) {
		GXPC_CullOff = 1;
		printf("L held: off-screen culling off\n");
		fprintf(stderr, "[3ds] L held at launch: off-screen culling off\n");
	}
	// Holding D-pad up at launch copies cached geometry into every frame again (resident
	// vertices off, to compare).
	if (testHeld & KEY_DUP) {
		GX3DS_SetArena(0);
		pica_set_big_ring(1);  // every vertex copied each frame
		printf("Up held: resident vertices off\n");
		fprintf(stderr, "[3ds] Up held at launch: cached geometry copied every frame (no resident vertices)\n");
	}
	// Holding D-pad left at launch draws on the game thread, as before the render thread (to compare).
	if (testHeld & KEY_DLEFT) {
		GX3DS_SetRenderThread(0);
		printf("Left held: no render thread\n");
		fprintf(stderr, "[3ds] Left held at launch: renderer on the game thread (no render thread)\n");
	}
	// Holding D-pad right at launch keeps GX command parsing on the game thread (no GX thread).
	if (testHeld & KEY_DRIGHT) {
		GX3DS_SetGxThread(0);
		printf("Right held: GX on the game thread\n");
		fprintf(stderr, "[3ds] Right held at launch: GX thread off\n");
	}
	// Holding ZL at launch keeps every texture in FCRAM, as before VRAM textures (to compare).
	if (testHeld & KEY_ZL) {
		pica_set_vram_textures(0);
		printf("ZL held: textures in FCRAM only\n");
		fprintf(stderr, "[3ds] ZL held at launch: textures in FCRAM only (no VRAM textures)\n");
	}
	// Holding ZR at launch checks VRAM texture copies against their originals ([vram] lines).
	if (testHeld & KEY_ZR) {
		pica_set_vram_check(1);
		printf("ZR held: VRAM copy self-check\n");
		fprintf(stderr, "[3ds] ZR held at launch: VRAM texture copies checked against their originals\n");
	}
	// Holding START at launch sends every GPU setting again whenever a draw's state differs at
	// all, as before unchanged settings were skipped piece by piece (to compare).
	if (testHeld & KEY_START) {
		pica_set_state_skip(0);
		printf("START held: GPU settings sent in full\n");
		fprintf(stderr, "[3ds] START held at launch: unchanged GPU settings sent again (no skipping)\n");
	}
	// Holding SELECT at launch sends every display-list call by address, as before copies of the
	// lists the game rewrites: each rewrite waits for the GX thread instead (to compare).
	if (testHeld & KEY_SELECT) {
		GX3DS_SetListCopies(0);
		printf("SELECT held: display lists by address\n");
		fprintf(stderr, "[3ds] SELECT held at launch: display-list copies off (rewrites wait for the GX thread)\n");
	}
	// Circle pad left at launch: every GX register write is sent, as before the write filter
	// (gx_api.cpp, patch 0023) left out those that change nothing (to compare).
	if (testHeld & KEY_CPAD_LEFT) {
		GXPC_SetWriteFilter(0);
		printf("Circle pad left: GX write filter off\n");
		fprintf(stderr, "[3ds] circle pad left at launch: GX write filter off (every register write sent)\n");
	}
	// Circle pad down at launch: the write filter's check mode (patch 0026): every write is sent, and
	// each one the filter would have left out is checked against the GX thread's register.
	if (testHeld & KEY_CPAD_DOWN) {
		GXPC_SetWriteFilter(2);
		printf("Circle pad down: GX write filter check\n");
		fprintf(stderr, "[3ds] circle pad down at launch: GX write filter check mode (nothing left out, each skip checked)\n");
	}
	// Circle pad up at launch: resident display-list runs' indices are copied into the GPU ring each
	// frame, rebased to the batch (the render thread's copy, as before), not drawn where stored.
	if (testHeld & KEY_CPAD_UP) {
		GX3DS_SetResidentIndicesInPlace(0);
		printf("Circle pad up: resident indices copied\n");
		fprintf(stderr, "[3ds] circle pad up at launch: resident indices copied into the ring, rebased (not in place)\n");
	}
	// Circle pad right at launch: resident display-list runs keep their duplicate vertices (each
	// stored as recorded), as before they were merged (to compare).
	if (testHeld & KEY_CPAD_RIGHT) {
		GX3DS_SetMergeDuplicates(0);
		printf("Circle pad right: duplicate vertices kept\n");
		fprintf(stderr, "[3ds] circle pad right at launch: resident runs keep their duplicate vertices (not merged)\n");
	}
	// A held through launch: resident runs keep their triangles in the order recorded (merged runs
	// are otherwise reordered for the vertex cache where order makes no difference).
	if (testHeld & KEY_A) {
		GX3DS_SetReorder(0);
		printf("A held: triangle order kept\n");
		fprintf(stderr, "[3ds] A held at launch: resident runs keep their triangle order (not reordered)\n");
	}
	if (testHeld & KEY_B) {
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
	if (gpuOk && (testHeld & KEY_TOUCH)) {
		printf("GPU vertex benchmark... ");
		pica_vertex_bench();
		printf("done\n");
	}
	// Overlapped frames (after the self-test, which runs one frame at a time): the CPU builds
	// the next frame while the GPU draws the last. D-pad down at launch keeps them apart.
	if (testHeld & KEY_DDOWN) {
		printf("Down held: frames not overlapped\n");
		fprintf(stderr, "[3ds] Down held at launch: CPU and GPU frames not overlapped\n");
	} else if (!(testHeld & (KEY_R | KEY_Y))) {
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
		if (s_nPress || s_nShot || s_traceAt >= 0.0f || s_goopDumpAt >= 0.0f || s_cpuPathAt >= 0.0f || s_nNewStage) {  // autorun.txt timed keys
			const float t = float(osGetTime() - s_launchMs) / 1000.0f;
			while (s_newStageNext < s_nNewStage && t >= s_newStageAt[s_newStageNext]) s_newStageNext++, GX3DS_NewStage();
			if (s_cpuPathAt >= 0.0f && t >= s_cpuPathAt) {
				GX3DS_SetGpuTransform(0);
				s_cpuPathAt = -1.0f;
			}
			if (s_memTestAt >= 0.0f && t >= s_memTestAt) {
				s_memTestAt = -1.0f;
				// Megabytes taken until operator new finds no room (out_of_memory gives the reserve back
				// and the last one is placed in it), then all given back: the game goes on.
				static char* blocks[128];
				int n = 0;
				while (n < 128 && !s_memReserveSpent) blocks[n++] = new (std::nothrow) char[1u << 20];
				for (int i = 0; i < n; i++) delete[] blocks[i];
				fprintf(stderr, "[mem] test: %d MB taken before the heap ran out, given back\n", n);
			}
			if (s_traceAt >= 0.0f && t >= s_traceAt) {
				GX3DS_TraceNextFrame();
				s_traceAt = -1.0f;
			}
			if (s_goopDumpAt >= 0.0f && t >= s_goopDumpAt) {
				pica_request_goop_dumps();
				s_goopDumpAt = -1.0f;
			}
			u32 keys = 0;
			for (int i = 0; i < s_nPress; i++)
				if (t >= s_press[i].at && t < s_press[i].at + s_press[i].hold) keys |= s_press[i].keys;
			pad3ds_script = keys;
			while (s_shotNext < s_nShot && t >= s_shotAt[s_shotNext]) pica_request_shot(int(s_shotAt[s_shotNext++] * 10.0f + 0.5f));
		}
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
		// SELECT on its own switches R between a soft and a hard press (pad_3ds.c), on its release, so
		// the first half of a START+SELECT quit pressed SELECT first does not switch it. With
		// debug_input = on, L + SELECT cycles the colours (material, debug, opaque) instead.
		static bool selectAlone;
		if (pressed & KEY_SELECT) selectAlone = true;
		if (held & KEY_START) selectAlone = false;
		if ((released & KEY_SELECT) && selectAlone) {
			selectAlone = false;
			if (s_debugInput && (held & KEY_L)) {
				GX3DS_SetDebugColors((GX3DS_GetDebugColors() + 1) % 3);
			} else {
				pad3ds_soft_r = !pad3ds_soft_r;
				fprintf(stderr, "[pad] R: %s press (SELECT)\n", pad3ds_soft_r ? "soft" : "hard");
			}
		}
		// Debug input stays on the touch screen, which the game does not use (its buttons all
		// reach the game through pad_3ds.c): the top half cycles the top screen through the last
		// frame's EFB copies, the bottom half logs the next frame's draws to sdmc:/sms/log.txt.
		if ((pressed & KEY_TOUCH) && s_debugInput) {
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

	// Closed from the HOME Menu the app is still suspended here, and every wait below is bounded: an
	// unbounded one kept the process from ending, and the next app the HOME Menu started had a black
	// screen until the console was turned off. Each step is logged as it ends, to tell which one.
	const bool homeClose = aptShouldClose();
	fprintf(stderr, "[3ds] exiting (%s)\n", homeClose ? "closed from the HOME Menu" : "START+SELECT or autorun");
	flush_log();
	if (s_statusThread) {
		s_statusRun = false;
		if (R_SUCCEEDED(threadJoin(s_statusThread, 1000000000ull)))  // at most 1 s
			threadFree(s_statusThread);
		else
			fprintf(stderr, "[3ds] quit: the status thread did not stop within 1 s\n");
	}
	// The game thread may be mid-frame: stop it submitting GPU work and let what is in flight
	// finish, so gfxExit does not free the screens under a running display transfer (a quit
	// that sometimes did not return cleanly).
	// Park the game thread at its next PADRead (once a frame; a slow frame is ~200 ms), so it is not
	// running game or GX code while the services and the heap go away.
	pad3ds_quit = 1;
	for (int i = 0; i < 40 && !pad3ds_parked; i++) svcSleepThread(10000000LL);  // at most 400 ms
	fprintf(stderr, "[3ds] quit: the game thread %s\n", pad3ds_parked ? "stopped at PADRead" : "still out after 400 ms (blocked)");
	flush_log();
	// From here the threads that wake up on their own (the render thread's GPU waits, the retrace
	// timer) park in the kernel: as a .cia the system can take the process's memory away before
	// svcExitProcess stops them (crash dump 49). Parked, none runs again.
	sms3ds_ending = 1;
	pica_request_stop();
	SMS3DS_AudioExit();  // the DSP stops playing what it was given
	svcSleepThread(100000000LL);
	fprintf(stderr, "[3ds] ending: %d threads parked (the game thread, then any that woke on their own)\n",
	        sms3ds_parked_threads);
	flush_log();
	// _exit, not a return from main: exit's atexit handlers and static destructors can wait on
	// the game thread (its locks, its emulated OS), and a stuck game thread then kept the
	// console from quitting. _exit still runs libctru's service cleanup.
	gfxExit();
	fprintf(stderr, "[3ds] quit: graphics stopped; the services next, then the process ends\n");
	flush_log();
	_exit(0);
}
