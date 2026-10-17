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
int GX3DS_GetDebugColors(void);
void GX3DS_GetCopyStats(uint32_t out[3]);
void GX3DS_GetPerf(float out[11]);
void GX3DS_SetGpuTransform(int on);
const char* pica_where(void);
const char* pica_queue_state(void);
uint32_t pica_frame_draws(void);
int pica_selftest(void);
void pica_set_gpu_check(int on);
void pica_set_split(int on);
void pica_set_overlap(int on);
void GX3DS_SetArena(int on);
void GX3DS_SetRenderThread(int on);
extern int GXPC_CullOff;  // patch 0014: skip off-screen display-list draws
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

static SceneState read_scene()
{
	const unsigned char* a = gpApplication;
	SceneState s;
	uintptr_t dir;
	memcpy(&dir, a + 0x04, sizeof dir);
	s.director      = dir;
	s.director_vtbl = dir ? *(const uintptr_t*)dir : 0;  // resolve with nm to get the class
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
	if (dir && s.director_vtbl == (uintptr_t)(_ZTV12TMarDirector + 8)) {
		const unsigned char* d = (const unsigned char*)dir;
		s.mar_state = d[0x64];
		memcpy(&s.cardload, d + 0x70, sizeof s.cardload);
		if (s.cardload) {
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
		if (state0_secs == 8 || (state0_secs > 8 && state0_secs % 30 == 8))
			dump_os_threads("stage setup not finished after 8 s");
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
	u64 now = osGetTime();
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

static void* game_thread(void*)
{
	static char a0[] = "sms3ds";
	static char a1[] = DISC_PATH;
	char* argv[] = {a0, a1, NULL};
	// No audio output on the 3DS yet, and sms-pc-port's software DSP mixer is slower than real
	// time on the ARM11: the audio thread (highest priority) then never waits for its next DMA
	// and the game thread is never scheduled again (seen as a freeze on entering the title
	// stage). Run silently until there is an ndsp backend and a cheaper mixer.
	setenv("SMS_NO_AUDIO", "1", 0);
	// Gameplay at the game's own 30 fps. sms-pc-port defaults to 60, which runs the game's logic
	// twice as often: on a New 3DS that doubled the CPU work per second of play and halved the
	// speed of a game that already cannot keep up. (An environment variable wins over
	// settings.txt's frame_rate.)
	setenv("SMS_FRAME_RATE", "30", 0);
	// Everything the game keeps lives under sdmc:/sms/, next to the disc image: the memory card
	// (sms-pc-port's card.cpp: SMS_SAVE_DIR), settings.txt (port_runtime.cpp: SMS_SETTINGS), and,
	// through the working directory, the relative lookups (mods/, mods/languages/,
	// mods/hd-cutscenes/). Without this they went to a PC-style ./.local/share/sms-port path.
	mkdir(SMS_DIR, 0777);
	chdir(SMS_DIR "/");
	setenv("SMS_SAVE_DIR", SMS_DIR "/card-a", 0);
	setenv("SMS_SETTINGS", SMS_DIR "/settings.txt", 0);
	g_phase = 1;
	port_init(2, argv);
	g_phase = 2;
	fprintf(stderr, "[3ds] entering SMS_main\n");
	SMS_main();
	fprintf(stderr, "[3ds] SMS_main returned\n");
	g_phase = 3;
	return NULL;
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
	// Holding X at launch logs each new TEV setup, unsupported TEV pattern and EFB copy kind.
	if (launchHeld & KEY_X) {
		GX3DS_SetVerboseLog(1);
		printf("X held: detailed renderer log\n");
		fprintf(stderr, "[3ds] X held at launch: [tevdump]/[tev]/[copy] lines on\n");
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
		printf("Up held: resident vertices off\n");
		fprintf(stderr, "[3ds] Up held at launch: cached geometry copied every frame (no resident vertices)\n");
	}
	// Holding D-pad left at launch draws on the game thread, as before the render thread (to compare).
	if (launchHeld & KEY_DLEFT) {
		GX3DS_SetRenderThread(0);
		printf("Left held: no render thread\n");
		fprintf(stderr, "[3ds] Left held at launch: renderer on the game thread (no render thread)\n");
	}
	if (launchHeld & KEY_B) {
		GX3DS_SetGpuTransform(0);
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
	// Overlapped frames (after the self-test, which runs one frame at a time): the CPU builds
	// the next frame while the GPU draws the last. D-pad down at launch keeps them apart.
	if (launchHeld & KEY_DDOWN) {
		printf("Down held: frames not overlapped\n");
		fprintf(stderr, "[3ds] Down held at launch: CPU and GPU frames not overlapped\n");
	} else if (!(launchHeld & (KEY_R | KEY_Y))) {
		pica_set_overlap(1);
	}
	flush_log();

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, 4u << 20);  // the game's call chains are deep
	pthread_t th;
	if (!gpuOk) {
		// A hung GPU would only stop the game at its first frame again.
	} else if (pthread_create(&th, &attr, game_thread, NULL) != 0) {
		printf("cannot start the game thread\n");
		fprintf(stderr, "[3ds] pthread_create failed\n");
	}
	pthread_attr_destroy(&attr);

	static const char* const phase[] = {"waiting", "port_init", "SMS_main (running)", "SMS_main returned"};
	while (aptMainLoop()) {
		// Edges are detected on the held state (the game's PADRead once scanned HID too, consuming
		// presses before hidKeysDown() saw them). START alone is the game's Start button, so quitting
		// needs START+SELECT.
		hidScanInput();
		pad3ds_poll();  // the game's PADRead takes this reading (pad_3ds.c)
		static u32 prevHeld;
		u32 held = hidKeysHeld(), pressed = held & ~prevHeld;
		prevHeld = held;
		if ((held & (KEY_START | KEY_SELECT)) == (KEY_START | KEY_SELECT))
			break;
		if ((pressed & KEY_SELECT) && !(held & KEY_START))
			GX3DS_SetDebugColors((GX3DS_GetDebugColors() + 1) % 3);  // material, debug, opaque
		// Debug input stays on the touch screen, which the game does not use (its buttons all
		// reach the game through pad_3ds.c): the top half cycles the top screen through the last
		// frame's EFB copies, the bottom half logs the next frame's draws to sdmc:/sms/log.txt.
		if (pressed & KEY_TOUCH) {
			touchPosition tp;
			hidTouchRead(&tp);
			if (tp.py < 120)
				pica_cycle_view();
			else if (tp.px < 107)
				GX3DS_TraceNextFrame();
			else if (tp.px < 213)
				GX3DS_SetGpuTransform(!GX3DS_GetGpuTransform());
			else
				pica_cycle_gpu_test();  // GPU test: pixels cut to 8x8 ([perf] gpu wait line)
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
		if (++statusTick % 30 != 0) {
			gspWaitForVBlank();
			continue;
		}
		const u64 busy0 = svcGetSystemTick();
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
		GX3DS_MainBusyTicks += svcGetSystemTick() - busy0;
		static u64 lastFlush;
		if (osGetTime() - lastFlush >= 1000) {
			lastFlush = osGetTime();
			flush_log();
		}
		gspWaitForVBlank();
	}

	fprintf(stderr, "[3ds] exiting\n");
	flush_log();
	// The game thread may be mid-frame: stop it submitting GPU work and let what is in flight
	// finish, so gfxExit does not free the screens under a running display transfer (a quit
	// that sometimes did not return cleanly).
	// Park the game thread at its next PADRead (once a frame; a slow frame is ~200 ms), so it is not
	// running game or GX code while the services and the heap go away.
	pad3ds_quit = 1;
	svcSleepThread(400000000LL);
	pica_request_stop();
	svcSleepThread(100000000LL);
	// _exit, not a return from main: exit's atexit handlers and static destructors can wait on
	// the game thread (its locks, its emulated OS), and a stuck game thread then kept the
	// console from quitting. _exit still runs libctru's service cleanup.
	gfxExit();
	_exit(0);
}
