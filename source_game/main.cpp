// Super Mario Sunshine 3DS - stage 2 boot attempt (MODE=game).
//
// Runs the real game (decomp + sms-pc-port platform layer) on a big-stack thread and shows
// progress on the bottom screen. Graphics are counted, not drawn yet (gx3ds_backend.cpp).
// Everything the port logs (port_log -> stderr) goes to sdmc:/sms/log.txt.
#include <3ds.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
void port_init(int argc, char** argv);
void GX3DS_GetStats(uint32_t out[7]);
uint32_t pica_dropped(void);
int pica_cycle_view(void);
int pica_view(void);
int pica_view_count(void);
void GX3DS_SetDebugColors(int on);
void GX3DS_TraceNextFrame(void);
int GX3DS_GetDebugColors(void);
void GX3DS_GetCopyStats(uint32_t out[3]);
void GX3DS_GetTexStats(uint32_t out[3]);
}
void SMS_main(void);  // the decomp's main(), renamed by port_compat.h (C++ linkage)

#define DISC_PATH "sdmc:/sms/GMSE01.iso"
#define LOG_PATH  "sdmc:/sms/log.txt"

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

	// Everything the port prints goes to the SD card, unbuffered, so a crash still leaves a log.
	freopen(LOG_PATH, "w", stderr);
	setvbuf(stderr, NULL, _IONBF, 0);

	bool isNew = false;
	APT_CheckNew3DS(&isNew);
	printf("SMS 3DS stage 2 (%s 3DS)\n", isNew ? "New" : "Old");
	printf("heap %lu MB, linear free %lu MB\n", (unsigned long)(envGetHeapSize() >> 20),
	       (unsigned long)(linearSpaceFree() >> 20));
	printf("app mem free %lu MB\n", (unsigned long)(osGetMemRegionFree(MEMREGION_APPLICATION) >> 20));
	fprintf(stderr, "[3ds] %s 3DS, heap %lu MB, linear free %lu MB\n", isNew ? "New" : "Old",
	        (unsigned long)(envGetHeapSize() >> 20), (unsigned long)(linearSpaceFree() >> 20));
	printf("log: %s\nSTART+SELECT = exit, SELECT = colours\n\n", LOG_PATH);

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, 4u << 20);  // the game's call chains are deep
	pthread_t th;
	if (pthread_create(&th, &attr, game_thread, NULL) != 0) {
		printf("cannot start the game thread\n");
		fprintf(stderr, "[3ds] pthread_create failed\n");
	}
	pthread_attr_destroy(&attr);

	static const char* const phase[] = {"waiting", "port_init", "SMS_main (running)", "SMS_main returned"};
	while (aptMainLoop()) {
		// The game thread scans HID too (pad_sdk.cpp), so a press can be consumed there before
		// hidKeysDown() sees it here: detect edges on the held state instead. START alone is the
		// game's Start button, so quitting needs START+SELECT.
		hidScanInput();
		static u32 prevHeld;
		u32 held = hidKeysHeld(), pressed = held & ~prevHeld;
		prevHeld = held;
		if ((held & (KEY_START | KEY_SELECT)) == (KEY_START | KEY_SELECT))
			break;
		if ((pressed & KEY_SELECT) && !(held & KEY_START))
			GX3DS_SetDebugColors((GX3DS_GetDebugColors() + 1) % 3);  // material, debug, opaque
		// Touching the bottom screen cycles the top screen through the last frame's EFB copies.
		if (pressed & KEY_L)  // log the next frame's draws (sdmc:/sms/log.txt)
			GX3DS_TraceNextFrame();
		if (pressed & KEY_TOUCH)
			pica_cycle_view();
		uint32_t s[7];
		GX3DS_GetStats(s);
		printf("\x1b[8;0Hphase: %-22s\nframes %6lu  batches %7lu\nverts %8lu  tris %8lu\n", phase[g_phase],
		       (unsigned long)s[0], (unsigned long)s[1], (unsigned long)s[2], (unsigned long)s[3]);
		if (g_phase == 2) {
			SceneState sc = read_scene();
			printf("\nscene: %-9s area %2u-%-2u next %2u-%-2u\nmovie %u  director %08lx\n",
			       app_state_name(sc.state), sc.curr_stage, sc.curr_scen, sc.next_stage, sc.next_scen,
			       sc.movie, (unsigned long)sc.director_vtbl);
			log_scene(s[0], s[1]);
		}
		printf("\x1b[16;0Hdropped draws %lu\ncolours: %-14s\n", (unsigned long)pica_dropped(),
		       GX3DS_GetDebugColors() == 1 ? "debug (per draw)" : GX3DS_GetDebugColors() == 2 ? "opaque" : "material");
		uint32_t cs[3];
		GX3DS_GetCopyStats(cs);
		printf("draws: %3lu before copy, %3lu after\ntexture copies/frame: %lu  \n", (unsigned long)cs[0],
		       (unsigned long)cs[1], (unsigned long)cs[2]);
		uint32_t ts[3];
		GX3DS_GetTexStats(ts);
		printf("\x1b[22;0Htop shows: %s %d of %d  (touch: next)   \n", pica_view() ? "EFB copy" : "frame",
		       pica_view(), pica_view_count());
		printf("\x1b[20;0Htextures: %lu uploads, %lu KB\nEFB-copy texture draws: %lu   \n", (unsigned long)ts[0],
		       (unsigned long)(ts[1] >> 10), (unsigned long)ts[2]);
		// No gfxSwapBuffers: citro3d swaps the top screen, and the console's bottom screen is
		// single-buffered, so writing back the CPU cache is enough.
		gfxFlushBuffers();
		gspWaitForVBlank();
	}

	fprintf(stderr, "[3ds] exiting\n");
	gfxExit();
	return 0;
}
