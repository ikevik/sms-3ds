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
}
void SMS_main(void);  // the decomp's main(), renamed by port_compat.h (C++ linkage)

#define DISC_PATH "sdmc:/sms/GMSE01.iso"
#define LOG_PATH  "sdmc:/sms/log.txt"

static volatile int g_phase;  // 0 not started, 1 port_init, 2 SMS_main, 3 SMS_main returned

static void* game_thread(void*)
{
	static char a0[] = "sms3ds";
	static char a1[] = DISC_PATH;
	char* argv[] = {a0, a1, NULL};
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
	consoleInit(GFX_TOP, NULL);

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
	printf("log: %s\nSTART = exit\n\n", LOG_PATH);

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
		hidScanInput();
		if (hidKeysDown() & KEY_START)
			break;
		uint32_t s[7];
		GX3DS_GetStats(s);
		printf("\x1b[8;0Hphase: %-22s\nframes %6lu  batches %7lu\nverts %8lu  tris %8lu\n", phase[g_phase],
		       (unsigned long)s[0], (unsigned long)s[1], (unsigned long)s[2], (unsigned long)s[3]);
		gfxFlushBuffers();
		gfxSwapBuffers();
		gspWaitForVBlank();
	}

	fprintf(stderr, "[3ds] exiting\n");
	gfxExit();
	return 0;
}
