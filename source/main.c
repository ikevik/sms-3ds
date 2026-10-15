// Super Mario Sunshine - 3DS port, bring-up entry point.
//
// Stage 0 (this file): boot, New 3DS speedup, find + validate the user's own
// GMSE01 disc image on the SD card, draw the test triangle on the top screen,
// and show pad/disc status on the bottom screen.
// Later stages replace the main loop body with the decomp's game entry point.
#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <string.h>

#include "gx_citro3d.h"
#include "gcm_disc.h"
#include "pad_3ds.h"

#define DISC_PATH "sdmc:/sms/GMSE01.iso"

int main(void)
{
	// New 3DS: run the ARM11 at 804 MHz. No-op (returns false) on Old 3DS.
	osSetSpeedupEnable(true);

	GX3DS gx;
	if (!gx3ds_init(&gx))
		return 1;

	irrstInit();                 // C-stick / ZL / ZR on New 3DS
	consoleInit(GFX_BOTTOM, NULL);

	printf("Super Mario Sunshine 3DS (WIP)\n\n");

	bool isNew = false;
	APT_CheckNew3DS(&isNew);
	printf("Console: %s 3DS\n", isNew ? "New" : "Old");
	if (!isNew)
		printf("  Old 3DS will likely be too slow/short on RAM.\n");

	bool discOk = gcm_open(DISC_PATH);
	if (discOk) {
		printf("Disc: %s rev %u\n", gcm_game_id(), gcm_revision());
		GcmFile f;
		if (gcm_boot_dol(&f))
			printf("boot dol: %lu bytes @ 0x%lX\n", (unsigned long)f.length, (unsigned long)f.offset);
		else
			printf("boot dol: not found\n");
		if (gcm_find("/opening.bnr", &f))
			printf("FST ok: opening.bnr %lu bytes\n", (unsigned long)f.length);
		else
			printf("FST lookup failed\n");
		if (strcmp(gcm_game_id(), "GMSE01") != 0)
			printf("Warning: expected GMSE01 (NTSC-U)\n");
	} else {
		printf("Disc not found/invalid:\n  %s\n", DISC_PATH);
	}
	printf("\nSTART = exit\n");

	while (aptMainLoop()) {
		hidScanInput();
		if (hidKeysDown() & KEY_START)
			break;

		Pad3DSStatus pad;
		pad3ds_read(&pad);
		printf("\x1b[15;0Hstick %4d %4d  c %4d %4d  btn %04X   ",
			pad.stickX, pad.stickY, pad.substickX, pad.substickY, pad.button);

		gx3ds_begin_frame(&gx);
		gx3ds_draw_test_triangle(&gx);
		gx3ds_end_frame(&gx);
	}

	gcm_close();
	irrstExit();
	gx3ds_shutdown(&gx);
	return 0;
}
