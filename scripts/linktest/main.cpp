// Link-only milestone: proves game + platform libraries resolve against newlib/libctru.
// It does not run the game (no GX backend yet); it just prints what it linked.
#include <3ds.h>
#include <stdio.h>
void SMS_main(void); // the decomp's main(), renamed by port_compat.h; C++ linkage
int main()
{
	gfxInitDefault();
	consoleInit(GFX_TOP, NULL);
	printf("sms3ds link test: SMS_main @ %p\n\nSTART = exit\n", (void*)&SMS_main);
	while (aptMainLoop()) {
		hidScanInput();
		if (hidKeysDown() & KEY_START) break;
		gspWaitForVBlank();
	}
	gfxExit();
	return 0;
}
