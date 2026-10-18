// GameCube SDK PAD layer for the 3DS: replaces sms-pc-port's platform/pad/ (SDL + keyboard +
// USB GameCube adapter) with the Circle Pad / buttons / C-stick read in pad_3ds.c, and
// defines the camera/rumble settings the game patches read from that module.
#include "port_compat.h"
#include <dolphin/pad.h>
#include <string.h>
#include <3ds.h>
#include "pad_3ds.h"

// Settings the PC port's game patches read (all neutral defaults).
extern "C" {
int port_free_camera = 0;
f32 port_camera_speed_x = 1.0f;
f32 port_camera_speed_y = 1.0f;
f32 port_rumble_boost = 1.0f;

// Port-only features (auto-press scripting, save-state labels, mouse camera): not on 3DS.
void port_pad_autopress_field(u32) {}
int port_pad_savestate_labels(int, char[2][16]) { return 0; }
int port_camera_take_mouse(f32*, f32*) { return 0; }
}

extern "C" BOOL PADInit() { return TRUE; }
extern "C" int PADReset(unsigned long) { return TRUE; }
extern "C" BOOL PADRecalibrate(u32) { return TRUE; }
extern "C" BOOL PADSync(void) { return TRUE; }
extern "C" void PADSetSpec(u32) {}
extern "C" void PADSetAnalogMode(u32) {}
extern "C" void PADSetSamplingRate(unsigned long) {}
extern "C" void PADControlMotor(s32, u32) {}
extern "C" void PADControlAllMotors(const u32*) {}

extern "C" u32 PADRead(PADStatus* status)
{
	// Quitting: the game thread stops here, before the services it uses are shut down.
	if (pad3ds_quit) pad3ds_parked = 1;
	while (pad3ds_quit) svcSleepThread(1000000000LL);
	Pad3DSStatus in;
	pad3ds_read(&in);

	for (int i = 0; i < 4; i++) {
		memset(&status[i], 0, sizeof status[i]);
		status[i].err = i == 0 ? PAD_ERR_NONE : PAD_ERR_NO_CONTROLLER;
	}
	PADStatus& s = status[0];
	// pad_3ds.h keeps the GameCube SDK's button bit layout, so the word copies straight over.
	s.button = in.button;
	s.stickX = in.stickX;
	s.stickY = in.stickY;
	s.substickX = in.substickX;
	s.substickY = in.substickY;
	s.triggerLeft = in.triggerL;
	s.triggerRight = in.triggerR;
	return PAD_CHAN0_BIT;
}
