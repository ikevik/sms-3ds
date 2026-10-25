// GameCube PAD -> 3DS input mapping (counterpart of the PAD layer in sms-pc-port/aurora).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bit layout matches the GameCube SDK's P3DS_BUTTON_* values.
#define P3DS_BUTTON_LEFT   0x0001
#define P3DS_BUTTON_RIGHT  0x0002
#define P3DS_BUTTON_DOWN   0x0004
#define P3DS_BUTTON_UP     0x0008
#define P3DS_TRIGGER_Z     0x0010
#define P3DS_TRIGGER_R     0x0020
#define P3DS_TRIGGER_L     0x0040
#define P3DS_BUTTON_A      0x0100
#define P3DS_BUTTON_B      0x0200
#define P3DS_BUTTON_X      0x0400
#define P3DS_BUTTON_Y      0x0800
#define P3DS_BUTTON_START  0x1000

typedef struct {
	uint16_t button;
	int8_t   stickX, stickY;       // -72..72 on real hardware
	int8_t   substickX, substickY; // C-stick
	uint8_t  triggerL, triggerR;   // analog 0..255 (digital on 3DS)
	int8_t   err;                  // 0 = OK
} Pad3DSStatus;

// Main thread, after each hidScanInput(): takes a reading. Game thread: the latest reading.
void pad3ds_poll(void);
void pad3ds_read(Pad3DSStatus* out);
// 3DS KEY_* buttons an autorun script holds now (main.cpp: autorun.txt buttons), as if pressed.
extern volatile uint32_t pad3ds_script;
// R as a soft press (1: the trigger's analog value only, so Mario sprays while he moves) or a hard
// one (0: the analog value and the click at the bottom, so he stops to aim, as the GameCube's R
// pushed all the way). The main thread switches it on SELECT (main.cpp).
extern volatile int pad3ds_soft_r;
// Set by the main thread when quitting: the game thread stops at its next PADRead.
extern volatile int pad3ds_quit;
// Set by the main thread once the game thread has stopped, as the services are about to shut down:
// from then on the threads that wake up on their own park (sms3ds_park).
extern volatile int sms3ds_ending;
extern volatile int sms3ds_parked_threads;  // how many have parked (the log's at the end)
// Never returns: the calling thread waits in the kernel, on an event nothing signals, until the
// process ends. For the threads that would otherwise wake up on their own while the app quits: as
// a .cia, the system can take the process's memory away while they still run (crash dump 49: the
// render thread, polling for a GPU completion on core 2, faulted on its own stack).
void sms3ds_park(void);
extern volatile int pad3ds_parked;  // set by the game thread once it stops for the quit

// Mapping:
//   Circle Pad -> main stick        C-stick (New 3DS) -> C-stick
//   A/B/X/Y    -> A/B/X/Y           L / R             -> L / R (full press; R soft: pad3ds_soft_r)
//   ZL         -> Z                 ZR (New 3DS)      -> Z
//   D-pad      -> D-pad             Start             -> Start

#ifdef __cplusplus
}
#endif
