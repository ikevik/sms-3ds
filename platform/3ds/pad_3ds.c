#include "pad_3ds.h"
#include <3ds.h>
#include <string.h>

static int8_t scale_axis(int16_t v)
{
	// Circle Pad reports about -156..156; GameCube stick range is about -72..72.
	int s = (v * 72) / 156;
	if (s > 72) s = 72;
	if (s < -72) s = -72;
	return (int8_t)s;
}

static void read_now(Pad3DSStatus* out)
{
	memset(out, 0, sizeof(*out));
	u32 held = hidKeysHeld() | pad3ds_script;

	if (held & KEY_DLEFT)  out->button |= P3DS_BUTTON_LEFT;
	if (held & KEY_DRIGHT) out->button |= P3DS_BUTTON_RIGHT;
	if (held & KEY_DDOWN)  out->button |= P3DS_BUTTON_DOWN;
	if (held & KEY_DUP)    out->button |= P3DS_BUTTON_UP;
	if (held & KEY_A)      out->button |= P3DS_BUTTON_A;
	if (held & KEY_B)      out->button |= P3DS_BUTTON_B;
	if (held & KEY_X)      out->button |= P3DS_BUTTON_X;
	if (held & KEY_Y)      out->button |= P3DS_BUTTON_Y;
	if (held & KEY_START)  out->button |= P3DS_BUTTON_START;
	if (held & (KEY_ZL | KEY_ZR)) out->button |= P3DS_TRIGGER_Z;
	if (held & KEY_L) { out->button |= P3DS_TRIGGER_L; out->triggerL = 255; }
	if (held & KEY_R) {
		/* Soft: past the analog level at which the game's spray pressure is full (150 of 255:
		 * TMario::getLRLevel, mAnalogLRToMaxVal), without the click. */
		if (pad3ds_soft_r) {
			out->triggerR = 180;
		} else {
			out->button |= P3DS_TRIGGER_R;
			out->triggerR = 255;
		}
	}

	circlePosition cp;
	hidCircleRead(&cp);
	out->stickX = scale_axis(cp.dx);
	out->stickY = scale_axis(cp.dy);

	circlePosition cs;
	irrstScanInput();
	irrstCstickRead(&cs);   // New 3DS C-stick (needs irrstInit)
	out->substickX = scale_axis(cs.dx);
	out->substickY = scale_axis(cs.dy);

	/* autorun.txt buttons: the Circle Pad and C-stick pushed all the way (main.cpp) */
	u32 sc = pad3ds_script;
	if (sc & KEY_CPAD_RIGHT) out->stickX = 72;
	if (sc & KEY_CPAD_LEFT)  out->stickX = -72;
	if (sc & KEY_CPAD_UP)    out->stickY = 72;
	if (sc & KEY_CPAD_DOWN)  out->stickY = -72;
	if (sc & KEY_CSTICK_RIGHT) out->substickX = 72;
	if (sc & KEY_CSTICK_LEFT)  out->substickX = -72;
	if (sc & KEY_CSTICK_UP)    out->substickY = 72;
	if (sc & KEY_CSTICK_DOWN)  out->substickY = -72;
}

/* The main thread reads the buttons (pad3ds_poll, after its hidScanInput) and the game thread takes
 * the latest reading (pad3ds_read): HID is touched by one thread only, and never after the main
 * thread starts shutting the services down. */
static Pad3DSStatus s_latest;
volatile uint32_t pad3ds_script;
static LightLock s_lock = 1; /* 1: unlocked */
volatile int pad3ds_quit;
volatile int pad3ds_soft_r;
volatile int sms3ds_ending;
volatile int sms3ds_parked_threads;

void sms3ds_park(void)
{
	__atomic_fetch_add(&sms3ds_parked_threads, 1, __ATOMIC_RELAXED);
	Handle never = 0;
	if (R_SUCCEEDED(svcCreateEvent(&never, RESET_ONESHOT)))
		for (;;) svcWaitSynchronization(never, U64_MAX);
	for (;;) svcSleepThread(3600000000000LL); /* (no event: an hour at a time) */
}
volatile int pad3ds_parked;

void pad3ds_poll(void)
{
	Pad3DSStatus st;
	read_now(&st);
	LightLock_Lock(&s_lock);
	s_latest = st;
	LightLock_Unlock(&s_lock);
}

void pad3ds_read(Pad3DSStatus* out)
{
	LightLock_Lock(&s_lock);
	*out = s_latest;
	LightLock_Unlock(&s_lock);
}
