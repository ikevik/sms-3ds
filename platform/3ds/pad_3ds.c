#include "pad_3ds.h"
#include <3ds.h>
#include <math.h>
#include <string.h>

/* A stick on the GameCube's scale (the Circle Pad and the C-stick report about -156..156, the
 * GameCube's stick about -72..72), with a dead zone: a pad that doesn't rest exactly at the centre
 * read as a slight push, and Mario drifted (issue #1). Within pad3ds_stick_deadzone of the centre
 * it reads centred; past it the rest of the travel is spread over the whole range, so all the way
 * still reads all the way. Measured as a radius, so that diagonals keep their direction. */
volatile int pad3ds_stick_deadzone = 15;
static void scale_stick(int dx, int dy, int8_t* ox, int8_t* oy)
{
	const float full = 156.0f, dz = (float)pad3ds_stick_deadzone;
	const float r = sqrtf((float)(dx * dx + dy * dy));
	*ox = *oy = 0;
	if (r <= dz || r <= 0.0f) return;
	const float k = (r >= full ? 1.0f : (r - dz) / (full - dz)) * 72.0f / r;
	int x = (int)lroundf((float)dx * k), y = (int)lroundf((float)dy * k);
	*ox = (int8_t)(x > 72 ? 72 : x < -72 ? -72 : x);
	*oy = (int8_t)(y > 72 ? 72 : y < -72 ? -72 : y);
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
	if (held & KEY_ZL) out->button |= P3DS_TRIGGER_Z;
	if (held & KEY_L) { out->button |= P3DS_TRIGGER_L; out->triggerL = 255; }
	/* R is a hard press (the analog value and the click: Mario stops to aim), or a soft one once
	 * SELECT has switched it (pad3ds_soft_r). ZR is the soft press, so FLUDD sprays while Mario
	 * moves; with settings.txt zr_press = opposite (pad3ds_zr_opposite) it is whichever press R is
	 * not, so hard while R is soft. Soft: past the analog level at which the game's spray pressure is
	 * full (150 of 255: TMario::getLRLevel, mAnalogLRToMaxVal), without the click. */
	const int zrHard = pad3ds_zr_opposite && pad3ds_soft_r;
	const int rHard = ((held & KEY_R) && !pad3ds_soft_r) || ((held & KEY_ZR) && zrHard);
	const int rSoft = ((held & KEY_R) && pad3ds_soft_r) || ((held & KEY_ZR) && !zrHard);
	if (rHard) {
		out->button |= P3DS_TRIGGER_R;
		out->triggerR = 255;
	} else if (rSoft) {
		out->triggerR = 180;
	}

	circlePosition cp;
	hidCircleRead(&cp);
	scale_stick(cp.dx, cp.dy, &out->stickX, &out->stickY);

	circlePosition cs;
	irrstScanInput();
	irrstCstickRead(&cs);   // New 3DS C-stick (needs irrstInit)
	scale_stick(cs.dx, cs.dy, &out->substickX, &out->substickY);

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
void (*volatile pad3ds_frame_hook)(void);
volatile int pad3ds_soft_r;
volatile int pad3ds_zr_opposite;
volatile int sms3ds_ending;
volatile int sms3ds_parked_threads;
volatile int sms3ds_suspended;
static LightEvent s_resumed;
static int s_resumedReady;

void sms3ds_set_suspended(int on)
{
	if (!s_resumedReady) LightEvent_Init(&s_resumed, RESET_STICKY), s_resumedReady = 1;
	if (on) {
		LightEvent_Clear(&s_resumed);
		sms3ds_suspended = 1;
	} else {
		sms3ds_suspended = 0;
		LightEvent_Signal(&s_resumed);
	}
}

void sms3ds_wait_resumed(void)
{
	while (sms3ds_suspended && !pad3ds_quit && !sms3ds_ending && s_resumedReady)
		LightEvent_WaitTimeout(&s_resumed, 100000000LL);
}

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
