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

void pad3ds_read(Pad3DSStatus* out)
{
	memset(out, 0, sizeof(*out));
	u32 held = hidKeysHeld();

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
	if (held & KEY_R) { out->button |= P3DS_TRIGGER_R; out->triggerR = 255; }

	circlePosition cp;
	hidCircleRead(&cp);
	out->stickX = scale_axis(cp.dx);
	out->stickY = scale_axis(cp.dy);

	circlePosition cs;
	irrstScanInput();
	irrstCstickRead(&cs);   // New 3DS C-stick (needs irrstInit)
	out->substickX = scale_axis(cs.dx);
	out->substickY = scale_axis(cs.dy);
}
