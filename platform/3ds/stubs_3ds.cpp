// Symbols sms-pc-port's executable gets from files this port does not build yet.
// Each is a deliberate placeholder (a real implementation replaces it as the feature lands).
#include <stdint.h>

// dsptask.c points a DSP task at the fixed GameCube address 0x80004600; with the DSP HLE
// (SMS_AUDIO_DSP_HLE=1) only the symbol's address is used, so a small object is enough.
// sms-pc-port does the same on macOS (src/lbl_80004600_darwin.c).
extern "C" unsigned char lbl_80004600[4];
unsigned char lbl_80004600[4];

// platform/presence (Discord rich presence) is not built for the 3DS.
extern "C" void port_presence_frame(const void*) {}

// src/savestate_menu.cpp draws through the PC GX path; no save-state menu on 3DS for now.
class TMarioGamePad;
namespace JDrama { struct TRect; }
bool SaveStateMenuIsOpen() { return false; }
void SaveStateMenuOpen() {}
void SaveStateMenuClose() {}
void SaveStateMenuMove(TMarioGamePad*) {}
void SaveStateMenuSetAlpha(unsigned char) {}
void SaveStateMenuDraw(const JDrama::TRect&) {}

// platform/gx/src/gx_prompts.cpp draws controller-button images over the system font; null
// keeps the font's own glyphs (JUTResFont.cpp and platform/savestate treat null as "none").
extern "C" const void* port_button_glyph(int) { return nullptr; }
