/* libctru's GPUCMD_Add (gpu.o), the call every GPU command goes through (citro3d's and ours), with
 * the same effects in fewer instructions: linked with -Wl,--wrap=GPUCMD_Add (Makefile), so every
 * call outside libctru's gpu.o comes here.
 *
 * libctru's version saves ten registers and calls memcpy for every write of more than one word
 * (memset when given no values); in Azahar, where each instruction costs about the same, it and
 * its memcpy calls were a fifth of the render thread. This one writes the same words in the same
 * order, as its disassembly does: the first value, the header with the extra count in bits 20-30,
 * the rest, a zero word after an odd number of extra words, at most 256 values a command (an
 * incremental write's next command starting at the register after the last), svcBreak when the
 * buffer has no room. */
#include <3ds.h>
#include <string.h>

void __real_GPUCMD_Add(u32 header, const u32* param, u32 paramlength);

/* libctru's own instead (pica_set_gpucmd_libctru; the profiler's revert steps). */
static int s_libctru;
void pica_set_gpucmd_libctru(int on) { s_libctru = on; }

void __wrap_GPUCMD_Add(u32 header, const u32* param, u32 paramlength)
{
	if (s_libctru) {
		__real_GPUCMD_Add(header, param, paramlength);
		return;
	}
	/* One value (most of citro3d's: GPUCMD_AddWrite, GPUCMD_AddMaskedWrite): the value, the header. */
	if (paramlength <= 1 && param) {
		u32* buf = gpuCmdBuf;
		const u32 off = gpuCmdBufOffset;
		if (buf && off + 2 <= gpuCmdBufSize) {
			buf[off] = param[0];
			buf[off + 1] = header;
			gpuCmdBufOffset = off + 2;
			return;
		}
	}
	u32 left = paramlength ? paramlength : 1;
	do {
		const u32 n = left < 256 ? left : 256;
		u32* buf = gpuCmdBuf;
		u32 off = gpuCmdBufOffset;
		if (!buf || off + 1 + n > gpuCmdBufSize) {
			svcBreak(USERBREAK_PANIC);
			buf = gpuCmdBuf;
			off = gpuCmdBufOffset;
		}
		const u32 extra = n - 1;
		buf[off] = param ? param[0] : 0;
		buf[off + 1] = header | (extra << 20);
		u32* out = buf + off + 2;
		if (extra) {
			if (param) {
				if (extra <= 16) {
					for (u32 i = 0; i < extra; i++) out[i] = param[1 + i];
				} else {
					memcpy(out, param + 1, extra * 4);
				}
			} else {
				memset(out, 0, extra * 4);
			}
			out += extra;
			if (extra & 1) *out++ = 0;  /* (alignment) */
		}
		gpuCmdBufOffset = (u32)(out - buf);
		if ((s32)header < 0) header += n;  /* an incremental write: on from the next register */
		left -= n;
		if (param) param += n;
	} while (left);
}
