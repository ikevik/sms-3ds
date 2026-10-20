// Logs game-heap allocations that fail. The game's operator new/new[] (JKRHeap.cpp) are
// renamed to __sms_game_new/__sms_game_new_array by scripts/build_game_lib.sh, and the game
// link wraps them (-Wl,--wrap, see Makefile), so every game `new` passes through here.
// JKRHeap returns null when the current heap is full, and most game code writes through the
// result unchecked, so a failure here is usually the first sign of a later crash.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

extern "C" {
void* __real___sms_game_new(size_t size);
void* __real___sms_game_new_array(size_t size);
}

static unsigned g_failures;

static void report(const char* what, size_t size, void* caller)
{
	// The first failures say the most; after that, one line per 256 keeps the log readable.
	unsigned n = ++g_failures;
	if (n <= 32 || (n & 255) == 0)
		fprintf(stderr, "[heap] %s(%lu) returned null, caller %08lx (failure #%u)\n", what,
		        (unsigned long)size, (unsigned long)(uintptr_t)caller, n);
}

extern "C" void* __wrap___sms_game_new(size_t size)
{
	void* p = __real___sms_game_new(size);
	if (!p)
		report("new", size, __builtin_return_address(0));
	return p;
}

extern "C" void* __wrap___sms_game_new_array(size_t size)
{
	void* p = __real___sms_game_new_array(size);
	if (!p)
		report("new[]", size, __builtin_return_address(0));
	return p;
}

// Decomp patch 3ds-0014: GD found its display list full (GDOverflowed), before the write that
// would pass its end. `caller` is the function doing the write (its GD writes are inline).
extern "C" void SMS3DS_GDOverflow(const void* dlp, const void* caller)
{
	struct GDLObj3ds {
		uint8_t* start;
		uint32_t length;
		uint8_t* ptr;
		uint8_t* top;
	};
	const GDLObj3ds* dl = (const GDLObj3ds*)dlp;
	static unsigned told;
	++told;
	if (told > 12 && (told & 1023)) return;
	if (!dl) {
		fprintf(stderr, "[gd] display list full with none current (writer fn %08lx)\n", (unsigned long)uintptr_t(caller));
	} else {
		fprintf(stderr, "[gd] display list at %08lx (%lu bytes, end %08lx) full: writing at %08lx, %ld bytes past its end, by fn %08lx (overflow write #%u)\n",
		        (unsigned long)uintptr_t(dl->start), (unsigned long)dl->length, (unsigned long)uintptr_t(dl->top),
		        (unsigned long)uintptr_t(dl->ptr), (long)(dl->ptr - dl->top), (unsigned long)uintptr_t(caller), told);
	}
	fflush(stderr);
	fsync(fileno(stderr));
}
