// Logs game-heap allocations that fail. The game's operator new/new[] (JKRHeap.cpp) are
// renamed to __sms_game_new/__sms_game_new_array by scripts/build_game_lib.sh, and the game
// link wraps them (-Wl,--wrap, see Makefile), so every game `new` passes through here.
// JKRHeap returns null when the current heap is full, and most game code writes through the
// result unchecked, so a failure here is usually the first sign of a later crash.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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
