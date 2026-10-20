// HEAPCHECK=1 builds (make MODE=game HEAPCHECK=1): free() and C++ operator delete are wrapped
// (-Wl,--wrap), and a freed block is filled with a poison byte and held back from reuse, in a
// quarantine of up to kQuarBytes. When it leaves the quarantine, and on each poll from the status
// thread, its poison is compared: a block written after it was freed is logged with its size, the
// code that freed it and the bytes written there. For finding a writer through a stale pointer
// (the sampling builds' crashes in the status text's float formatting, 19 Oct). The game's own
// operator new/delete (JKRHeap, renamed) are not affected.
#if defined(SMS3DS_HEAPCHECK)
#include <3ds.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" void __real_free(void* p);

namespace {

const uint32_t kPoison = 0xDBDBDBDBu;
const uint32_t kQuarBytes = 6u << 20;  // held back in all
const uint32_t kQuarMax = 1u << 20;    // bigger blocks go straight back
const uint32_t kRing = 16384;
const int kReports = 16;

struct Held {
	uint8_t* p;
	uint32_t n;       // usable size
	uint32_t lo, hi;  // the poisoned bytes: [lo, hi)
	void* caller;
};
struct Report {
	uint8_t* p;
	uint32_t n, off;
	void* caller;
	uint32_t words[8];
};

LightLock s_lock = 1;  // libctru: 1 is unlocked
Held s_ring[kRing];
uint32_t s_head, s_tail, s_bytes;
uint32_t s_freed, s_checked, s_bad, s_straight;
Report s_rep[kReports];
int s_nRep, s_printed;

// First poisoned word that changed, or -1.
int32_t firstChange(const Held& h) {
	const uint32_t* w = (const uint32_t*)(h.p + h.lo);
	const uint32_t n = (h.hi - h.lo) / 4;
	for (uint32_t i = 0; i < n; i++)
		if (w[i] != kPoison) return int32_t(h.lo + i * 4);
	return -1;
}
void note(const Held& h, int32_t off) {
	s_bad++;
	if (s_nRep >= kReports) return;
	Report& r = s_rep[s_nRep++];
	r.p = h.p;
	r.n = h.n;
	r.off = uint32_t(off);
	r.caller = h.caller;
	for (int i = 0; i < 8; i++) {
		const uint32_t o = uint32_t(off) + i * 4;
		r.words[i] = o + 4 <= h.hi ? *(const uint32_t*)(h.p + o) : 0;
	}
}
// Under s_lock: checks the oldest block and gives it back.
void evict() {
	Held& h = s_ring[s_tail % kRing];
	s_checked++;
	const int32_t off = firstChange(h);
	if (off >= 0) note(h, off);
	s_bytes -= h.n;
	s_tail++;
	__real_free(h.p);
}

void hold(void* ptr, void* caller) {
	if (!ptr) return;
	uint8_t* p = (uint8_t*)ptr;
	const uint32_t n = uint32_t(malloc_usable_size(ptr));
	if (n > kQuarMax || n < 8) {
		__atomic_add_fetch(&s_straight, 1, __ATOMIC_RELAXED);
		__real_free(ptr);
		return;
	}
	// A thread freeing the block it runs on (libctru: a detached thread at its exit): only
	// what lies below its stack pointer, and past the Thread record, is poisoned.
	uint32_t lo = 0, hi = n & ~3u;
	const uint8_t* sp = (const uint8_t*)__builtin_frame_address(0);
	if (sp >= p && sp < p + n) {
		lo = 512;
		const uint32_t below = uint32_t(sp - p);
		hi = below > 256 + lo ? (below - 256) & ~3u : lo;
	}
	for (uint32_t* w = (uint32_t*)(p + lo); w < (uint32_t*)(p + hi); w++) *w = kPoison;
	LightLock_Lock(&s_lock);
	s_freed++;
	while (s_head - s_tail >= kRing || (s_head != s_tail && s_bytes + n > kQuarBytes)) evict();
	s_ring[s_head % kRing] = {p, n, lo, hi, caller};
	s_head++;
	s_bytes += n;
	LightLock_Unlock(&s_lock);
}

}  // namespace

extern "C" {
void __wrap_free(void* p) { hold(p, __builtin_return_address(0)); }
void __wrap__ZdlPv(void* p) { hold(p, __builtin_return_address(0)); }
void __wrap__ZdaPv(void* p) { hold(p, __builtin_return_address(0)); }
void __wrap__ZdlPvj(void* p, unsigned) { hold(p, __builtin_return_address(0)); }
void __wrap__ZdaPvj(void* p, unsigned) { hold(p, __builtin_return_address(0)); }

// Status thread, twice a second: compares every block held (a few at a time under the lock), then
// logs what has been found since the last call.
void SMS3DS_HeapCheckPoll(uint32_t frame) {
	for (uint32_t i = 0;; i++) {
		LightLock_Lock(&s_lock);
		const uint32_t at = s_tail + i;
		if (int32_t(at - s_head) >= 0) {
			LightLock_Unlock(&s_lock);
			break;
		}
		Held& h = s_ring[at % kRing];
		const int32_t off = firstChange(h);
		if (off >= 0) {
			note(h, off);
			for (uint32_t* w = (uint32_t*)(h.p + h.lo); w < (uint32_t*)(h.p + h.hi); w++) *w = kPoison;  // once
		}
		LightLock_Unlock(&s_lock);
	}
	Report rep[kReports];
	int from, to;
	uint32_t freed, checked, bad, held, bytes, straight;
	LightLock_Lock(&s_lock);
	from = s_printed;
	to = s_nRep;
	memcpy(rep, s_rep, sizeof rep);
	s_printed = s_nRep;
	freed = s_freed;
	checked = s_checked;
	bad = s_bad;
	held = s_head - s_tail;
	bytes = s_bytes;
	straight = s_straight;
	LightLock_Unlock(&s_lock);
	static uint32_t lastFreed;
	if (from < to || freed - lastFreed > 0) {
		static int polls;
		if (from < to || ++polls % 20 == 1)
			fprintf(stderr, "[heap] quarantine at frame %lu: %lu blocks freed so far (%lu too big to hold), %lu held (%lu KB), %lu given back after checking, %lu found written after being freed\n",
			        (unsigned long)frame, (unsigned long)freed, (unsigned long)straight, (unsigned long)held,
			        (unsigned long)(bytes >> 10), (unsigned long)checked, (unsigned long)bad);
	}
	lastFreed = freed;
	for (int i = from; i < to; i++) {
		const Report& r = rep[i];
		fprintf(stderr, "[heap] written after it was freed: block %08lx (%lu bytes), freed from fn %08lx, first change at byte %lu:",
		        (unsigned long)uintptr_t(r.p), (unsigned long)r.n, (unsigned long)uintptr_t(r.caller), (unsigned long)r.off);
		for (int k = 0; k < 8; k++) fprintf(stderr, " %08lx", (unsigned long)r.words[k]);
		fprintf(stderr, "\n");
	}
}
}
#endif
