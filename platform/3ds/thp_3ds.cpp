// Movie (THP) video decoding on core 2.
//
// The game decodes each movie frame on its video-decode thread, one of the emulated threads that
// take turns on the game's core (sms-pc-port's os_thread.cpp). Here THPVideoDecode is wrapped
// (-Wl,--wrap=THPVideoDecode): the decode thread hands the frame to a worker on core 2 and sleeps
// like any blocked thread, so the game's other threads run meanwhile. When the worker finishes, an
// interrupt source wakes the decode thread, which then tells the GX layer the frame's textures
// changed (decomp patch 0010 keeps the decoder itself away from the GX layer).
//
// Within a frame the work is split again (THP3DS_DecodeRows): the worker reads the bit stream
// one MCU row at a time while a helper on core 1 turns the rows already read into pixels.
#include <3ds.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>

extern "C" {
struct OSThreadQueue3ds {
	void* head;
	void* tail;
};
int __real_THPVideoDecode(void* file, void* tileY, void* tileU, void* tileV, void* work);
int __wrap_THPVideoDecode(void* file, void* tileY, void* tileU, void* tileV, void* work);
void THP3DS_DecodeOutput(void* out[3], u32 len[3]);
void THP3DS_DecodeWaitIdle(void);
int THP3DS_DecodeRows(u32 rows, u32 blocksPerRow);
void THP3DS_HuffRow(s16* blocks);
void THP3DS_IdctRow(const s16* blocks, u32 row);
int OSDisableInterrupts(void);
int OSRestoreInterrupts(int level);
void OSSleepThread(OSThreadQueue3ds* queue);
void OSWakeupThread(OSThreadQueue3ds* queue);
void DCFlushRange(void* addr, u32 nBytes);
typedef void (*port_irq_poll_fn)(void);
void port_irq_add_source(port_irq_poll_fn fn);
void port_irq_kick(void);
}

namespace {

struct Job {
	void *file, *y, *u, *v, *work;
};

Job s_job;
int s_result;
std::atomic<int> s_busy(0);  // a job was handed to the worker and is not finished
std::atomic<int> s_wake(0);  // finished; the decode thread still has to be woken
LightEvent s_go;
OSThreadQueue3ds s_waiters;
int s_state;  // 0 not started, 1 worker running, -1 no worker (decode in place)
uint64_t s_ticks, s_frames, s_logTick;

// Row pipeline: kSlots row buffers; the worker fills row r into slot r % kSlots once the helper
// has finished row r - kSlots.
const u32 kSlots = 4;
s16* s_rowBuf;
u32 s_rowBlocks, s_rows;
std::atomic<u32> s_read(0), s_built(0);  // rows read from the bit stream / turned into pixels
LightEvent s_rowRead, s_rowBuilt, s_helperGo;
int s_helper;  // 0 not started, 1 running, -1 none (rows decoded in place)
uint64_t s_waitTicks, s_helperTicks;  // worker waiting for a free slot; helper busy

void helper(void*) {
	for (;;) {
		LightEvent_Wait(&s_helperGo);
		const u32 rows = s_rows;
		for (u32 r = 0; r < rows; r++) {
			while (s_read.load(std::memory_order_acquire) <= r) LightEvent_Wait(&s_rowRead);
			const uint64_t t0 = svcGetSystemTick();
			THP3DS_IdctRow(s_rowBuf + (r % kSlots) * s_rowBlocks * 64, r);
			s_helperTicks += svcGetSystemTick() - t0;
			s_built.store(r + 1, std::memory_order_release);
			LightEvent_Signal(&s_rowBuilt);
		}
	}
}

bool startHelper() {
	LightEvent_Init(&s_rowRead, RESET_ONESHOT);
	LightEvent_Init(&s_rowBuilt, RESET_ONESHOT);
	LightEvent_Init(&s_helperGo, RESET_ONESHOT);
	s32 prio = 0x30;
	svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
	// Core 1 (the system core) runs app threads only with a time limit set; the renderer may
	// already have asked for 80% when core 2 was not available.
	if (R_FAILED(APT_SetAppCpuTimeLimit(80)) || !threadCreate(helper, nullptr, 32u << 10, prio, 1, true)) {
		fprintf(stderr, "[thp] no helper thread on core 1: each frame decodes on one core\n");
		return false;
	}
	fprintf(stderr, "[thp] frames split across cores 2 (bit stream) and 1 (pixels)\n");
	return true;
}

void worker(void*) {
	for (;;) {
		LightEvent_Wait(&s_go);
		const uint64_t t0 = svcGetSystemTick();
		s_result = __real_THPVideoDecode(s_job.file, s_job.y, s_job.u, s_job.v, s_job.work);
		s_ticks += svcGetSystemTick() - t0;
		s_busy.store(0);
		s_wake.store(1);
		port_irq_kick();
	}
}

// Interrupt source, polled on the game's core with the emulated CPU held.
void poll() {
	if (s_wake.exchange(0)) OSWakeupThread(&s_waiters);
}

bool start() {
	LightEvent_Init(&s_go, RESET_ONESHOT);
	s32 prio = 0x30;
	svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
	// Core 2, below the GX render thread (which runs one step above the game thread).
	if (!threadCreate(worker, nullptr, 64u << 10, prio, 2, true)) {
		fprintf(stderr, "[thp] no worker thread on core 2: movies decode on the game's core\n");
		return false;
	}
	port_irq_add_source(poll);
	fprintf(stderr, "[thp] movie frames decode on core 2\n");
	return true;
}

}  // namespace

// Called on the worker by __THPDecompressYUV; 0: no helper, the caller decodes the rows itself.
int THP3DS_DecodeRows(u32 rows, u32 blocksPerRow) {
	if (s_helper == 0) s_helper = startHelper() ? 1 : -1;
	if (s_helper < 0 || rows == 0) return 0;
	if (blocksPerRow > s_rowBlocks) {
		free(s_rowBuf);
		s_rowBuf = (s16*)aligned_alloc(32, kSlots * blocksPerRow * 64 * sizeof(s16));
		if (!s_rowBuf) {
			s_rowBlocks = 0;
			return 0;
		}
		s_rowBlocks = blocksPerRow;
	}
	s_rows = rows;
	s_read.store(0);
	s_built.store(0);
	LightEvent_Signal(&s_helperGo);
	for (u32 r = 0; r < rows; r++) {
		if (r >= kSlots && s_built.load(std::memory_order_acquire) <= r - kSlots) {
			const uint64_t t0 = svcGetSystemTick();
			while (s_built.load(std::memory_order_acquire) <= r - kSlots) LightEvent_Wait(&s_rowBuilt);
			s_waitTicks += svcGetSystemTick() - t0;
		}
		s16* blocks = s_rowBuf + (r % kSlots) * s_rowBlocks * 64;
		THP3DS_HuffRow(blocks);
		s_read.store(r + 1, std::memory_order_release);
		LightEvent_Signal(&s_rowRead);
	}
	const uint64_t t0 = svcGetSystemTick();
	while (s_built.load(std::memory_order_acquire) < rows) LightEvent_Wait(&s_rowBuilt);
	s_waitTicks += svcGetSystemTick() - t0;
	return 1;
}

int __wrap_THPVideoDecode(void* file, void* tileY, void* tileU, void* tileV, void* work) {
	if (s_state == 0) s_state = start() ? 1 : -1;
	int result;
	if (s_state < 0) {
		result = __real_THPVideoDecode(file, tileY, tileU, tileV, work);
	} else {
		s_job = {file, tileY, tileU, tileV, work};
		s_busy.store(1);
		LightEvent_Signal(&s_go);
		const int level = OSDisableInterrupts();
		while (s_busy.load()) OSSleepThread(&s_waiters);
		OSRestoreInterrupts(level);
		result = s_result;
	}
	// The textures changed behind the GX layer's back (LCStoreData no longer flushes).
	void* out[3];
	u32 len[3];
	THP3DS_DecodeOutput(out, len);
	for (int i = 0; i < 3; i++)
		if (out[i] && len[i]) DCFlushRange(out[i], len[i]);
	if (s_state > 0 && s_frames == 0) s_logTick = svcGetSystemTick();
	if (s_state > 0 && ++s_frames % 150 == 0) {
		const uint64_t now = svcGetSystemTick();
		const double ms = 150.0 * (SYSCLOCK_ARM11 / 1000.0);
		fprintf(stderr,
		        "[thp] decode %.1f ms/frame (last 150 frames): bit stream on core 2 %.1f, waiting for core 1 %.1f; "
		        "pixels on core 1 %.1f; %.1f frames decoded per second\n",
		        s_ticks / ms, (s_ticks - s_waitTicks) / ms, s_waitTicks / ms, s_helperTicks / ms,
		        150.0 * SYSCLOCK_ARM11 / double(now - s_logTick));
		s_ticks = s_waitTicks = s_helperTicks = 0;
		s_logTick = now;
	}
	return result;
}

// Before the player frees its buffers: the frame on the worker must be finished.
void THP3DS_DecodeWaitIdle(void) {
	while (s_busy.load()) svcSleepThread(200000);
}
