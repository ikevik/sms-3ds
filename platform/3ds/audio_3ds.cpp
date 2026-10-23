// Game audio on the 3DS's DSP (settings.txt audio; docs/audio-3ds.md).
//
// JAudio runs as on the GameCube: its sequencer and channel updates on the audio thread (one of
// the emulated threads that take turns on the game's core), which fills the 64 voice blocks the
// GameCube's DSP would mix. sms-pc-port's DSP HLE hands each 80-sample subframe here
// (port_audio_hw_render, from DSPReleaseHalt) before its software mixer (patch 0041):
//  - The waves are AFC ADPCM, which the 3DS DSP cannot decode. The CPU decodes them, bit for bit
//    as the software mixer does, a few hundred samples at a time into small PCM16 buffers; the 3DS
//    DSP resamples them, applies the volumes and mixes them, on ndsp channels 0-22.
//  - What the 3DS DSP is not given (streams, oscillators, voices past the 23rd) stays with the
//    software mixer, whose output reaches the AI blocks. Those play on channel 23 with the movies'
//    sound, and the DSP's progress through them is the game's audio clock (port_audio_out_*).
// The audio thread records each subframe's wanted channel state (a snapshot); ndsp's thread
// (every 4.9 ms DSP frame, on the game's core, above every game thread) applies the snapshots at
// the DSP's pace, about a DAC frame (17.5 ms) after the audio thread wrote them, and keeps each
// channel's buffers decoded ahead (ndsp_frame). A voice that runs to its end comes back through
// s_finished, and the audio thread then marks it done in its block, as the GameCube's DSP would.
//
// settings.txt `audio`: off (no sound data, as before), on (the above), software (sms-pc-port's
// mixer for every voice, out through channel 23: for comparing), silent (the sound driver runs,
// no voice plays: for measuring). `audio_cycle = N` turns through on, software and silent every
// N seconds, each voice restarted under the new way, and logs each one's cost ([audio] lines).
#include <3ds.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio/dsp_mixer.h"
#if defined(__arm__) && __ARM_ARCH >= 6
#include <arm_acle.h>
#endif

extern "C" {
u8* port_aram_ptr(u32 addr);  // platform/ar: ARAM is one 16 MB block
void port_irq_kick(void);
void* pica_linear_alloc(uint32_t bytes);
void GX3DS_GetStats(uint32_t out[7]);
int SMS3DS_AudioInit(int mode, int cycleSeconds);
void SMS3DS_AudioSetUpdateInterval(int subframes);
void SMS3DS_AudioExit(void);
void port_audio_hw_setup(void* voices, uint32_t nvoices, const uint32_t* adpcm_filter);
int port_audio_hw_render(int16_t* outA, int16_t* outB, int n, uint16_t master);
int port_audio_out_open(void);
int port_audio_out_queued(void);
void port_audio_out_write(const s16* lr, u32 frames);
void port_os_switch_hook(void* from, void* to);
// JAudio's turns on core 2 (settings.txt audio_thread = core2; sms-pc-port patch 0043, Makefile wraps,
// scripts/build_game_lib.sh's lock hooks).
void SMS3DS_AudioSetThread(int core2);
void port_os_set_off_cpu(int on);
void port_os_resume_hook(void);
void port_audio_checkpoint(void);
int port_audio_parallel_owns(int idle);
void port_os_idle_hook(int idle);
void port_ai_block(void);
int port_ai_running(void);
void __audio_lock_enter(void* fn, void* site);
void __audio_lock_exit(void* fn, void* site);
extern volatile int port_audio_turn_waiting;  // os_thread.cpp looks at it at interrupt check points
int __real__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(s32 (*cb)(void*), void* data);
int __wrap__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(s32 (*cb)(void*), void* data);
void __real__ZN8JASystem6Kernel19registerMixCallbackEPFPslEh(s16* (*cb)(s32), u8 mode);
void __wrap__ZN8JASystem6Kernel19registerMixCallbackEPFPslEh(s16* (*cb)(s32), u8 mode);
// MSound::mainLoop, the game thread's sound work once a frame (-Wl,--wrap in the Makefile).
void __real__ZN6MSound8mainLoopEv(void* self);
void __wrap__ZN6MSound8mainLoopEv(void* self);
// The audio thread's two big parts, timed the same way: JASystem::Kernel::subframeCallback (the
// sequencer and port commands) and JASystem::TDSPChannel::updateAll (the channels' updates).
void __real__ZN8JASystem6Kernel16subframeCallbackEv(void);
void __wrap__ZN8JASystem6Kernel16subframeCallbackEv(void);
void __real__ZN8JASystem11TDSPChannel9updateAllEv(void);
void __wrap__ZN8JASystem11TDSPChannel9updateAllEv(void);
// JAudio's function hooks in AUDIO_SAMPLE=1 builds (scripts/build_game_lib.sh renames them from
// __cyg_profile_func_enter/_exit), and the emulated thread that owns the CPU (os_thread.cpp).
void __audio_prof_enter(void* fn, void* site);
void __audio_prof_exit(void* fn, void* site);
extern void* __gCurrentThread;
}
namespace JASystem {
namespace AudioThread {
extern u8 jac_audioThread[];  // the audio thread's OSThread (JASAudioThread.cpp)
}
namespace AudioThread {
u32 getDSPSyncCount();             // subframes of the DSP frame still to come (JASAudioThread.cpp)
void setDSPSyncCount(u32 count);
}
namespace Driver {
void setUpdateInterval(u8 interval);  // JASDriverIF.cpp: subframes between a voice's updates
}
namespace Kernel {
void updateDac();                                  // JASAiCtrl.cpp: an AI block's work
int checkCallback(s32 (*cb)(void*), void* data);   // JASCallback.cpp: still registered (its index)
}
namespace DSPBuf {
void updateDSP();       // JASDSPBuf.cpp: one subframe (sequencer, channels, the DSP's halt released)
void finishDSPFrame();  // ... a DSP frame's end, and the next's start if a buffer is free
}
struct TTrack {
	static s32 rootCallback(void* track);  // JASTrack.cpp: a root track's subframe (the sequencer)
};
}

namespace {

// --- the game's voice block (JASystem::DSPInterface::DSPBuffer; as sms-pc-port's dsp_mixer.cpp) ---
struct MixSlot {
	u16 bus;      // DSP bus address (connect_table), 0 = unconnected
	u16 target;   // volume the slot ramps to over one subframe (Q14)
	u16 current;  // volume at the start of the subframe (the DSP writes it back)
	u16 level;
};
struct Voice {
	u16 enabled;            // 0x000
	u16 done;               // 0x002 the DSP sets it when the voice has finished
	u16 ratio;              // 0x004 Q12 resampling ratio (0x1000 = 1:1 at 32028.5 Hz)
	u16 pad006;
	u16 resetVpb;           // 0x008 set by playStart: (re)start decoding
	u16 endReached;         // 0x00A the DSP sets it at a one-shot's end
	u16 pause;              // 0x00C
	u16 delayMax;           // 0x00E
	MixSlot slot[6];        // 0x010
	u8 pad040[0x10];
	u16 amPan;              // 0x050 auto mixer: pan << 8 | dolby
	u16 amFx;               // 0x052
	u16 amVolCurrent;       // 0x054
	u16 amVolTarget;        // 0x056 (Q15)
	u16 amEnabled;          // 0x058
	u8 pad05A[0xA6];
	u16 sourceType;         // 0x100 9 ADPCM4, 5 ADPCM2, 8 PCM8, 16 PCM16, 0x21 stream, else oscillator
	u16 looping;            // 0x102
	s16 loopYN1;            // 0x104
	s16 loopYN2;            // 0x106
	s16 filterMode;         // 0x108
	u16 endRequested;       // 0x10A
	u32 cpuTicks;           // 0x10C
	u32 loopStart;          // 0x110 sample index
	u32 endPosition;        // 0x114 loop end, or sample count
	u32 base;               // 0x118 ARAM byte address of the wave
	s32 sampleCount;        // 0x11C
	u8 pad120[0x60];
};
static_assert(sizeof(Voice) == 0x180, "DSPBuffer is 0x180 bytes");
static_assert(__builtin_offsetof(Voice, slot) == 0x10, "mixChannels");
static_assert(__builtin_offsetof(Voice, amPan) == 0x50, "dolbyVoicePosition");
static_assert(__builtin_offsetof(Voice, sourceType) == 0x100, "samplesSourceType");
static_assert(__builtin_offsetof(Voice, endRequested) == 0x10A, "endRequested");
static_assert(__builtin_offsetof(Voice, base) == 0x118, "baseAddress");

enum { kOff = 0, kDsp = 1, kSoft = 2, kSilent = 3 };  // settings.txt audio
const char* const kModeName[] = {"off", "on the 3DS DSP", "in the software mixer", "silent"};

const int kHwChans    = 23;     // ndsp channels 0-22 play voices
const int kSinkChan   = 23;     // ... and 23 the AI blocks
const int kBufs       = 4;      // decoded buffers per voice channel
const int kBufCap     = 2048;   // samples in each (about 32 ms of the wave is decoded into one)
const int kSinkBufs   = 4;      // AI blocks the DSP can hold
const int kSinkCap    = 1024;   // stereo frames in each (JAudio's are 560)
const int kSinkTarget = 3;      // blocks ai.cpp keeps queued (its kBaseBlocks)
const int kSnaps      = 32;     // subframe snapshots in flight
const u32 kAramSize   = 16u << 20;
const float kDacRate  = 32028.5f;
// Subframes (80 samples at 32028.5 Hz) per DSP frame (160 samples at the DSP's 32728 Hz).
const float kSubPerFrame = float(160.0 * 32028.5 / (80.0 * NDSP_SAMPLE_RATE));
const double kTicksPerMs = SYSCLOCK_ARM11 / 1000.0;

// How a voice plays: on a DSP channel; in the software mixer; on the DSP path but followed without a
// channel while at zero volume (kQuiet: it gets one when it is heard); silent mode (kUnheard).
enum { kIdle, kOnDsp, kInMixer, kQuiet, kUnheard };
const int kQuietSubs = 16;     // subframes at zero volume (40 ms) before a voice gives its channel up
const u32 kYoung = 2048;       // samples: a voice heard this soon after its start, with no channel
                               // free, takes the quietest voice's channel; later ones wait for one
const float kQuieter = 0.8f;   // ... if that voice is at most this loud, relative to the new one
const int kRefillEvery = 4;    // DSP frames between top-ups of the buffers (sooner when nearly dry)
const u32 kSeekExact = 4096;   // samples decoded and dropped at most to restart a voice exactly

// What the audio thread wants of one voice channel (a snapshot entry).
struct Want {
	u32 gen;                // the voice start it plays (0: none)
	u32 base;               // ARAM byte address of the wave
	u32 loopStart, endPos;  // samples
	u32 startPos;           // the sample to start from (a voice that was followed without a channel)
	float l, r;             // volumes
	u16 ratio;              // Q12
	u16 type;               // 9, 5, 8 or 16
	s16 yn1, yn2;           // ADPCM history at the loop start's frame
	u8 play;                // 0: stop (finished, released or disabled)
	u8 paused;
	u8 looping;
	u8 pad;
};
struct Snap {
	volatile u32 seq;
	Want ch[kHwChans];
};

// The audio thread's view of each voice block.
struct Slot {
	u8 how;    // kIdle, kOnDsp, kInMixer, kQuiet, kUnheard
	s8 ch;     // kOnDsp: its channel
	u8 quiet;  // kOnDsp: subframes in a row at zero volume
	u8 ended;  // the position ran past a one-shot's end
	u32 gen;   // kOnDsp: the channel start it plays
	u64 pos;   // Q12 position in the wave at the subframe's start, wrapped into its loop
};

// ndsp's thread's view of each voice channel.
struct Feed {
	Want on;           // the state applied
	bool playing;      // a voice is decoding into the channel
	bool ended;        // its one-shot ran out of samples
	bool queuedAny;    // a buffer has been queued since the start
	u8 next;           // the next buffer to fill
	u32 idx;           // the next source sample to decode
	s32 h1, h2;        // ADPCM history
	const u8* wave;    // ARAM at the wave's base
	u32 waveBytes;     // ARAM from there to its end
	s16* mem;          // kBufs * kBufCap samples, linear memory
	ndspWaveBuf wb[kBufs];
};

int s_mode;                 // settings.txt audio
volatile int s_play = kDsp; // how voices play now (audio_cycle changes it)
int s_cycleSeconds;
bool s_ready;               // ndsp is running
// How the buffers are written back from the CPU's caches to RAM for the DSP: the cheapest that
// works, as the bench measured it at start.
enum { kWayStore, kWayFlush, kWayService };
const char* const kWayName[] = {"svcStoreProcessDataCache", "svcFlushProcessDataCache", "DSP_FlushDataCache"};
int s_way = kWayService;
u32 s_spanTicks, s_regionTicks;  // the bench: writing back a 4 KB span, and all the voices' buffers
s16* s_voiceMem;                 // the voices' buffers, kHwChans * kBufs * kBufCap samples
const u32 kVoiceBytes = kHwChans * kBufs * kBufCap * sizeof(s16);
Voice* s_voices;
u32 s_nvoices;
s16 s_coef[16][2];
u32 s_coefPacked[16];  // coef2 << 16 | coef1, for SMLAD against the history (h2 << 16 | h1)
float s_panL[256], s_panR[256];
const u8* s_aram;

Slot s_slot[64];
s8 s_owner[kHwChans];  // the voice block playing on each channel, or -1
Want s_want[kHwChans];
u32 s_gen;
Snap s_snaps[kSnaps];
volatile u32 s_published;            // snapshots written
volatile u32 s_finished[kHwChans];   // ndsp's thread: the voice start that ran to its end

// ndsp's thread
Feed s_feed[kHwChans];
u32 s_read;           // snapshots applied
float s_credit;
u32 s_minLeft = ~0u;  // the fewest snapshots left waiting after a frame's, in this window
u32 s_paceFrames;
u32 s_lastPublished, s_quietFrames;  // DSP frames since the audio thread last wrote a snapshot
bool s_held;                          // ... so many that the channels are paused
ndspWaveBuf s_sinkWb[kSinkBufs];
s16* s_sinkMem;
int s_sinkNext;
bool s_sinkOpen;
volatile bool s_sinkStarted;  // ai.cpp has written a block: its DMA runs

// Counters. Those ndsp's thread adds to are 32-bit (single stores); the audio thread reads them
// for the [audio] lines, taking differences, which stay right across a wrap.
volatile u32 s_threadTicks;      // the audio thread holding the emulated CPU
u32 s_threadIn;
u32 s_renderTicks;               // in port_audio_hw_render, less the software mixer
u32 s_mixerTicks;                // the software mixer
volatile u32 s_mainLoopTicks;    // the game thread in MSound::mainLoop, less the audio thread's turns
volatile u32 s_cbTicks;          // ndsp's thread in ndsp_frame
volatile u32 s_decodeTicks;      // ... decoding
volatile u32 s_flushTicks;       // ... writing the buffers back for the DSP
volatile u32 s_flushCalls, s_regionFlushes, s_exactSeeks, s_approxSeeks;
volatile u32 s_seqTicks;         // the audio thread in Kernel::subframeCallback (the sequencer)
volatile u32 s_chanTicks;        // ... in TDSPChannel::updateAll (the channels' updates)
volatile u32 s_dspFrames, s_refills, s_dry, s_waits, s_skips, s_holds, s_sinkDry, s_sinkFull;
u32 s_subframes, s_voiceSum, s_voiceMax, s_mixerSum, s_unheardSum, s_starts, s_overflow;
u32 s_silentDspSum, s_silentMixerSum, s_silentUnheardSum, s_mixerSeSum;  // per subframe: at zero volume; auto-mixer
u32 s_quietSum, s_dspStarts, s_steals, s_stealsHeard, s_toQuiet, s_unplaced;  // followed without a channel; channel starts

// JAudio's turns on core 2 (their code and comment: further down, before the benchmark).
enum { kSerial, kDriver };
bool s_parallel;                    // audio_thread = core2, and the thread runs
volatile int s_blocksBy = kSerial;  // who runs the AI blocks now
s16* (*volatile s_thpMix)(s32);     // THP's mix callback (JAudio is given thp_mix in its place)
s32 (*volatile s_foreignCb)(void*); // the last subframe callback not the sequencer's (a stream)
void* volatile s_foreignData;
s32 (*volatile s_pendingCb)(void*); // ... registered while the turns ran: the driver registers it at
void* volatile s_pendingData;       // its next block and hands the blocks to the audio thread
LightLock s_alock;                  // the audio lock
__thread int t_lockDepth;           // this host thread's depth in the game's sound code
__thread bool t_turn;               // this is the turn's thread
Thread s_driver;
LightEvent s_blockDue;
volatile bool s_driverRun;
// With audio_thread = core2 the voices' decoding moves to core 2 too (feed_main): ndsp's thread runs
// on the game's core, above the game thread, and its frame callback's decoding took ~1.2 ms of every
// game frame there. The callback counts the DSP frames due; this thread on core 2 works through
// them in turn, each as the callback did (pace, refill, write-back), and is the only one that does.
Thread s_feeder;
LightEvent s_feedDue;
volatile bool s_feedRun;
volatile u32 s_feedFrames, s_feedDone;  // DSP frames given to the feeder, and done
volatile u32 s_turnTicks, s_turnWaitTicks, s_emuWaitTicks, s_turnBlocks, s_serialBlocks, s_handoffs;
volatile u32 s_thpLate;             // turns' blocks a movie's sound started in (left silent)
u32 s_hookCalls;                    // the game's side: calls into its sound code (the lock hooks)
const char* serial_reason();

struct Report {
	u32 sub, thread, render, mixer, mainLoop, cb, decode, flush, flushCalls, regionFlushes, seq, chan;
	u32 frames, dspFrames, refills, dry, waits, skips, holds;
	u32 sinkDry, sinkFull, dropped, starts, overflow, gameFrames;
	u32 dspStarts, steals, stealsHeard, toQuiet, unplaced, exactSeeks, approxSeeks;
	u32 turn, turnWait, emuWait, turnBlocks, serialBlocks, handoffs, hookCalls, thpLate;
	u64 voiceSum, mixerSum, unheardSum, silentDsp, silentMixer, silentUnheard, mixerSe, quiet;
};
Report s_last;

inline u32 ticks() { return (u32)svcGetSystemTick(); }
inline s32 sat16(s32 v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; }
// ARMv6: SMLAD (c + a.lo * b.lo + a.hi * b.hi, signed halves) and SSAT; elsewhere (the offline
// checks on a PC) the same in C.
#if defined(__arm__) && __ARM_ARCH >= 6
inline s32 smlad(u32 a, u32 b, s32 c) { return __smlad((int16x2_t)a, (int16x2_t)b, c); }
inline s32 sat16_asr11(s32 v) { return __ssat(v >> 11, 16); }
#else
inline s32 smlad(u32 a, u32 b, s32 c) { return c + (s16)a * (s16)b + (s16)(a >> 16) * (s16)(b >> 16); }
inline s32 sat16_asr11(s32 v) { return sat16(v >> 11); }
#endif
inline float rate_hz(u16 ratio) { return ratio * (kDacRate / 4096.0f); }
bool playable(u16 type) { return type == 9 || type == 5 || type == 8 || type == 16; }

// Bus addresses (connect_table): 1 = 0x0D00 is the left output, 2 = 0x0D60 the right.
const u16 kBusL = 0x0D00, kBusR = 0x0D60;

// --- ndsp's thread -----------------------------------------------------------------------------

Result write_back(int way, const void* p, u32 bytes)
{
	switch (way) {
	case kWayStore: return svcStoreProcessDataCache(CUR_PROCESS_HANDLE, (u32)(uintptr_t)p, bytes);
	case kWayFlush: return svcFlushProcessDataCache(CUR_PROCESS_HANDLE, (u32)(uintptr_t)p, bytes);
	default: return DSP_FlushDataCache(p, bytes);
	}
}

void flush(const void* p, u32 bytes)
{
	if (R_FAILED(write_back(s_way, p, bytes))) DSP_FlushDataCache(p, bytes);
}

// The spans a DSP frame decoded into, written back together at its end (before ndsp hands the
// new buffers to the DSP): spans next to each other (a channel's buffers lie in a row) as one,
// and all of them as one call over every channel's buffers when that is cheaper.
struct Span {
	const u8* p;
	u32 n;
};
Span s_pend[2 * kHwChans];
int s_npend;

void pend(const void* p, u32 bytes)
{
	const u8* b = (const u8*)p;
	if (s_npend) {
		Span& l = s_pend[s_npend - 1];
		if (b >= l.p + l.n && b <= l.p + l.n + 4096) {  // after the last, at most a buffer's tail between
			l.n = (u32)(b + bytes - l.p);
			return;
		}
	}
	s_pend[s_npend].p = b;
	s_pend[s_npend].n = bytes;
	s_npend++;
}

void flush_pending()
{
	if (!s_npend) return;
	if (s_regionTicks && (u32)s_npend * s_spanTicks > s_regionTicks) {
		flush(s_voiceMem, kVoiceBytes);
		s_regionFlushes++;
		s_flushCalls++;
	} else {
		for (int i = 0; i < s_npend; i++) flush(s_pend[i].p, s_pend[i].n);
		s_flushCalls += s_npend;
	}
	s_npend = 0;
}

// Samples j to end (end <= 16) of the AFC frame holding sample f.idx into out, as dsp_mixer.cpp's
// decode_at: a header byte (scale exponent, predictor), then 16 4-bit (9-byte frames) or 2-bit
// (5-byte frames) residuals, high bits first; y = sat16(((r << scale) << 11) + c1 y1 + c2 y2) >> 11).
// Each sample is one SMLAD (both predictor products and the residual) and one SSAT (the shift and
// the saturation), the history kept packed for the next.
void afc_frame(Feed& f, u32 fbytes, s16* out, u32 j, u32 end)
{
	const u32 fr = f.idx >> 4;
	f.idx = (fr << 4) + end;
	if ((fr + 1) * fbytes > f.waveBytes) {
		memset(out, 0, (end - j) * sizeof *out);
		return;
	}
	const u8* p = f.wave + fr * fbytes;
	const int sh = (p[0] >> 4) + 11;
	const u32 coef = s_coefPacked[p[0] & 15];
	u32 hist = (u32)(u16)f.h2 << 16 | (u16)f.h1;
	const u8* d = p + 1;
#define AFC_STEP(r)                                              \
	do {                                                         \
		const s32 v_ = sat16_asr11(smlad(coef, hist, (r) << sh)); \
		hist = hist << 16 | (u16)v_;                             \
		*out++ = (s16)v_;                                        \
	} while (0)
	if (fbytes == 9) {
		if (j & 1) {
			AFC_STEP((s32)((u32)d[j >> 1] << 28) >> 28);
			j++;
		}
		for (; j + 1 < end; j += 2) {
			const u32 b = d[j >> 1];
			AFC_STEP((s32)(b << 24) >> 28);
			AFC_STEP((s32)(b << 28) >> 28);
		}
		if (j < end) AFC_STEP((s32)((u32)d[j >> 1] << 24) >> 28);
	} else {
		for (; j < end; j++)  // 2-bit residuals on the 4-bit scale
			AFC_STEP(((s32)((u32)d[j >> 2] << (24 + 2 * (j & 3))) >> 30) * 4);
	}
#undef AFC_STEP
	f.h1 = (s16)hist;
	f.h2 = (s16)(hist >> 16);
}

// Up to n samples of the channel's wave into out: the samples dsp_mixer.cpp's pull gives, a frame
// at a time. Fewer at a one-shot's end.
int decode(Feed& f, s16* out, int n)
{
	const Want& w = f.on;
	const bool afc = w.type == 9 || w.type == 5;
	int k = 0;
	while (k < n) {
		if (f.idx >= w.endPos) {
			if (!w.looping || w.loopStart >= w.endPos) {
				f.ended = true;
				break;
			}
			f.idx = w.loopStart;
			if (afc) {
				// The stored history is y[-1], y[-2] of the frame holding the loop start: resume
				// at that frame and decode up to the loop start.
				f.idx &= ~15u;
				f.h1 = w.yn1;
				f.h2 = w.yn2;
				s16 skip[16];
				if (w.loopStart & 15) afc_frame(f, w.type, skip, 0, w.loopStart & 15);
			}
			continue;
		}
		if (afc) {
			const u32 first = f.idx & ~15u, j = f.idx & 15;
			u32 end = 16;
			if (end - j > (u32)(n - k)) end = j + (n - k);
			if (first + end > w.endPos) end = w.endPos - first;
			afc_frame(f, w.type, out + k, j, end);
			k += end - j;
		} else if (w.type == 8) {
			out[k++] = f.idx < f.waveBytes ? (s16)((s8)f.wave[f.idx] * 256) : 0;
			f.idx++;
		} else {  // 16: big-endian in ARAM
			const u32 b = f.idx * 2;
			out[k++] = b + 1 < f.waveBytes ? (s16)((f.wave[b] << 8) | f.wave[b + 1]) : 0;
			f.idx++;
		}
	}
	return k;
}

// Decodes and drops samples up to pos, for a voice that was followed without a channel: exactly
// when it is at most kSeekExact samples on from a known state (the wave's start, or its loop start
// with the stored history), else from the frame holding pos with no history. That settles within a
// few samples, while the voice fades in from silence.
void seek(Feed& f, u32 pos)
{
	const Want& w = f.on;
	if (pos >= w.endPos) return;
	const bool afc = w.type == 9 || w.type == 5;
	u32 from = 0;
	s32 h1 = 0, h2 = 0;
	if (w.looping && w.loopStart < w.endPos && pos >= w.loopStart) {
		from = afc ? w.loopStart & ~15u : w.loopStart;
		if (afc) {
			h1 = w.yn1;
			h2 = w.yn2;
		}
	}
	if (pos - from > kSeekExact) {
		from = afc ? pos & ~15u : pos;
		h1 = h2 = 0;
		s_approxSeeks++;
	} else {
		s_exactSeeks++;
	}
	f.idx = from;
	f.h1 = h1;
	f.h2 = h2;
	static s16 scratch[256];
	while (f.idx < pos && !f.ended) {
		const u32 k = pos - f.idx;
		decode(f, scratch, k > 256 ? 256 : (int)k);
	}
}

void set_mix(int c, float l, float r)
{
	float mix[12] = {l, r};
	ndspChnSetMix(c, mix);
}

void start_chan(int c, const Want& w)
{
	Feed& f = s_feed[c];
	ndspChnReset(c);  // its buffers are marked done
	ndspChnSetInterp(c, NDSP_INTERP_POLYPHASE);
	ndspChnSetFormat(c, NDSP_FORMAT_MONO_PCM16);
	ndspChnSetRate(c, rate_hz(w.ratio));
	set_mix(c, w.l, w.r);
	if (w.paused) ndspChnSetPaused(c, true);
	f.on = w;
	f.playing = true;
	f.ended = false;
	f.queuedAny = false;
	f.idx = 0;
	f.h1 = f.h2 = 0;
	f.wave = w.base < kAramSize ? s_aram + w.base : s_aram;
	f.waveBytes = w.base < kAramSize ? kAramSize - w.base : 0;
	if (w.startPos) seek(f, w.startPos);
}

void stop_chan(int c)
{
	ndspChnWaveBufClear(c);
	s_feed[c].playing = false;
}

void apply_chan(int c, const Want& w)
{
	Feed& f = s_feed[c];
	if (w.gen != f.on.gen) {
		if (f.playing) stop_chan(c);
		f.on = w;
		if (w.gen && w.play) start_chan(c, w);
		return;
	}
	if (!f.playing) {  // ran to its end, stopped, or never started
		f.on = w;
		return;
	}
	if (!w.play) {
		stop_chan(c);
		f.on = w;
		return;
	}
	if (w.ratio != f.on.ratio) ndspChnSetRate(c, rate_hz(w.ratio));
	if (w.l != f.on.l || w.r != f.on.r) set_mix(c, w.l, w.r);
	if (w.paused != f.on.paused) ndspChnSetPaused(c, w.paused != 0);
	f.on = w;
}

// Keeps the channel's buffers decoded ahead, each about 32 ms of the wave at its rate: topped up every
// kRefillEvery DSP frames, so that one write-back covers several frames' buffers, or at once when one
// buffer or none is left (a voice just started, or a fast one); two more at most each time.
void refill(int c, bool due)
{
	Feed& f = s_feed[c];
	if (!f.playing) return;
	int busy = 0;
	for (int b = 0; b < kBufs; b++)
		busy += f.wb[b].status == NDSP_WBUF_QUEUED || f.wb[b].status == NDSP_WBUF_PLAYING;
	if (!busy && f.queuedAny && !f.ended) s_dry++;  // the DSP played everything given to it
	if (!due && busy > 1 && !f.ended) return;
	int n = (int)(rate_hz(f.on.ratio) * 0.032f);
	n = n < 256 ? 256 : n > kBufCap ? kBufCap : n;
	for (int filled = 0; filled < 2 && !f.ended; filled++) {
		ndspWaveBuf& b = f.wb[f.next];
		if (b.status == NDSP_WBUF_QUEUED || b.status == NDSP_WBUF_PLAYING) break;
		s16* out = f.mem + f.next * kBufCap;
		const int got = decode(f, out, n);
		if (got <= 0) break;
		pend(out, got * 2);
		b.data_pcm16 = out;
		b.nsamples = got;
		b.looping = false;
		b.adpcm_data = NULL;
		b.offset = 0;
		ndspChnWaveBufAdd(c, &b);
		f.next = (f.next + 1) % kBufs;
		f.queuedAny = true;
		busy++;
		s_refills++;
	}
	if (f.ended && !busy) {  // played to its end
		s_finished[c] = f.on.gen;
		f.playing = false;
	}
}

// The next snapshot at the DSP's pace: 1.96 subframes a frame, holding the last state while the
// audio thread is late. Every 32 frames (156 ms), if more than two snapshots were always waiting,
// the ones beyond two are skipped: after the audio thread catches up on a stall it writes several
// DAC frames' at once (7 each), which would otherwise stay in hand as latency.
void pace()
{
	const u32 p = s_published;
	__dmb();
	if (!p) return;
	// No snapshot for 40 frames (196 ms): the audio thread is held up (a load reads the disc with
	// the emulated CPU held). Pause the channels rather than hold their notes on: on the GameCube the
	// sequencer would have moved on, and the software mixer would have run dry and gone quiet.
	if (p != s_lastPublished) {
		s_lastPublished = p;
		s_quietFrames = 0;
		if (s_held) {
			s_held = false;
			for (int c = 0; c < kHwChans; c++)
				if (s_feed[c].playing) ndspChnSetPaused(c, s_feed[c].on.paused != 0);
		}
	} else if (++s_quietFrames == 40) {
		s_held = true;
		s_holds++;
		for (int c = 0; c < kHwChans; c++)
			if (s_feed[c].playing) ndspChnSetPaused(c, true);
	}
	if (p - s_read > (u32)kSnaps - 2) {  // the audio thread lapped us (we were held up)
		s_read = p - 7;
		s_skips++;
	}
	const u32 from = s_read, avail = p - s_read;
	s_credit += kSubPerFrame;
	u32 take = (u32)s_credit;
	if (take > avail) take = avail;
	if (!avail) {
		s_waits++;
		if (s_credit > 2) s_credit = 2;
	}
	s_read += take;
	s_credit -= take;
	if (avail - take < s_minLeft) s_minLeft = avail - take;
	if (++s_paceFrames >= 32) {
		if (s_minLeft > 2 && s_minLeft != ~0u && s_minLeft <= p - s_read) s_read += s_minLeft - 2;
		s_minLeft = ~0u;
		s_paceFrames = 0;
	}
	if (s_read == from) return;
	const Snap& sn = s_snaps[(s_read - 1) % kSnaps];
	static Want ch[kHwChans];
	const u32 seq = sn.seq;
	__dmb();
	memcpy(ch, (const void*)sn.ch, sizeof ch);
	__dmb();
	if (seq != s_read - 1 || sn.seq != seq) return;  // overwritten meanwhile
	for (int c = 0; c < kHwChans; c++) apply_chan(c, ch[c]);
}

void feed_frame(u32 n);
void ndsp_frame(void*)
{
	const u32 t0 = ticks();
	if (s_dspFrames++ == 0) fprintf(stderr, "[audio] the ndsp thread's frame callback runs on core %d\n", (int)svcGetProcessorID());
	if (s_sinkStarted) {
		const int q = port_audio_out_queued();
		if (q < kSinkTarget) {
			if (s_blocksBy == kDriver)
				LightEvent_Signal(&s_blockDue);  // JAudio's next block, on core 2
			else
				port_irq_kick();  // ai.cpp starts the next block
		}
		if (!q) s_sinkDry++;
	}
	if (s_feeder) {  // the feeder on core 2 does the rest
		s_feedFrames++;
		LightEvent_Signal(&s_feedDue);
		s_cbTicks += ticks() - t0;
		return;
	}
	feed_frame(s_dspFrames);
	s_cbTicks += ticks() - t0;
}

// DSP frame n's work on the voices' buffers: the next snapshot's channel changes, the buffers topped
// up, the new ones written back.
void feed_frame(u32 n)
{
	pace();
	const u32 t1 = ticks();
	const bool due = n % kRefillEvery == 0;
	for (int c = 0; c < kHwChans; c++) refill(c, due);
	const u32 t2 = ticks();
	flush_pending();
	const u32 t3 = ticks();
	s_decodeTicks += t2 - t1;
	s_flushTicks += t3 - t2;
}

void feed_main(void*)
{
	while (s_feedRun) {
		LightEvent_Wait(&s_feedDue);
		while (s_feedRun && s_feedDone != s_feedFrames) {
			const u32 t0 = ticks();
			feed_frame(++s_feedDone);
			s_cbTicks += ticks() - t0;
		}
	}
}

// --- the audio thread (emulated CPU held) ------------------------------------------------------

// Whether a voice block's volumes (at their targets) reach the left or right output.
bool audible(const Voice& v)
{
	if (v.amEnabled) return v.amVolTarget != 0;
	for (int k = 0; k < 6; k++)
		if ((v.slot[k].bus == kBusL || v.slot[k].bus == kBusR) && v.slot[k].target != 0) return true;
	return false;
}

// Its channel freed: the voice followed without one from here (how), or forgotten (kIdle).
void unplace(Slot& s, u8 how)
{
	if (s.how == kOnDsp && s.ch >= 0) {
		s_want[(int)s.ch].play = 0;
		s_want[(int)s.ch].gen = 0;
		s_owner[(int)s.ch] = -1;
	}
	s.how = how;
	s.ch = -1;
	s.quiet = 0;
}

void release(Slot& s) { unplace(s, kIdle); }

// A free channel, or else the one whose voice has been at zero volume longest, that voice then
// followed without it. -1: every channel's voice is being heard.
int take_chan()
{
	int best = -1, quietest = -1;
	for (int c = 0; c < kHwChans; c++) {
		if (s_owner[c] < 0) return c;
		const Slot& o = s_slot[(int)s_owner[c]];
		if (s_want[c].l == 0 && s_want[c].r == 0 && (int)o.quiet > quietest) {
			quietest = o.quiet;
			best = c;
		}
	}
	if (best >= 0) {
		unplace(s_slot[(int)s_owner[best]], kQuiet);
		s_steals++;
	}
	return best;
}

// For a voice just started and heard at `gain`, with every channel's voice heard: the channel of the
// quietest, if it is at most kQuieter as loud, that voice then followed without it (as the DSP of a
// console drops its quietest voices). -1: none is quieter.
int take_quieter(float gain)
{
	int best = -1;
	float lowest = gain * kQuieter;
	for (int c = 0; c < kHwChans; c++) {
		if (s_owner[c] < 0) continue;
		const float g = fmaxf(fabsf(s_want[c].l), fabsf(s_want[c].r));
		if (g < lowest) {
			lowest = g;
			best = c;
		}
	}
	if (best >= 0) {
		unplace(s_slot[(int)s_owner[best]], kQuiet);
		s_stealsHeard++;
	}
	return best;
}

// A voice followed so far (kQuiet) is heard: onto channel c, from its position.
void place(int i, const Voice& v, Slot& s, int c, float l, float r)
{
	if (++s_gen == 0) s_gen = 1;
	s.gen = s_gen;
	s.how = kOnDsp;
	s.ch = (s8)c;
	s.quiet = 0;
	s_owner[c] = (s8)i;
	Want& w = s_want[c];
	memset(&w, 0, sizeof w);
	w.gen = s.gen;
	w.base = v.base;
	w.type = v.sourceType;
	w.looping = v.looping != 0;
	w.loopStart = v.loopStart;
	w.endPos = v.endPosition;
	w.startPos = (u32)(s.pos >> 12);
	w.yn1 = v.loopYN1;
	w.yn2 = v.loopYN2;
	w.ratio = v.ratio;
	w.l = l;
	w.r = r;
	w.paused = v.pause != 0;
	w.play = 1;
	s_dspStarts++;
}

// A voice block was (re)started: decide how it plays. A wave on the DSP path starts followed
// (kQuiet) and gets a channel in the same subframe if it is heard.
void start_voice(Voice& v, Slot& s, int play)
{
	s_starts++;
	s.pos = 0;
	s.ended = 0;
	s.quiet = 0;
	s.gen = 0;
	s.ch = -1;
	if (playable(v.sourceType) && (play == kDsp || play == kSilent)) {
		v.resetVpb = 0;  // the DSP clears it
		s.how = play == kDsp ? kQuiet : kUnheard;
		return;
	}
	// The software mixer: everything when it is the way to play, otherwise streams and
	// oscillators. It sees resetVpb and starts from the beginning.
	s.how = kInMixer;
}

// The volumes the block asks for this subframe (times the master), written back as the software
// mixer does (dsp_mixer.cpp render_voice), but at the subframe's end value instead of ramped over
// it, stop requests' halving included.
void voice_gains(Voice& v, u16 master, float& l, float& r)
{
	const bool stopping = v.endRequested != 0;
	l = r = 0;
	if (v.amEnabled) {
		const s32 cur = v.amVolCurrent, tgt = stopping ? cur / 2 : v.amVolTarget;
		if (stopping) {
			v.amVolTarget = (u16)tgt;
			v.done = tgt == 0;
		}
		l = tgt * s_panL[v.amPan >> 8] * (1.0f / 32768);
		r = tgt * s_panR[v.amPan >> 8] * (1.0f / 32768);
		v.amVolCurrent = (u16)tgt;
	} else {
		bool released = true;
		for (int k = 0; k < 6; k++) {
			MixSlot& m = v.slot[k];
			const s32 tgt = stopping ? (s16)m.current / 2 : (s16)m.target;
			if (stopping) {
				m.target = (u16)tgt;
				released = released && tgt == 0;
			}
			// Buses 3-6 (FX sends) and the surround buses are not played on the 3DS DSP yet.
			if (m.bus == kBusL) l += tgt * (1.0f / 16384);
			if (m.bus == kBusR) r += tgt * (1.0f / 16384);
			m.current = (u16)tgt;
		}
		if (stopping) v.done = released;
	}
	l *= master * (1.0f / 32768);
	r *= master * (1.0f / 32768);
}

// The position one subframe on: past a loop's end back into the loop, past a one-shot's end ended.
void advance(Slot& s, const Voice& v, int n)
{
	s.pos += (u64)v.ratio * n;
	const u64 end = (u64)v.endPosition << 12;
	if (s.pos < end) return;
	if (v.looping && v.loopStart < v.endPosition) {
		const u64 start = (u64)v.loopStart << 12;
		s.pos = start + (s.pos - end) % (end - start);
	} else {
		s.pos = end;
		s.ended = 1;
	}
}

// A playing voice's subframe (not in the software mixer).
void update_voice(int i, Voice& v, Slot& s, u16 master, int n)
{
	float l = 0, r = 0;
	if (v.pause) {
		if (v.endRequested) v.done = 1;
		if (s.how == kOnDsp) {
			l = s_want[(int)s.ch].l;
			r = s_want[(int)s.ch].r;
		}
	} else {
		voice_gains(v, master, l, r);
	}
	const bool heard = l != 0 || r != 0;
	if (s.how == kOnDsp) {
		Want& w = s_want[(int)s.ch];
		w.paused = v.pause != 0;
		if (!v.pause) {
			w.l = l;
			w.r = r;
			w.ratio = v.ratio;
		}
		if (s_finished[(int)s.ch] == s.gen) {
			v.endReached = 1;
			v.done = 1;
		}
		if (heard || v.pause) {
			s.quiet = 0;
		} else if (++s.quiet >= kQuietSubs && !v.done) {
			unplace(s, kQuiet);  // its channel and decoding go to voices being heard
			s_toQuiet++;
		}
	} else if (s.how == kQuiet && heard && !v.done && !s.ended) {
		int c = take_chan();
		if (c < 0 && s.pos < ((u64)kYoung << 12)) c = take_quieter(fmaxf(fabsf(l), fabsf(r)));
		if (c >= 0) {
			place(i, v, s, c, l, r);
		} else {
			s_unplaced++;  // the quietest heard: heard once a channel comes free
		}
	}
	if (!v.pause) advance(s, v, n);
	if ((s.how == kQuiet || s.how == kUnheard) && s.ended && !v.done) {
		v.endReached = 1;
		v.done = 1;
	}
	if (s.how == kOnDsp && v.done) s_want[(int)s.ch].play = 0;
}

// audio_cycle: every voice restarted the new way.
void set_play(int play)
{
	for (u32 i = 0; i < s_nvoices; i++) {
		Slot& s = s_slot[i];
		if (s.how == kIdle) continue;
		release(s);
		Voice& v = s_voices[i];
		if (v.enabled && !v.done) v.resetVpb = 1;
	}
	s_play = play;
}

// This subframe's snapshot for ndsp's thread.
void publish()
{
	if (!s_ready) return;
	const u32 w = s_published;
	Snap& sn = s_snaps[w % kSnaps];
	sn.seq = ~0u;
	__dmb();
	memcpy((void*)sn.ch, s_want, sizeof s_want);
	__dmb();
	sn.seq = w;
	__dmb();
	s_published = w + 1;
}

void report(bool cycleEnd)
{
	uint32_t st[7];
	GX3DS_GetStats(st);
	Report now;
	now.sub = s_subframes;
	now.thread = s_threadTicks;
	now.render = s_renderTicks;
	now.mixer = s_mixerTicks;
	now.mainLoop = s_mainLoopTicks;
	now.cb = s_cbTicks;
	now.decode = s_decodeTicks;
	now.flush = s_flushTicks;
	now.flushCalls = s_flushCalls;
	now.regionFlushes = s_regionFlushes;
	now.seq = s_seqTicks;
	now.chan = s_chanTicks;
	now.silentDsp = s_silentDspSum;
	now.silentMixer = s_silentMixerSum;
	now.silentUnheard = s_silentUnheardSum;
	now.mixerSe = s_mixerSeSum;
	now.quiet = s_quietSum;
	now.dspStarts = s_dspStarts;
	now.steals = s_steals;
	now.turn = s_turnTicks;
	now.turnWait = s_turnWaitTicks;
	now.emuWait = s_emuWaitTicks;
	now.turnBlocks = s_turnBlocks;
	now.serialBlocks = s_serialBlocks;
	now.handoffs = s_handoffs;
	now.thpLate = s_thpLate;
	now.hookCalls = s_hookCalls;
	now.stealsHeard = s_stealsHeard;
	now.toQuiet = s_toQuiet;
	now.unplaced = s_unplaced;
	now.exactSeeks = s_exactSeeks;
	now.approxSeeks = s_approxSeeks;
	now.dspFrames = s_dspFrames;
	now.refills = s_refills;
	now.dry = s_dry;
	now.waits = s_waits;
	now.skips = s_skips;
	now.holds = s_holds;
	now.sinkDry = s_sinkDry;
	now.sinkFull = s_sinkFull;
	now.dropped = s_ready ? ndspGetDroppedFrames() : 0;
	now.starts = s_starts;
	now.overflow = s_overflow;
	now.gameFrames = st[0];
	now.voiceSum = s_voiceSum;
	now.mixerSum = s_mixerSum;
	now.unheardSum = s_unheardSum;
	const Report& a = s_last;
	const u32 sub = now.sub - a.sub;
	const u32 frames = now.gameFrames - a.gameFrames;
	if (sub && frames) {
		const double perFrame = 1.0 / (frames * kTicksPerMs);
		const double secs = sub * 80 / 32028.5;
		// JAudio's work wherever it ran: its audio thread on the game's core, and the turns on core 2.
		const double thread = ((now.thread - a.thread) + (now.turn - a.turn)) * perFrame;
		const double render = (now.render - a.render) * perFrame;
		const double mixer = (now.mixer - a.mixer) * perFrame;
		const double seq = (now.seq - a.seq) * perFrame, chan = (now.chan - a.chan) * perFrame;
		const double jaudio = thread - render - mixer;
		fprintf(stderr,
		        "[audio] %.1f s %s, %lu game frames (%.1f/s): JAudio's blocks %.2f ms/frame (JAudio %.2f: sequencer %.2f, "
		        "channel updates %.2f, the rest %.2f; voices to the DSP %.2f, software mixer %.2f), game thread's "
		        "MSound::mainLoop %.2f ms/frame, ndsp thread's frame callback %.2f ms/frame (decoding %.2f, writing "
		        "back %.2f in %lu calls, %lu of them all the buffers at once)%s\n",
		        secs, kModeName[s_play], (unsigned long)frames, frames / secs, thread, jaudio, seq, chan,
		        jaudio - seq - chan, render, mixer, (now.mainLoop - a.mainLoop) * perFrame, (now.cb - a.cb) * perFrame,
		        (now.decode - a.decode) * perFrame, (now.flush - a.flush) * perFrame,
		        (unsigned long)(now.flushCalls - a.flushCalls), (unsigned long)(now.regionFlushes - a.regionFlushes),
		        cycleEnd ? " (audio_cycle: next way)" : "");
		fprintf(stderr,
		        "[audio]   voices: %lu started, on DSP channels %.1f on average (%.1f at zero volume) and %lu at most, "
		        "%.1f followed without one at zero volume (%lu channel starts, %lu of them restarts: %lu exact, %lu from "
		        "a frame; %lu channels taken from a voice at zero volume, %lu from a quieter voice heard, %lu given up after "
		        "40 ms at zero volume, %lu heard with none free), %.1f in the software mixer (streams, oscillators; %.1f "
		        "at zero volume, %.1f sound effects), %.1f unheard (%.1f at zero volume) | %lu buffers decoded, %lu ran dry | snapshots: %lu waits, "
		        "%lu skips, %lu pauses for a stalled audio thread | AI blocks: %lu ran dry, %lu found no room | DSP "
		        "frames %lu, %lu dropped\n",
		        (unsigned long)(now.starts - a.starts), double(now.voiceSum - a.voiceSum) / sub,
		        double(now.silentDsp - a.silentDsp) / sub, (unsigned long)s_voiceMax,
		        double(now.quiet - a.quiet) / sub, (unsigned long)(now.dspStarts - a.dspStarts),
		        (unsigned long)((now.exactSeeks - a.exactSeeks) + (now.approxSeeks - a.approxSeeks)),
		        (unsigned long)(now.exactSeeks - a.exactSeeks), (unsigned long)(now.approxSeeks - a.approxSeeks),
		        (unsigned long)(now.steals - a.steals), (unsigned long)(now.stealsHeard - a.stealsHeard),
		        (unsigned long)(now.toQuiet - a.toQuiet),
		        (unsigned long)(now.unplaced - a.unplaced),
		        double(now.mixerSum - a.mixerSum) / sub, double(now.silentMixer - a.silentMixer) / sub,
		        double(now.mixerSe - a.mixerSe) / sub,
		        double(now.unheardSum - a.unheardSum) / sub, double(now.silentUnheard - a.silentUnheard) / sub,
		        (unsigned long)(now.refills - a.refills), (unsigned long)(now.dry - a.dry),
		        (unsigned long)(now.waits - a.waits), (unsigned long)(now.skips - a.skips),
		        (unsigned long)(now.holds - a.holds), (unsigned long)(now.sinkDry - a.sinkDry),
		        (unsigned long)(now.sinkFull - a.sinkFull), (unsigned long)(now.dspFrames - a.dspFrames),
		        (unsigned long)(now.dropped - a.dropped));
		fprintf(stderr, "[audio]   the game's sound code: %lu calls a frame (each passes the audio lock's hooks)\n",
		        (unsigned long)((now.hookCalls - a.hookCalls) / frames));
		if (s_parallel) {
			const char* why = s_blocksBy == kDriver ? NULL : serial_reason();
			fprintf(stderr,
			        "[audio]   JAudio on core 2: %.2f ms/frame in %lu blocks (the turns waited %.2f ms/frame for the "
			        "game's sound code), on the game's core %.2f; the game's sound code waited %.2f ms/frame for "
			        "turns, let %lu wait no longer; %lu blocks for the audio thread (movies, streams), %lu a movie's "
			        "sound started in on core 2; now %s%s%s\n",
			        (now.turn - a.turn) * perFrame, (unsigned long)(now.turnBlocks - a.turnBlocks),
			        (now.turnWait - a.turnWait) * perFrame, (now.thread - a.thread) * perFrame,
			        (now.emuWait - a.emuWait) * perFrame, (unsigned long)(now.handoffs - a.handoffs),
			        (unsigned long)(now.serialBlocks - a.serialBlocks), (unsigned long)(now.thpLate - a.thpLate),
			        s_blocksBy == kDriver ? "on core 2" : "the audio thread's", why ? ", for " : "", why ? why : "");
		}
	}
	s_voiceMax = 0;
	s_last = now;
}

// --- JAudio's turns on core 2 (settings.txt audio_thread = core2) ---------------------------------
//
// On the GameCube, JAudio's audio thread ran on the one CPU above the game thread: woken by the AI
// DMA interrupt and by each DSP subframe, it preempted the game wherever interrupts were enabled, and
// the game's sound code (JAI, MSound) disabled interrupts where it shared state with it. Here a thread
// on core 2 does the audio thread's work for each AI block (Kernel::updateDac, then the subframes as
// the DSP's replies would have it), while the game runs on core 0. One lock stands in for that
// preemption: the game's side holds it while it runs sound code (the lock hooks compiled into
// JAInterface, JALibrary, MSound and JAudio's DVD thread, scripts/build_game_lib.sh), the turn holds
// it for each step. The game's side lets it go where its thread switches, and at interrupt check
// points while a turn waits: the places a GameCube's audio thread could have run. Every interleaving
// is therefore one the GameCube allowed. On the turn's thread the emulated OS's interrupt calls do
// nothing (patch 0043) and the DSP does not reply; it calls nothing else of the emulated OS.
// Movies (THP's mix callback) and streams (a subframe callback other than the sequencer's) use the
// emulated OS from inside a block: while one is registered the blocks go back to JAudio's own audio
// thread, as before.
void emu_lock()
{
	if (LightLock_TryLock(&s_alock) == 0) return;
	const u32 t0 = ticks();
	LightLock_Lock(&s_alock);
	s_emuWaitTicks += ticks() - t0;
}

void turn_lock()
{
	port_audio_turn_waiting = 1;
	__dmb();
	const u32 t0 = ticks();
	LightLock_Lock(&s_alock);
	s_turnWaitTicks += ticks() - t0;
	port_audio_turn_waiting = 0;
}

void turn_unlock() { LightLock_Unlock(&s_alock); }

// THPPlayer.c's player; the fields its mix callback tests (the ARM build's offsets: open, internalState,
// audioExist).
extern "C" unsigned char ActivePlayer[];
bool thp_would_mix()
{
	s32 open;
	memcpy(&open, ActivePlayer + 0xA0, sizeof open);
	return open && ActivePlayer[0xA5] == 2 && ActivePlayer[0xA7];
}

// Why a block must go to JAudio's own audio thread (NULL: it need not).
const char* serial_reason()
{
	if (s_thpMix && thp_would_mix()) return "a movie's sound";
	s32 (*cb)(void*) = s_foreignCb;
	if (cb && JASystem::Kernel::checkCallback(cb, s_foreignData) >= 0) return "a stream";
	return NULL;
}

bool serial_needed() { return serial_reason() != NULL; }

// THP's mix callback as JAudio calls it (Kernel::vframeWork). It mixes only while a movie with sound
// plays (and then runs the emulated OS), so the turns test for that before each block; a movie's sound
// that starts inside a turn's block leaves that block without it, and the next block is the audio
// thread's. Otherwise it returns NULL, as THP's callback does.
s16* thp_mix(s32 samples)
{
	s16* (*cb)(s32) = s_thpMix;
	if (!cb) return NULL;
	if (!t_turn) return cb(samples);
	if (thp_would_mix()) s_thpLate++;
	return NULL;
}

void driver_main(void*)
{
	port_os_set_off_cpu(1);
	t_turn = true;
	while (s_driverRun) {
		LightEvent_Wait(&s_blockDue);
		while (s_driverRun && s_blocksBy == kDriver && port_audio_out_queued() < kSinkTarget) {
			turn_lock();
			const u32 t0 = ticks();
			if (s_pendingCb) {
				__real__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(s_pendingCb, s_pendingData);
				s_pendingCb = NULL;
			}
			if (serial_needed()) {
				s_blocksBy = kSerial;  // ai.cpp's ai_poll starts the next block, for the audio thread
				turn_unlock();
				port_irq_kick();
				break;
			}
			port_ai_block();                // the block JAudio made last time starts playing
			JASystem::Kernel::updateDac();  // the audio thread's message 0
			s_turnTicks += ticks() - t0;
			turn_unlock();
			for (;;) {  // its message 1s: a DSP reply for each subframe the DSP was given
				turn_lock();
				const u32 t1 = ticks();
				const u32 left = JASystem::AudioThread::getDSPSyncCount();
				if (left) {
					JASystem::AudioThread::setDSPSyncCount(left - 1);
					if (left == 1)
						JASystem::DSPBuf::finishDSPFrame();
					else
						JASystem::DSPBuf::updateDSP();
				}
				s_turnTicks += ticks() - t1;
				turn_unlock();
				if (!left) break;
			}
			s_turnBlocks++;
		}
	}
}

// --- a benchmark at start (game's core, nothing else running yet) ------------------------------

u8 s_benchWave[8192 * 9];  // 8192 AFC frames: random residuals, mid scales
s16 s_benchDelay[2][48 * 80];  // the FX lines' delay buffers
const u8* bench_aram(u32 addr) { return addr < sizeof s_benchWave ? s_benchWave + addr : NULL; }

// What writing buffers back costs, each way (us a call; -1: refused), and the cheapest way chosen.
void bench_write_back()
{
	static const u32 kSizes[4] = {1024, 4096, 32768, kVoiceBytes};
	double us[3][4];
	for (int way = 0; way < 3; way++) {
		for (int z = 0; z < 4; z++) {
			const int reps = z < 2 ? 16 : z == 2 ? 6 : 3;
			u32 total = 0;
			bool ok = true;
			for (int r = 0; r < reps && ok; r++) {
				memset(s_voiceMem, r, kSizes[z]);  // dirty, as freshly decoded samples are
				const u32 t0 = ticks();
				ok = R_SUCCEEDED(write_back(way, s_voiceMem, kSizes[z]));
				total += ticks() - t0;
			}
			us[way][z] = ok ? total / (reps * kTicksPerMs / 1000) : -1;
		}
	}
	int best = -1;
	for (int way = 0; way < 3; way++)
		if (us[way][1] >= 0 && (best < 0 || us[way][1] < us[best][1])) best = way;
	if (best < 0) best = kWayService;  // none worked here: flush() still falls back to the service
	s_way = best;
	s_spanTicks = us[best][1] > 0 ? (u32)(us[best][1] * kTicksPerMs / 1000) : 0;
	s_regionTicks = us[best][3] > 0 ? (u32)(us[best][3] * kTicksPerMs / 1000) : 0;
	fprintf(stderr, "[audio] bench, writing buffers back for the DSP, us a call for 1 KB / 4 KB / 32 KB / all %lu KB: ",
	        (unsigned long)(kVoiceBytes >> 10));
	for (int way = 0; way < 3; way++)
		fprintf(stderr, "%s %.1f / %.1f / %.1f / %.1f%s", kWayName[way], us[way][0], us[way][1], us[way][2], us[way][3],
		        way < 2 ? "; " : "");
	fprintf(stderr, " (-1: refused). Using %s\n", kWayName[best]);
}

// What a second of sound costs the game's core: sms-pc-port's software mixer with 16 voices (half
// music voices with four sends, half auto-mixer voices, the game's two FX lines) against this
// file's way, the same 16 voices decoded into buffers (and written back for the DSP). First the
// decoder is checked against the mixer's own, on this CPU.
void bench()
{
	port_audio_hw_setup(NULL, 0, NULL);  // the ADPCM table (the game's setup comes later)
	bench_write_back();
	u32 seed = 12345;
	for (u32 i = 0; i < sizeof s_benchWave; i++) {
		seed = seed * 1103515245u + 12345u;
		s_benchWave[i] = (u8)(seed >> 16);
		if (i % 9 == 0) s_benchWave[i] = (u8)((4 + (seed >> 24) % 5) << 4 | ((seed >> 20) & 15));
	}
	static Voice voices[64];
	static u8 fx[4][0x20];
	memset(voices, 0, sizeof voices);
	memset(fx, 0, sizeof fx);
	const int kVoices = 16;
	for (int i = 0; i < kVoices; i++) {
		Voice& v = voices[i];
		v.enabled = 1;
		v.resetVpb = 1;
		v.ratio = (u16)(0x0C00 + i * 0x0080);
		v.sourceType = 9;
		v.looping = 1;
		v.endPosition = 8192 * 16;
		if (i & 1) {
			v.amEnabled = 1;
			v.amVolTarget = v.amVolCurrent = 0x4000;
			v.amPan = 64 << 8;
			v.amFx = 0x20 << 8;
		} else {
			static const u16 kBus[4] = {0x0D00, 0x0D60, 0x0DC0, 0x0E20};
			for (int k = 0; k < 4; k++) {
				v.slot[k].bus = kBus[k];
				v.slot[k].target = v.slot[k].current = k < 2 ? 0x2000 : 0x0800;
			}
		}
	}
	static const s16 kTaps[8] = {0, 0, 846, 846, 4715, 8343, 13421, -4096};
	for (int l = 0; l < 2; l++) {  // as the game's FX scenes: lines 0/1 return to buses 1/2
		u16* f = (u16*)fx[l];
		f[0] = 1;
		f[1] = 48;
		*(u32*)(fx[l] + 4) = (u32)(uintptr_t)s_benchDelay[l];
		f[4] = l ? 0x0D60 : 0x0D00;
		f[5] = 4096;
		f[6] = 0x0F40;
		f[7] = 4096;
		memcpy(fx[l] + 0x10, kTaps, sizeof kTaps);
	}
	port_dspmix_set_aram(bench_aram);
	port_dspmix_setup(64, voices, NULL, NULL, fx);
	// The decoder against the mixer's (port_dspmix_decode_wave: its pull and decode_at), ADPCM4 and 2.
	u32 checked = 0, differ = 0;
	{
		const int kN = 8192 * 16;
		s16* ref = (s16*)malloc(kN * sizeof(s16));
		s16* got = (s16*)malloc(kN * sizeof(s16));
		for (int fmt = 0; fmt < 2 && ref && got; fmt++) {
			const int n = kN;
			port_dspmix_decode_wave(fmt, 0, n, ref);
			Feed f;
			memset(&f, 0, sizeof f);
			f.on.type = fmt ? 5 : 9;
			f.on.endPos = (u32)n;
			f.wave = s_benchWave;
			f.waveBytes = sizeof s_benchWave;
			for (int k = 0; k < n;) k += decode(f, got + k, n - k < 777 ? n - k : 777);
			for (int k = 0; k < n; k++) differ += ref[k] != got[k];
			checked += (u32)n;
		}
		free(ref);
		free(got);
	}
	static s16 a[80], b[80];
	const int kSubs = 400;  // a second
	const u32 t0 = ticks();
	for (int k = 0; k < kSubs; k++) port_dspmix_render(a, b, 80, 0x5000);
	const u32 t1 = ticks();
	port_dspmix_setup(64, NULL, NULL, NULL, NULL);  // the game's own setup comes later
	// This file's way: each voice's second of samples at its rate, 1024 at a time; once decoded only,
	// once also written back as ndsp_frame does.
	u32 samples = 0, tDecode = 0, tBoth = 0;
	for (int pass = 0; pass < 2; pass++) {
		const u32 p0 = ticks();
		for (int i = 0; i < kVoices; i++) {
			Feed f;
			memset(&f, 0, sizeof f);
			f.on.type = 9;
			f.on.looping = 1;
			f.on.endPos = 8192 * 16;
			f.wave = s_benchWave;
			f.waveBytes = sizeof s_benchWave;
			const u32 want = (u32)(rate_hz(voices[i].ratio) * kSubs * 80 / kDacRate);
			s16* out = s_feed[i % kHwChans].mem;
			for (u32 done = 0; done < want; done += 1024) {
				const int got = decode(f, out, 1024);
				if (pass) {
					pend(out, got * 2);
					flush_pending();
				} else {
					samples += got;
				}
			}
		}
		(pass ? tBoth : tDecode) = ticks() - p0;
	}
	const double mixMs = (t1 - t0) / kTicksPerMs, decMs = tDecode / kTicksPerMs, bothMs = tBoth / kTicksPerMs;
	fprintf(stderr,
	        "[audio] bench, a second of 16 voices on the game's core: software mixer %.1f ms (%.1f%% of the core, %.0f "
	        "cycles a voice-sample); for the DSP, decoding %.2f ms (%.0f cycles a sample), with writing back %.2f ms "
	        "(%.2f%% of the core) | decoder against the mixer's: %lu samples, %lu differ\n",
	        mixMs, mixMs / 10, (t1 - t0) * (804.0 / 268.111856) / (kVoices * kSubs * 80.0), decMs,
	        tDecode * (804.0 / 268.111856) / samples, bothMs, bothMs / 10, (unsigned long)checked, (unsigned long)differ);
}

// --- the audio thread's profile (AUDIO_SAMPLE=1 builds) ----------------------------------------
//
// JAudio's functions are compiled with -finstrument-functions there, their hooks renamed to
// __audio_prof_enter/_exit: on the audio thread they push and pop a shadow stack. A thread on core 2
// reads its top every 100 us while the audio thread owns the emulated CPU: the function it is in
// (calls into code that is not JAudio's count for the JAudio function that made them), and the one
// that called that. Every 20 s the audio thread logs the most sampled ([audio-prof]; resolve the
// addresses with addr2line on the build's elf). Other builds never call the hooks, and the sampler
// never starts.
const int kProfDepth = 64, kProfTab = 2048;
struct ProfStack {
	void* fn[kProfDepth];
	volatile u32 depth;
	volatile u32 calls;
};
ProfStack s_prof;
struct ProfEntry {
	u32 fn, caller, n;
};
ProfEntry s_profSelf[kProfTab], s_profPair[kProfTab];
volatile u32 s_profSamples, s_profAudio, s_profLost;
volatile bool s_profRun;
u32 s_profSub;

void prof_count(ProfEntry* tab, u32 fn, u32 caller)
{
	u32 h = (fn * 2654435761u) ^ (caller * 40503u);
	for (int k = 0; k < 16; k++, h++) {
		ProfEntry& e = tab[h & (kProfTab - 1)];
		if (e.n && (e.fn != fn || e.caller != caller)) continue;
		e.fn = fn;
		e.caller = caller;
		e.n++;
		return;
	}
	s_profLost++;
}

void prof_sampler(void*)
{
	void* const audio = JASystem::AudioThread::jac_audioThread;
	while (s_profRun) {
		svcSleepThread(100000);
		s_profSamples++;
		if (__gCurrentThread != audio || !s_threadIn) continue;
		const u32 d = s_prof.depth;
		if (!d) continue;
		const u32 fn = (u32)(uintptr_t)s_prof.fn[(d - 1) % kProfDepth];
		const u32 caller = d >= 2 ? (u32)(uintptr_t)s_prof.fn[(d - 2) % kProfDepth] : 0;
		s_profAudio++;
		prof_count(s_profSelf, fn, 0);
		prof_count(s_profPair, fn, caller);
	}
}

void prof_top(const char* what, ProfEntry* tab, int count, u32 total)
{
	fprintf(stderr, "[audio-prof]   %s:", what);
	for (int k = 0; k < count; k++) {
		int best = -1;
		for (int i = 0; i < kProfTab; i++)
			if (tab[i].n && (best < 0 || tab[i].n > tab[best].n)) best = i;
		if (best < 0) break;
		if (tab[best].caller)
			fprintf(stderr, " %08lx<%08lx %.1f%%", (unsigned long)tab[best].fn, (unsigned long)tab[best].caller,
			        100.0 * tab[best].n / total);
		else
			fprintf(stderr, " %08lx %.1f%%", (unsigned long)tab[best].fn, 100.0 * tab[best].n / total);
		tab[best].n = 0;
	}
	fprintf(stderr, "\n");
}

// The audio thread, each subframe: starts the sampler once the hooks have run, and reports.
void prof_tick()
{
	if (!s_prof.calls) return;
	if (!s_profRun) {
		s_profRun = true;
		if (!threadCreate(prof_sampler, NULL, 8u << 10, 0x18, 2, true) &&
		    !(R_SUCCEEDED(APT_SetAppCpuTimeLimit(80)) && threadCreate(prof_sampler, NULL, 8u << 10, 0x18, 1, true)))
			fprintf(stderr, "[audio-prof] no thread for the sampler\n");
		else
			fprintf(stderr, "[audio-prof] sampling the audio thread every 100 us (JAudio instrumented)\n");
		s_profSub = s_subframes;
		return;
	}
	if (s_subframes - s_profSub < 8008) return;  // 20 s
	s_profSub = s_subframes;
	const u32 total = s_profAudio, all = s_profSamples, lost = s_profLost;
	fprintf(stderr, "[audio-prof] 20 s: %lu samples, %lu (%.1f%%) on the audio thread, %lu lost; %lu hook calls so far\n",
	        (unsigned long)all, (unsigned long)total, all ? 100.0 * total / all : 0.0, (unsigned long)lost,
	        (unsigned long)s_prof.calls);
	if (total) {
		prof_top("self", s_profSelf, 40, total);
		prof_top("self < caller", s_profPair, 40, total);
	}
	memset(s_profSelf, 0, sizeof s_profSelf);
	memset(s_profPair, 0, sizeof s_profPair);
	s_profSamples = s_profAudio = s_profLost = 0;
}

}  // namespace

// --- entry points ------------------------------------------------------------------------------

void __audio_prof_enter(void* fn, void*)
{
	if (__gCurrentThread != (void*)JASystem::AudioThread::jac_audioThread) return;
	const u32 d = s_prof.depth;
	s_prof.fn[d % kProfDepth] = fn;
	s_prof.depth = d + 1;
	s_prof.calls++;
}

void __audio_prof_exit(void*, void*)
{
	if (__gCurrentThread != (void*)JASystem::AudioThread::jac_audioThread) return;
	const u32 d = s_prof.depth;
	if (d) s_prof.depth = d - 1;
}

// From the game thread before port_init. mode: settings.txt audio (0 off, 1 on, 2 software,
// 3 silent). 0 when the DSP cannot be used: the caller then runs without sound data, as before.
int SMS3DS_AudioInit(int mode, int cycleSeconds)
{
	s_mode = mode;
	s_cycleSeconds = cycleSeconds > 0 ? cycleSeconds : 0;
	s_play = mode == kSoft ? kSoft : mode == kSilent ? kSilent : kDsp;
	if (mode == kOff) return 1;
	for (int p = 0; p < 256; p++) {
		const double pan = p / 127.0;
		s_panL[p] = (float)cos(pan * M_PI / 2);
		s_panR[p] = (float)sin(pan * M_PI / 2);
	}
	for (int i = 0; i < 64; i++) s_slot[i].ch = -1;
	for (int c = 0; c < kHwChans; c++) s_owner[c] = -1;
	if (mode == kSilent && !s_cycleSeconds) {
		fprintf(stderr, "[audio] audio = silent: the sound driver runs, no voice plays, no output\n");
		return 1;
	}
	const Result rc = ndspInit();
	if (R_FAILED(rc)) {
		fprintf(stderr, "[audio] ndspInit failed (%08lx; is there a sdmc:/3ds/dspfirm.cdc?): running without sound, as "
		        "with audio = off\n", (unsigned long)rc);
		s_mode = kOff;
		return 0;
	}
	s16* mem = (s16*)pica_linear_alloc(kHwChans * kBufs * kBufCap * sizeof(s16));
	s_sinkMem = (s16*)pica_linear_alloc(kSinkBufs * kSinkCap * 2 * sizeof(s16));
	if (!mem || !s_sinkMem) {
		fprintf(stderr, "[audio] no linear memory for the DSP's buffers: running without sound\n");
		ndspExit();
		s_mode = kOff;
		return 0;
	}
	s_voiceMem = mem;
	memset(s_feed, 0, sizeof s_feed);
	for (int c = 0; c < kHwChans; c++) s_feed[c].mem = mem + c * kBufs * kBufCap;
	memset(s_sinkWb, 0, sizeof s_sinkWb);
	ndspSetMasterVol(1.0f);
	for (int c = 0; c < 24; c++) ndspChnReset(c);
	bench();
	ndspSetCallback(ndsp_frame, NULL);
	s_ready = true;
	fprintf(stderr, "[audio] audio = %s: ndsp running, %d voice channels, %lu KB of linear memory, buffers written "
	        "back by %s%s\n", mode == kSoft ? "software" : mode == kSilent ? "silent" : "on", kHwChans,
	        (unsigned long)((kVoiceBytes + kSinkBufs * kSinkCap * 4) >> 10), kWayName[s_way],
	        s_cycleSeconds ? " (audio_cycle)" : "");
	return 1;
}

// settings.txt audio_update_interval: JAudio updates each voice (envelopes, vibrato, volumes, pan)
// every this many subframes. The game's own is 1, every 2.5 ms; JAudio scales its envelope and slide
// rates by the interval, so their timing is kept. The 3DS DSP takes a voice's state once per 4.9 ms
// frame (about two subframes) in any case, so 2, the 3DS default (main.cpp), saved 0.5-0.6 ms a frame
// in Delfino Plaza and sounded the same, though its values are not the GameCube's.
void SMS3DS_AudioSetUpdateInterval(int subframes)
{
	if (subframes < 1) subframes = 1;
	if (subframes > 4) subframes = 4;
	JASystem::Driver::setUpdateInterval((u8)subframes);
	fprintf(stderr, "[audio] voices updated every %d subframe%s (audio_update_interval; the game's own: 1)\n", subframes,
	        subframes == 1 ? "" : "s");
}

void SMS3DS_AudioExit(void)
{
	if (!s_ready) return;
	if (s_feeder) {  // (finished before ndsp's callback works the frames itself again)
		s_feedRun = false;
		LightEvent_Signal(&s_feedDue);
		threadJoin(s_feeder, 200000000ull);
		threadFree(s_feeder);
		s_feeder = NULL;
	}
	if (s_driver) {
		s_driverRun = false;
		LightEvent_Signal(&s_blockDue);
		threadJoin(s_driver, 200000000ull);  // at most 200 ms: it may wait for the lock
		s_driver = NULL;
	}
	s_ready = false;
	s_sinkOpen = false;
	s_sinkStarted = false;
	ndspSetCallback(NULL, NULL);
	for (int c = 0; c < 24; c++) ndspChnWaveBufClear(c);
	ndspExit();
}

// sms-pc-port's DSP HLE, at the 0x81 setup command.
void port_audio_hw_setup(void* voices, uint32_t nvoices, const uint32_t* adpcm_filter)
{
	s_voices = (Voice*)voices;
	s_nvoices = nvoices > 64 ? 64 : nvoices;
	static const u32 kDefault[16] = {0x0,        0x08000000, 0x00000800, 0x04000400, 0x1000f800, 0x0e00fa00,
	                                 0x0c00fc00, 0x1200f600, 0x1068f738, 0x12c0f704, 0x1400f400, 0x0800f800,
	                                 0x0400fc00, 0xfc000400, 0xfc000000, 0xf8000000};
	const u32* t = adpcm_filter ? adpcm_filter : kDefault;
	for (int i = 0; i < 16; i++) {
		s_coef[i][0] = (s16)(t[i] >> 16);
		s_coef[i][1] = (s16)(t[i] & 0xFFFF);
		s_coefPacked[i] = (u32)(u16)s_coef[i][1] << 16 | (u16)s_coef[i][0];
	}
	s_aram = port_aram_ptr(0);
}

// sms-pc-port's DSP HLE, for each subframe (DSPReleaseHalt). 0: the software mixer does it all.
int port_audio_hw_render(int16_t* outA, int16_t* outB, int n, uint16_t master)
{
	if (s_mode == kOff || !s_voices || !s_aram) return 0;
	const u32 t0 = ticks();
	s_subframes++;
	prof_tick();
	if (s_cycleSeconds && s_subframes % (u32)(s_cycleSeconds * 32028.5 / 80) == 0) {
		report(true);
		set_play(s_play == kDsp ? kSoft : s_play == kSoft ? kSilent : kDsp);
	} else if (s_subframes - s_last.sub >= 2002) {  // 5 s
		report(false);
	}
	const int play = s_play;
	u64 mixer = 0;
	u32 onDsp = 0, unheard = 0;
	if (play == kSoft) {  // as sms-pc-port: every voice and the FX lines in the software mixer
		for (u32 i = 0; i < s_nvoices; i++) {
			const Voice& v = s_voices[i];
			if (!v.enabled || v.done) continue;
			mixer++;
			s_silentMixerSum += !audible(v);
			s_mixerSeSum += v.amEnabled != 0;
		}
		publish();
		s_mixerSum += (u32)mixer;
		const u32 t1 = ticks();
		port_dspmix_render_voices(outA, outB, n, master, ~0ull, 1);
		s_mixerTicks += ticks() - t1;
		s_renderTicks += t1 - t0;
		return 1;
	}
	for (u32 i = 0; i < s_nvoices; i++) {
		Voice& v = s_voices[i];
		Slot& s = s_slot[i];
		if (!v.enabled) {
			if (s.how != kIdle) release(s);
			continue;
		}
		if (v.resetVpb || s.how == kIdle) {
			if (s.how != kIdle) release(s);
			start_voice(v, s, play);
		}
		if (s.how == kInMixer) {
			mixer |= 1ull << i;
			if (!v.done) {
				s_silentMixerSum += !audible(v);
				s_mixerSeSum += v.amEnabled != 0;
			}
			continue;
		}
		if (v.done) {  // finished: JAudio clears it at its next update
			if (s.how == kOnDsp) s_want[(int)s.ch].play = 0;
			continue;
		}
		update_voice((int)i, v, s, master, n);
		if (s.how == kQuiet) {
			s_quietSum++;
		} else if (s.how == kUnheard) {
			unheard++;
			s_silentUnheardSum += !audible(v);
		}
	}
	for (int c = 0; c < kHwChans; c++) {  // by channel: a voice may have taken another's this subframe
		if (s_owner[c] < 0 || !s_want[c].play) continue;
		onDsp++;
		s_silentDspSum += s_want[c].l == 0 && s_want[c].r == 0;
	}
	publish();
	s_voiceSum += onDsp;
	if (onDsp > s_voiceMax) s_voiceMax = onDsp;
	s_unheardSum += unheard;
	s_mixerSum += __builtin_popcountll(mixer);
	const u32 t1 = ticks();
	if (mixer) {  // FX lines left out: the voices on the DSP send them nothing
		port_dspmix_render_voices(outA, outB, n, master, mixer, 0);
		s_mixerTicks += ticks() - t1;
	} else {
		memset(outA, 0, n * sizeof *outA);
		memset(outB, 0, n * sizeof *outB);
	}
	s_renderTicks += t1 - t0;
	return 1;
}

// ai.cpp: the AI blocks on channel 23.
int port_audio_out_open(void)
{
	if (!s_ready) return 0;
	// settings.txt volume (sms-pc-port's SMS_VOLUME, 0 to 100), for the voices and the AI blocks.
	if (const char* e = getenv("SMS_VOLUME")) {
		const int v = atoi(e);
		ndspSetMasterVol((v < 0 ? 0 : v > 100 ? 100 : v) / 100.0f);
	}
	ndspChnReset(kSinkChan);
	ndspChnSetInterp(kSinkChan, NDSP_INTERP_POLYPHASE);
	ndspChnSetFormat(kSinkChan, NDSP_FORMAT_STEREO_PCM16);
	ndspChnSetRate(kSinkChan, kDacRate);
	set_mix(kSinkChan, 1.0f, 1.0f);
	s_sinkOpen = true;
	return 1;
}

int port_audio_out_queued(void)
{
	int q = 0;
	for (int b = 0; b < kSinkBufs; b++)
		q += s_sinkWb[b].status == NDSP_WBUF_QUEUED || s_sinkWb[b].status == NDSP_WBUF_PLAYING;
	return q;
}

void port_audio_out_write(const s16* lr, u32 frames)
{
	if (!s_sinkOpen) return;
	ndspWaveBuf& b = s_sinkWb[s_sinkNext];
	if (b.status == NDSP_WBUF_QUEUED || b.status == NDSP_WBUF_PLAYING) {
		s_sinkFull++;
		return;
	}
	if (frames > (u32)kSinkCap) frames = kSinkCap;
	s16* out = s_sinkMem + s_sinkNext * kSinkCap * 2;
	memcpy(out, lr, frames * 4);
	flush(out, frames * 4);
	b.data_pcm16 = out;
	b.nsamples = frames;
	b.looping = false;
	b.adpcm_data = NULL;
	b.offset = 0;
	ndspChnWaveBufAdd(kSinkChan, &b);
	s_sinkNext = (s_sinkNext + 1) % kSinkBufs;
	s_sinkStarted = true;
}

// os_thread.cpp, at every switch of the emulated CPU.
void port_os_switch_hook(void* from, void* to)
{
	if (s_mode == kOff) return;
	void* audio = JASystem::AudioThread::jac_audioThread;
	if (s_parallel && t_lockDepth) LightLock_Unlock(&s_alock);  // switched away inside sound code
	if (s_parallel && to == audio) s_serialBlocks++;  // JAudio's audio thread runs a block (a movie, a stream)
	if (to == audio) {
		s_threadIn = ticks();
	} else if (from == audio && s_threadIn) {
		s_threadTicks += ticks() - s_threadIn;
		s_threadIn = 0;
	}
}

void __wrap__ZN6MSound8mainLoopEv(void* self)
{
	if (s_mode == kOff) {
		__real__ZN6MSound8mainLoopEv(self);
		return;
	}
	const u32 t0 = ticks(), a0 = s_threadTicks;
	__real__ZN6MSound8mainLoopEv(self);
	s_mainLoopTicks += (ticks() - t0) - (s_threadTicks - a0);
}

void __wrap__ZN8JASystem6Kernel16subframeCallbackEv(void)
{
	const u32 t0 = ticks();
	__real__ZN8JASystem6Kernel16subframeCallbackEv();
	s_seqTicks += ticks() - t0;
}

void __wrap__ZN8JASystem11TDSPChannel9updateAllEv(void)
{
	const u32 t0 = ticks();
	__real__ZN8JASystem11TDSPChannel9updateAllEv();
	s_chanTicks += ticks() - t0;
}

// --- JAudio's turns on core 2: the entry points ------------------------------------------------

volatile int port_audio_turn_waiting;

// settings.txt audio_thread = core2, from the game thread after SMS3DS_AudioInit: JAudio's blocks are
// run by a thread on core 2 (above the render thread there) instead of JAudio's audio thread.
void SMS3DS_AudioSetThread(int core2)
{
	if (!core2) return;
	if (!s_ready || (s_mode != kDsp && s_mode != kSoft)) {
		fprintf(stderr, "[audio] audio_thread = core2 needs the 3DS DSP's output (audio = on or software): JAudio "
		        "stays on the game's core\n");
		return;
	}
	LightLock_Init(&s_alock);
	LightEvent_Init(&s_blockDue, RESET_ONESHOT);
	s32 prio = 0x30;
	svcGetThreadPriority(&prio, CUR_THREAD_HANDLE);
	prio = prio > 0x1A ? prio - 2 : 0x18;  // the render thread runs one step above the game thread
	s_driverRun = true;
	s_parallel = true;  // before the thread: the lock hooks are live from here on
	s_driver = threadCreate(driver_main, NULL, 32u << 10, prio, 2, false);
	if (!s_driver) {
		s_parallel = false;
		fprintf(stderr, "[audio] no thread on core 2: JAudio stays on the game's core\n");
		return;
	}
	fprintf(stderr, "[audio] audio_thread = core2: JAudio's blocks run on core 2 (priority 0x%lx); the game's sound "
	        "code takes the audio lock\n", (unsigned long)prio);
	// The voices' decoding, at ndsp's own thread's priority (0x18), from the next DSP frame on.
	LightEvent_Init(&s_feedDue, RESET_ONESHOT);
	s_feedRun = true;
	s_feedFrames = s_feedDone = s_dspFrames;
	__dmb();
	s_feeder = threadCreate(feed_main, NULL, 16u << 10, 0x18, 2, false);
	fprintf(stderr, "[audio] the voices' decoding %s\n", s_feeder ? "runs on core 2" : "stays in ndsp's thread (no thread on core 2)");
}

// The lock hooks in the game's sound code (scripts/build_game_lib.sh).
void __audio_lock_enter(void*, void*)
{
	s_hookCalls++;
	if (!s_parallel || t_turn) return;
	if (t_lockDepth++ == 0) emu_lock();
}

void __audio_lock_exit(void*, void*)
{
	if (!s_parallel || t_turn || !t_lockDepth) return;
	if (--t_lockDepth == 0) LightLock_Unlock(&s_alock);
}

// os_thread.cpp: the switched-in thread resumes; inside sound code it takes the lock back.
void port_os_resume_hook(void)
{
	if (s_parallel && t_lockDepth) emu_lock();
}

// os_thread.cpp: an interrupt check point (interrupts on) while a turn waits for the lock.
void port_audio_checkpoint(void)
{
	if (!s_parallel || t_turn || !t_lockDepth) return;
	LightLock_Unlock(&s_alock);
	const u32 t0 = ticks();
	while (port_audio_turn_waiting && ticks() - t0 < (u32)(2 * kTicksPerMs)) {}  // until the turn has it
	emu_lock();
	s_handoffs++;
}

// ai.cpp's ai_poll, at each AI interrupt: 1 while the turns run the blocks. They take them over from
// JAudio's audio thread when it is idle (between DSP frames) and no movie or stream needs it.
int port_audio_parallel_owns(int idle)
{
	if (!s_parallel) return 0;
	if (s_blocksBy == kDriver) return 1;
	if (!idle || !port_ai_running() || serial_needed() || s_pendingCb) return 0;
	s_blocksBy = kDriver;
	LightEvent_Signal(&s_blockDue);
	return 1;
}

// os_thread.cpp: the emulated CPU idles (every thread waits) and comes back; inside sound code the
// lock is let go meanwhile.
void port_os_idle_hook(int idle)
{
	if (!s_parallel || !t_lockDepth) return;
	if (idle)
		LightLock_Unlock(&s_alock);
	else
		emu_lock();
}

// A subframe callback other than the sequencer's (JAIGFrameStream's stream) runs the emulated OS from
// inside a subframe: while the turns run, it waits for the next block's start, where the driver
// registers it and gives the blocks to the audio thread.
int __wrap__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(s32 (*cb)(void*), void* data)
{
	if (!s_parallel || cb == &JASystem::TTrack::rootCallback)
		return __real__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(cb, data);
	__audio_lock_enter(NULL, NULL);
	int r = 0;
	s_foreignCb = cb;
	s_foreignData = data;
	if (s_blocksBy == kDriver) {
		s_pendingCb = cb;
		s_pendingData = data;
	} else {
		r = __real__ZN8JASystem6Kernel24registerSubframeCallbackEPFlPvES1_(cb, data);
	}
	__audio_lock_exit(NULL, NULL);
	return r;
}

// THP's mix callback goes through thp_mix: on the audio thread it is THP's, on a turn it stays out
// (a movie's sound sends the blocks to the audio thread; see serial_reason).
void __wrap__ZN8JASystem6Kernel19registerMixCallbackEPFPslEh(s16* (*cb)(s32), u8 mode)
{
	if (!s_parallel) {
		__real__ZN8JASystem6Kernel19registerMixCallbackEPFPslEh(cb, mode);
		return;
	}
	__audio_lock_enter(NULL, NULL);
	s_thpMix = cb;
	__real__ZN8JASystem6Kernel19registerMixCallbackEPFPslEh(cb ? &thp_mix : NULL, mode);
	__audio_lock_exit(NULL, NULL);
}
