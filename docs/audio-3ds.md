# Game audio on the 3DS (design note)

Status: written 2026-10-21 against c1e062d. Sound is set by `settings.txt audio`, `on` by default since
2026-10-21 for the first public release (`off` while it was a prototype). Code: `platform/3ds/audio_3ds.cpp`,
`patches/sms-port/0041-3ds-audio-ndsp.patch`, `0042-3ds-mixer-no-divisions.patch`. Four versions ran on a New 3DS (logs `log-audio-2.txt` to
`log-audio-5.txt`): music, effects and movie sound play, with no cut-outs in the last two. Section 6 has
what they measured. Numbers not marked as
measured are estimates or offline measurements.

An earlier research note (`docs/audio-plan.md` in worktree `blissful-snyder-bf9a6f`, not committed)
recommended keeping sms-pc-port's software mixer and moving it to core 1. This note goes the other way, as
the frame-rate work now asks: every core is over budget (game thread ~42 ms, GX thread ~40 on core 1,
render thread ~37 on core 2, GPU ~40 against a 33 ms frame), so the mixing goes to the 3DS's DSP.

## Summary

- **The waves are not DSP-ADPCM.** SMS's sound driver (JAudio, with Nintendo's early "Zelda" DSP
  microcode, not AX) stores almost every wave as **AFC** ADPCM: 9-byte frames of 16 samples with a fixed
  table of 16 predictors (some 5-byte 2-bit frames, a few PCM16). The 3DS DSP decodes GameCube
  **DSP-ADPCM** (8-byte frames of 14 samples, 8 predictors per sound), which is a different format, so
  ndsp cannot play SMS's waves as they are.
- **Design.** The CPU decodes AFC into small PCM16 buffers, a few hundred samples at a time, bit for bit
  as sms-pc-port's software mixer decodes it; the 3DS DSP does the expensive part: resampling (pitch),
  volume and pan, and mixing, on ndsp channels 0-22. Streams, oscillators and voices past the 23rd stay in
  the software mixer, whose output goes out with the movies' sound through JAudio's AI blocks on channel
  23. The DSP's progress through those blocks is the game's audio clock.
- **Cost, measured in Delfino Plaza (first version, section 6).** With sound off Plaza ran at 24-26.5 fps.
  With JAudio running and no voice played, 20-23 fps: JAudio itself costs **4.4-5.4 ms a frame**, far more
  than the 1-3 ms estimated. With the voices on the 3DS DSP, 17.6-18.2 fps: on top of JAudio, 4.2 ms of
  decoding and writing buffers back (75 cycles a sample, most of it the cache write-backs), and 5.4 ms of
  software mixing, because Plaza plays **26-28 voices** on average and only 23 fit on the DSP. With every
  voice in the software mixer (422 cycles a voice-sample, measured), 10-15 fps, and the sound broke up.
- **Second version, measured.** Decoding with ARMv6 SMLAD/SSAT (24 cycles a sample), 32 ms buffers
  written back in batches by the cheapest kernel call (`svcStoreProcessDataCache`, 11-15 us a call):
  decoding plus write-backs 2.1 % of the core for 16 voices instead of 4.8 %. The mixer without
  divisions: 344 cycles a voice-sample instead of 422. Plaza with the voices on the DSP: **19.8-23.8 fps**
  (was 17.6-18.2), against 21.4-26 with JAudio alone. JAudio splits into the sequencer (2.0-2.6 ms) and
  the channel updates (1.6-2.2 ms). The overflow into the mixer was still 3.3-8.2 ms.
- **Third version, measured.** Voices at zero volume give their channel up and are followed without
  decoding (section 3). Plaza: **23.3 fps on average** (20-27.6), against 24-26.5 with sound off; per
  frame JAudio 4.66 ms (sequencer 2.26, channel updates 1.81), this file's voice handling 0.42, the
  software mixer 0.93 (a voice heard with every channel's voice heard), ndsp's callback 1.94 (decoding
  1.15, write-backs 0.55). No cut-outs, no buffer ran dry.
- **Fourth version, measured.** No overflow into the software mixer: a voice just started takes the
  channel of the quietest voice being heard if that one is at most 80 % as loud (it is followed and comes
  back when a channel frees), else waits. Buffers are topped up every 4th DSP frame unless nearly dry, so
  one write-back covers four frames' buffers. Plaza: **~24 fps**, sound's work 6.5 ms a frame (was 8.2):
  JAudio 4.33, ndsp callback 1.46 (decoding 0.93, write-backs 0.30), voice handling 0.44, mixer 0.
- **Memory.** The waves stay where they are (the 16 MB ARAM block, already allocated in no-audio
  builds); the DSP path adds 384 KB of linear memory. Transcoding to DSP-ADPCM instead would need 11-15 MB
  of linear memory with 5.3 MB free in Delfino Plaza.
- **Cores.** The sequencer stays on JAudio's audio thread, one of the emulated threads that take turns on
  the game's core (it cannot run in parallel with the game thread; section 4). The decoding runs in ndsp's
  own thread, also on the game's core, every 4.9 ms.
- **Checked offline.** The decoder reproduces the software mixer's samples exactly for all 1680 waves on
  the disc (75 M samples, random buffer sizes, loops played several times over); the build checks its ARM
  version against the mixer's decoder again at start. A simulation with a fake ndsp plays every voice's
  samples exactly, in order, with no buffer running dry, through scheduling jitter and stalls of up to
  800 ms; a voice starts sounding 12-23 ms after the audio thread starts it, on average. The mixer change
  gives the same hash of every output sample and voice block over 40,000 random subframes.
- **Not done yet.** Reverb/echo (JAudio's FX lines) on the DSP path, per-voice filters, surround.

## 1. Loading the game's real sound data

Today `source_game/main.cpp` sets `SMS_NO_AUDIO=1`. sms-pc-port's `noaudio.cpp` then gives `MSound` an
empty sound configuration instead of `mSound.aaf`, and the AI DMA stays idle: no wave, bank or sequence is
loaded and the audio thread never mixes. The DSP task still boots and is set up, which is why the log says
"DSP HLE: audio task booted (software mixer)" and "DSP setup: 64 voices".

The three decomp patches written for no-audio (`3ds-0003`, `-0004`, `-0007`) test `port_no_audio` at run
time, so leaving `SMS_NO_AUDIO` unset is all it takes to load the real data. The prototype does that when
`audio` is `on`, `software` or `silent`.

What JAudio loads (`/AudioRes` on the disc, sizes from the GMSE01 image):

| File | Size | When |
| --- | --- | --- |
| `mSound.aaf` (in `data/nintendo.szs`) | | at boot: the init-data command stream (sound tables, banks, wave systems, FX scenes) |
| `Banks/w1stLoad_0.aw` | 63 KB | first-stay waves, at boot |
| `Banks/w2ndLoad_0.aw` | 10.8 MB | second-stay waves, when the SE sequence starts (`JAIBasic::processFrameWork` → `loadSecondStayWave`); resident from then on |
| `Banks/wScene_N.aw` | 57 KB-4.7 MB (22 files) | the stage's scene waves |
| `Seqs/sequence.arc` | 782 KB | sequences, loaded one at a time into JAI's heaps |
| `Streams/title.afc` | 1.5 MB | a stream, read from disc while it plays |

- **Memory.** ARAM is one 16 MB `calloc` in the regular heap (`platform/ar/ar.cpp`), allocated in
  no-audio builds too, so the waves add nothing. JAudio's heaps (`MSound`'s 0x151800 solid heap) are also
  allocated already. The DSP path adds 368 KB of linear memory for voice buffers and 16 KB for the AI
  blocks; ndsp keeps the DSP firmware (`dspfirm.cdc`) in a heap block. Plaza today has ~17 MB of regular
  heap and 5.3 MB of linear heap free.
- **Load time.** sms-pc-port's DVD reads are synchronous host reads with the emulated CPU held, and JAudio
  loads `w2ndLoad_0.aw` (10.8 MB) through its DVD thread in chunks. Expect a hitch of several seconds early
  on (SD reads of a few MB/s, estimate); the log's `[audio]` lines show when the sound starts. Moving the
  DVD layer's reads off the emulated CPU would remove it, for every load, not only audio.
- **The DSP firmware.** ndsp needs `sdmc:/3ds/dspfirm.cdc` (dumped once by the DSP1 homebrew) or a loader
  that provides `hb:ndsp`. Without either, `ndspInit` fails, the log says so, and the game runs as with
  `audio = off`.

## 2. What the software mixer would cost

sms-pc-port's mixer (`platform/audio/dsp_mixer.cpp`) renders each 80-sample subframe synchronously inside
`DSPReleaseHalt`, on the audio thread, so on the 3DS it would run on the game's core and its time would add
to the game thread's frame. Per voice and output sample it decodes one AFC sample (re-reading the frame
header, through a function-pointer ARAM lookup), runs a 4-tap polyphase resampler, and for each of 2-6
sends does a linear ramp with an integer **division** per sample (`(tgt - cur) * (k + 1) / n`; the ARM11
has no divide instruction, so it is a libgcc call) and a 64-bit multiply. The FX lines add an 8-tap FIR and
a `%` per sample on each active line. The earlier note counted 180-290 cycles per voice-sample.

**Measured** (first version): **422 cycles per voice-sample**, 26.9 % of the core for 16 voices (the bench
at start: half music voices with four sends, half auto-mixer voices, the game's two FX lines). In Delfino
Plaza with every voice in the mixer it took 23.5-48.8 ms a frame (22-29 voices), the frame rate fell to
10-15 fps, and the output ran dry 201 times in 5 s. Even with no voice playing, the FX lines alone cost
1.4 ms a frame.

Patch 0042 removes the per-sample divisions: a ramp whose target equals its start is that value
throughout, and SMS's 80-sample subframes divide by a constant (a multiply); the FX lines wrap their
delay position by subtraction. The output is the same bit for bit. Its new cost is measured by the same
bench in the next run. The software mixer still matters on the DSP path: it mixes streams, oscillators and
the voices past the 23rd.

## 3. The game's voices on the 3DS DSP

### Why not hand ndsp the waves

| Option | Memory (linear) | CPU | Exact | Verdict |
| --- | --- | --- | --- | --- |
| Transcode AFC → DSP-ADPCM at load or into an SD cache | 11-15 MB more (5.3 MB free); or ARAM moved into linear memory and its layout remapped | none while playing; seconds to transcode 11 MB, or a cache | no: frames of 14 vs 16 samples, 8 vs 16 predictors, rounding (`+1024`) differs; a second generation of ADPCM noise | too much memory now |
| Decode whole waves to PCM16 on first use, cached | 2 bytes a sample: the resident set is 40+ MB | a burst at each first use | yes | does not fit |
| **Decode AFC into small rolling PCM16 buffers** | 384 KB | 75 cycles a sample measured with write-backs (first version) | **yes** | **chosen** |

The transcoding route stays open: if linear memory is rebalanced later (ARAM moved out of the regular
heap), the voices could become fire-and-forget DSP-ADPCM buffers and the decode cost would go too.

### How each voice block maps

JAudio writes 64 voice blocks (`DSPInterface::DSPBuffer`, 0x180 bytes; `PROTOCOL.md` in sms-pc-port's
`platform/audio/`). Each subframe, `port_audio_hw_render` reads them where the software mixer would:

| Voice block | 3DS DSP (ndsp) |
| --- | --- |
| `resetVpb` / `enabled` (a start) | a free channel of 0-22; `ndspChnReset`, PCM16 mono, polyphase interpolation, and decoding from sample 0. No channel free, or not a wave (`samplesSourceType` 0x21 stream, oscillator): the software mixer |
| `samplesSourceType` 9 / 5 / 8 / 16 | decoded on the CPU into 4 buffers of up to 2048 samples (~32 ms each at the wave's rate), two refilled per DSP frame at most; the frame's new buffers written back from the CPU's caches together at its end (adjacent ones as one span, or all buffers in one call when that is cheaper), by the cheapest of `svcStoreProcessDataCache`, `svcFlushProcessDataCache` and `DSP_FlushDataCache`, as measured at start |
| `isLooping`, `loopAddress`, `endPosition`, `loopYN1/2` | done in the decoder exactly as the mixer does: at the loop end, back to the frame holding the loop start with the stored history, then decode up to the loop start |
| `resamplingRatio` (Q12 at 32028.5 Hz) | `ndspChnSetRate(ratio × 32028.5 / 4096)` Hz; the DSP resamples with its own polyphase filter instead of the game's `DSPRES_FILTER` (tone differs slightly) |
| `mixChannels[6]` (Q14, ramp to target over the subframe) | sends to bus 1 (0x0D00) add to the left gain, bus 2 (0x0D60) to the right; × master (`DSP_MIXERLEVEL` 0x5000, Q15). The target is applied, not ramped: ndsp changes a gain once per 4.9 ms DSP frame. `current` is written back as the mixer does |
| auto mixer (`useDolbyVolume`; Q15 volume, pan byte) | the mixer's sine law (`cos`/`sin` of pan/127 × π/2, as a table); written back the same |
| `endRequested` (forceStop) | volumes halved each subframe (written back, as the mixer does) until they reach zero, then `done` and the channel stops |
| `useConstantSample` (pause) | `ndspChnSetPaused` |
| `done` / `endReached` (the DSP sets them) | set when the decoder has reached a one-shot's end **and** the DSP has played every buffer: about a DAC frame later than the software mixer would |
| FX sends (buses 3-6), FX lines | **not played** on this path yet: the music and effects are dry |
| surround buses 8/9, per-voice FIR/IIR/low-pass | not played (the software mixer does not do these either) |
| DirectPCM streams (0x21), oscillators | software mixer (FX lines off), into the AI blocks |
| AI blocks (JAudio's DAC output: the software-mixed voices, THP movie sound) | channel 23, stereo PCM16 at 32028.5 Hz, 3 blocks (52 ms) kept queued; each block the DSP finishes kicks the emulated CPU, which starts the next (ai.cpp), so the DSP clocks JAudio |

### Timing

The audio thread runs a DAC frame's 7 subframes in one burst, every 17.5 ms. Applying each subframe's
changes straight away would put every note start and volume change of a frame at the same instant (up to
17.5 ms of timing error, audible in fast rhythms). Instead each subframe's wanted channel state goes into
a snapshot (32 in a ring), and ndsp's thread applies them at the DSP's own pace, 1.96 subframes per 4.9 ms
DSP frame, about a DAC frame behind. If the audio thread is late the channels keep their last state (and
keep playing: the decoding does not wait for the audio thread). If it catches up in a burst after a stall,
the snapshots beyond two in hand are skipped within ~150 ms, so latency does not build up. After 196 ms
with no new snapshot (a load: sms-pc-port reads the disc with the emulated CPU held) the channels are
paused, so held notes do not drone on through the load; they resume where they were.

Offline simulation (fake ndsp that plays the queued buffers at their rates; ~1900 voices over 120 s from
the resident waves, 4-20 at a time, random pitch changes and stops):

| Audio thread | Start latency, average / most | Buffers that ran dry |
| --- | --- | --- |
| on time | 11.6 / 22 ms | 0 |
| up to 8 ms late, a 60 ms stall every 3 s | 20.7 / 75 ms | 0 |
| up to 12 ms late, a 150 ms stall every 2 s | 16.3 / 36 ms | 0 |
| up to 4 ms late, a 30 ms stall every 0.5 s | 23.1 / 52 ms | 0 |
| up to 4 ms late, an 800 ms stall every 4 s (29 pauses) | 14.3 / 35 ms | 0 |

Every voice that ran to its end played all its samples in order; every stopped one played a prefix of
them. With up to 32 voices wanted, the ones without a channel went to the (stand-in) software mixer and the
bookkeeping held.

### Voices at zero volume (third version)

JAudio keeps many voices running at zero volume (distant or muted sounds, notes waiting for their
envelope): in Plaza about a third of its 28-31 voices. They no longer hold a channel:
- A wave starts followed (position kept from its ratio, loops wrapped, a one-shot's end marked done) and
  gets a channel in the same subframe if it is audible. A voice on a channel that has been at zero volume
  for 16 subframes (40 ms) gives it up; an audible voice with no channel free takes the one whose voice has
  been at zero volume longest.
- Taken from where it is: the decoder is started at the voice's position. Exactly (decoded from the wave's
  start, or from its loop start with the stored history, and dropped up to the position) when that is at
  most 4096 samples; otherwise from the ADPCM frame holding the position with no history, which settles
  within a few samples, and the voice is fading in from silence at that moment.
- Heard with no channel free: a voice that started less than 2048 samples ago takes the channel of the
  quietest voice being heard, if that one is at most 80 % as loud (fourth version; the third sent it to the
  software mixer); otherwise it waits, unheard, for a channel, like a console DSP's dropped voices.

Simulated (18-34 voices wanted, voices fading out and back at random): every channel start that the rule
makes exact (~880) played exactly the reference samples from its position; 129 channels were taken from
voices at zero volume; nothing ran dry.

## 4. Which core and thread runs what

| Work | Thread | Core | Notes |
| --- | --- | --- | --- |
| Sequencer (BMS tracks), channel updates (`TDSPChannel::updateAll`), voice blocks → snapshot | JAudio's audio thread (OSThread, priority above the game thread) | 0, taking turns with the game thread | 7 times per 17.5 ms DAC frame |
| JAI frame work (`MSound::mainLoop`), the game's sound starts | game thread | 0 | once a frame |
| Applying snapshots, decoding into buffers, writing them back | ndsp's thread (priority 0x18), our frame callback | 0 | every 4.9 ms |
| ndsp's own channel/state updates | ndsp's thread | 0 | not measurable without vendoring ndsp; ~24 channels × two short loops a frame, estimate ~5-10 µs |
| Resampling, volumes, mixing, output | the 3DS DSP | - | free for the CPU |
| Streams, oscillators, overflow voices | audio thread (software mixer) | 0 | only while such voices play |

**Why the sequencer cannot leave core 0.** sms-pc-port runs every OSThread on one emulated CPU
(`os_thread.cpp`): one host thread at a time holds `g_cpu`, and `OSDisableInterrupts` only stops thread
switches. That is the GameCube's model, and JAudio relies on it: JAI (game thread) and JASystem (audio
thread) share tracks, channels and port command queues guarded by nothing more than disabled interrupts.
Running the audio thread on another core at the same time as the game thread would turn every such
section into a race. It could be done by making `OSDisableInterrupts` a real lock the audio thread holds
for each of its turns, but the game's lock-free patterns would then also meet the ARM11's weak memory
ordering; it is a large, risky change for the 1-3 ms JAudio is estimated to take. Measure first
(section 6).

**Why the decoding stays on core 0.** libctru starts ndsp's thread on the application's default core, and
the callback must run in that thread, before ndsp hands the frame's changes to the DSP. Moving it means
vendoring libctru's `ndsp.c` (zlib licence). Cores 1 and 2 are both over budget (the GX and render
threads), and core 1 gives the application far less than its nominal 80 %, so a fraction of a millisecond
gains nothing there. Vendoring would also give the aux-bus callback libctru leaves out (below), so it is
the natural next step if the callback's time matters.

## 5. The prototype

`settings.txt`:

| Key | Values |
| --- | --- |
| `audio` | `on` (the default: this design), `off` (no sound data, as before: a little faster), `software` (sms-pc-port's mixer for every voice, with the FX lines, out through channel 23: to compare), `silent` (the sound driver runs, no voice plays, no output: to measure JAudio alone) |
| `audio_cycle` | seconds; with `audio` not off, turn through on → software → silent and back every N seconds, each playing voice restarted the new way, logging each way's cost |
| `volume` | sms-pc-port's master volume, 0-100, applied to the DSP's output |
| `audio_update_interval` | subframes between JAudio's updates of each voice: 2 (the default: 0.5-0.6 ms a frame less in Plaza, the same to the ear) or 1 (the game's own) |

Log lines (`sdmc:/sms/log.txt`):

- `[audio] bench, writing buffers back for the DSP, us a call for 1 KB / 4 KB / 32 KB / all ...`: each of the
  three ways, and the one chosen. Then `[audio] bench, a second of 16 voices on the game's core: software
  mixer X ms (Y % of the core, Z cycles a voice-sample); for the DSP, decoding X ms (C cycles a sample),
  with writing back X ms | decoder against the mixer's: N samples, M differ` (M must be 0). Once, before
  the game starts.
- Every 5 s (and at each `audio_cycle` turn):
  `[audio] 5.0 s on the 3DS DSP, N game frames (F/s): audio thread A ms/frame (JAudio J: sequencer, channel
  updates, the rest; voices to the DSP V, software mixer M), game thread's MSound::mainLoop L ms/frame,
  ndsp thread's frame callback C ms/frame (decoding, writing back in K calls)`. The audio thread's time is measured at every switch of the emulated CPU (patch 0041's
  hook in `os_thread.cpp`), so it includes everything JAudio does there; "JAudio" is that less the other
  two. `MSound::mainLoop` is wrapped (`-Wl,--wrap`), less any audio-thread turns inside it.
- `[audio]   voices: S started, on DSP channels avg (at zero volume) and max, in the software mixer (at zero
  volume, sound effects; found no channel), unheard (at zero volume) | buffers decoded, ran dry |
  snapshots: waits, skips, pauses for a stalled audio thread | AI blocks: ran dry, found no room | DSP
  frames, dropped`.

What changed:

- `platform/3ds/audio_3ds.cpp` (new): ndsp set-up, the voice mapping, the decoder, the AI block output,
  the timers and the bench.
- `patches/sms-port/0042-3ds-mixer-no-divisions.patch`: the mixer's ramps and FX wraps without per-sample
  divisions, bit for bit the same.
- `patches/sms-port/0041-3ds-audio-ndsp.patch`: weak hooks in `dsp_hle.cpp` (each subframe and the setup
  go to `audio_3ds.cpp` first), `ai.cpp` (the 3DS output, which is also the clock), `os_thread.cpp` (a
  switch hook), and `port_dspmix_render_voices` in `dsp_mixer.cpp` (the mixer for a subset of voices,
  FX lines optional; `port_dspmix_render` is unchanged in what it does).
- `source_game/main.cpp`: the settings (`audio` defaults to `on`, `SMS3DS_AUDIO_DEFAULT`), `SMS_NO_AUDIO` only for
  `audio = off` or when ndsp cannot start (no `dspfirm.cdc`), ndsp stopped at exit.
- `scripts/build_platform_lib.sh`, `Makefile`: the new unit; the wraps of `MSound::mainLoop`,
  `Kernel::subframeCallback` and `TDSPChannel::updateAll` (timers only).

The offline checks (decoder against the mixer over the disc's waves; the simulation) are host programs
that compile `audio_3ds.cpp` against a stand-in for libctru. They are not in the repository; a copy is in
the side sessions' notes folder (`audio-tests/`, with how to build and run them).

## 6. Measured on a New 3DS

First version, one run with `audio = on`, `audio_cycle = 20` (`log-audio-2.txt`): boot, the opening
stage, the Plaza arrival movie, then Delfino Plaza. Per game frame; Delfino Plaza's windows only (area
1-0, 5 s each). With sound off (`log-audio.txt`, same build), Plaza ran at 24.1-26.5 fps.

| | on (3DS DSP) | software mixer | silent (JAudio only) |
| --- | --- | --- | --- |
| game frames per second | 17.6-18.2 | 9.8-14.8 | 20.2-23.4 |
| audio thread, ms/frame | 11.9-12.1 | 29.7-58.3 | 4.8-5.8 |
| of which JAudio | 6.0-6.3 | 6.0-9.1 | 4.4-5.4 |
| of which voices to the DSP | 0.41-0.43 | 0.23-0.43 | 0.32-0.42 |
| of which software mixer | 5.4 (the 6.3 voices with no channel) | 23.5-48.8 | 0 |
| ndsp callback (decoding and write-backs), ms/frame | 4.4-4.5 | 0.25-0.32 | 0.19-0.21 |
| MSound::mainLoop (game thread), ms/frame | 0.26-0.27 | 0.33-0.39 | 0.24-0.27 |
| voices | 20.5-20.8 on the DSP (23 at most) + 6.3 in the mixer | 22-29 | 26-28 (not played) |
| buffers that ran dry / DSP frames dropped | 0 / 0 | - | - |
| AI blocks that ran dry, per 5 s | 1 | 0-201 | 0-9 |

The start-up bench (nothing else running): software mixer 422 cycles a voice-sample; decoding plus
write-back 94 ns (75 cycles) a sample, against ~30 cycles counted for the decode loop alone.

Memory in Plaza with sound on: 5.17 MB of linear memory free (5.34 MB with it off; the second version
takes 184 KB more), regular heap 70.2 MB in use (69.5 MB off).

What it shows:
- The DSP path's own work is small next to two other costs: **JAudio** (4.4-6.3 ms) and the **overflow**
  (Plaza wants ~27 voices, 23 fit, the rest went to the slow mixer: 5.4 ms).
- The cut-outs heard in that run were the test itself: `audio_cycle` silences the game's sound for 20 s of
  every minute (movies keep theirs), and its software-mixer phase could not keep up in Plaza. The DSP
  phases logged no dry buffer and no stall pause.
- In the opening stage (area 15-0) JAudio took 8.5-9.5 ms a frame with fewer voices: not explained yet;
  the new split timers will show which part.

Second version (`log-audio-3.txt`, `audio_cycle` still on): Delfino Plaza windows.

| | on (3DS DSP) | software mixer | silent (JAudio only) |
| --- | --- | --- | --- |
| game frames per second | 19.8-23.8 | 13.6-19.2 | 21.4-26.0 |
| JAudio, ms/frame (sequencer + channel updates + rest) | 4.3-5.8 (2.2-2.8 + 1.6-2.3 + 0.5) | 4.6-7.0 | 4.1-5.1 (2.0-2.5 + 1.6-2.0 + 0.5) |
| voices to the DSP (this file, audio thread) | 0.31-0.41 | 0.18-0.27 | 0.33-0.44 |
| software mixer | 3.3-8.2 (5.5-11.8 voices with no channel) | 15.0-29.9 | 0 |
| ndsp callback (decoding + write-back) | 1.7-2.8 (1.0-1.8 + 0.5-0.75) | 0.2 | 0.2 |
| voices | 19.6-21.2 on the DSP, 4-11 of them at zero volume | 22-30 | 28-31, 10-16 at zero volume |

Start-up bench: write-backs per call (1 KB / 4 KB / 32 KB / 368 KB) `svcStoreProcessDataCache` 10.9 / 15.2 /
51.8 / 63.9 us, `svcFlushProcessDataCache` 20.6 / 24.8 / 89.8 / 254 us, `DSP_FlushDataCache` (a request to the
DSP service) 197 / 247 / 941 / 1577 us. Software mixer 344 cycles a voice-sample; decoding 24 cycles a sample;
the decoder matched the mixer's on all 262,144 samples checked on the 3DS.

The crash entering Pianta Village (crash dump 37) was the port's own scene probe (`read_scene` in
`source_game/main.cpp`, on the main thread), which followed the stage director's `TCardLoad` pointer while
the stage load had left garbage there (0x7a060). It now follows pointers only into the heap or the
program image.

Third version (`log-audio-4.txt`, `audio = on`, ~50 windows of 5 s in Delfino Plaza), averages (range):

| | per frame |
| --- | --- |
| game frames per second | 23.3 (19.6-27.6); 24.1-26.5 with sound off |
| JAudio: sequencer + channel updates + rest | 4.66 (3.6-6.1): 2.26 + 1.81 + 0.58 |
| this file's voice handling (audio thread) | 0.42 |
| software mixer (voices heard with none free) | 0.93 (0-3.7) |
| ndsp callback: decoding + write-backs + rest | 1.94: 1.15 + 0.55 + 0.24 |
| MSound::mainLoop (game thread) | 0.28 |
| voices on DSP channels / followed without one | 16.5 (0.1 at zero volume) / 11.1 |
| channel restarts, per 5 s | 1.1 (0.2 exact); channels given up 1.4, taken from silent voices 2.8 |
| buffers that ran dry; AI blocks that ran dry, per 5 s | 0; 5 |

No cut-outs were heard, and the stage change to Pianta Village no longer crashed.

Fourth version (`log-audio-5.txt`, Plaza, then the secret slide, then Plaza again), Plaza's steady
windows: 22.4-29.4 fps, about 24 on average. Per frame: JAudio 4.33 (sequencer 2.14, channel updates
1.67), voice handling 0.44, software mixer 0, ndsp callback 1.46 (decoding 0.93, write-backs 0.30 in
~80 calls a second), MSound::mainLoop 0.29. Voices: ~20 on DSP channels, ~10 followed without one;
13-40 a second take the channel of a quieter voice being heard, which then restarts, mostly from a frame,
when one frees.

**JAudio profile** (`/3ds/sms-audio-prof/`, APP_TITLE "SMS audio prof", `log-audioprof.txt`): the same
build with JAudio's functions instrumented (`AUDIO_SAMPLE=1 scripts/build_game_lib.sh`, into its own object
folder), sampled every 100 us on core 2; `[audio-prof]` lines every 20 s list the most sampled functions
and their callers. Six Plaza windows, self time (instrumentation inflates small functions):

| Share | Function (part) |
| --- | --- |
| 7.8 % | `DSPBuf::updateDSP` (mostly this file's voice handling, which runs inside it) |
| 6.1 + 3.8 + 1.3 % | `TOscillator::calc`, `getOffset`, `getOffsetNoCount` (envelopes, vibrato; channels and tracks) |
| 6.1 + 3.0 + 2.7 + 2.5 + 2.4 + 2.2 + 1.5 % | `TTrack::updateSeq`, `updateTrack`, `TSeqParser::mainProc`, `readRegDirect`, `TTrack::mainProc`, `TSeqCtrl::waitCountDown`, `Cmd_Process` (the sequencer) |
| 3.5 + 2.8 + 2.6 + 1.7 % | `TChannel::isOsc`, `Driver::getUpdateInterval`, `updatecallDSPChannel`, `JASDSPChannelGetLogicalChannel` (each voice's update) |
| 3.4 + 2.7 + 2.2 % | `__UpdateJcToDSP`, `DSPBuffer::setMixerVolume`, `flushChannel` (writing the voice block) |
| 2.8 + 2.7 + 1.1 % | `updateEffectorParam`, `calcEffect`, `calcPan` |

About 780,000 JAudio calls a second, ~2,000 a subframe, spread over many small functions: no single
hot spot, no slow library calls (JAudio's sines are tables). Rewriting them one by one would give little
for its risk. The one large lever is how often each voice is updated: SMS leaves JAudio's
`JAS_UPDATE_INTERVAL` at 1, every subframe, but the 3DS DSP takes a voice's state only once per 4.9 ms
frame. `settings.txt audio_update_interval = 2` (default 1, the game's own) updates voices every second
subframe; JAudio divides its envelope and slide rates by the interval, so their timing holds.

Measured (`log-audio-6.txt`, interval 2, against `log-audio-5.txt`, interval 1; Plaza, per frame): channel
updates 1.16 ms instead of 1.77, JAudio 4.02 instead of 4.51, the audio thread 4.54 instead of 4.97. About
0.5-0.6 ms a frame saved: less than half of the channel updates, because `TDSPChannel::updateAll` still
walks all 64 voice slots every subframe. It sounded the same, so 2 is the default since 2026-10-21
(`SMS3DS_AUDIO_UPDATE_INTERVAL_DEFAULT` in `source_game/main.cpp`); 1 gives the game's own updates.

## 7. Next steps

1. Measure the second version (section 6) and listen: music and effects present, pitches and levels
   right, no clicks.
2. **Release default**: `audio = on` (the user's choice for the first public release); without
   `sdmc:/3ds/dspfirm.cdc` the game runs silently. With sound on, Plaza loses ~1-2 fps (JAudio's cost, mostly).
3. **JAudio's own ~4.3 ms a frame** in Plaza, two thirds of what sound costs now and the floor with
   sound on: the sequencer (`Kernel::subframeCallback`, ~2.1 ms) and the channel updates
   (`TDSPChannel::updateAll`, ~1.7 ms), both 400 times a second. The profile build above says which
   functions; then cheaper code paths with the same results.
4. Reverb: JAudio's FX lines (a 120 ms delay and an 8-tap FIR per line, two stereo pairs). The 3DS DSP
   can send each channel to two aux buses and return them; libctru never calls the aux callback (a TODO in
   `ndsp.c`), so this needs a vendored ndsp: the CPU would then filter only the summed aux bus (160
   samples × 4 channels a frame, a few thousand cycles), not each voice. The firmware also has built-in
   delay and reverb effects (`DspMasterStatus`), undocumented.
5. If ndsp's callback still matters: vendor ndsp and move its thread off core 0.
6. The wave load's hitch: DVD reads off the emulated CPU (helps every load).
