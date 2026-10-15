# Stage 1: compile the decomp for ARM

Goal: link a static library of the game's code (starting with JSystem's JKernel/JUtility)
for armv6k + hard-float with devkitARM, and call into it from `main.c`.

## Steps

1. **Add sms-pc-port as `ext/sms-pc-port`** (submodule, MIT). We reuse its patches, endian layer,
   math/paired-single emulation, compat headers and build flags rather than copying them.
   `scripts/setup.sh` adds it.
2. **Syntax check on ARM**: `scripts/apply_patches.sh` then `scripts/check_slices.sh` (add `--game` for the game sources). Verified so far: devkitARM g++ on the unpatched decomp passes 22/24 JKernel files; with all 263 pc-port patches applied, every JSystem slice (JKernel, JUtility, JSupport, JGadget, J3D, J2D: 79 files) passes with host g++.
   Host g++ gave 22/24 on JKernel; the 2 failures are fixed by pc-port patch 0018.
   Next slices: `libs/JSystem/src/JUtility/*.cpp`, `JSupport`, `JGadget`, then `J3D`, `J2D`, then game `src/`.
3. **Apply patches**: port `cmake/patches.cmake` to a small shell/Python step that copies touched files
   to `build/patched/` and applies `decomp-patches/*.patch` in name order.
4. **Static library**: compile the passing slice to `libsms_game.a` with the same flags + `-march=armv6k -mfloat-abi=hard`.
   Beware `-ffp-contract=off` (kept) and the objcopy rename of the game's global `operator new/delete`.
5. **Platform stubs**: link against minimal `platform/os` + `dvd` (backed by `gcm_disc`) so JKR heaps and
   archive loading run; print results on the bottom screen.
6. **Smoke test**: init a JKR heap, mount `/scene/...` from the disc through `gcm_disc`, decompress one Yaz0 file, print its size.

## Decisions needed from you
- Plain Makefile (current) vs CMake with `$DEVKITPRO/cmake/3DS.cmake` (pc-port is CMake; reusing `gen_sources.py` output is easier with CMake).
  Recommendation: switch to CMake at step 4.
- Memory target: `.3dsx` via Homebrew Launcher (simplest, less RAM) vs CIA with extended memory (more RAM, needs install).

## Progress log
- Syntax check (devkitARM g++, patched decomp): JSystem 79/79; game src 383/383 once the PCH
  (`SMS.pch`) and GX override include dirs are passed (what `syntax_check_arm.sh` now does).
- `scripts/build_game_lib.sh` compiles all 588 units to `build/libsms_game.a` and prints text/data/bss totals.
  Not included yet (as in sms-pc-port): `platform/*.cpp` (OS, DVD, audio, endian...), the THP decoder, and the
  `objcopy --redefine-syms` rename of the game's global `operator new/delete`.
- ARM compile of all 588 units succeeded on the user's Mac (devkitARM): text 3,878,830 / data 62,451 / bss 1,564,946 bytes.
- Link gap (`scripts/undefined_symbols.sh`, measured on a host build): 491 undefined symbols.
  226 have real implementations in sms-pc-port `platform/` (OS, DVD, CARD, PAD, VI, AR/AI, MTX...).
  171 only have weak stubs: 150 GX/GD (the renderer), 6 DSP, 4 font, 3 THP, PADClamp.
  ~94 come from elsewhere: libc/libstdc++ (newlib on 3DS), the sms_gx vertex writers/`GXPC_*`,
  `port_*` feature flags and `sms_mod_*` hooks, `SaveStateMenu*`, `lbl_80004600` (a link-time address symbol).
- `scripts/build_platform_lib.sh` compiles the platform layer (40 units; all compile on host) to `build/libsms_platform.a`.
- Next milestone ("null GX"): link game + platform + weak GX stubs + newlib into the .3dsx and run the game's init on a
  thread with no rendering. Decide the MEM1 window first (the game wants 0x80000000).

## Progress: 3DS runtime layer (route: pc-port base, BE8 only if needed)
- `scripts/prepare_pcport.sh` copies `ext/sms-pc-port` platform/ + src/ to `build/pcport-3ds` and applies `patches/sms-port/*.patch`.
- `platform/3ds/`: `posix_shims.cpp` (weak dl*/backtrace/mmap/uname/sysconf/pread), `pad_sdk.cpp` (replaces platform/pad/), `include/` fallbacks, `sms_compat_3ds.h` (clock_nanosleep).
- `build_platform_lib.sh` now builds the patched copy, excludes platform/pad, trace, presence, window_icon, port_main.
- Next: run prepare + build_platform_lib + undefined_symbols on ARM; then null-GX link.

## Milestone: link test passes (game + platform + weak GX placeholders)
- build/linktest.elf links with 0 unresolved / 0 duplicate symbols: text 3.98 MB, data 72 KB, bss 1.37 MB.
- Stage 2 starts: `scripts/build_gx_lib.sh` builds sms-pc-port's renderer-independent GX front half
  (gx_api/gx_fifo/gx_vert/gx_trace) + GD + `platform/3ds/gx/gx3ds_backend.cpp`, which implements the
  ~25 renderer entry points the front half calls (batching, indices, display-copy = frame end). It draws
  nothing yet; the PICA200 draw is added in `drawBatch()`.

## Stage 2: first boot attempt (MODE=game)
- `decomp-patches/3ds-0001-mem1-relative.patch`: OSPhysicalToCached & co. are offsets into `port_mem1_base`
  on 3DS (Horizon can't map MEM1 at 0x80000000). `patches/sms-port/0003`: port_init tells the GX layer
  the window via `GXPC_SetMemoryWindow`.
- `make MODE=game` builds `sms3ds.3dsx` from `source_game/main.cpp` + the libraries (output dir `build3ds/`).
  It runs port_init + SMS_main on a 4 MB-stack thread, logs to `sdmc:/sms/log.txt`, shows phase/frame/vertex counters.
