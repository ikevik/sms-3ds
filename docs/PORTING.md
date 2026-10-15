# Porting notes

## Prior art this follows

| Project | What we borrow |
|---|---|
| **chasem-dev/sms-pc-port** | Overall shape: `decomp/` as a submodule of the matching GMSE01 decomp (`chasem-dev/sms-english`), `decomp-patches/` for port-only edits applied to copies at configure time, `platform/` holding host replacements for the GC SDK (graphics, disc, audio, input, OS), `src/` for entry + compat headers, reads assets from the user's disc image at runtime. Also its option/settings model (`settings.txt`, env vars). |
| **chasem-dev/sms-launcher** | Distribution model: user supplies their own GMSE01 rev 0 image; a launcher validates it. On 3DS this becomes the disc check in `main.c`. |
| **encounter/aurora** | Used by Metaforce and melee-native, but **not by sms-pc-port** (correction to an earlier version of this doc). sms-pc-port's own README explains why it wrote its own GX layer instead: Aurora is WebGPU/Dawn + SDL3 + C++20. Aurora is also unusable on 3DS. |
| **sms-pc-port `platform/gx` (`sms_gx`)** | The real reference for the GX layer: GX over OpenGL 3.3, modelled at register level (API calls -> BP/CP/XF register writes -> one command parser shared with display lists and raw FIFO writes -> renderer). The front half (`gx_api`, `gx_fifo`, `gx_texture` decoders) is renderer-independent; the back half (`gx_shader` GLSL generation, `gx_render` FBO/batching, `gx_platform` SDL window) is what a PICA200 backend replaces. MIT licensed. |
| **2gifts/melee-3ds** (`native-3ds/`) | Correction: a real Melee 3DS port exists (New 3DS only, GPLv3 for its 3DS integration). It runs the decomp's game code as **big-endian ARM (BE8)**: built with clang + `ld.lld -m armelfb`, objects converted for a little-endian `.3dsx` (`tools/be8_object.py`), and `setend be`/`setend le` thunks (`port/3ds/game_bridge.S`) switch modes around every call between game and platform. Game data and GX display lists are read natively, no byte-swap layer. Rendering: the game's GX commands are recorded on the main core and translated + drawn by citro3d on another core (`port/3ds/renderer.c`, `render_worker.c`, `gx_thread.c`), with PICA vertex shader variants (`vertex*.v.pica`) and a geometry shader. Hardware-tested: 1v1 at 60 FPS on most stages, 4-player casual at 30 FPS. Unsupported: Old 3DS, multiplayer, movies. |
| **melee-Vita** | Native PS Vita port reusing the Melee decomp and PC-port fixes with its own graphics backend. |
| **chasem-dev/sms-switch, sms-android** | The downstream-layer pattern: `sms-port/` submodule pinned to a pc-port commit, `patches/sms-port/` applied by the build script, and a platform runtime dir (`switch/`: `low_memory.cpp`, `stack_switch.cpp`, `posix_shims.cpp`, shim headers `dlfcn.h`, `execinfo.h`, `sys/mman.h`...). Android reuses pc-port's relocatable MEM1 (`port_mem1_base`, `GXPC_SetMemoryWindow`). The Switch shims are exactly what our 11 failing platform files need. |
| **SM64 3DS ports** (e.g. sm64_3ds) | Working devkitARM + citro3d build setup and a render backend that turns a console GPU's display lists into PICA200 draws. |
| **gameblabla/CSE2** | CMake flags/toolchain pattern for devkitARM + citro3d + NDSP audio backends (an alternative to the plain Makefile used here). |

## What sms-pc-port already solves (found by reading its source)

Reusable as-is or nearly so on a 32-bit little-endian ARM target:
- `decomp-patches/` (263 files): host-compiler leniency, byte-order, bounds and audio fixes.
- `platform/endian/`: in-place big-endian -> native conversion of J3D, JPA, JAudio and SMS formats as resources load.
- `platform/mtx`, `platform/misc/msl_math*`, `src/port_fpu.h`: Gekko paired-single, MSL math and quantised-store emulation.
- `platform/os`, `dvd`, `disc`, `card`, `ar`, `pad`, `audio`, `vi`: host versions of the SDK modules (replace the host-specific parts: our `gcm_disc`, `pad_3ds`, NDSP).
- Build flags (`-std=gnu++03 -fpermissive -fno-strict-aliasing -fwrapv ... -include src/port_compat.h`) and the `tools/gen_sources.py` list of the 588 game units.

Must be new for 3DS: the GX renderer back half, texture upload (GX formats -> PICA formats), audio output, windowing/presentation.

Open risks: (1) the game expects MEM1 at `0x80000000` (`OSCachedToPhysical`); as far as I know a 3DS app can't map that range, so the port needs `GXPC_SetMemoryWindow` plus a consistent physical-address macro (sms_gx's docs describe both). (2) arm-none-eabi-g++ may lack the CP932 exec charset (script checks). (3) Performance is unknown: paired-single emulation and the GX parser run on an ARM11 (VFP11, no NEON).

## Architecture

```
decomp (C++, matching GMSE01)  ->  GC SDK API (GX, DVD, PAD, AX, OS, CARD)
                                     |
                      platform/3ds replaces each module:
                         GX   -> citro3d / PICA200
                         DVD  -> SD card GCM reader        (done: gcm_disc)
                         PAD  -> HID / irrst               (done: pad_3ds)
                         AX   -> NDSP                       (todo)
                         CARD -> SD folder                  (todo)
                         OS   -> libctru (threads, alarms, heap) (todo)
```

## Hard constraints (be realistic)

- **CPU**: Gekko 485 MHz vs ARM11 268 MHz (Old 3DS) / 804 MHz (New 3DS). New 3DS is the
  realistic target; Old 3DS is unlikely to reach playable speed.
- **RAM**: GC has 24 MB main + 16 MB ARAM + 2 MB EFB/1 MB texture cache. Applications on
  New 3DS can get ~124 MB with the right `APPMEMTYPE` (CIA/extended memory); `.3dsx` via
  hbmenu gets less, so test under both. Old 3DS: 64 MB at most.
- **GPU**: PICA200 has 6 texenv stages (GX: 16), 3 texture units (GX: 8 tex gens), no
  geometry shaders for GX-style indexed vertex formats, ETC1 as the only compressed format
  (GX uses CMPR/S3TC-like). Texture conversion at load time is mandatory.
- **Resolution**: 400x240 top screen vs 640x480 (SMS native). Render at 400x240 or letterbox.
- **Endianness / pointers**: the decomp expects big-endian data and 32-bit pointers. ARM11
  is 32-bit little-endian, which helps pointers; the pc-port's handling of big-endian game
  data should be studied (`platform/` in sms-pc-port) before choosing byte-swap strategy.
  I have not verified how it does this.
- **Licensing**: decomp and port contain no assets; never commit the ISO or extracted data.

## Roadmap

1. (done) Toolchain, SD disc reader, input, triangle through citro3d.
2. Add `decomp/` submodule; get a subset (e.g. `JSystem`, `JUtility`) compiling for ARM with a stub SDK.
3. Implement OS/DVD module against the decomp's expectations (heap arenas, threads, alarms).
4. GX backend: viewport, matrices, immediate-mode draw, then textures, TEV collapse, lighting, EFB copies.
5. Load `main.dol` equivalents' assets (SZS archives, BMD/BTI textures) via DVD layer; render title screen.
6. NDSP audio (AX), memory card to SD, then performance work (display lists, LOD, frame cap at 30).
7. Optional: 3D mode, bottom-screen HUD/map, settings file.
