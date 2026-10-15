# Super Mario Sunshine – Nintendo 3DS port (WIP)

A native 3DS port of Super Mario Sunshine (GMSE01, NTSC-U rev 0), modelled on how
the existing handheld/PC ports of GameCube decompilations are structured. No game
data is included: the port reads everything from **your own disc image** on the SD card.

## Status

Stage 0 – bring-up. Builds a `.3dsx` that:
- boots, enables New 3DS 804 MHz mode,
- finds and validates `sdmc:/sms/GMSE01.iso` and walks its file system,
- reads the pad (Circle Pad, C-stick, ZL/ZR) into a GameCube-style status,
- draws the test triangle through the citro3d path.

Not implemented yet: the decomp integration, the GX backend, audio, memory card. See `docs/PORTING.md`.

## Build

```sh
# 1. devkitPro + 3ds-dev (macOS: use the devkitPro pkg installer, then)
sudo dkp-pacman -S 3ds-dev
export DEVKITPRO=/opt/devkitpro
export DEVKITARM=$DEVKITPRO/devkitARM

# 2. build
make -j

# 3. run in Azahar/Citra, or copy sms3ds.3dsx to sdmc:/3ds/ and launch from the Homebrew Launcher
```

Put your disc image at `sdmc:/sms/GMSE01.iso` (uncompressed ISO/GCM).

## Layout

```
source/            entry point (later: glue into the decomp's main)
platform/3ds/      replacements for the GameCube SDK, same idea as sms-pc-port's platform/
  gcm_disc.*         DVD layer: GCM/FST reader over the SD card
  pad_3ds.*          PAD layer: 3DS input -> GC pad status
  gx_citro3d.*       GX layer: GX -> citro3d / PICA200 (stub + bring-up)
shaders/           PICA200 vertex shaders (.v.pica)
decomp/            (add) matching decomp as a git submodule
decomp-patches/    (add) 3DS-only edits applied to copies of decomp sources
docs/              PORTING.md: architecture, prior art, constraints, roadmap
scripts/           setup.sh
```
