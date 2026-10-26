# Super Mario Sunshine – New Nintendo 3DS port (beta)

A native port of Super Mario Sunshine (GMSE01, NTSC-U) to the New Nintendo 3DS. The decompiled
game code is compiled for the 3DS's ARM CPU, its GameCube graphics (GX) are translated to the
3DS's PICA200 GPU, and its sound plays on the 3DS's DSP. No game data, art or music is included:
the port reads everything from **your own disc image** on the SD card.

## Status: 0.1.1-beta

What works:

- The game, from the opening to the final Bowser fight, on a New 3DS, New 3DS XL or New 2DS XL.
- Music and sound on the DSP, with JAudio on its own core; the movies.
- Saves, as memory card files on the SD card (`SD:/sms/card-a/`).
- Launching from the Homebrew Launcher, or from the HOME Menu once the CIA is installed.

Speed, measured on a New 3DS with the release's `settings.txt` (its `lite = plus` trades): Delfino
Plaza mostly at the game's own cap of 30 frames a second, Ricco Harbor about 29, Bianco Hills about
25, and the slowest moments around 24. The final Bowser fight is slower (below).

Known issues:

- The final Bowser fight runs slowly: its hot-tub water is drawn by the CPU.
- The Old 3DS is not supported: it is too slow.
- The bottom screen shows the port's status text (frame times and counts), not a game screen.

## Installing

You need:

- A New 3DS, New 3DS XL or New 2DS XL with custom firmware (Luma3DS).
- Your own Super Mario Sunshine disc image: USA (GMSE01), a plain 1.4 GB `.iso`.
- For sound, the DSP firmware dump `SD:/3ds/dspfirm.cdc` (run DSP1 once if you never have).

Each release has two zips:

| File | What it is |
|---|---|
| `sms3ds-<version>.zip` | The game for the Homebrew Launcher (`3ds/sms3ds/sms3ds.3dsx`), the tested `sms/settings.txt`, the README and the licences. |
| `SMS-3DS-CIA-Builder-<version>.zip` | Makes the `.cia` for the HOME Menu from your disc image (below). |

Copy these to the SD card:

```
SD:/sms/GMSE01.iso            your disc image
SD:/sms/settings.txt          from the zip (without it the game writes slower defaults)
SD:/3ds/sms3ds/sms3ds.3dsx    for the Homebrew Launcher
```

### The HOME Menu (.cia)

No `.cia` is published: its icon (the Shine Sprite), its banner (the title logo) and the banner's
sound (the first seconds of the title theme) are made from your own disc, so the release carries
none of Nintendo's art or music. The builder takes about half a minute and needs Python 3.8 or
newer (no packages).

- Windows: unzip the builder and double-click `Build SMS CIA.bat`, then choose your disc image.
- macOS: double-click `Build SMS CIA.command` and drag your disc image into its window. The first
  time, macOS refuses it as from an unidentified developer: right-click it and choose Open.
- Linux: `python3 build_cia.py /path/to/GMSE01.iso`, with `makerom` and `bannertool` on your
  `PATH` (built from the sources in the credits below).

It writes `output/sms3ds.cia` and `output/sd-card/` (the same SD files as above, the `.3dsx` with
the same icon). Copy `sms3ds.cia` to `SD:/cias/`, then in FBI choose SD → cias → sms3ds.cia →
Install CIA. Please don't share the built `.cia`: it carries art and music from your disc.

## Controls

| 3DS | Game |
|---|---|
| Circle Pad | Control stick |
| C-Stick (New 3DS) | C-stick (camera) |
| A, B, X, Y | A, B, X, Y |
| L, R | L and R, pressed all the way |
| ZL | Z |
| ZR | R pressed part of the way: FLUDD sprays while Mario moves (or R's opposite: `zr_press`) |
| D-pad, START | D-pad, START |
| SELECT | Switches R between the hard press (Mario stops to aim) and the soft one (as ZR) |
| START + SELECT | Quit |

## Settings

`SD:/sms/settings.txt` is read at start, and each setting there has a comment saying what it does.
The ones a player may want:

- `lite`: `plus` (the release's) trades some detail for speed: Mario's and Yoshi's shadows only, a
  quarter of the particles, a coarser sea, no mirror or Shine shadow. Leave it out for the full
  detail, which is slower.
- `audio_thread = core2`: JAudio on its own core (the release's).
- `etc1_textures = on`: compressed textures as ETC1, so more of them fit in the GPU's memory
  (faster in Bianco Hills and Delfino Plaza; a little colour loss, about a second more per stage
  load).
- `r_press`: `hard` or `soft`, the press R starts as (SELECT switches it).
- `zr_press`: `soft` (ZR is always the soft press) or `opposite` (ZR is whichever press R isn't).
- `stick_deadzone`: how far a stick must move before it counts (15 of about 156); raise it if
  Mario drifts.
- `debug_input = on`: the port's debug and test controls (the touch screen, L + SELECT, buttons
  held at launch). Off by default.

## Building from source

The build runs on macOS or Linux. It needs:

- devkitPro with the `3ds-dev` group (devkitARM, libctru, citro3d, picasso, 3dsxtool):
  https://devkitpro.org/wiki/Getting_Started, then `sudo dkp-pacman -S 3ds-dev`.
- git, make, patch and Python 3.

```sh
git clone --recursive https://github.com/ikevik/sms-3ds.git
cd sms-3ds
export DEVKITPRO=/opt/devkitpro DEVKITARM=/opt/devkitpro/devkitARM

scripts/apply_patches.sh            # build/decomp-patched: the decomp with sms-pc-port's and decomp-patches/ applied
scripts/prepare_pcport.sh           # build/pcport-3ds: sms-pc-port with patches/sms-port/ applied
scripts/build_game_lib.sh -j8       # build/libsms_game.a: the game, JSystem and the movie player
scripts/build_platform_lib.sh -j8   # build/libsms_platform.a: sms-pc-port's SDK layer and the 3DS's
scripts/undefined_symbols.sh        # what the game needs from the layers below it
scripts/link_test.sh                # build/linktest/gx_stubs.o (its trial link then fails: expected)
scripts/build_gx_lib.sh             # build/libsms_gx.a: GX's front half and the 3DS renderer
make MODE=game -j8                  # sms3ds.3dsx
```

The submodules (`decomp/`, `ext/sms-pc-port/`) are never modified: the scripts patch copies of
them under `build/`. What to run again after a change:

- `source_game/`, `platform/3ds/pica/` or `shaders/`: `make MODE=game` only (they are the app's).
- `platform/3ds/*.c*`: `scripts/build_platform_lib.sh`; `platform/3ds/gx/`: `scripts/build_gx_lib.sh`.
- `decomp-patches/`: `scripts/apply_patches.sh`, then the game library.
- `patches/sms-port/`: `scripts/prepare_pcport.sh`, then all three libraries.

A local build's icon can be any 48x48 PNG at `meta/icon.png` (not in the repository). With the
tools that `scripts/cia/build_tools.sh` builds (macOS, with mingw-w64 for the Windows ones),
`make MODE=game cia ISO=/path/to/GMSE01.iso` writes `build/cia/sms3ds.cia`, and
`scripts/cia/package.sh <version>` writes a release's two zips to `build/release/<version>/`.

## Layout

```
source_game/       the 3DS app: entry point, settings, the main loop
platform/3ds/      3DS stand-ins for the GameCube SDK: pad, DSP audio, disc, movies, profiler
  gx/                GX -> PICA200: the renderer (gx3ds_backend.cpp) and the TEV compiler
  pica/              citro3d drawing, EFB copies, the GPU command ring
shaders/           PICA200 vertex shaders (.v.pica)
decomp/            the decompilation (chasem-dev/sms-english), a submodule
ext/sms-pc-port/   the PC port (chasem-dev/sms-pc-port), a submodule
decomp-patches/    3DS-only edits to the decomp
patches/sms-port/  3DS-only edits to sms-pc-port
scripts/           the build steps; scripts/cia/: the CIA builder and the release packaging
docs/              PORTING.md (prior art, design), STAGE1.md, audio-3ds.md, native-j3d.md
source/            the first bring-up app (make without MODE=game)
```

## Contributing

Pull requests and bug reports are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) for how to build,
test and submit a change, and what to include in a bug report.

## AI disclosure

This port was made with heavy AI assistance, and tested rigorously.

- **Written with AI:** nearly all of its code, scripts, patches and documentation were written by
  Claude, Anthropic's model, working through Claude Code: the 3DS platform layer, the GX-to-PICA
  renderer and its TEV compiler, the 3DS patches to sms-pc-port and the decompilation, the CIA
  builder, and this README. Every commit except one merge carries a `Co-Authored-By: Claude` line.
- **Directed and tested by a person:** the project's author (ikevik) decided what to build and what
  to trade away, and tested the work throughout, in the Azahar emulator and on a real New 3DS:
  play-testing through the whole game on the console, profiling runs there, before-and-after
  builds of each change, a checker that tests the renderer's materials against the GameCube's own
  arithmetic, and builds from a clean copy of the tree.

The game's own code is the decompilation and the PC port in the credits below: this repository
builds a 3DS port on top of them. Expect mistakes all the same; please report problems in the
issue tracker.

## Credits

The port is built from, or takes code from:

| Project | What the port takes from it |
|---|---|
| [doldecomp/sms](https://github.com/doldecomp/sms) | The Super Mario Sunshine decompilation (CC0). sms-english forked from it; game code that looks wrong in sms-english is checked against it. |
| [chasem-dev/sms-english](https://github.com/chasem-dev/sms-english) | The decompilation the game code is built from (`decomp/`, CC0), complete for the US version (GMSE01). |
| [chasem-dev/sms-pc-port](https://github.com/chasem-dev/sms-pc-port) | The native PC port this one is built on (`ext/sms-pc-port/`, MIT): its stand-ins for the GameCube SDK, its GX front end and texture decoders, and its decomp patches. The 3DS changes to it are in `patches/sms-port/`. |
| [chasem-dev/sms-switch](https://github.com/chasem-dev/sms-switch) | The POSIX shim headers in `platform/3ds/include/` (MIT), and the layout of a console port on top of sms-pc-port. |
| [chasem-dev/sms-launcher](https://github.com/chasem-dev/sms-launcher) | The bring-your-own-disc model: the player's disc image is checked, and nothing from it ships. |
| [devkitPro](https://devkitpro.org): devkitARM, [libctru](https://github.com/devkitPro/libctru), [citro3d](https://github.com/devkitPro/citro3d) | The toolchain, the 3DS system library and the GPU library. `platform/3ds/pica/c3d_uniforms.c` stands in for citro3d's uniforms object (zlib licence). |
| [3DSGuy/Project_CTR](https://github.com/3DSGuy/Project_CTR) | makerom, which builds the .cia (`scripts/cia/`). |
| [diasurgical/bannertool](https://github.com/diasurgical/bannertool) | bannertool (Steveice10's, in diasurgical's fork), which makes the HOME Menu icon and banner. |

It also learned from:

| Project | What it showed |
|---|---|
| [2gifts/melee-3ds](https://github.com/2gifts/melee-3ds) | A decompilation ported to the New 3DS. Its CIA builder, which makes the HOME Menu art from the player's own disc, is the model for `scripts/cia/`. |
| [MiniBandit/MKW-3DS](https://github.com/MiniBandit/MKW-3DS) | Renderer ideas for citro3d (translate once rather than every frame; leave out GPU state that has not changed) and the same builder approach. |
| [Dolphin](https://github.com/dolphin-emu/dolphin) | How the GameCube's GPU behaves: TEV arithmetic (the TEV compiler's checker), lighting and texture formats. |
| [Azahar](https://github.com/azahar-emu/azahar) | The 3DS emulator every change is tested in first. |

More prior art, and what was taken from each, is in `docs/PORTING.md`.

## Licence

The port's own code is under the MIT licence: see [`LICENSE`](LICENSE). The parts taken from other
projects keep their own licences and notices:

- `platform/3ds/gx/gx3ds_texdecode.cpp` and the shim headers in `platform/3ds/include/` come from
  sms-pc-port and sms-switch (MIT, SoftwareAura LLC; `ext/sms-pc-port/LICENSE`);
- `platform/3ds/pica/c3d_uniforms.c` stands in for citro3d's uniforms object (zlib licence);
- the decompilation (`decomp/`) is CC0, and sms-pc-port (`ext/sms-pc-port/`) is MIT.

None of it covers Super Mario Sunshine itself, which belongs to Nintendo: the port plays your own
disc image, and nothing from the game is in this repository or its releases.

