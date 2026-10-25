# Contributing

Thanks for helping with the port. This page is how to get a change from your fork into it.

## Before you start

Open an issue (or comment on one) saying what you want to take on, so that two people don't build
the same thing. Good places to start:

- the bottom screen, which needs development: for example touch to skip cutscenes rather than
  pressing A, and bringing over what chasem-dev's sms-pc-port does with its radar and the progress
  tracking in its pause menu (today the bottom screen shows the port's status text);
- CISO disc images, next to the plain `.iso`;
- the GameCube's 4:3 picture as an option (the top screen is stretched now);
- the final Bowser fight, which may need further testing;
- the occasional boot stall in Azahar (only ever seen there, never on a console), for further
  investigation by beta testers.

## Building

The steps are in the README's [Building from source](README.md#building-from-source), including
which script to run again after a change to which folder. Two rules:

- **The submodules are never edited.** Changes to the game go in `decomp-patches/`, changes to
  sms-pc-port in `patches/sms-port/`, as patch files the scripts apply to copies under `build/`.
- **A `.cia` for your own tests:** `make MODE=game cia ISO=/path/to/GMSE01.iso`, after
  `scripts/cia/build_tools.sh` (macOS) or with `makerom` and `bannertool` on your `PATH`.

## Testing

**In Azahar** first. `sdmc:/sms/autorun.txt` drives unattended runs; the main keys:

| Key | What it does |
|---|---|
| `boot_stage = 1-0` | Boot straight into a stage and episode (1 Delfino Plaza, 2 Bianco Hills, 3 Ricco Harbor, ...) |
| `skip_movies = on` | Skip the movies |
| `quit_after = 30` | Quit after that many seconds of play |
| `buttons = 28:ZR+SUP:4` | Hold buttons at a time, for a while (A B X Y L R ZL ZR START SELECT, D-pad UP DOWN LEFT RIGHT, Circle Pad SUP SDOWN SLEFT SRIGHT, C-stick CUP CDOWN CLEFT CRIGHT) |
| `shots = 29,31.5` | Save the frame at those times (`sdmc:/sms/shot-<tenths>.ppm`) |
| `debug_input = on` | The debug controls (also in `settings.txt`) |

The others are described where `source_game/main.cpp` reads them. Every run writes
`sdmc:/sms/log.txt` (the previous one is kept as `log-prev.txt`); its `[perf]` lines give the frame
time, the GPU's, and each thread's every few seconds, and `[scene]` lines the stage.

**Then on a console.** Azahar doesn't enforce much of what a New 3DS does, so a change that
touches the GPU, threads, memory or the `.cia` isn't done until it has run on hardware. Things that
were fine in Azahar and broke on a console:

- a vertex shader writing an output register more than once, or in pieces, hangs the GPU;
- as a `.cia`, the exheader's `MaxCpu` decides whether the app gets core 1 at all;
- threads still running as the app quits from the HOME Menu kept the process from ending;
- Azahar's speeds aren't a console's: compare frame times measured on the same device.

On a console, `debug_input = on` in `SD:/sms/settings.txt` turns on the touch-screen profiler
(touch the right of the bottom screen, the lowest quarter excepted) and the test modes. Crash dumps
land in `SD:/luma/dumps/arm11/`.

## Pull requests

- One topic per pull request, described in plain words: what was wrong or missing, what changed,
  and why it is correct.
- Say what you tested: Azahar or which console, which stages, and for anything about speed the
  frame times before and after.
- Match the code around your change. Comments say why, not what. Where a change trades accuracy
  for speed, put it behind a setting (as `lite` does) rather than making it the default.
- Nothing from the game: no disc files, extracted assets, art, music or links to them.
- Nothing personal: no `/Users/<name>` paths or real names in files, patches or commit messages.
- AI assistance is welcome, as long as you say so (a `Co-Authored-By:` line in the commit is
  enough) and the result is tested like any other change. See the README's AI disclosure.

## Bugs

Open an issue with: the console (New 3DS, New 3DS XL or New 2DS XL), the `.cia` or the `.3dsx`,
where it happened (stage and episode) and what you saw, and attach `SD:/sms/log.txt` (or
`log-prev.txt` if the game has been started again since), plus the newest crash dump if there is
one.

## Licence

Contributions are under the repository's [MIT licence](LICENSE), the same as the rest of the
port's own code.
