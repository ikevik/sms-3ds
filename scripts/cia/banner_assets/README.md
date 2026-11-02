# Relief banner assets

The fixed artwork and authored relief profile reproduce the compact HOME banner:
Mario faces the screen with F.L.U.D.D., the title stays on the left, and Mario and
the water bob together. Both the character and the lettering have stereoscopic
depth. This is a frontal relief, not a complete model intended to rotate.

## Artwork source

`mario.png` is the 1116 × 1199 rendition of **Mario and Hover Nozzle SMS** from
[Super Mario Wiki](https://www.mariowiki.com/File:Mario_and_Hover_Nozzle_SMS.png).
The artwork is by Nintendo; the source page credits Dracarys for the PSD-to-PNG
version. The artwork is separate from this project's MIT-licensed code.

`mario.etc1a4` is the same artwork prepared as a bottom-up 256 × 256 ETC1A4 texture
with tex3ds, preserving the water's alpha. It is included so the builder needs no
texture compiler. It contains raw GPU data, without tex3ds' four-byte compression
header. `SHA256SUMS` pins the artwork, native texture and profile together.

## Authored depth and layout

`relief.json` contains sampled depths, visible grid cells, layout and the six-key
eight-second bob. The depths were estimated with Depth Anything V2 Small and
then edited, including the nose/shoes and the title's rainbow backing. The large
upward bob was removed to keep Mario inside the HOME safe area. No depth model,
model weights, PyTorch or Blender is downloaded or run by the builder.

The logo image is **not** included here. Each build extracts
`option/timg/title_logo.bti` from the player's `/data/option.szs` and applies the
authored lettering depths; the rainbow paint remains behind the letters.

Do not replace just the PNG: its aspect ratio, coverage, depth grid and native
texture belong together. New artwork needs an authored matching profile and
hardware testing. No disc image, game executable or audio is stored here.
