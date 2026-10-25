Super Mario Sunshine for New 3DS - beta
=======================================

An unofficial native port: the game's decompiled code, running on the 3DS. It plays your own
disc image. No game data, art or music comes with it.

What you need
-------------
- A New 3DS, New 3DS XL or New 2DS XL with custom firmware (Luma3DS). The Old 3DS is too slow.
- Your own Super Mario Sunshine disc image: USA (GMSE01), a plain 1.4 GB .iso.
- For sound, the DSP firmware dump SD:/3ds/dspfirm.cdc. If you have never made it, run DSP1 once.

Setting up the SD card
----------------------
1. Put your disc image at SD:/sms/GMSE01.iso.
2. Copy sms/settings.txt to SD:/sms/settings.txt. It holds the settings tested on a New 3DS (the
   lite = plus speed trades, sound on its own core, ETC1 textures). Without it, the game writes
   its own defaults there, which are slower.
3. Then either:
   - Homebrew Launcher: copy 3ds/sms3ds/sms3ds.3dsx to SD:/3ds/sms3ds/sms3ds.3dsx.
   - HOME Menu: build the .cia (below) and install it with FBI.

Building the .cia (SMS-3DS-CIA-Builder)
---------------------------------------
The .cia is not in the release: its icon (the Shine Sprite), banner (the title logo) and banner
sound (the first seconds of the title theme) are made from your disc. The builder takes about
half a minute and needs Python 3.8 or newer (https://www.python.org/downloads/).

- Windows: double-click "Build SMS CIA.bat" and choose your disc image.
- macOS: double-click "Build SMS CIA.command" and drag your disc image into the window. The first
  time, macOS refuses it as from an unidentified developer: right-click it, choose Open, then Open.
- Linux: python3 build_cia.py /path/to/GMSE01.iso, with makerom and bannertool on your PATH
  (build them from https://github.com/3DSGuy/Project_CTR and https://github.com/diasurgical/bannertool).

It writes output/sms3ds.cia and output/sd-card/, the same SD files as above with the .3dsx given
the same icon. Install sms3ds.cia with FBI. Please don't share the built .cia: it carries art
and music from your disc.

Playing
-------
- The buttons are the GameCube controller's: the Circle Pad is the stick, the C-Stick moves the
  camera, L and R are the triggers, ZL is Z.
- R is a hard press, as the GameCube's R pushed all the way: Mario stops to aim FLUDD. ZR is a
  soft press: FLUDD sprays while Mario moves. SELECT makes R a soft press too, and SELECT again
  switches it back; the bottom screen shows which, and settings.txt's r_press sets the one R
  starts with.
- START + SELECT quits.
- Saves go to SD:/sms/card-a/, and the log of the last run to SD:/sms/log.txt.
- settings.txt is read at start; each setting there has a comment saying what it does.

AI disclosure
-------------
Nearly all of this port (its code, scripts, patches and documentation) was written by an AI:
Claude, Anthropic's model, working through Claude Code. ikevik directed the work and tested every
build, in the Azahar emulator and on a New 3DS. The game's own code is the decompilation and PC
port credited below. Expect mistakes; please report problems at
https://github.com/ikevik/sms-3ds/issues.

Credits
-------
Built from, or with code from:
- doldecomp/sms (https://github.com/doldecomp/sms): the Super Mario Sunshine decompilation.
- chasem-dev/sms-english (https://github.com/chasem-dev/sms-english): its fork that the game code
  is built from.
- chasem-dev/sms-pc-port (https://github.com/chasem-dev/sms-pc-port): the native PC port this one
  is built on.
- chasem-dev/sms-switch (https://github.com/chasem-dev/sms-switch): shim headers and the layout of
  a console port.
- chasem-dev/sms-launcher (https://github.com/chasem-dev/sms-launcher): the bring-your-own-disc
  model.
- devkitPro's devkitARM, libctru and citro3d (https://devkitpro.org).
- The CIA builder: makerom (3DSGuy's Project_CTR, https://github.com/3DSGuy/Project_CTR) and
  bannertool (Steveice10's, in diasurgical's fork: https://github.com/diasurgical/bannertool).

Learned from:
- 2gifts/melee-3ds (https://github.com/2gifts/melee-3ds): the CIA builder that makes the HOME
  Menu art from the player's own disc.
- MiniBandit/MKW-3DS (https://github.com/MiniBandit/MKW-3DS): renderer ideas for citro3d.
- Dolphin (https://github.com/dolphin-emu/dolphin): how the GameCube's GPU behaves.
- Azahar (https://github.com/azahar-emu/azahar): the emulator every change is tested in first.

The licences of the code this release includes are in LICENSES.
