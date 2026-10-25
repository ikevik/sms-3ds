#!/bin/bash
# The release's files in build/release/<version>/: the Homebrew Launcher package (sms3ds-<version>.zip:
# the .3dsx, settings.txt, README.txt) and the CIA builder (SMS-3DS-CIA-Builder-<version>.zip: the
# stripped ELF, the .3dsx, build_cia.py and the makerom and bannertool binaries).
#   scripts/cia/package.sh 0.1.0-beta
# Needs the game, platform and GX libraries built (scripts/build_*_lib.sh) and the tools in
# build/tools/dist (scripts/cia/build_tools.sh). The .3dsx is linked again here with the default
# icon: the release carries no art (meta/icon.png is for local builds only).
set -euo pipefail
cd "$(dirname "$0")/../.."
VER=${1:?usage: scripts/cia/package.sh <version, as 0.1.0-beta>}
for t in macos/makerom macos/bannertool windows/makerom.exe windows/bannertool.exe LICENSES; do
	[ -e "build/tools/dist/$t" ] || { echo "build/tools/dist/$t is missing: run scripts/cia/build_tools.sh" >&2; exit 1; }
done

# REUSE_BUILD=1: package the .3dsx and ELF already here (an earlier package.sh's, tested as they are).
if [ -z "${REUSE_BUILD:-}" ]; then
	rm -f build3ds/main.o sms3ds.smdh
	make MODE=game ICON= -j8
fi

R="build/release/$VER"
rm -rf "$R"
H="$R/sms3ds-$VER"
mkdir -p "$H/3ds/sms3ds" "$H/sms"
cp sms3ds.3dsx "$H/3ds/sms3ds/"
cp scripts/cia/settings.txt "$H/sms/"
cp scripts/cia/README.txt "$H/"
# The game's binary carries sms-pc-port's and sms-switch's code (the same MIT licence and holder).
mkdir -p "$H/LICENSES"
cp ext/sms-pc-port/LICENSE "$H/LICENSES/sms-pc-port-and-sms-switch.txt"

K="$R/SMS-3DS-CIA-Builder"
mkdir -p "$K/tools"
cp scripts/cia/build_cia.py scripts/cia/gcdisc.py scripts/cia/sms3ds.rsf scripts/cia/settings.txt \
	scripts/cia/README.txt "scripts/cia/Build SMS CIA.command" "scripts/cia/Build SMS CIA.bat" "$K/"
"$DEVKITARM/bin/arm-none-eabi-strip" -o "$K/sms3ds.elf" sms3ds.elf
cp sms3ds.3dsx "$K/"
echo "$VER" > "$K/version.txt"
cp -R build/tools/dist/macos build/tools/dist/windows "$K/tools/"
mkdir -p "$K/LICENSES"
cp "$H/LICENSES/"* build/tools/dist/LICENSES/* "$K/LICENSES/"

# zip -X: no extra Mac attributes; the executables keep their mode bits.
(cd "$R" && zip -X -r -q "sms3ds-$VER.zip" "sms3ds-$VER" -x '*.DS_Store' &&
	zip -X -r -q "SMS-3DS-CIA-Builder-$VER.zip" SMS-3DS-CIA-Builder -x '*.DS_Store')
ls -la "$R"
