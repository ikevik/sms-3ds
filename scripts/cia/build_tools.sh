#!/bin/bash
# The CIA builder's tools, from source, into build/tools/dist/<macos|windows>/: makerom (3DSGuy's
# Project_CTR) and bannertool (diasurgical's fork; Steveice10's original is gone). macOS binaries are
# universal (arm64 and x86_64, macOS 11 and later); Windows ones are cross-compiled with mingw-w64
# (brew install mingw-w64). Linux users build them from the same sources.
set -euo pipefail
cd "$(dirname "$0")/../.."
T=build/tools
mkdir -p "$T"
[ -d "$T/Project_CTR" ] || git clone https://github.com/3DSGuy/Project_CTR.git "$T/Project_CTR"
[ -d "$T/bannertool" ] || git clone --depth 1 https://github.com/diasurgical/bannertool.git "$T/bannertool"
git -C "$T/Project_CTR" checkout -q d52c2b0
mkdir -p "$T/dist/macos" "$T/dist/windows"

# makerom: its makefile builds in place, so one copy per target.
MINGW_AR=$(mktemp -d)
ln -sf "$(command -v x86_64-w64-mingw32-ar)" "$MINGW_AR/ar"
for t in win x86_64 arm64; do
	rm -rf "$T/makerom-$t"
	cp -R "$T/Project_CTR/makerom" "$T/makerom-$t"
	# user_settings.c includes makerom's version.h, whose VER_MINOR (19, makerom's own version) hides
	# the title version's index (1): -minor wrote past titleVersion[] and the minor number was lost
	# (0.1.1 became 0.0.1).
	sed -i '' 's/titleVersion\[VER_MINOR\]/titleVersion[1]/g' "$T/makerom-$t/src/user_settings.c"
	(cd "$T/makerom-$t" && make clean >/dev/null 2>&1 && make clean_deps >/dev/null 2>&1) || true
done
(cd "$T/makerom-win" && PATH="$MINGW_AR:$PATH" make PROJECT_PLATFORM=WIN32 deps && PATH="$MINGW_AR:$PATH" make PROJECT_PLATFORM=WIN32 -j8)
for a in x86_64 arm64; do
	(cd "$T/makerom-$a" && MACOSX_DEPLOYMENT_TARGET=11.0 make PROJECT_PLATFORM_ARCH=$a deps &&
		MACOSX_DEPLOYMENT_TARGET=11.0 make PROJECT_PLATFORM_ARCH=$a -j8)
done
lipo -create "$T/makerom-arm64/bin/makerom" "$T/makerom-x86_64/bin/makerom" -output "$T/dist/macos/makerom"
cp "$T/makerom-win/bin/makerom.exe" "$T/dist/windows/"
x86_64-w64-mingw32-strip "$T/dist/windows/makerom.exe"

# bannertool: a handful of sources (its buildtools submodule is gone too).
B="$T/bannertool"
DEFS="-DVERSION_MAJOR=1 -DVERSION_MINOR=2 -DVERSION_MICRO=0"
SRCS="$B/source/*.cpp $B/source/3ds/*.cpp $B/source/pc/wav.cpp"
O=$(mktemp -d)
x86_64-w64-mingw32-gcc -O2 -w -c "$B/source/pc/stb_image.c" -o "$O/stb_image_win.o"
x86_64-w64-mingw32-gcc -O2 -w -c "$B/source/pc/stb_vorbis.c" -o "$O/stb_vorbis_win.o"
x86_64-w64-mingw32-g++ -O2 -std=gnu++11 -w -I"$B/source" $DEFS $SRCS "$O/stb_image_win.o" "$O/stb_vorbis_win.o" \
	-static -s -o "$T/dist/windows/bannertool.exe"
for a in arm64 x86_64; do
	F="-arch $a -mmacosx-version-min=11.0 -O2 -w"
	cc $F -c "$B/source/pc/stb_image.c" -o "$O/stb_image_$a.o"
	cc $F -c "$B/source/pc/stb_vorbis.c" -o "$O/stb_vorbis_$a.o"
	c++ $F -std=gnu++11 -I"$B/source" $DEFS $SRCS "$O/stb_image_$a.o" "$O/stb_vorbis_$a.o" -o "$O/bannertool_$a"
done
lipo -create "$O/bannertool_arm64" "$O/bannertool_x86_64" -output "$T/dist/macos/bannertool"
strip "$T/dist/macos/bannertool"
# strip leaves the arm64 slice unsigned, and macOS will not run an unsigned arm64 binary.
codesign -s - -f "$T/dist/macos/makerom" "$T/dist/macos/bannertool"

mkdir -p "$T/dist/LICENSES"
cp "$T/Project_CTR/makerom/LICENSE" "$T/dist/LICENSES/makerom.txt"
cp "$B/LICENSE.txt" "$T/dist/LICENSES/bannertool.txt"
ls -la "$T/dist"/*
