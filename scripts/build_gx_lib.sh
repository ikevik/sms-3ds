#!/usr/bin/env bash
# Stage 2: build the GX/GD layer for the 3DS -> build/libsms_gx.a
#   * sms-pc-port's renderer-independent front half (gx_api, gx_fifo, gx_vert, gx_trace)
#   * GD, compiled from the decomp's SDK sources exactly as sms-pc-port does
#   * platform/3ds/gx/gx3ds_backend.cpp: the 3DS renderer side (replaces gx_render/gx_shader/
#     gx_texture/gx_platform, which are OpenGL + SDL)
# Needs scripts/prepare_pcport.sh and scripts/apply_patches.sh first.
# Usage: scripts/build_gx_lib.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECOMP="${DECOMP:-$ROOT/build/decomp-patched}"
PCPORT="${PCPORT:-$ROOT/build/pcport-3ds}"
P3DS="$ROOT/platform/3ds"
OBJ="$ROOT/build/gxobj"
DEVKITARM="${DEVKITARM:-/opt/devkitpro/devkitARM}"
DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
CXX="${CXX:-$DEVKITARM/bin/arm-none-eabi-g++}"; CC="${CC:-$DEVKITARM/bin/arm-none-eabi-gcc}"
AR="${AR:-$DEVKITARM/bin/arm-none-eabi-ar}"; SIZE="${SIZE:-$DEVKITARM/bin/arm-none-eabi-size}"
ARCHFLAGS="${ARCHFLAGS--march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft -mword-relocations}"
OPT="${OPT:--O2}"
GX="$PCPORT/platform/gx"
[ -d "$GX/src" ] && [ -d "$DECOMP/include" ] || { echo "run prepare_pcport.sh and apply_patches.sh first"; exit 1; }
mkdir -p "$OBJ"; rm -f "$OBJ"/*.o "$OBJ"/*.err

INCS="-I$GX/src -I$GX/include -I$GX/port_include -I$PCPORT/platform -I$PCPORT/src -I$PCPORT/src/port_include -I$P3DS/include -I$DECOMP/include -I$DECOMP/libs/dolphin/include"
for d in "$DECOMP"/libs/*/include; do INCS="$INCS -I$d"; done
# GameCube ABI: enums are 4 bytes and plain char is signed (MWCC). arm-none-eabi defaults to
# variable-size enums and unsigned char, which breaks every struct read straight from game
# files (e.g. GXVtxAttrFmtList in J3D models) and char comparisons. Units that talk to
# libctru keep the toolchain defaults (see build_platform_lib.sh).
GC_ABI="-fno-short-enums -fsigned-char"
COMMON="$ARCHFLAGS $OPT -g0 -w -ffunction-sections -fdata-sections -fno-strict-aliasing -fwrapv -ffp-contract=off -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DGEKKO -DTARGET_PC -DVERSION_GMSE01 -DBUILD_VERSION=2 -DNDEBUG=1 -D__3DS__ -DSMS_PLATFORM_3DS=1 $GC_ABI"
fail=0
cxx() { # src out
  $CXX $COMMON -std=gnu++17 -fpermissive -include "$PCPORT/src/port_compat.h" -include "$P3DS/sms_compat_3ds.h" $INCS -c "$1" -o "$2" 2>"$2.err" \
    && rm -f "$2.err" || { echo "FAIL $(basename "$1"): $(grep -m1 error "$2.err" | cut -c1-160)"; fail=$((fail+1)); }
}
for f in gx_api gx_fifo gx_vert gx_trace; do cxx "$GX/src/$f.cpp" "$OBJ/$f.o"; done
cxx "$P3DS/gx/gx3ds_backend.cpp" "$OBJ/gx3ds_backend.o"
for f in GDBase GDGeometry GDLight GDPixel GDTev GDTransform; do
  $CC $COMMON -std=gnu99 -fcommon -include "$GX/src/gd_host_prefix.h" $INCS -c "$DECOMP/libs/dolphin/src/gd/$f.c" -o "$OBJ/$f.o" 2>"$OBJ/$f.o.err" \
    && rm -f "$OBJ/$f.o.err" || { echo "FAIL $f.c: $(grep -m1 error "$OBJ/$f.o.err" | cut -c1-160)"; fail=$((fail+1)); }
done
n=$(ls "$OBJ"/*.o 2>/dev/null | wc -l | tr -d ' ')
echo "compiled: $n objects, failed: $fail"
[ "$n" -gt 0 ] && { rm -f "$ROOT/build/libsms_gx.a"; "$AR" rcs "$ROOT/build/libsms_gx.a" "$OBJ"/*.o; "$SIZE" -t "$OBJ"/*.o | tail -1 | awk '{printf "totals  text=%d data=%d bss=%d (bytes)\n",$1,$2,$3}'; }
exit $fail
