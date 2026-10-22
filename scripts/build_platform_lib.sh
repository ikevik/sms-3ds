#!/usr/bin/env bash
# Stage 1, step 5: compile sms-pc-port's platform/ layer (everything except platform/gx,
# platform/mods/eclipse, tests, and Windows/macOS-only files) plus src/port_main.cpp for the 3DS CPU.
# Same source rules and flags as sms-pc-port's CMakeLists (gnu++11, SMS_AUDIO_DSP_HLE=1).
# The point is to learn which host-specific files need a 3DS replacement.
#
# Output: build/libsms_platform.a, build/platform_failures.txt
# Usage: scripts/build_platform_lib.sh [-j N]
#   CXX/CC/AR/SIZE/ARCHFLAGS/OPT/SAMPLE as in build_game_lib.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECOMP="${DECOMP:-$ROOT/build/decomp-patched}"
PCPORT="${PCPORT:-$ROOT/build/pcport-3ds}"   # patched copy: scripts/prepare_pcport.sh
P3DS="$ROOT/platform/3ds"
DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
OBJ="${OBJ:-$ROOT/build/platobj}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)"
[ "${1:-}" = "-j" ] && JOBS="${2:-$JOBS}"

DEVKITARM="${DEVKITARM:-/opt/devkitpro/devkitARM}"
CXX="${CXX:-$DEVKITARM/bin/arm-none-eabi-g++}"
CC="${CC:-$DEVKITARM/bin/arm-none-eabi-gcc}"
AR="${AR:-$DEVKITARM/bin/arm-none-eabi-ar}"
SIZE="${SIZE:-$DEVKITARM/bin/arm-none-eabi-size}"
ARCHFLAGS="${ARCHFLAGS--march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft -mword-relocations}"
OPT="${OPT:--O2}"
[ -d "$DECOMP/include" ] || { echo "run scripts/apply_patches.sh first"; exit 1; }
[ -d "$PCPORT/platform" ] || { echo "run scripts/prepare_pcport.sh first"; exit 1; }

mkdir -p "$ROOT/build" "$OBJ"; rm -f "$OBJ"/*.o "$OBJ"/*.err
# Same filter as CMake, plus files that can only build on Windows/macOS.
( cd "$PCPORT" && { find platform -name '*.c' -o -name '*.cpp'; echo src/port_main.cpp; } \
  | grep -vE '^platform/(gx|mods/eclipse|pad)/|/tests/|windows|darwin|window_icon|trace|presence|^src/port_main' | sort ) > "$ROOT/build/platform_units.txt"
echo "platform units: $(wc -l < "$ROOT/build/platform_units.txt"), jobs: $JOBS, compiler: $CXX"

INCS="-I$PCPORT/platform -I$PCPORT/platform/mods/include -I$PCPORT/src -I$PCPORT/src/port_include -I$P3DS/include -I$P3DS -I$PCPORT/platform/gx/port_include -I$PCPORT/platform/gx/include -I$DECOMP/include"
for d in "$DECOMP"/libs/*/include; do INCS="$INCS -I$d"; done
# No -include port_compat.h here: as in sms-pc-port's CMakeLists, only game code gets it. Its
# math macros (sqrtf -> sms_msl_sqrtf, ...) would turn platform/misc/msl_math.c's own
# implementations into calls to themselves (sms_msl_sqrtf compiled to a branch-to-self).
COMMON="$ARCHFLAGS $OPT -g0 -w -ffunction-sections -fdata-sections -fno-strict-aliasing -fwrapv -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DGEKKO -DTARGET_PC -DVERSION_GMSE01 -DBUILD_VERSION=2 -DNDEBUG=1 -DSMS_AUDIO_DSP_HLE=1 -include $P3DS/sms_compat_3ds.h -D__3DS__ -DSMS_PLATFORM_3DS=1"
# GameCube ABI for sms-pc-port's units (4-byte enums, signed plain char), matching the game
# and GX libraries. The @3ds units below include libctru headers, whose structs use the
# toolchain's variable-size enums, so they keep the defaults; they share no enum-typed
# structs with the game (PADStatus is all integer fields).
GC_ABI="-fno-short-enums -fsigned-char"
# SAMPLE=1: see build_game_lib.sh; sms-pc-port's units only (the @3ds ones hold the hooks), less
# the audio, which runs on its own thread. SMS3DS_SAMPLE doubles the host threads' stacks
# (os_thread.cpp, patch 0029): every instrumented frame is bigger.
INSTR=""; [ -n "${SAMPLE:-}" ] && INSTR="-finstrument-functions -finstrument-functions-exclude-file-list=.h -DSMS3DS_SAMPLE=1"
export CXX CC INCS OBJ PCPORT P3DS GC_ABI INSTR
export CXXF="$COMMON -std=gnu++11"
export CF="$COMMON -std=gnu99 -fcommon"

# 3DS-side units (shims + pad) live in platform/3ds and need libctru headers.
for f in posix_shims.cpp pad_sdk.cpp stubs_3ds.cpp game_new_3ds.cpp pad_3ds.c prof_3ds.cpp thp_3ds.cpp audio_3ds.cpp; do echo "@3ds/$f" >> "$ROOT/build/platform_units.txt"; done
CTRU="-I$DEVKITPRO/libctru/include"
export CTRU

compile_one() {
  rel="$1"; out="$OBJ/$(echo "$rel" | tr '/' '_').o"
  case "$rel" in platform/audio/*) INSTR="" ;; esac
  case "$rel" in
    @3ds/*) f="${rel#@3ds/}"; out="$OBJ/3ds_$f.o"
            case "$f" in *.cpp) CMP="$CXX $CXXF" ;; *) CMP="$CC $CF" ;; esac
            $CMP $INCS $CTRU -c "$P3DS/$f" -o "$out" 2>"$out.err" || { echo "FAIL $rel: $(grep -m1 'error' "$out.err" | cut -c1-140)"; return 1; }
            rm -f "$out.err"; return 0 ;;
    *.cpp) $CXX $CXXF $GC_ABI $INSTR $INCS -c "$PCPORT/$rel" -o "$out" 2>"$out.err" ;;
    *)     $CC  $CF   $GC_ABI $INSTR $INCS -c "$PCPORT/$rel" -o "$out" 2>"$out.err" ;;
  esac || { echo "FAIL $rel: $(grep -m1 'error' "$out.err" | cut -c1-140)"; return 1; }
  rm -f "$out.err"
}
export -f compile_one
tr '\n' '\0' < "$ROOT/build/platform_units.txt" | xargs -0 -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} | sort | tee "$ROOT/build/platform_failures.txt"

nobj=$(ls "$OBJ"/*.o 2>/dev/null | wc -l | tr -d ' ')
nfail=$(grep -c '^FAIL' "$ROOT/build/platform_failures.txt" || true)
echo "compiled: $nobj objects, failed: $nfail"
if [ "$nobj" -gt 0 ]; then
  rm -f "$ROOT/build/libsms_platform.a"; "$AR" rcs "$ROOT/build/libsms_platform.a" "$OBJ"/*.o
  "$SIZE" -t "$OBJ"/*.o 2>/dev/null | tail -1 | awk '{printf "totals  text=%d data=%d bss=%d (bytes)\n",$1,$2,$3}'
fi
