#!/usr/bin/env bash
# Stage 1, step 4: compile the decomp's game + JSystem + THPPlayer units (the same 588-unit
# set sms-pc-port builds) from build/decomp-patched into build/libsms_game.a for the 3DS CPU.
#
# Usage: scripts/build_game_lib.sh [-j N]
#   CXX/CC/AR/SIZE/OBJCOPY override toolchain (default: devkitARM arm-none-eabi-*)
#   ARCHFLAGS        default: -march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft -mword-relocations
#   LIMIT=N          compile only the first N units (quick test)
#   OPT=-O2          optimization level
#   SAMPLE=1         instrument every function for profile runs' sampler (platform/3ds/prof_3ds.cpp)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECOMP="${DECOMP:-$ROOT/build/decomp-patched}"
PCPORT="${PCPORT:-$ROOT/ext/sms-pc-port}"
OBJ="${OBJ:-$ROOT/build/obj}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)"
[ "${1:-}" = "-j" ] && JOBS="${2:-$JOBS}"

DEVKITARM="${DEVKITARM:-/opt/devkitpro/devkitARM}"
CXX="${CXX:-$DEVKITARM/bin/arm-none-eabi-g++}"
CC="${CC:-$DEVKITARM/bin/arm-none-eabi-gcc}"
AR="${AR:-$DEVKITARM/bin/arm-none-eabi-ar}"
SIZE="${SIZE:-$DEVKITARM/bin/arm-none-eabi-size}"
OBJCOPY="${OBJCOPY:-$DEVKITARM/bin/arm-none-eabi-objcopy}"
ARCHFLAGS="${ARCHFLAGS--march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft -mword-relocations}"
OPT="${OPT:--O2}"

[ -d "$DECOMP/src" ] || { echo "run scripts/apply_patches.sh first"; exit 1; }
[ -f "$PCPORT/tools/gen_sources.py" ] || { echo "sms-pc-port missing at $PCPORT"; exit 1; }

mkdir -p "$ROOT/build" "$OBJ"
LIST="$ROOT/build/decomp_sources.cmake"
python3 "$PCPORT/tools/gen_sources.py" "$DECOMP" "$LIST" || exit 1

# Emit "kind<TAB>relpath" lines (kind: cxx|c, pch flag as third column).
python3 - "$LIST" "$DECOMP" > "$ROOT/build/units.tsv" <<'PY'
import re, sys
txt = open(sys.argv[1]).read(); root = sys.argv[2]
def block(n):
    m = re.search(r'set\(%s\n(.*?)\n\)' % n, txt, re.S)
    return [l.strip().replace('${SMS_DECOMP}/', '') for l in m.group(1).splitlines() if l.strip()] if m else []
pch = set(block('SMS_DECOMP_PCH_SOURCES'))
for kind, name in (('cxx', 'SMS_DECOMP_CXX_SOURCES'), ('c', 'SMS_DECOMP_C_SOURCES')):
    for p in block(name):
        print('%s\t%s\t%d' % (kind, p, 1 if p in pch else 0))
# THP movie decoders: sms-pc-port links these via platform/thp/thp.cmake rather than the
# generated list. Their strong THPInit/THPVideoDecode/THPAudioDecode replace the SDK stubs.
for p in ('libs/dolphin/src/thp/THPDec.c', 'libs/dolphin/src/thp/THPAudio.c'):
    print('c\t%s\t0' % p)
PY
TOTAL=$(wc -l < "$ROOT/build/units.tsv")
[ -n "${LIMIT:-}" ] && head -n "$LIMIT" "$ROOT/build/units.tsv" > "$ROOT/build/units.limit.tsv" && mv "$ROOT/build/units.limit.tsv" "$ROOT/build/units.tsv"
echo "units: $(wc -l < "$ROOT/build/units.tsv") of $TOTAL, jobs: $JOBS, compiler: $CXX"

INCS="-I$PCPORT/src/port_include -I$PCPORT/src -I$PCPORT/platform/gx/port_include -I$PCPORT/platform/gx/include -I$DECOMP/include"
for d in "$DECOMP"/libs/*/include; do INCS="$INCS -I$d"; done
# GameCube ABI: enums are 4 bytes and plain char is signed (MWCC). arm-none-eabi defaults to
# variable-size enums and unsigned char, which breaks every struct read straight from game
# files (e.g. GXVtxAttrFmtList in J3D models) and char comparisons. Units that talk to
# libctru keep the toolchain defaults (see build_platform_lib.sh).
GC_ABI="-fno-short-enums -fsigned-char"
# SAMPLE=1: each function's entry and exit call prof_3ds.cpp's hooks (inline ones from headers
# are left out), for the profile run's sampler. A measurement build: the hooks cost time.
INSTR=""; [ -n "${SAMPLE:-}" ] && INSTR="-finstrument-functions -finstrument-functions-exclude-file-list=.h"
COMMON="$ARCHFLAGS $OPT $INSTR -g0 -w -ffunction-sections -fdata-sections -fno-strict-aliasing -fwrapv -fno-stack-protector -ffp-contract=off -DGEKKO -DTARGET_PC -DVERSION_GMSE01 -DBUILD_VERSION=2 -DNDEBUG=1 -D__3DS__ -DSMS_PLATFORM_3DS=1 $GC_ABI -include $PCPORT/src/port_compat.h"
CXXF="$COMMON -std=gnu++03 -fno-gnu-keywords -fpermissive -fno-threadsafe-statics"
CF="$COMMON -std=gnu99 -fcommon"
CP932=""
if echo 'int x;' | "$CXX" -x c++ -fsyntax-only -finput-charset=UTF-8 -fexec-charset=CP932 - 2>/dev/null; then
  CP932="-finput-charset=UTF-8 -fexec-charset=CP932"
fi

compile_one() {
  IFS=$'\t' read -r kind rel pch <<< "$1"
  out="$OBJ/$(echo "$rel" | tr '/' '_').o"
  extra=""; [ "$pch" = "1" ] && extra="-include $DECOMP/include/SMS.pch"
  if [ "$kind" = "cxx" ]; then
    $CXX $CXXF $CP932 $extra $INCS -c "$DECOMP/$rel" -o "$out" 2>"$out.err" || { echo "FAIL $rel"; return 1; }
  else
    $CC $CF $INCS -c "$DECOMP/$rel" -o "$out" 2>"$out.err" || { echo "FAIL $rel"; return 1; }
  fi
  rm -f "$out.err"
}
export -f compile_one
export CXX CC CXXF CF CP932 INCS OBJ DECOMP

rm -f "$OBJ"/*.o "$OBJ"/*.err
tr '\n' '\0' < "$ROOT/build/units.tsv" | xargs -0 -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} | tee "$ROOT/build/compile_failures.txt"

nobj=$(ls "$OBJ"/*.o 2>/dev/null | wc -l | tr -d ' ')
nfail=$(grep -c '^FAIL' "$ROOT/build/compile_failures.txt" || true)
echo "compiled: $nobj objects, failed: $nfail"
if [ "$nobj" -gt 0 ]; then
  rm -f "$ROOT/build/libsms_game.a"
  "$AR" rcs "$ROOT/build/libsms_game.a" "$OBJ"/*.o
  # The game replaces the global operator new/delete with JKRHeap allocation, which returns
  # null until the first JKRHeap exists and would also capture libstdc++, static constructors
  # and the platform layer. Rename the game's definitions and its references to them (as
  # sms-pc-port's CMakeLists does), so only game code allocates from JKR heaps.
  printf '%s\n' "_Znwj __sms_game_new" "_Znaj __sms_game_new_array" \
    "_ZdlPv __sms_game_delete" "_ZdaPv __sms_game_delete_array" > "$ROOT/build/game_new_syms.txt"
  "$OBJCOPY" --redefine-syms="$ROOT/build/game_new_syms.txt" "$ROOT/build/libsms_game.a" || exit 1
  echo "archive: build/libsms_game.a (game operator new/delete renamed)"
  "$SIZE" -t "$OBJ"/*.o 2>/dev/null | tail -1 | awk '{printf "totals  text=%d data=%d bss=%d (bytes)\n",$1,$2,$3}'
fi
if [ "$nfail" -gt 0 ]; then
  echo "--- top error messages ---"
  cat "$OBJ"/*.err 2>/dev/null | grep "error" | sed 's/^[^ ]* //' | sort | uniq -c | sort -rn | head -15
fi
