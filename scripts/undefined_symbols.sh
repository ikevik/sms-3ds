#!/usr/bin/env bash
# Stage 1, step 5 prep: which symbols does libsms_game.a need that nothing inside it defines?
# That is the exact list of SDK/runtime functions the 3DS platform layer must provide.
# Output: build/undefined_symbols.txt (demangled, sorted) and a summary grouped by prefix.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVKITARM="${DEVKITARM:-/opt/devkitpro/devkitARM}"
NM="${NM:-$DEVKITARM/bin/arm-none-eabi-nm}"
CXXFILT="${CXXFILT:-$DEVKITARM/bin/arm-none-eabi-c++filt}"
# Game + platform libraries are examined together: what is still undefined is what the
# GX backend, newlib/libctru or new 3DS code has to supply. LIB=... overrides (one file).
LIBS="${LIB:-$ROOT/build/libsms_game.a $ROOT/build/libsms_platform.a $ROOT/build/libsms_gx.a}"
LIBS_OK=""
for l in $LIBS; do [ -f "$l" ] && LIBS_OK="$LIBS_OK $l"; done
[ -n "$LIBS_OK" ] || { echo "build the library first: scripts/build_game_lib.sh"; exit 1; }
echo "libraries:$LIBS_OK"

TMP="$(mktemp -d)"
"$NM" -g --defined-only $LIBS_OK 2>/dev/null | awk 'NF==3{print $3}' | sort -u > "$TMP/defined.txt"
"$NM" -g --undefined-only $LIBS_OK 2>/dev/null | awk 'NF==2{print $2}' | sort -u > "$TMP/undef.txt"
comm -23 "$TMP/undef.txt" "$TMP/defined.txt" > "$TMP/missing.txt"

"$CXXFILT" < "$TMP/missing.txt" | sort -u > "$ROOT/build/undefined_symbols.txt"
total=$(wc -l < "$ROOT/build/undefined_symbols.txt" | tr -d ' ')
grep -vE "^(GX|GD)" "$ROOT/build/undefined_symbols.txt" > "$ROOT/build/undefined_non_gx.txt"
echo "undefined symbols: $total  (full list: build/undefined_symbols.txt; without GX/GD: build/undefined_non_gx.txt)"
echo
echo "count  group"
# Group by well-known SDK prefixes; everything else by first 3 letters.
awk '{
  s=$0; g="other";
  if (s ~ /^(OS|__OS)/) g="OS (os/arena/thread/alarm/cache)";
  else if (s ~ /^(DVD|__DVD)/) g="DVD";
  else if (s ~ /^(GX|GD)/) g="GX/GD (graphics)";
  else if (s ~ /^(PAD|SI)/) g="PAD/SI";
  else if (s ~ /^(AX|AI|AR|ARQ|DSP|DTK)/) g="Audio (AX/AI/AR/DSP/DTK)";
  else if (s ~ /^(PS|C_|MTX|VEC|QUAT)/) g="MTX/VEC/QUAT";
  else if (s ~ /^(VI|__VI)/) g="VI (video)";
  else if (s ~ /^CARD/) g="CARD (memory card)";
  else if (s ~ /^(THP|THPPlayer)/) g="THP (movies)";
  else if (s ~ /^(DC|IC|LC)/) g="Cache (DC/IC/LC)";
  else if (s ~ /^(EXI|__EXI)/) g="EXI";
  else if (s ~ /^(mem|str|sprintf|printf|snprintf|vsnprintf|atoi|abs|rand|sin|cos|tan|sqrt|pow|exp|floor|ceil|fabs|atan|acos|asin|fmod|log|__)/) g="libc/libm/MSL/runtime";
  else if (s ~ /^(operator|std::|__cxa|__gxx|vtable|typeinfo)/) g="C++ runtime";
  else if (s ~ /^(lbl_|fn_|sub_)/) g="decomp lbl_/fn_ placeholders (unlinked asm)";
  c[g]++
} END { for (k in c) printf "%5d  %s\n", c[k], k }' "$ROOT/build/undefined_symbols.txt" | sort -rn
rm -rf "$TMP"
