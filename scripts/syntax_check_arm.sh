#!/usr/bin/env bash
# Stage 1, step 1: does the decomp's source even parse with devkitARM's g++?
# Runs `-fsyntax-only` over a slice of decomp units using the same flags sms-pc-port
# uses for its 32-bit build (decomp-patches are NOT applied yet, so some errors are
# expected; the point is to see what class of errors we get).
#
# Usage: scripts/syntax_check_arm.sh [glob-under-decomp]   (default: libs/JSystem/src/JKernel/*.cpp)
#   CXX=...      compiler (default: arm-none-eabi-g++ from devkitARM)
#   PCPORT=...   path to a sms-pc-port checkout (default: ext/sms-pc-port, else ../sms-pc-port)
#   DECOMP=...   path to the decomp (default: decomp)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECOMP="${DECOMP:-$ROOT/decomp}"
PCPORT="${PCPORT:-$ROOT/ext/sms-pc-port}"
[ -d "$PCPORT" ] || PCPORT="$ROOT/../sms-pc-port"
GLOB="${1:-libs/JSystem/src/JKernel/*.cpp}"

if [ -z "${CXX:-}" ]; then
  DEVKITARM="${DEVKITARM:-/opt/devkitpro/devkitARM}"
  CXX="$DEVKITARM/bin/arm-none-eabi-g++"
fi
[ -x "$(command -v "$CXX")" ] || { echo "compiler not found: $CXX"; exit 1; }
[ -f "$PCPORT/src/port_compat.h" ] || { echo "sms-pc-port not found at $PCPORT (git submodule add https://github.com/chasem-dev/sms-pc-port ext/sms-pc-port)"; exit 1; }

# Include order mirrors sms-pc-port's CMakeLists: its port_include and the GX override
# headers (platform/gx/port_include, platform/gx/include) come BEFORE the decomp's headers.
INCS=(-I"$PCPORT/src/port_include" -I"$PCPORT/src"
      -I"$PCPORT/platform/gx/port_include" -I"$PCPORT/platform/gx/include"
      -I"$DECOMP/include")

# Units that MWCC built with the game PCH get SMS.pch force-included (see tools/gen_sources.py).
PCH_UNITS="$(python3 - "$DECOMP/configure.py" <<'PY'
import re, sys
cfg = open(sys.argv[1]).read()
print("\n".join(sorted(set(re.findall(r'PCHObject\(\s*[^,]+,\s*"([^"]+)"', cfg)))))
PY
)"
for d in "$DECOMP"/libs/*/include; do INCS+=(-I"$d"); done

FLAGS=(-fsyntax-only -w -fno-strict-aliasing -fwrapv -fno-stack-protector -ffp-contract=off
       -std=gnu++03 -fno-gnu-keywords -fpermissive -fno-threadsafe-statics
       -DGEKKO -DTARGET_PC -DVERSION_GMSE01 -DBUILD_VERSION=2 -DNDEBUG=1
       -include "$PCPORT/src/port_compat.h")
# Shift-JIS string literals, as in the MWCC build. Skipped if this g++ has no iconv/CP932.
if echo 'int x;' | "$CXX" -x c++ -fsyntax-only -finput-charset=UTF-8 -fexec-charset=CP932 - 2>/dev/null; then
  FLAGS+=(-finput-charset=UTF-8 -fexec-charset=CP932)
  echo "CP932 exec charset: supported"
else
  echo "CP932 exec charset: NOT supported by $CXX (will need a source mirror like tools/darwin_cp932_mirror.py)"
fi

ok=0; bad=0; : > /tmp/sms_syntax_errors.txt
shopt -s nullglob
for f in "$DECOMP"/$GLOB; do
  extra=()
  rel="${f#$DECOMP/src/}"
  if [ "$rel" != "$f" ] && printf '%s\n' "$PCH_UNITS" | grep -qxF "$rel"; then
    extra=(-include "$DECOMP/include/SMS.pch")
  fi
  if "$CXX" "${FLAGS[@]}" "${extra[@]+"${extra[@]}"}" "${INCS[@]}" -c "$f" 2>>/tmp/sms_syntax_errors.txt; then
    ok=$((ok+1))
  else
    bad=$((bad+1)); echo "FAIL ${f#$DECOMP/}"
  fi
done
echo "ok=$ok fail=$bad  (error log: /tmp/sms_syntax_errors.txt)"
grep -h "error:" /tmp/sms_syntax_errors.txt | sed 's/^[^ ]* //' | sort | uniq -c | sort -rn | head -15
