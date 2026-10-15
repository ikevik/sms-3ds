#!/usr/bin/env bash
# Copy the pinned sms-pc-port submodule (platform/, src/) to build/pcport-3ds and apply
# patches/sms-port/*.patch there. The submodule itself is never modified.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${PCPORT_SRC:-$ROOT/ext/sms-pc-port}"
DST="$ROOT/build/pcport-3ds"
[ -d "$SRC/platform" ] || { echo "sms-pc-port missing at $SRC (run scripts/setup.sh)"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
( cd "$SRC" && tar cf - platform src ) | ( cd "$DST" && tar xf - )
n=0
for p in "$ROOT"/patches/sms-port/*.patch; do
  [ -f "$p" ] || continue
  ( cd "$DST" && patch -p1 --no-backup-if-mismatch < "$p" >/dev/null ) \
    && { echo "applied $(basename "$p")"; n=$((n+1)); } \
    || { echo "FAILED  $(basename "$p")"; exit 1; }
done
echo "pcport-3ds ready: $DST ($n patches)"
