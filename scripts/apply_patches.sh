#!/usr/bin/env bash
# Stage 1, step 3: make build/decomp-patched = a copy of decomp/ with sms-pc-port's
# decomp-patches/*.patch applied in name order (decomp/ itself is never modified).
# Then: DECOMP=build/decomp-patched scripts/syntax_check_arm.sh '<glob>'
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECOMP="${DECOMP:-$ROOT/decomp}"
PCPORT="${PCPORT:-$ROOT/ext/sms-pc-port}"
OUT="${OUT:-$ROOT/build/decomp-patched}"
PATCHES="$PCPORT/decomp-patches"
[ -d "$PATCHES" ] || { echo "no patches at $PATCHES"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
# Only sources and headers are needed; skip .git, disc data and assets.
(cd "$DECOMP" && tar cf - --exclude=.git --exclude=./orig --exclude=./assets --exclude=./build .) | (cd "$OUT" && tar xf -)

applied=0; failed=0
: > "$ROOT/build/patch_failures.txt"
for p in $(ls "$PATCHES"/*.patch | sort); do
  if (cd "$OUT" && patch -p1 --forward --no-backup-if-mismatch -s < "$p" >/dev/null 2>&1); then
    applied=$((applied+1))
  else
    failed=$((failed+1)); echo "$(basename "$p")" >> "$ROOT/build/patch_failures.txt"
    # undo any partial hunks by re-trying in reverse is risky; just record it.
  fi
done
# This port's own decomp patches (decomp-patches/*.patch), after sms-pc-port's.
ours=0
for p in $(ls "$ROOT"/decomp-patches/*.patch 2>/dev/null | sort); do
  if (cd "$OUT" && patch -p1 --forward --no-backup-if-mismatch -s < "$p" >/dev/null 2>&1); then ours=$((ours+1));
  else failed=$((failed+1)); echo "3ds:$(basename "$p")" >> "$ROOT/build/patch_failures.txt"; fi
done
echo "3DS patches applied=$ours"
echo "patches applied=$applied failed=$failed -> $OUT"
[ "$failed" -gt 0 ] && { echo "failed patches (see build/patch_failures.txt):"; head -20 "$ROOT/build/patch_failures.txt"; }
exit 0
