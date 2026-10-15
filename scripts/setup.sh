#!/usr/bin/env bash
# Installs this project into $HOME/3ds-dev/sms-3ds and checks the toolchain.
# Usage: ./scripts/setup.sh [dest-parent-dir]
set -euo pipefail

DEST_PARENT="${1:-$HOME/3ds-dev}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$DEST_PARENT/sms-3ds"

mkdir -p "$DEST_PARENT"
if [ "$SRC_DIR" != "$DEST" ]; then
  mkdir -p "$DEST"
  rsync -a --exclude build --exclude '*.3dsx' --exclude '*.elf' "$SRC_DIR/" "$DEST/"
fi
echo "Project at: $DEST"

export DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
export DEVKITARM="${DEVKITARM:-$DEVKITPRO/devkitARM}"
if [ ! -d "$DEVKITARM" ]; then
  echo "devkitARM not found at $DEVKITARM."
  echo "Install devkitPro: https://devkitpro.org/wiki/Getting_Started then run: sudo dkp-pacman -S 3ds-dev"
  exit 1
fi

cd "$DEST"
git init -q 2>/dev/null || true
if [ ! -d decomp ]; then
  echo "Adding decomp submodule (chasem-dev/sms-english, as used by sms-pc-port)..."
  git submodule add https://github.com/chasem-dev/sms-english decomp || echo "Skipped (add manually later)."
fi
if [ ! -d ext/sms-pc-port ]; then
  echo "Adding sms-pc-port submodule (MIT; patches, endian layer, compat headers)..."
  git submodule add https://github.com/chasem-dev/sms-pc-port ext/sms-pc-port || echo "Skipped (add manually later)."
fi
mkdir -p rom
echo "Next: put your GMSE01 ISO on the SD card at sdmc:/sms/GMSE01.iso, then run: make -j"
