#!/usr/bin/env bash
# Runs syntax_check_arm.sh over the JSystem slices (and optionally the game sources)
# against the patched decomp copy. Usage: scripts/check_slices.sh [--game]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$ROOT/build/decomp-patched" ] || "$ROOT/scripts/apply_patches.sh"
export DECOMP="$ROOT/build/decomp-patched"

SLICES=(
  'libs/JSystem/src/JKernel/*.cpp'
  'libs/JSystem/src/JUtility/*.cpp'
  'libs/JSystem/src/JSupport/*.cpp'
  'libs/JSystem/src/JGadget/*.cpp'
  'libs/JSystem/src/J3D/*/*.cpp'
  'libs/JSystem/src/J2D/*.cpp'
)
if [ "${1:-}" = "--game" ]; then
  for d in Animal Camera Enemy GC2D M3DUtil MSound Map MarioUtil MoveBG NPC Player Strategic System; do
    SLICES+=("src/$d/*.cpp")
  done
  SLICES+=('src/*.cpp')
fi
for g in "${SLICES[@]}"; do
  echo "### $g"
  "$ROOT/scripts/syntax_check_arm.sh" "$g" 2>&1 | grep -v CP932
done
