#!/bin/bash
set -e

MODULE_ID="spectra"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

echo "Building $MODULE_ID for ARM64 (aarch64)..."

docker build -t ${MODULE_ID}-builder -f "$ROOT/scripts/Dockerfile" "$ROOT"

docker run --rm \
  -v "$ROOT:/build" \
  ${MODULE_ID}-builder \
  bash -c "
    dos2unix /build/src/dsp/*.c /build/src/dsp/*.h 2>/dev/null || true
    mkdir -p /build/dist/$MODULE_ID
    aarch64-linux-gnu-gcc \
      -O2 -shared -fPIC -ffast-math \
      -o /build/dist/$MODULE_ID/$MODULE_ID.so \
      /build/src/dsp/$MODULE_ID.c \
      -lm
    cp /build/src/module.json /build/dist/$MODULE_ID/module.json
    cd /build/dist && tar -czf ${MODULE_ID}-module.tar.gz $MODULE_ID/
    echo '=== Build complete ==='
    ls -la /build/dist/$MODULE_ID/
  "

echo "Built: dist/$MODULE_ID/"
echo "Tarball: dist/${MODULE_ID}-module.tar.gz"
