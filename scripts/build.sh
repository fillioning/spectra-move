#!/bin/bash
set -e

MODULE_ID="spectra"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
OUT_DIR="$PROJECT_DIR/dist/$MODULE_ID"

# Cross-compiler prefix (Docker or local toolchain)
CC="${CROSS_PREFIX:-aarch64-linux-gnu-}gcc"

echo "=== Building Spectra for Ableton Move (ARM64) ==="
echo "Compiler: $CC"

mkdir -p "$OUT_DIR"

# Compile spectra.c → shared library
echo "  CC src/dsp/spectra.c"
$CC -O2 -shared -fPIC -ffast-math \
    -o "$OUT_DIR/$MODULE_ID.so" \
    "$SRC_DIR/dsp/$MODULE_ID.c" \
    -lm \
    -Wall -Wextra -Wno-unused-parameter

# Copy module files
cp "$PROJECT_DIR/module.json" "$OUT_DIR/"

echo "=== Build complete: $OUT_DIR/$MODULE_ID.so ==="
ls -la "$OUT_DIR/"
