FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . /build/

RUN dos2unix /build/src/dsp/*.c 2>/dev/null || true && \
    mkdir -p /build/dist/spectra && \
    aarch64-linux-gnu-gcc \
      -O2 -shared -fPIC -ffast-math \
      -o /build/dist/spectra/spectra.so \
      /build/src/dsp/spectra.c \
      -lm \
      -Wall -Wextra -Wno-unused-parameter && \
    cp /build/module.json /build/dist/spectra/module.json
