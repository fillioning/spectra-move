# Spectra

Multiband spectral resonator with pitch tracking and scale quantization for [Ableton Move](https://www.ableton.com/move/),
built for the [Schwung](https://github.com/charlesvestal/schwung) framework.

## What it does

Spectra analyzes incoming audio in real-time using 8-band multiband analysis — detecting onsets, frequency content, and noise-vs-tonal ratio per band — then excites a bank of musically-tuned resonant filters quantized to a user-selected scale and root note. Different frequency bands trigger different notes: a kick triggers low notes, a snare mid notes, a hi-hat high notes. Noisy inputs produce darker resonances; tonal inputs ring bright.

## Parameters

### Main (Knobs 1-8)

| Knob | Parameter  | Description |
|------|-----------|-------------|
| 1    | Onset      | Onset sensitivity — how hard a transient must be to trigger a new note |
| 2    | Frequency  | Band selectivity — 0% = all bands trigger root note, 100% = full frequency mapping |
| 3    | Brightness | Resonator brightness — controls Q (bandwidth) and output gain, added to auto-detected tonal/noise ratio |
| 4    | Timbre     | Harmonic series (0%) to scale-degree spread (100%) — crossfades resonator tuning |
| 5    | Decay      | Resonator ring time — 50ms to 10 seconds |
| 6    | Root       | Root note (C through B) |
| 7    | Scale      | Scale type (12 options × 12 roots = 144 combinations) |
| 8    | Mix        | Equal-power dry/wet crossfade |

### Control (Knobs 1-8 on Page 2)

| Knob | Parameter    | Description |
|------|-------------|-------------|
| 1    | Chord Drift  | Probability of ±1-2 semitone offset from scale (0-100%) |
| 2    | Resonators   | Filters per voice: 5 (pentatonic), 7 (diatonic), 12 (chromatic) |
| 3    | Oct Range    | Octave spread of resonator bank: 1-4 octaves |
| 4    | Compress     | RMS compressor strength — 0% = off, 100% = full dynamics flattening |
| 5    | Pre Gain     | Input gain (-12 to +12 dB) |
| 6    | Post Gain    | Output gain (-12 to +12 dB) |
| 7    | HPF          | High-pass filter (20 Hz - 2 kHz, 10ms smoothed) |
| 8    | LPF          | Low-pass filter (500 Hz - 20 kHz, 10ms smoothed) |

### Menu-only parameters

| Parameter    | Description |
|-------------|-------------|
| Limiter      | Tape soft clipper on/off |
| Polyphony    | Simultaneous voices: 1, 2, or 4 |

### Available Scales

Major, Natural Minor, Harmonic Minor, Melodic Minor, Dorian, Phrygian, Lydian, Mixolydian, Pentatonic Major, Pentatonic Minor, Blues, Chromatic

## Building

```
./scripts/build.sh
```

Requires Docker or an `aarch64-linux-gnu-gcc` cross-compiler.

## Installation

```
./scripts/install.sh
```

Or install via the Module Store in Schwung.

## DSP Architecture

All DSP is self-contained C with no external dependencies:

- **8-band multiband analysis** — per-band onset detection via spectral flux (512-point FFT)
- **Spectral flatness** analysis for auto-brightness (band min/max energy ratio)
- **Q-normalized SVF bandpass resonator bank** (up to 4 voices × 12 filters = 48 SVFs)
- **Harmonic/scale-degree crossfade** tuning (Timbre parameter)
- **Scale quantizer** with chord drift
- **RMS compressor** (Dissolver-style, per-block gain = RMS^(-power))
- **10ms smoothed HPF/LPF** for zipper-free filter sweeps

## Credits

Algorithmic references: Mutable Instruments Elements/Rings resonator (MIT), spectral flux onset detection (standard MIR technique).

## License

MIT — see [LICENSE](LICENSE)
