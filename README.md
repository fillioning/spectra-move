# Spectra

Multiband spectral resonator with pitch tracking and scale quantization for [Ableton Move](https://www.ableton.com/move/),
built for the [Schwung](https://github.com/charlesvestal/schwung) framework.

## What it does

Spectra analyzes incoming audio in real-time — detecting pitch, onsets, and noise-vs-tonal ratio — then excites a bank of musically-tuned resonant filters quantized to a user-selected scale and root note. Noisy inputs produce darker resonances; tonal inputs ring bright. Complex sounds generate chords across multiple polyphonic voices.

## Parameters

### Main (Knobs 1-8)

| Knob | Parameter  | Description |
|------|-----------|-------------|
| 1    | Onset      | Spectral flux threshold — how hard a transient triggers a new note |
| 2    | Frequency  | Pitch detection confidence — how clear a pitch must be to track |
| 3    | Brightness | Manual brightness offset, added to auto-detected tonal/noise ratio |
| 4    | Timbre     | Inharmonicity — pure harmonic (0%) to metallic/bell-like (100%) |
| 5    | Decay      | Resonator ring time — 50ms to 10 seconds |
| 6    | Root       | Root note (C through B) |
| 7    | Scale      | Scale type (12 options × 12 roots = 144 combinations) |
| 8    | Mix        | Equal-power dry/wet crossfade |

### Control (Menu Page 2)

| Parameter    | Description |
|-------------|-------------|
| Chord Drift  | Probability of ±1-2 semitone offset from scale (0-100%) |
| Resonators   | Filters per voice: 5 (pentatonic), 7 (diatonic), 12 (chromatic) |
| Polyphony    | Simultaneous voices: 1, 2, or 4 |
| Oct Range    | Octave spread of resonator bank: 1-4 octaves |
| Pre Gain     | Input gain (-12 to +12 dB) |
| Post Gain    | Output gain (-12 to +12 dB) |
| HPF          | High-pass filter (20 Hz - 2 kHz) |
| LPF          | Low-pass filter (500 Hz - 20 kHz) |
| Limiter      | Tape soft clipper on/off |

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

- **YIN pitch detection** (2048-sample buffer)
- **Spectral flux onset detection** (512-point FFT)
- **Spectral flatness** analysis for auto-brightness
- **SVF bandpass resonator bank** (up to 4 voices × 12 filters = 48 SVFs)
- **Scale quantizer** with chord drift

## Credits

Algorithmic references: Mutable Instruments Elements/Rings resonator (MIT), YIN algorithm (de Cheveigné & Kawahara 2002).

## License

MIT — see [LICENSE](LICENSE)
