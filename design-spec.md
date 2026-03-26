# Spectra — Design Spec

## Overview
**Spectra** is a multiband spectral resonator audio FX for Ableton Move (Schwung framework).
It analyzes incoming audio in real-time — detecting pitch, onsets, and noise/tonal ratio — then
excites a bank of musically-tuned resonant SVF bandpass filters quantized to a user-selected
scale and root note. The noise-vs-tonal ratio of the input inversely controls brightness, so
noisy sources produce darker resonances and tonal sources ring bright.

## Module identity
- **Module ID:** `spectra`
- **Name:** Spectra
- **Abbreviation:** SPCTRA
- **Author:** fillioning
- **License:** MIT
- **Language:** C (self-contained, no external dependencies)
- **API:** audio_fx_api_v2
- **Component type:** audio_fx

## Signal flow
```
Audio In
  │
  ├──► Analysis Stage (per block)
  │      ├── Onset detection (spectral flux, 512-pt FFT)
  │      ├── Pitch detection (YIN, 2048-sample circular buffer)
  │      └── Noise ratio (spectral flatness from same FFT)
  │
  ├──► Scale Quantizer
  │      ├── Input: detected freq (Hz) + onset gate + inverse flatness
  │      ├── Params: Root note (12) × Scale (12) = 144 combos
  │      ├── Chord Drift: probability of ±1–2 semitone offset
  │      └── Output: quantized MIDI note(s), gate, brightness
  │
  ├──► Voice Allocator (1 / 2 / 4 polyphony, round-robin with steal)
  │
  ├──► Resonator Bank (per voice)
  │      ├── 5 / 7 / 12 SVF bandpass filters tuned to scale degrees
  │      ├── Timbre: shifts partial ratios (harmonic → inharmonic)
  │      ├── Decay: controls Q / ring time (50ms – 10s)
  │      ├── Brightness: inverse flatness + manual offset → filter bandwidth
  │      └── Octave range: 1–4 octaves of resonator spread
  │
  └──► Output Chain
         ├── HPF (20Hz – 2kHz)
         ├── LPF (500Hz – 20kHz)
         ├── Limiter (on/off)
         └── Equal-power dry/wet mix (Knob 8)
```

## Parameters

### Page 1 — Main (knob-mapped)

| Knob | Key         | Label      | Type  | Range      | Default | Notes |
|------|-------------|------------|-------|------------|---------|-------|
| 1    | onset       | Onset      | float | 0.0 – 1.0 | 0.3     | Spectral flux threshold |
| 2    | frequency   | Frequency  | float | 0.0 – 1.0 | 0.5     | YIN confidence threshold |
| 3    | brightness  | Brightness | float | 0.0 – 1.0 | 0.5     | Manual offset added to auto-detected inverse flatness |
| 4    | timbre      | Timbre     | float | 0.0 – 1.0 | 0.0     | Inharmonicity — 0=pure harmonic, 1=metallic/bell |
| 5    | decay       | Decay      | float | 0.0 – 1.0 | 0.5     | Ring time — maps exponentially to 50ms–10s |
| 6    | root_note   | Root       | enum  | C..B (12)  | 0 (C)   | Root note of scale |
| 7    | scale       | Scale      | enum  | 12 modes   | 0 (Maj) | Scale type |
| 8    | mix         | Mix        | float | 0.0 – 1.0 | 0.5     | Equal-power dry/wet |

### Page 2 — Control (menu only)

| Key          | Label       | Type  | Range              | Default |
|--------------|-------------|-------|--------------------|---------|
| chord_drift  | Chord Drift | float | 0.0 – 1.0         | 0.0     |
| resonators   | Resonators  | enum  | 5 / 7 / 12        | 1 (7)   |
| polyphony    | Polyphony   | enum  | 1 / 2 / 4         | 1 (2)   |
| octave_range | Oct Range   | enum  | 1 / 2 / 3 / 4     | 3 (4)   |
| pre_gain     | Pre Gain    | float | -12.0 – 12.0 dB   | 0.0     |
| post_gain    | Post Gain   | float | -12.0 – 12.0 dB   | 0.0     |
| hpf          | HPF         | float | 20 – 2000 Hz       | 20.0    |
| lpf          | LPF         | float | 500 – 20000 Hz     | 20000.0 |
| limiter      | Limiter     | enum  | Off / On           | 1 (On)  |

### Scale definitions (12 scales)

| Index | Name           | Intervals (semitones from root)      |
|-------|----------------|--------------------------------------|
| 0     | Major          | 0, 2, 4, 5, 7, 9, 11                |
| 1     | Natural Minor  | 0, 2, 3, 5, 7, 8, 10                |
| 2     | Harmonic Minor | 0, 2, 3, 5, 7, 8, 11                |
| 3     | Melodic Minor  | 0, 2, 3, 5, 7, 9, 11                |
| 4     | Dorian         | 0, 2, 3, 5, 7, 9, 10                |
| 5     | Phrygian       | 0, 1, 3, 5, 7, 8, 10                |
| 6     | Lydian         | 0, 2, 4, 6, 7, 9, 11                |
| 7     | Mixolydian     | 0, 2, 4, 5, 7, 9, 10                |
| 8     | Penta Major    | 0, 2, 4, 7, 9                        |
| 9     | Penta Minor    | 0, 3, 5, 7, 10                       |
| 10    | Blues          | 0, 3, 5, 6, 7, 10                    |
| 11    | Chromatic      | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 |

## DSP components (all self-contained C)

1. **512-point FFT** — shared between onset detection and spectral flatness
2. **YIN pitch detector** — 2048-sample circular buffer, autocorrelation difference
3. **Spectral flux onset detector** — positive magnitude differences frame-to-frame
4. **Spectral flatness** — geometric/arithmetic mean ratio of FFT magnitude bins
5. **Scale quantizer** — lookup table, 12 patterns × 12 root offsets
6. **Chord Drift** — per-onset random ±1–2 semitone offset based on drift probability
7. **SVF bandpass resonator bank** — per-voice, up to 12 filters
8. **Voice allocator** — round-robin with steal, 1/2/4 polyphony
9. **Output chain** — pre-gain → HPF → LPF → limiter → post-gain → equal-power mix

## CPU budget
- Max: 4 voices × 12 resonators = 48 SVFs + 1 FFT(512) + 1 YIN(2048)
- Target: <30% CPU on CM4 at 44.1kHz / 128-sample blocks
- Resonator count and polyphony are user-adjustable to trade quality for CPU

## Open-source references (algorithmic, not ported)
- Elements/Rings resonator (pichenettes/eurorack) — brightness-to-bandwidth mapping, MIT
- YIN paper (de Cheveigné & Kawahara 2002) — pitch detection algorithm
- Spectral flux onset detection — standard MIR technique
