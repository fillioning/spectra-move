# Spectra — Claude Code context

## What this is
Multiband spectral resonator: analyzes incoming audio (pitch, onset, noise ratio) and excites scale-quantized SVF bandpass filters.
Schwung audio FX module. API: audio_fx_api_v2. Language: C.

## Repo structure
- `src/dsp/spectra.c` — Move API wrapper + params + all DSP (YIN, FFT, onset, resonator bank)
- `src/dsp/audio_fx_api_v2.h` — API header (CI needs this)
- `src/module.json` — module metadata (desktop installer reads this)
- `module.json` — root copy (for builds, must stay in sync with src/)
- `scripts/build.sh` — Docker ARM64 cross-compile
- `scripts/install.sh` — deploys to Move via scp + fixes ownership
- `scripts/Dockerfile` — build container
- `.github/workflows/release.yml` — CI: verifies version, builds, releases

## Shadow UI integration (CRITICAL)
- `ui_hierarchy` lives in **module.json ONLY** — do NOT return it from get_param
- The DSP MUST implement these get_param responses:
  - `chain_params` — JSON array of parameter metadata (type, min, max, step, options)
  - `knob_N_name` / `knob_N_value` — popup label and formatted value for knobs 1-8
- Enum params: get_param MUST return the name string (not integer index)
- No ui_chain.js — Shadow UI handles everything from module.json + DSP responses

## Parameters

### Page 1 — Main (knob-mapped)
| Knob | Key        | Type  | Range/Options | Default |
|------|------------|-------|---------------|---------|
| 1    | onset      | float | 0.0 – 1.0    | 0.3     |
| 2    | frequency  | float | 0.0 – 1.0    | 0.5     |
| 3    | brightness | float | 0.0 – 1.0    | 0.5     |
| 4    | timbre     | float | 0.0 – 1.0    | 0.0     |
| 5    | decay      | float | 0.0 – 1.0    | 0.5     |
| 6    | root_note  | enum  | C..B (12)     | 0 (C)   |
| 7    | scale      | enum  | 12 modes      | 0 (Maj) |
| 8    | mix        | float | 0.0 – 1.0    | 0.5     |

### Page 2 — Control (menu only)
| Key          | Type  | Range              | Default |
|--------------|-------|--------------------|---------|
| chord_drift  | float | 0.0 – 1.0         | 0.0     |
| resonators   | enum  | 5/7/12             | 1 (7)   |
| polyphony    | enum  | 1/2/4              | 1 (2)   |
| octave_range | enum  | 1/2/3/4            | 3 (4)   |
| pre_gain     | float | -12.0 – 12.0 dB   | 0.0     |
| post_gain    | float | -12.0 – 12.0 dB   | 0.0     |
| hpf          | float | 20 – 2000 Hz       | 20.0    |
| lpf          | float | 500 – 20000 Hz     | 20000.0 |
| limiter      | enum  | Off/On             | 1 (On)  |

## Critical constraints
- NEVER allocate memory in `process_block` — all DSP state lives in instance struct
- NEVER call printf/log/mutex in `process_block`
- Compile with `-ffast-math` — ARM has no FTZ; denormals WILL stall the pipeline
- Files on Move must be owned by `ableton:users`
- `module.json` must have `component_type` at BOTH root level AND inside `capabilities`
- `release.json` is auto-updated by CI — never edit manually
- Git tag `vX.Y.Z` must match `version` in `module.json` exactly
- Constants and static arrays must be declared BEFORE the functions that use them
- State serialization format specifiers must match types: `%f` for float, `%d` for int
- Equal-power crossfade for Mix (cos/sin), never linear
- `get_param` MUST return -1 for unknown keys (not 0)

## DSP architecture
- 512-pt FFT: real-only, split-radix, shared between onset + flatness
- YIN: 2048-sample circular buffer, autocorrelation difference function
- Spectral flux: sum of positive magnitude differences between frames
- Spectral flatness: geometric/arithmetic mean of magnitude bins → inverted for brightness
- Scale quantizer: 12 scale patterns × 12 roots, MIDI note snapping
- Chord Drift: per-onset random ±1–2 semitone offset
- SVF resonator bank: per-voice, 5/7/12 bandpass filters
- Voice allocator: round-robin with steal, 1/2/4 polyphony
- Output chain: pre-gain → HPF → LPF → limiter → post-gain → mix

## Build & deploy
```bash
./scripts/build.sh          # Docker ARM64 cross-compile
./scripts/install.sh        # Deploy to move.local
```

## Release
Use the `/move-schwung-release` skill.

## Source / license
MIT — all original code. Algorithmic references: Mutable Instruments Elements/Rings resonator (MIT), YIN paper (de Cheveigné & Kawahara 2002).
