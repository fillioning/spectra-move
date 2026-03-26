/**
 * Spectra — Multiband Spectral Resonator
 * Author: fillioning
 * License: MIT
 *
 * Analyzes incoming audio (pitch, onset, noise ratio) and excites a bank of
 * musically-tuned SVF bandpass filters quantized to a user-selected scale.
 *
 * API: audio_fx_api_v2 (in-place stereo processing, int16 interleaved, 44100Hz, 128 frames/block)
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

/* ── host_api_v1_t — MUST match chain_host ABI exactly ──────────────────────── */
typedef int (*move_mod_emit_value_fn)(void *ctx,
                                      const char *source_id,
                                      const char *target,
                                      const char *param,
                                      float signal, float depth, float offset,
                                      int bipolar, int enabled);
typedef void (*move_mod_clear_source_fn)(void *ctx, const char *source_id);

typedef struct host_api_v1 {
    uint32_t api_version;
    int sample_rate;
    int frames_per_block;
    uint8_t *mapped_memory;
    int audio_out_offset;
    int audio_in_offset;
    void (*log)(const char *msg);
    int (*midi_send_internal)(const uint8_t *msg, int len);
    int (*midi_send_external)(const uint8_t *msg, int len);
    int (*get_clock_status)(void);
    move_mod_emit_value_fn mod_emit_value;
    move_mod_clear_source_fn mod_clear_source;
    void *mod_host_ctx;
} host_api_v1_t;

typedef struct audio_fx_api_v2 {
    uint32_t api_version;
    void* (*create_instance)(const char *module_dir, const char *config_json);
    void  (*destroy_instance)(void *instance);
    void  (*process_block)(void *instance, int16_t *audio_inout, int frames);
    void  (*set_param)(void *instance, const char *key, const char *val);
    int   (*get_param)(void *instance, const char *key, char *buf, int buf_len);
    void  (*on_midi)(void *instance, const uint8_t *msg, int len, int source);
} audio_fx_api_v2_t;

static const host_api_v1_t *g_host = NULL;

/* ── Helpers ─────────────────────────────────────────────────────────────────── */

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ── Constants ───────────────────────────────────────────────────────────────── */

#define SAMPLE_RATE     44100
#define BLOCK_SIZE      128
#define FFT_SIZE        512
#define MAX_VOICES      4
#define MAX_RESONATORS  12

/* ── Multiband analysis ─────────────────────────────────────────────────────── */
/* 8 bands, roughly octave-spaced. 512-pt FFT at 44.1k = 86.13 Hz/bin.
 * Each band covers a frequency range relevant to drum/instrument separation:
 *   0: sub-bass (kick fundamental)    1: bass (kick body, bass)
 *   2: low-mid (toms, low snare)      3: mid (snare body, vocals)
 *   4: upper-mid (snare crack, clap)  5: presence (hi-hat attack)
 *   6: brilliance (hi-hat, cymbals)   7: air (shimmer, noise)
 */
#define NUM_BANDS 8
/* Bin ranges [start, end) for each band */
static const int BAND_BIN_START[NUM_BANDS] = {  1,   2,   4,   8,  16,  32,  64, 128 };
static const int BAND_BIN_END[NUM_BANDS]   = {  2,   4,   8,  16,  32,  64, 128, 256 };
/* Center frequency of each band in Hz (geometric mean of bin range) */
static const float BAND_CENTER_HZ[NUM_BANDS] = {
    86.0f, 258.0f, 516.0f, 1033.0f, 2067.0f, 4134.0f, 8268.0f, 16536.0f
};

/* ── Enum definitions ────────────────────────────────────────────────────────── */

#define NUM_ROOT_NOTES 12
static const char *ROOT_NOTE_NAMES[NUM_ROOT_NOTES] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};

#define NUM_SCALES 12
static const char *SCALE_NAMES[NUM_SCALES] = {
    "Major", "Nat Minor", "Harm Minor", "Mel Minor",
    "Dorian", "Phrygian", "Lydian", "Mixolydian",
    "Penta Maj", "Penta Min", "Blues", "Chromatic"
};

/* Scale interval patterns — each row is intervals in semitones, terminated by -1 */
static const int SCALE_INTERVALS[NUM_SCALES][13] = {
    { 0, 2, 4, 5, 7, 9, 11, -1 },             /* Major */
    { 0, 2, 3, 5, 7, 8, 10, -1 },             /* Natural Minor */
    { 0, 2, 3, 5, 7, 8, 11, -1 },             /* Harmonic Minor */
    { 0, 2, 3, 5, 7, 9, 11, -1 },             /* Melodic Minor */
    { 0, 2, 3, 5, 7, 9, 10, -1 },             /* Dorian */
    { 0, 1, 3, 5, 7, 8, 10, -1 },             /* Phrygian */
    { 0, 2, 4, 6, 7, 9, 11, -1 },             /* Lydian */
    { 0, 2, 4, 5, 7, 9, 10, -1 },             /* Mixolydian */
    { 0, 2, 4, 7, 9, -1 },                     /* Pentatonic Major */
    { 0, 3, 5, 7, 10, -1 },                    /* Pentatonic Minor */
    { 0, 3, 5, 6, 7, 10, -1 },                /* Blues */
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1 } /* Chromatic */
};

/* Number of notes per scale */
static const int SCALE_SIZES[NUM_SCALES] = {
    7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 6, 12
};

/* Resonator count options */
static const int RESONATOR_COUNTS[3] = { 5, 7, 12 };

/* Polyphony options */
static const int POLYPHONY_OPTIONS[3] = { 1, 2, 4 };

/* Limiter option names */
#define NUM_LIMITER_OPTIONS 2
static const char *LIMITER_NAMES[NUM_LIMITER_OPTIONS] = { "Off", "On" };

/* ── Knob mapping ────────────────────────────────────────────────────────────── */

typedef struct {
    const char *key;
    const char *label;
    float min, max, step;
    int is_enum;
} knob_def_t;

static const knob_def_t KNOB_MAP[8] = {
    { "onset",      "Onset",      0, 1, 0.01f, 0 },
    { "frequency",  "Frequency",  0, 1, 0.01f, 0 },
    { "brightness", "Brightness", 0, 1, 0.01f, 0 },
    { "timbre",     "Timbre",     0, 1, 0.01f, 0 },
    { "decay",      "Decay",      0, 1, 0.01f, 0 },
    { "root_note",  "Root",       0, 11, 1.0f, 1 },
    { "scale",      "Scale",      0, 11, 1.0f, 1 },
    { "mix",        "Mix",        0, 1, 0.01f, 0 },
};

/* Page 2 knob mapping (Control page)
 * 1=Chord Drift, 2=Resonators, 3=Oct Range, 4=Compress,
 * 5=Pre Gain, 6=Post Gain, 7=HPF, 8=LPF
 * Polyphony is menu-only (not on a knob). */
static const knob_def_t KNOB_MAP_P2[8] = {
    { "chord_drift",  "Chrd Drft",  0, 1, 0.01f, 0 },
    { "resonators",   "Resonatrs",  0, 2, 1.0f,  1 },
    { "octave_range", "Oct Range",  0, 3, 1.0f,  1 },
    { "compress",     "Compress",   0, 1, 0.01f, 0 },
    { "pre_gain",     "Pre Gain",  -12, 12, 0.5f, 0 },
    { "post_gain",    "Post Gain", -12, 12, 0.5f, 0 },
    { "hpf",          "HPF",        20, 2000, 20.0f, 0 },
    { "lpf",          "LPF",       500, 20000, 200.0f, 0 },
};

/* ── FFT (real-only, radix-2 DIT) ────────────────────────────────────────────
 * Minimal in-place FFT for spectral analysis. Only computes magnitude spectrum.
 * Not the fastest, but simple and correct for 512-point real FFT at audio rate.
 */

static void fft_bit_reverse(float *re, float *im, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }
}

static void fft_forward(float *re, float *im, int n) {
    fft_bit_reverse(re, im, n);
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / (float)len;
        float wre = cosf(angle);
        float wim = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float tre = cur_re * re[i + j + len/2] - cur_im * im[i + j + len/2];
                float tim = cur_re * im[i + j + len/2] + cur_im * re[i + j + len/2];
                re[i + j + len/2] = re[i + j] - tre;
                im[i + j + len/2] = im[i + j] - tim;
                re[i + j] += tre;
                im[i + j] += tim;
                float new_re = cur_re * wre - cur_im * wim;
                cur_im = cur_re * wim + cur_im * wre;
                cur_re = new_re;
            }
        }
    }
}

/* ── SVF (State Variable Filter) ─────────────────────────────────────────────
 * Chamberlin SVF — bandpass output used for resonator.
 * Each filter maintains its own state.
 */

/* Denormal guard — ARM has no FTZ, denormals stall the pipeline */
#define DENORMAL_GUARD 1e-25f

typedef struct {
    float lp;     /* lowpass state */
    float bp;     /* bandpass state */
    float freq;   /* target frequency Hz */
    float q;      /* resonance (Q factor) */
    float fc;     /* precomputed coefficient — call svf_update_coeff when freq/q change */
    float fb;     /* precomputed 1/q */
} svf_state_t;

/* Call this when freq or q changes (once per block, NOT per sample) */
static inline void svf_update_coeff(svf_state_t *f) {
    f->fc = 2.0f * sinf(M_PI * f->freq / (float)SAMPLE_RATE);
    if (f->fc > 0.95f) f->fc = 0.95f; /* stability limit */
    f->fb = (f->q > 0.01f) ? 1.0f / f->q : 100.0f;
}

static inline float svf_process_bp(svf_state_t *f, float input) {
    f->lp += f->fc * f->bp + DENORMAL_GUARD;
    float hp = input - f->lp - f->fb * f->bp;
    f->bp += f->fc * hp + DENORMAL_GUARD;
    /* Normalize by Q: raw bandpass gain ≈ Q at resonance.
     * Multiply by fb (= 1/Q) to keep output level roughly constant. */
    return f->bp * f->fb;
}

/* One-pole filter for HPF/LPF output chain */
typedef struct {
    float state;
    float coeff; /* precomputed — call onepole_set_freq to update */
} onepole_t;

static inline void onepole_set_freq(onepole_t *f, float cutoff_hz) {
    f->coeff = 1.0f - expf(-2.0f * M_PI * cutoff_hz / (float)SAMPLE_RATE);
}

static inline float onepole_lp(onepole_t *f, float input) {
    f->state += f->coeff * (input - f->state) + DENORMAL_GUARD;
    return f->state;
}

static inline float onepole_hp(onepole_t *f, float input) {
    f->state += f->coeff * (input - f->state) + DENORMAL_GUARD;
    return input - f->state;
}

/* ── Voice state ─────────────────────────────────────────────────────────────── */

typedef struct {
    int active;
    float midi_note;          /* quantized MIDI note */
    float amplitude;          /* current amplitude (envelope) */
    float decay_rate;         /* per-sample decay multiplier */
    svf_state_t resonators[MAX_RESONATORS]; /* resonator bank */
    int num_resonators;       /* active resonator count */
} voice_t;

/* ── Instance state ──────────────────────────────────────────────────────────── */

typedef struct {
    /* Page 1 — Main (knob-mapped) */
    float onset;           /* spectral flux threshold 0-1 */
    float frequency;       /* band selectivity 0-1: 0=all→root, 1=full freq mapping */
    float brightness;      /* manual brightness offset 0-1 */
    float timbre;          /* inharmonicity 0-1 */
    float decay;           /* ring time 0-1 (maps to 50ms-10s) */
    int   root_note;       /* 0-11 (C..B) */
    int   scale;           /* 0-11 scale index */
    float mix;             /* dry/wet 0-1 */

    /* Page 2 — Control (menu only) */
    float chord_drift;     /* 0-1 drift probability */
    int   resonators_idx;  /* 0=5, 1=7, 2=12 */
    int   polyphony_idx;   /* 0=1, 1=2, 2=4 */
    int   octave_range_idx; /* 0=1, 1=2, 2=3, 3=4 */
    float pre_gain;        /* -12 to +12 dB */
    float post_gain;       /* -12 to +12 dB */
    int   hpf;             /* 20-2000 Hz (integer target) */
    int   lpf;             /* 500-20000 Hz (integer target) */
    float hpf_smooth;      /* smoothed HPF for coefficient computation */
    float lpf_smooth;      /* smoothed LPF for coefficient computation */
    int   limiter;         /* 0=Off, 1=On */
    float compress;        /* RMS compressor strength 0-1 */

    /* FFT analysis state */
    float fft_re[FFT_SIZE];
    float fft_im[FFT_SIZE];
    float prev_magnitude[FFT_SIZE / 2]; /* previous frame magnitudes for spectral flux */
    float fft_window[FFT_SIZE];         /* Hann window */
    float mono_buffer[FFT_SIZE];        /* accumulate mono samples for FFT */
    int   mono_write_pos;
    int   fft_ready;                    /* flag: enough samples accumulated */

    /* Multiband analysis state */
    float band_energy[NUM_BANDS];       /* current energy per band */
    float band_prev_energy[NUM_BANDS];  /* previous frame energy per band */
    float band_flux_avg[NUM_BANDS];     /* running average of per-band flux */
    int   band_onset[NUM_BANDS];        /* onset flag per band per frame */

    /* Global analysis */
    float spectral_flatness;   /* 0-1 */
    float input_rms;           /* smoothed RMS of input */

    /* Voice allocator */
    voice_t voices[MAX_VOICES];
    int next_voice;         /* round-robin index */

    /* Output filters */
    onepole_t hp_l, hp_r;
    onepole_t lp_l, lp_r;

    /* Compressor state */
    float comp_env_l;      /* RMS envelope follower L */
    float comp_env_r;      /* RMS envelope follower R */

    /* UI page tracking (0=Spectra/Main, 1=Control) */
    int current_page;

    /* PRNG state (xorshift32 for chord drift) */
    uint32_t rng_state;

} plugin_instance_t;

/* ── PRNG ────────────────────────────────────────────────────────────────────── */

static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand_float(uint32_t *state) {
    return (float)(xorshift32(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* ── Param pointer helpers ───────────────────────────────────────────────────── */

/* Page 1 float param pointers */
static float *get_float_param_ptr(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 0: return &inst->onset;
        case 1: return &inst->frequency;
        case 2: return &inst->brightness;
        case 3: return &inst->timbre;
        case 4: return &inst->decay;
        case 7: return &inst->mix;
        default: return NULL;
    }
}

/* Page 1 int param pointers */
static int *get_int_param_ptr(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 5: return &inst->root_note;
        case 6: return &inst->scale;
        default: return NULL;
    }
}

/* Page 2 float param pointers */
static float *get_float_param_ptr_p2(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 0: return &inst->chord_drift;
        case 3: return &inst->compress;
        case 4: return &inst->pre_gain;
        case 5: return &inst->post_gain;
        default: return NULL;
    }
}

/* Page 2 int param pointers */
static int *get_int_param_ptr_p2(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 1: return &inst->resonators_idx;
        case 2: return &inst->octave_range_idx;
        default: return NULL;
    }
}

/* ── Scale quantizer ─────────────────────────────────────────────────────────── */

static float freq_to_midi(float freq) {
    if (freq <= 0.0f) return 0.0f;
    return 69.0f + 12.0f * log2f(freq / 440.0f);
}

static float midi_to_freq(float midi) {
    return 440.0f * powf(2.0f, (midi - 69.0f) / 12.0f);
}

/* Quantize a MIDI note to the nearest note in the given scale */
static float quantize_to_scale(float midi_note, int root, int scale_idx) {
    if (scale_idx < 0 || scale_idx >= NUM_SCALES) return midi_note;

    int note = (int)roundf(midi_note);
    int octave = note / 12;
    int pc = note % 12; /* pitch class */
    if (pc < 0) { pc += 12; octave--; }

    /* Relative to root */
    int rel = (pc - root + 12) % 12;

    /* Find nearest scale degree */
    int best = 0;
    int best_dist = 999;
    const int *intervals = SCALE_INTERVALS[scale_idx];
    int size = SCALE_SIZES[scale_idx];
    for (int i = 0; i < size; i++) {
        int dist = abs(rel - intervals[i]);
        /* Also check wrapping */
        int wrap_dist = 12 - dist;
        if (wrap_dist < dist) dist = wrap_dist;
        if (dist < best_dist) {
            best_dist = dist;
            best = intervals[i];
        }
    }

    int quantized_pc = (root + best) % 12;
    return (float)(octave * 12 + quantized_pc);
}

/* Apply chord drift: randomly offset ±1 or ±2 semitones with probability */
static float apply_chord_drift(float midi_note, float drift_amount, uint32_t *rng) {
    if (drift_amount <= 0.0f) return midi_note;
    float r = rand_float(rng);
    if (r < drift_amount) {
        float r2 = rand_float(rng);
        int offset;
        if (r2 < 0.7f) {
            offset = (rand_float(rng) < 0.5f) ? 1 : -1;
        } else {
            offset = (rand_float(rng) < 0.5f) ? 2 : -2;
        }
        return midi_note + (float)offset;
    }
    return midi_note;
}

/* ── Resonator bank tuning ───────────────────────────────────────────────────
 * Timbre controls the character:
 *   0.0 = pure harmonic series (f, 2f, 3f...) — synth-like, clear pitch
 *   0.5 = blend between harmonics and scale degrees
 *   1.0 = scale degrees across octaves — chordal, bell-like, spread
 * This crossfade is the key to making Timbre audible and musical.
 */

static void tune_resonator_bank(voice_t *v, float base_midi, int root, int scale_idx,
                                 int res_count, int octave_range, float timbre, float q_value) {
    if (res_count > MAX_RESONATORS) res_count = MAX_RESONATORS;
    float base_freq = midi_to_freq(base_midi);

    /* ── Compute harmonic series frequencies (timbre=0 target) ── */
    float harm_freq[MAX_RESONATORS];
    for (int i = 0; i < res_count; i++) {
        /* Harmonic series: f, 2f, 3f, 4f... clamped to audible range */
        harm_freq[i] = clampf(base_freq * (float)(i + 1), 20.0f, 18000.0f);
    }

    /* ── Compute scale-degree frequencies (timbre=1 target) ── */
    float scale_freq[MAX_RESONATORS];
    int size = SCALE_SIZES[scale_idx];
    const int *intervals = SCALE_INTERVALS[scale_idx];
    int base_octave = (int)base_midi / 12;
    int idx = 0;
    for (int oct = 0; oct < octave_range && idx < res_count; oct++) {
        for (int i = 0; i < size && idx < res_count; i++) {
            float midi = (float)((base_octave + oct) * 12 + (root + intervals[i]) % 12);
            if (midi < base_midi - 0.5f) midi += 12.0f;
            scale_freq[idx] = clampf(midi_to_freq(midi), 20.0f, 18000.0f);
            idx++;
        }
    }
    /* Fill any remaining with octave-folded harmonics */
    for (int i = idx; i < res_count; i++) {
        scale_freq[i] = harm_freq[i < MAX_RESONATORS ? i : MAX_RESONATORS - 1];
    }

    /* ── Crossfade between harmonic and scale-degree tuning ── */
    for (int i = 0; i < res_count; i++) {
        float freq = harm_freq[i] * (1.0f - timbre) + scale_freq[i] * timbre;
        v->resonators[i].freq = clampf(freq, 20.0f, 18000.0f);
        v->resonators[i].q = q_value;
        svf_update_coeff(&v->resonators[i]);
    }

    /* Silence unused resonators */
    for (int i = res_count; i < MAX_RESONATORS; i++) {
        v->resonators[i].freq = 0.0f;
        v->resonators[i].fc = 0.0f;
    }
    v->num_resonators = res_count;
}

/* ── Tape soft clipper (limiter) ─────────────────────────────────────────────── */

static inline float tape_soft_clip(float x) {
    if (x > 1.5f) return 1.0f;
    if (x < -1.5f) return -1.0f;
    if (x > 1.0f) { float t = x - 1.0f; return 1.0f - t * t * 0.5f; }
    if (x < -1.0f) { float t = x + 1.0f; return -1.0f + t * t * 0.5f; }
    return x - x * x * x / 3.0f;
}

/* ── Lifecycle ───────────────────────────────────────────────────────────────── */

static void *create_instance(const char *module_dir, const char *json_defaults) {
    plugin_instance_t *inst = calloc(1, sizeof(plugin_instance_t));
    if (!inst) return NULL;

    /* Page 1 defaults */
    inst->onset = 0.7f;
    inst->frequency = 0.3f;
    inst->brightness = 0.3f;
    inst->timbre = 0.0f;
    inst->decay = 0.5f;
    inst->root_note = 0;    /* C */
    inst->scale = 0;        /* Major */
    inst->mix = 1.0f;

    /* Page 2 defaults */
    inst->chord_drift = 0.0f;
    inst->resonators_idx = 2;   /* 12 */
    inst->polyphony_idx = 2;    /* 4 */
    inst->octave_range_idx = 1; /* 2 octaves — lighter CPU than 4 */
    inst->pre_gain = 0.0f;
    inst->post_gain = 0.0f;
    inst->hpf = 20;
    inst->lpf = 20000;
    inst->hpf_smooth = 20.0f;
    inst->lpf_smooth = 20000.0f;
    inst->limiter = 1;     /* On */
    inst->compress = 0.2f; /* 20% — same default as Dissolver */

    /* Initialize Hann window for FFT */
    for (int i = 0; i < FFT_SIZE; i++) {
        inst->fft_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)(FFT_SIZE - 1)));
    }

    /* PRNG seed */
    inst->rng_state = 0xDEADBEEF;

    if (g_host && g_host->log) g_host->log("[spectra] instance created");
    return inst;
}

static void destroy_instance(void *instance) {
    free(instance);
}

/* ── Parameters ──────────────────────────────────────────────────────────────── */

static void set_param(void *instance, const char *key, const char *val) {
    plugin_instance_t *inst = (plugin_instance_t *)instance;
    if (!inst || !key || !val) return;

    /* ── Page tracking (Schwung sends _level when user navigates pages) ── */
    if (strcmp(key, "_level") == 0) {
        if (strcmp(val, "Control") == 0) inst->current_page = 1;
        else inst->current_page = 0; /* "Spectra" or root */
        return;
    }

    /* ── knob_N_adjust (page-aware) ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_adjust")) {
        int knob_num = atoi(key + 5);
        int idx = knob_num - 1;
        if (idx < 0 || idx >= 8) return;

        float delta = atof(val);
        const knob_def_t *k;
        float *fp = NULL;
        int *ip = NULL;

        if (inst->current_page == 1) {
            /* Control page */
            k = &KNOB_MAP_P2[idx];
            /* HPF/LPF are int with clamping (not wrapping) */
            if (idx == 6) { inst->hpf = clampi(inst->hpf + (int)(delta * k->step), 20, 2000); return; }
            if (idx == 7) { inst->lpf = clampi(inst->lpf + (int)(delta * k->step), 500, 20000); return; }
            if (k->is_enum) ip = get_int_param_ptr_p2(inst, idx);
            else fp = get_float_param_ptr_p2(inst, idx);
        } else {
            /* Main page */
            k = &KNOB_MAP[idx];
            if (k->is_enum) ip = get_int_param_ptr(inst, idx);
            else fp = get_float_param_ptr(inst, idx);
        }

        if (k->is_enum && ip) {
            int new_val = *ip + (int)delta;
            if (new_val > (int)k->max) new_val = (int)k->min;
            if (new_val < (int)k->min) new_val = (int)k->max;
            *ip = new_val;
        } else if (fp) {
            *fp = clampf(*fp + delta * k->step, k->min, k->max);
        }
        return;
    }

    /* ── Page 1 params ── */
    if (strcmp(key, "onset") == 0)       inst->onset = clampf(atof(val), 0, 1);
    else if (strcmp(key, "frequency") == 0)  inst->frequency = clampf(atof(val), 0, 1);
    else if (strcmp(key, "brightness") == 0) inst->brightness = clampf(atof(val), 0, 1);
    else if (strcmp(key, "timbre") == 0)     inst->timbre = clampf(atof(val), 0, 1);
    else if (strcmp(key, "decay") == 0)      inst->decay = clampf(atof(val), 0, 1);
    else if (strcmp(key, "mix") == 0)        inst->mix = clampf(atof(val), 0, 1);
    else if (strcmp(key, "root_note") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_ROOT_NOTES; i++) {
            if (strcmp(val, ROOT_NOTE_NAMES[i]) == 0) { inst->root_note = i; found = 1; break; }
        }
        if (!found) inst->root_note = clampi(atoi(val), 0, NUM_ROOT_NOTES - 1);
    }
    else if (strcmp(key, "scale") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_SCALES; i++) {
            if (strcmp(val, SCALE_NAMES[i]) == 0) { inst->scale = i; found = 1; break; }
        }
        if (!found) inst->scale = clampi(atoi(val), 0, NUM_SCALES - 1);
    }
    /* ── Page 2 params ── */
    else if (strcmp(key, "chord_drift") == 0)  inst->chord_drift = clampf(atof(val), 0, 1);
    else if (strcmp(key, "resonators") == 0) {
        if (strcmp(val, "5") == 0) inst->resonators_idx = 0;
        else if (strcmp(val, "7") == 0) inst->resonators_idx = 1;
        else if (strcmp(val, "12") == 0) inst->resonators_idx = 2;
        else inst->resonators_idx = clampi(atoi(val), 0, 2);
    }
    else if (strcmp(key, "polyphony") == 0) {
        if (strcmp(val, "1") == 0) inst->polyphony_idx = 0;
        else if (strcmp(val, "2") == 0) inst->polyphony_idx = 1;
        else if (strcmp(val, "4") == 0) inst->polyphony_idx = 2;
        else inst->polyphony_idx = clampi(atoi(val), 0, 2);
    }
    else if (strcmp(key, "octave_range") == 0) {
        if (strcmp(val, "1") == 0) inst->octave_range_idx = 0;
        else if (strcmp(val, "2") == 0) inst->octave_range_idx = 1;
        else if (strcmp(val, "3") == 0) inst->octave_range_idx = 2;
        else if (strcmp(val, "4") == 0) inst->octave_range_idx = 3;
        else inst->octave_range_idx = clampi(atoi(val), 0, 3);
    }
    else if (strcmp(key, "pre_gain") == 0)   inst->pre_gain = clampf(atof(val), -12, 12);
    else if (strcmp(key, "post_gain") == 0)  inst->post_gain = clampf(atof(val), -12, 12);
    else if (strcmp(key, "hpf") == 0)        inst->hpf = clampi(atoi(val), 20, 2000);
    else if (strcmp(key, "lpf") == 0)        inst->lpf = clampi(atoi(val), 500, 20000);
    else if (strcmp(key, "compress") == 0)  inst->compress = clampf(atof(val), 0, 1);
    else if (strcmp(key, "limiter") == 0) {
        if (strcmp(val, "Off") == 0) inst->limiter = 0;
        else if (strcmp(val, "On") == 0) inst->limiter = 1;
        else inst->limiter = clampi(atoi(val), 0, 1);
    }
    /* ── State restore ── */
    else if (strcmp(key, "state") == 0) {
        sscanf(val,
            "onset=%f;frequency=%f;brightness=%f;timbre=%f;decay=%f;"
            "root_note=%d;scale=%d;mix=%f;"
            "chord_drift=%f;resonators=%d;polyphony=%d;octave_range=%d;"
            "pre_gain=%f;post_gain=%f;hpf=%d;lpf=%d;limiter=%d",
            &inst->onset, &inst->frequency, &inst->brightness, &inst->timbre, &inst->decay,
            &inst->root_note, &inst->scale, &inst->mix,
            &inst->chord_drift, &inst->resonators_idx, &inst->polyphony_idx, &inst->octave_range_idx,
            &inst->pre_gain, &inst->post_gain, &inst->hpf, &inst->lpf, &inst->limiter);
    }
}

static int get_param(void *instance, const char *key, char *buf, int buf_len) {
    plugin_instance_t *inst = (plugin_instance_t *)instance;
    if (!inst || !key || !buf || buf_len < 1) return -1;

    if (strcmp(key, "name") == 0)
        return snprintf(buf, buf_len, "Spectra");

    /* ── chain_params ── */
    if (strcmp(key, "chain_params") == 0) {
        return snprintf(buf, buf_len,
            "["
            "{\"key\":\"onset\",\"name\":\"Onset\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"frequency\",\"name\":\"Frequency\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"brightness\",\"name\":\"Brightness\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"timbre\",\"name\":\"Timbre\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"decay\",\"name\":\"Decay\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"root_note\",\"name\":\"Root\",\"type\":\"enum\",\"options\":[\"C\",\"C#\",\"D\",\"D#\",\"E\",\"F\",\"F#\",\"G\",\"G#\",\"A\",\"A#\",\"B\"]},"
            "{\"key\":\"scale\",\"name\":\"Scale\",\"type\":\"enum\",\"options\":[\"Major\",\"Nat Minor\",\"Harm Minor\",\"Mel Minor\",\"Dorian\",\"Phrygian\",\"Lydian\",\"Mixolydian\",\"Penta Maj\",\"Penta Min\",\"Blues\",\"Chromatic\"]},"
            "{\"key\":\"mix\",\"name\":\"Mix\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"chord_drift\",\"name\":\"Chord Drift\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"resonators\",\"name\":\"Resonators\",\"type\":\"enum\",\"options\":[\"5\",\"7\",\"12\"]},"
            "{\"key\":\"polyphony\",\"name\":\"Polyphony\",\"type\":\"enum\",\"options\":[\"1\",\"2\",\"4\"]},"
            "{\"key\":\"octave_range\",\"name\":\"Oct Range\",\"type\":\"enum\",\"options\":[\"1\",\"2\",\"3\",\"4\"]},"
            "{\"key\":\"pre_gain\",\"name\":\"Pre Gain\",\"type\":\"float\",\"min\":-12,\"max\":12,\"step\":0.5},"
            "{\"key\":\"post_gain\",\"name\":\"Post Gain\",\"type\":\"float\",\"min\":-12,\"max\":12,\"step\":0.5},"
            "{\"key\":\"compress\",\"name\":\"Compress\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"hpf\",\"name\":\"HPF\",\"type\":\"int\",\"min\":20,\"max\":2000,\"step\":1},"
            "{\"key\":\"lpf\",\"name\":\"LPF\",\"type\":\"int\",\"min\":500,\"max\":20000,\"step\":1},"
            "{\"key\":\"limiter\",\"name\":\"Limiter\",\"type\":\"enum\",\"options\":[\"Off\",\"On\"]}"
            "]");
    }

    /* ── Regular param values ── */
    if (strcmp(key, "onset") == 0)      return snprintf(buf, buf_len, "%.2f", inst->onset);
    if (strcmp(key, "frequency") == 0)  return snprintf(buf, buf_len, "%.2f", inst->frequency);
    if (strcmp(key, "brightness") == 0) return snprintf(buf, buf_len, "%.2f", inst->brightness);
    if (strcmp(key, "timbre") == 0)     return snprintf(buf, buf_len, "%.2f", inst->timbre);
    if (strcmp(key, "decay") == 0)      return snprintf(buf, buf_len, "%.2f", inst->decay);
    if (strcmp(key, "mix") == 0)        return snprintf(buf, buf_len, "%.2f", inst->mix);
    if (strcmp(key, "root_note") == 0)
        return snprintf(buf, buf_len, "%s", ROOT_NOTE_NAMES[clampi(inst->root_note, 0, NUM_ROOT_NOTES - 1)]);
    if (strcmp(key, "scale") == 0)
        return snprintf(buf, buf_len, "%s", SCALE_NAMES[clampi(inst->scale, 0, NUM_SCALES - 1)]);

    /* Page 2 */
    if (strcmp(key, "chord_drift") == 0)  return snprintf(buf, buf_len, "%.2f", inst->chord_drift);
    if (strcmp(key, "resonators") == 0)
        return snprintf(buf, buf_len, "%d", RESONATOR_COUNTS[clampi(inst->resonators_idx, 0, 2)]);
    if (strcmp(key, "polyphony") == 0)
        return snprintf(buf, buf_len, "%d", POLYPHONY_OPTIONS[clampi(inst->polyphony_idx, 0, 2)]);
    if (strcmp(key, "octave_range") == 0)
        return snprintf(buf, buf_len, "%d", clampi(inst->octave_range_idx + 1, 1, 4));
    if (strcmp(key, "pre_gain") == 0)    return snprintf(buf, buf_len, "%.1f", inst->pre_gain);
    if (strcmp(key, "post_gain") == 0)   return snprintf(buf, buf_len, "%.1f", inst->post_gain);
    if (strcmp(key, "hpf") == 0)         return snprintf(buf, buf_len, "%d", inst->hpf);
    if (strcmp(key, "lpf") == 0)         return snprintf(buf, buf_len, "%d", inst->lpf);
    if (strcmp(key, "compress") == 0)  return snprintf(buf, buf_len, "%.2f", inst->compress);
    if (strcmp(key, "limiter") == 0)
        return snprintf(buf, buf_len, "%s", LIMITER_NAMES[clampi(inst->limiter, 0, 1)]);

    /* ── knob_N_name (page-aware) ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_name")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8) {
            const knob_def_t *k = (inst->current_page == 1) ? &KNOB_MAP_P2[idx] : &KNOB_MAP[idx];
            if (k->label) return snprintf(buf, buf_len, "%s", k->label);
        }
        return 0;
    }

    /* ── knob_N_value (page-aware) ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_value")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8) {
            const knob_def_t *k;
            if (inst->current_page == 1) {
                k = &KNOB_MAP_P2[idx];
                /* HPF/LPF are int — display directly */
                if (idx == 6) return snprintf(buf, buf_len, "%d Hz", inst->hpf);
                if (idx == 7) return snprintf(buf, buf_len, "%d Hz", inst->lpf);
                if (k->is_enum) {
                    return get_param(instance, k->key, buf, buf_len);
                } else {
                    float *p = get_float_param_ptr_p2(inst, idx);
                    if (p) {
                        if (idx == 4 || idx == 5) return snprintf(buf, buf_len, "%.1f dB", *p);
                        return snprintf(buf, buf_len, "%d%%", (int)(*p * 100));
                    }
                }
            } else {
                k = &KNOB_MAP[idx];
                if (k->is_enum) {
                    return get_param(instance, k->key, buf, buf_len);
                } else {
                    float *p = get_float_param_ptr(inst, idx);
                    if (p) return snprintf(buf, buf_len, "%d%%", (int)(*p * 100));
                }
            }
        }
        return 0;
    }

    /* ── State serialization ── */
    if (strcmp(key, "state") == 0) {
        return snprintf(buf, buf_len,
            "onset=%.6f;frequency=%.6f;brightness=%.6f;timbre=%.6f;decay=%.6f;"
            "root_note=%d;scale=%d;mix=%.6f;"
            "chord_drift=%.6f;resonators=%d;polyphony=%d;octave_range=%d;"
            "pre_gain=%.6f;post_gain=%.6f;hpf=%d;lpf=%d;limiter=%d;compress=%.6f",
            inst->onset, inst->frequency, inst->brightness, inst->timbre, inst->decay,
            inst->root_note, inst->scale, inst->mix,
            inst->chord_drift, inst->resonators_idx, inst->polyphony_idx, inst->octave_range_idx,
            inst->pre_gain, inst->post_gain, inst->hpf, inst->lpf, inst->limiter, inst->compress);
    }

    return -1; /* unknown key */
}

/* ── Audio processing ────────────────────────────────────────────────────────── */

static void process_block(void *instance, int16_t *audio_buf, int frames) {
    plugin_instance_t *inst = (plugin_instance_t *)instance;
    if (!inst || frames <= 0) return;
    if (frames > 128) frames = 128;

    /* ── Derived params ─────────────────────────────────────────────────── */
    float pre_gain_lin = powf(10.0f, inst->pre_gain / 20.0f);
    float post_gain_lin = powf(10.0f, inst->post_gain / 20.0f);
    int num_voices = POLYPHONY_OPTIONS[clampi(inst->polyphony_idx, 0, 2)];
    int res_count = RESONATOR_COUNTS[clampi(inst->resonators_idx, 0, 2)];
    int oct_range = clampi(inst->octave_range_idx + 1, 1, 4);

    float decay_seconds = 0.05f * powf(200.0f, inst->decay);
    float decay_rate = expf(-1.0f / (decay_seconds * (float)SAMPLE_RATE));
    /* Onset sensitivity: ratio above per-band running average to trigger.
     * 0% = 1.0x (triggers on everything)
     * 100% = 6.0x (only hard transients) */
    float onset_sensitivity = 1.0f + inst->onset * 5.0f;
    /* Frequency knob: band selectivity.
     * 0.0 = all bands map to root note (unison resonance)
     * 1.0 = each band maps to its natural frequency in the scale (full separation)
     * This is what lets a kick trigger a low note and a hihat trigger a high note. */
    float freq_selectivity = inst->frequency;

    /* ── Precompute per-block constants ──────────────────────────────────── */
    float mix_angle = inst->mix * 1.5707963f;
    float dry_gain = cosf(mix_angle);
    float wet_gain = sinf(mix_angle);
    /* Smooth HPF/LPF targets (~10ms smoothing at 44.1k/128 = 345 blocks/s)
     * coeff ≈ 1 - exp(-1 / (0.010 * 345)) ≈ 0.25 */
    float filt_smooth = 0.25f;
    inst->hpf_smooth += filt_smooth * ((float)inst->hpf - inst->hpf_smooth);
    inst->lpf_smooth += filt_smooth * ((float)inst->lpf - inst->lpf_smooth);
    onepole_set_freq(&inst->hp_l, inst->hpf_smooth);
    onepole_set_freq(&inst->hp_r, inst->hpf_smooth);
    onepole_set_freq(&inst->lp_l, inst->lpf_smooth);
    onepole_set_freq(&inst->lp_r, inst->lpf_smooth);

    /* Root note MIDI reference (octave 3 = C3..B3) */
    float root_midi = (float)(48 + inst->root_note);

    /* ── Convert input to mono float ────────────────────────────────────── */
    float mono[BLOCK_SIZE];
    for (int i = 0; i < frames; i++) {
        float l = (float)audio_buf[i * 2] / 32768.0f * pre_gain_lin;
        float r = (float)audio_buf[i * 2 + 1] / 32768.0f * pre_gain_lin;
        mono[i] = (l + r) * 0.5f;
    }

    /* ── Feed FFT buffer ────────────────────────────────────────────────── */
    for (int i = 0; i < frames; i++) {
        inst->mono_buffer[inst->mono_write_pos] = mono[i];
        inst->mono_write_pos++;
        if (inst->mono_write_pos >= FFT_SIZE) {
            inst->mono_write_pos = 0;
            inst->fft_ready = 1;
        }
    }

    /* ── Measure input RMS for excitation gain ──────────────────────────── */
    float rms_sum = 0.0f;
    for (int i = 0; i < frames; i++) rms_sum += mono[i] * mono[i];
    float block_rms = sqrtf(rms_sum / (float)frames);
    inst->input_rms = inst->input_rms * 0.9f + block_rms * 0.1f;

    /* ── Multiband analysis (runs when FFT buffer is full) ──────────────── */
    /* Clear per-frame band onset flags */
    for (int b = 0; b < NUM_BANDS; b++) inst->band_onset[b] = 0;

    if (inst->fft_ready) {
        inst->fft_ready = 0;

        /* Window and FFT */
        for (int i = 0; i < FFT_SIZE; i++) {
            inst->fft_re[i] = inst->mono_buffer[i] * inst->fft_window[i];
            inst->fft_im[i] = 0.0f;
        }
        fft_forward(inst->fft_re, inst->fft_im, FFT_SIZE);

        /* Compute magnitude² spectrum — skip sqrtf to save 256 sqrtf calls per frame */
        float mag_sq[FFT_SIZE / 2];
        for (int i = 0; i < FFT_SIZE / 2; i++) {
            mag_sq[i] = inst->fft_re[i] * inst->fft_re[i] +
                        inst->fft_im[i] * inst->fft_im[i];
        }

        /* ── Per-band energy + onset detection ─────────────────────────── */
        for (int b = 0; b < NUM_BANDS; b++) {
            float energy = 0.0f;
            int start = BAND_BIN_START[b];
            int end = BAND_BIN_END[b];
            if (end > FFT_SIZE / 2) end = FFT_SIZE / 2;
            for (int i = start; i < end; i++) {
                energy += mag_sq[i]; /* using mag² — still works for relative comparison */
            }

            /* Per-band spectral flux (positive differences only) */
            float flux = energy - inst->band_prev_energy[b];
            if (flux < 0.0f) flux = 0.0f;

            /* Normalize by band energy for level-independence */
            float norm_flux = (energy > 1e-10f) ? flux / energy : 0.0f;

            /* Update per-band running average */
            inst->band_flux_avg[b] = inst->band_flux_avg[b] * 0.93f + norm_flux * 0.07f;

            /* Onset: flux above adaptive threshold AND minimum energy gate */
            float min_energy = 1e-6f; /* silence gate (mag² scale) */
            if (norm_flux > inst->band_flux_avg[b] * onset_sensitivity
                && norm_flux > 0.02f && energy > min_energy) {
                inst->band_onset[b] = 1;
            }

            inst->band_energy[b] = energy;
            inst->band_prev_energy[b] = energy;
        }

        /* ── Spectral flatness (simplified — use band energy variance) ─── */
        /* Instead of expensive log/exp geometric mean, estimate flatness from
         * the ratio of min-band to max-band energy. Fast and good enough. */
        {
            float emin = 1e30f, emax = 0.0f;
            for (int b = 0; b < NUM_BANDS; b++) {
                if (inst->band_energy[b] > emax) emax = inst->band_energy[b];
                if (inst->band_energy[b] < emin) emin = inst->band_energy[b];
            }
            /* Flat spectrum (noise): emin ≈ emax → ratio ≈ 1
             * Tonal spectrum: one band dominates → ratio ≈ 0 */
            inst->spectral_flatness = (emax > 1e-10f) ? clampf(emin / emax, 0.0f, 1.0f) : 0.0f;
        }
    }

    /* ── Compute brightness + Q ─────────────────────────────────────────── */
    float auto_brightness = 1.0f - inst->spectral_flatness;
    float brightness_final = clampf(auto_brightness + (inst->brightness - 0.5f), 0.0f, 1.0f);
    /* Q: exponential mapping for more dramatic range.
     * 0.0 → Q=3 (wide, warm, muted)  1.0 → Q=80 (narrow, ringing, bright) */
    float q_value = 3.0f * powf(80.0f / 3.0f, brightness_final);
    /* Brightness also boosts resonator output: dull=0.5x, bright=2x */
    float brightness_gain = 0.5f + brightness_final * 1.5f;

    /* ── Voice allocation from multiband onsets ─────────────────────────── */
    /* Collect triggered bands, sorted by energy (strongest first).
     * Allocate up to num_voices from the strongest band onsets. */
    {
        int triggered[NUM_BANDS];
        float triggered_energy[NUM_BANDS];
        int num_triggered = 0;

        for (int b = 0; b < NUM_BANDS; b++) {
            if (inst->band_onset[b]) {
                triggered[num_triggered] = b;
                triggered_energy[num_triggered] = inst->band_energy[b];
                num_triggered++;
            }
        }

        /* Simple insertion sort by energy (descending) — max 8 elements */
        for (int i = 1; i < num_triggered; i++) {
            for (int j = i; j > 0 && triggered_energy[j] > triggered_energy[j-1]; j--) {
                float te = triggered_energy[j]; triggered_energy[j] = triggered_energy[j-1]; triggered_energy[j-1] = te;
                int tb = triggered[j]; triggered[j] = triggered[j-1]; triggered[j-1] = tb;
            }
        }

        /* Allocate up to num_voices from strongest triggers (cap at 2 per frame to save CPU) */
        int max_per_frame = num_voices < 2 ? num_voices : 2;
        int to_alloc = num_triggered < max_per_frame ? num_triggered : max_per_frame;
        for (int t = 0; t < to_alloc; t++) {
            int band = triggered[t];

            /* Map band center frequency to MIDI note.
             * freq_selectivity controls how much the band frequency matters:
             *   0.0 → all bands produce root_midi (unison)
             *   1.0 → each band maps to its natural frequency (full spread) */
            float band_midi = freq_to_midi(BAND_CENTER_HZ[band]);
            float target_midi = root_midi + freq_selectivity * (band_midi - root_midi);

            /* Quantize to scale + drift */
            float quantized = quantize_to_scale(target_midi, inst->root_note, inst->scale);
            quantized = apply_chord_drift(quantized, inst->chord_drift, &inst->rng_state);

            /* Allocate voice (round-robin with steal) */
            int vi = inst->next_voice % num_voices;
            inst->next_voice = (inst->next_voice + 1) % num_voices;

            voice_t *v = &inst->voices[vi];
            v->active = 1;
            v->midi_note = quantized;
            v->amplitude = 1.0f;
            v->decay_rate = decay_rate;

            tune_resonator_bank(v, quantized, inst->root_note, inst->scale,
                                res_count, oct_range, inst->timbre, q_value);

            /* Clear filter state for new note */
            for (int r = 0; r < v->num_resonators; r++) {
                v->resonators[r].lp = 0.0f;
                v->resonators[r].bp = 0.0f;
            }
        }
    }

    /* ── Update active voices with current timbre + brightness (real-time) ── */
    /* Without this, turning Timbre/Brightness only affects NEW notes.
     * Re-tune all active voices every block so knobs respond immediately. */
    for (int vi = 0; vi < MAX_VOICES; vi++) {
        voice_t *v = &inst->voices[vi];
        if (!v->active) continue;
        tune_resonator_bank(v, v->midi_note, inst->root_note, inst->scale,
                            res_count, oct_range, inst->timbre, q_value);
        /* NOTE: do NOT clear filter state here — that would cause clicks.
         * svf_update_coeff just updates fc/fb, filter state (lp/bp) continues. */
    }

    /* ── Process audio through resonator bank ───────────────────────────── */
    float wet_l_buf[BLOCK_SIZE];
    float wet_r_buf[BLOCK_SIZE];
    memset(wet_l_buf, 0, sizeof(wet_l_buf));
    memset(wet_r_buf, 0, sizeof(wet_r_buf));

    for (int vi = 0; vi < num_voices; vi++) {
        voice_t *v = &inst->voices[vi];
        if (!v->active) continue;

        float pan = 0.5f + (float)(vi - num_voices / 2) * 0.15f;
        pan = clampf(pan, 0.0f, 1.0f);
        float pan_l = 1.0f - pan;
        float pan_r = pan;
        /* With Q-normalized SVF, output level is stable regardless of Q.
         * Scale by num_resonators and apply brightness gain for tonal control. */
        float res_gain = (v->num_resonators > 0) ? 3.0f / sqrtf((float)v->num_resonators) : 0.0f;
        res_gain *= brightness_gain;
        float excitation_gain = 1.0f + inst->input_rms * 2.0f;

        for (int i = 0; i < frames; i++) {
            float input_sample = mono[i] * excitation_gain;
            float resonator_sum = 0.0f;

            for (int r = 0; r < v->num_resonators; r++) {
                if (v->resonators[r].fc > 0.0f) {
                    resonator_sum += svf_process_bp(&v->resonators[r], input_sample);
                }
            }

            resonator_sum *= res_gain;
            float voiced = resonator_sum * v->amplitude;
            v->amplitude *= v->decay_rate;

            wet_l_buf[i] += voiced * pan_l;
            wet_r_buf[i] += voiced * pan_r;
        }

        if (v->amplitude < 0.0001f) {
            v->active = 0;
            v->amplitude = 0.0f;
        }
    }

    /* ── RMS compressor (Dissolver-style, computed per-block for CPU) ────── */
    float comp_gain_l = 1.0f, comp_gain_r = 1.0f;
    if (inst->compress > 0.01f) {
        /* Compute block RMS of wet signal */
        float sum_l = 0.0f, sum_r = 0.0f;
        for (int i = 0; i < frames; i++) {
            sum_l += wet_l_buf[i] * wet_l_buf[i];
            sum_r += wet_r_buf[i] * wet_r_buf[i];
        }
        float rms_l = sqrtf(sum_l / (float)frames);
        float rms_r = sqrtf(sum_r / (float)frames);
        /* Smooth RMS envelope across blocks (~10ms) */
        inst->comp_env_l = inst->comp_env_l * 0.7f + rms_l * 0.3f;
        inst->comp_env_r = inst->comp_env_r * 0.7f + rms_r * 0.3f;
        /* gain = RMS^(-power), capped */
        if (inst->comp_env_l > 1e-6f) {
            comp_gain_l = powf(inst->comp_env_l, -inst->compress);
            if (comp_gain_l > 20.0f) comp_gain_l = 20.0f;
        }
        if (inst->comp_env_r > 1e-6f) {
            comp_gain_r = powf(inst->comp_env_r, -inst->compress);
            if (comp_gain_r > 20.0f) comp_gain_r = 20.0f;
        }
    }

    /* ── Output chain: HPF → LPF → Compress → Limiter → Post-gain → Mix ── */
    for (int i = 0; i < frames; i++) {
        float dry_l = (float)audio_buf[i * 2] / 32768.0f;
        float dry_r = (float)audio_buf[i * 2 + 1] / 32768.0f;

        float wl = wet_l_buf[i];
        float wr = wet_r_buf[i];

        if (inst->hpf > 20.5f) {
            wl = onepole_hp(&inst->hp_l, wl);
            wr = onepole_hp(&inst->hp_r, wr);
        }
        if (inst->lpf < 19500.0f) {
            wl = onepole_lp(&inst->lp_l, wl);
            wr = onepole_lp(&inst->lp_r, wr);
        }

        /* Apply compressor gain */
        wl *= comp_gain_l;
        wr *= comp_gain_r;

        if (inst->limiter) {
            wl = tape_soft_clip(wl);
            wr = tape_soft_clip(wr);
        }

        wl *= post_gain_lin;
        wr *= post_gain_lin;

        float out_l = dry_gain * dry_l + wet_gain * wl;
        float out_r = dry_gain * dry_r + wet_gain * wr;

        int32_t il = (int32_t)(out_l * 32767.0f);
        int32_t ir = (int32_t)(out_r * 32767.0f);
        if (il > 32767) il = 32767;
        if (il < -32768) il = -32768;
        if (ir > 32767) ir = 32767;
        if (ir < -32768) ir = -32768;
        audio_buf[i * 2]     = (int16_t)il;
        audio_buf[i * 2 + 1] = (int16_t)ir;
    }
}

/* ── API v2 export ───────────────────────────────────────────────────────────── */

static audio_fx_api_v2_t g_api = {
    .api_version      = 2,
    .create_instance  = create_instance,
    .destroy_instance = destroy_instance,
    .process_block    = process_block,
    .set_param        = set_param,
    .get_param        = get_param,
    .on_midi          = NULL,
};

__attribute__((visibility("default")))
audio_fx_api_v2_t* move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;
    if (host && host->log) host->log("[spectra] loaded");
    return &g_api;
}
