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
#define YIN_BUFFER_SIZE 2048
#define MAX_VOICES      4
#define MAX_RESONATORS  12

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
    return f->bp;
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

/* ── YIN pitch detector state ────────────────────────────────────────────────── */

typedef struct {
    float buffer[YIN_BUFFER_SIZE];
    int write_pos;
    float diff[YIN_BUFFER_SIZE / 2];
    float cmnd[YIN_BUFFER_SIZE / 2]; /* cumulative mean normalized difference */
} yin_state_t;

/* ── Instance state ──────────────────────────────────────────────────────────── */

typedef struct {
    /* Page 1 — Main (knob-mapped) */
    float onset;           /* spectral flux threshold 0-1 */
    float frequency;       /* YIN confidence threshold 0-1 */
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
    float hpf;             /* 20-2000 Hz */
    float lpf;             /* 500-20000 Hz */
    int   limiter;         /* 0=Off, 1=On */

    /* Analysis state */
    yin_state_t yin;
    int yin_blocks_since_last;  /* run YIN every N blocks to save CPU */
    float fft_re[FFT_SIZE];
    float fft_im[FFT_SIZE];
    float prev_magnitude[FFT_SIZE / 2]; /* previous frame magnitudes for spectral flux */
    float fft_window[FFT_SIZE];         /* Hann window */
    float mono_buffer[FFT_SIZE];        /* accumulate mono samples for FFT */
    int   mono_write_pos;
    int   fft_ready;                    /* flag: enough samples accumulated */

    /* Detected analysis values */
    float detected_freq;    /* Hz, from YIN */
    float detected_confidence; /* 0-1 from YIN */
    float spectral_flatness;   /* 0-1 */
    int   onset_detected;      /* flag per block */

    /* Voice allocator */
    voice_t voices[MAX_VOICES];
    int next_voice;         /* round-robin index */

    /* Output filters */
    onepole_t hp_l, hp_r;
    onepole_t lp_l, lp_r;

    /* Temp buffer for YIN reordering (avoid 8KB stack alloc) */
    float yin_temp[YIN_BUFFER_SIZE];

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

static int *get_int_param_ptr(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 5: return &inst->root_note;
        case 6: return &inst->scale;
        default: return NULL;
    }
}

/* ── YIN pitch detection ─────────────────────────────────────────────────────
 * Implements the YIN algorithm (de Cheveigné & Kawahara 2002).
 * Operates on 2048-sample buffer, returns frequency in Hz and confidence.
 */

static void yin_detect(yin_state_t *yin, float threshold, float *out_freq, float *out_confidence) {
    int half = YIN_BUFFER_SIZE / 2;

    /* Step 2: Difference function */
    for (int tau = 0; tau < half; tau++) {
        yin->diff[tau] = 0.0f;
        for (int j = 0; j < half; j++) {
            float delta = yin->buffer[j] - yin->buffer[j + tau];
            yin->diff[tau] += delta * delta;
        }
    }

    /* Step 3: Cumulative mean normalized difference function */
    yin->cmnd[0] = 1.0f;
    float running_sum = 0.0f;
    for (int tau = 1; tau < half; tau++) {
        running_sum += yin->diff[tau];
        yin->cmnd[tau] = (running_sum > 0.0f) ? yin->diff[tau] * tau / running_sum : 1.0f;
    }

    /* Step 4: Absolute threshold — find first dip below threshold */
    int tau_estimate = -1;
    /* Skip tau 0 and 1 (too short for valid pitch) */
    for (int tau = 2; tau < half; tau++) {
        if (yin->cmnd[tau] < threshold) {
            /* Find local minimum */
            while (tau + 1 < half && yin->cmnd[tau + 1] < yin->cmnd[tau]) {
                tau++;
            }
            tau_estimate = tau;
            break;
        }
    }

    if (tau_estimate < 0) {
        *out_freq = 0.0f;
        *out_confidence = 0.0f;
        return;
    }

    /* Step 5: Parabolic interpolation for sub-sample accuracy */
    float better_tau = (float)tau_estimate;
    if (tau_estimate > 0 && tau_estimate < half - 1) {
        float s0 = yin->cmnd[tau_estimate - 1];
        float s1 = yin->cmnd[tau_estimate];
        float s2 = yin->cmnd[tau_estimate + 1];
        float denom = 2.0f * (2.0f * s1 - s2 - s0);
        if (fabsf(denom) > 1e-10f) {
            better_tau = (float)tau_estimate + (s0 - s2) / denom;
        }
    }

    *out_freq = (better_tau > 0.0f) ? (float)SAMPLE_RATE / better_tau : 0.0f;
    *out_confidence = 1.0f - yin->cmnd[tau_estimate];
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
 * Given a base note, scale, and resonator count, compute frequencies for the bank.
 * Timbre shifts partial ratios from harmonic (0) to inharmonic (1).
 */

static void tune_resonator_bank(voice_t *v, float base_midi, int root, int scale_idx,
                                 int res_count, int octave_range, float timbre, float q_value) {
    v->num_resonators = res_count;
    if (res_count > MAX_RESONATORS) res_count = MAX_RESONATORS;

    int size = SCALE_SIZES[scale_idx];
    const int *intervals = SCALE_INTERVALS[scale_idx];

    /* Build the resonator frequencies across the octave range */
    int idx = 0;
    int base_octave = (int)base_midi / 12;

    for (int oct = 0; oct < octave_range && idx < res_count; oct++) {
        for (int i = 0; i < size && idx < res_count; i++) {
            float midi = (float)((base_octave + oct) * 12 + (root + intervals[i]) % 12);
            /* Only include notes at or above the base */
            if (midi < base_midi - 0.5f) midi += 12.0f;

            /* Apply inharmonicity (timbre): stretch ratio away from integer harmonics */
            if (timbre > 0.0f && idx > 0) {
                float ratio = midi_to_freq(midi) / midi_to_freq(base_midi);
                /* Stretch the ratio: higher partials get stretched more */
                float stretch = 1.0f + timbre * 0.02f * ratio * ratio;
                float stretched_freq = midi_to_freq(base_midi) * ratio * stretch;
                v->resonators[idx].freq = clampf(stretched_freq, 20.0f, 20000.0f);
            } else {
                v->resonators[idx].freq = clampf(midi_to_freq(midi), 20.0f, 20000.0f);
            }
            v->resonators[idx].q = q_value;
            svf_update_coeff(&v->resonators[idx]);
            idx++;
        }
    }

    /* Fill remaining with silence */
    for (int i = idx; i < MAX_RESONATORS; i++) {
        v->resonators[i].freq = 0.0f;
        v->resonators[i].fc = 0.0f;
    }
    v->num_resonators = idx;
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
    inst->onset = 0.3f;
    inst->frequency = 0.5f;
    inst->brightness = 0.5f;
    inst->timbre = 0.0f;
    inst->decay = 0.5f;
    inst->root_note = 0;    /* C */
    inst->scale = 0;        /* Major */
    inst->mix = 0.5f;

    /* Page 2 defaults */
    inst->chord_drift = 0.0f;
    inst->resonators_idx = 1;   /* 7 */
    inst->polyphony_idx = 1;    /* 2 */
    inst->octave_range_idx = 3; /* 4 */
    inst->pre_gain = 0.0f;
    inst->post_gain = 0.0f;
    inst->hpf = 20.0f;
    inst->lpf = 20000.0f;
    inst->limiter = 1;     /* On */

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

    /* ── knob_N_adjust ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_adjust")) {
        int knob_num = atoi(key + 5);
        int idx = knob_num - 1;
        if (idx >= 0 && idx < 8 && KNOB_MAP[idx].key) {
            float delta = atof(val);
            const knob_def_t *k = &KNOB_MAP[idx];
            if (k->is_enum) {
                int *p = get_int_param_ptr(inst, idx);
                if (p) {
                    int new_val = *p + (int)delta;
                    if (new_val > (int)k->max) new_val = (int)k->min;
                    if (new_val < (int)k->min) new_val = (int)k->max;
                    *p = new_val;
                }
            } else {
                float *p = get_float_param_ptr(inst, idx);
                if (p) *p = clampf(*p + delta * k->step, k->min, k->max);
            }
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
    else if (strcmp(key, "hpf") == 0)        inst->hpf = clampf(atof(val), 20, 2000);
    else if (strcmp(key, "lpf") == 0)        inst->lpf = clampf(atof(val), 500, 20000);
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
            "pre_gain=%f;post_gain=%f;hpf=%f;lpf=%f;limiter=%d",
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
            "{\"key\":\"hpf\",\"name\":\"HPF\",\"type\":\"float\",\"min\":20,\"max\":2000,\"step\":1},"
            "{\"key\":\"lpf\",\"name\":\"LPF\",\"type\":\"float\",\"min\":500,\"max\":20000,\"step\":1},"
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
    if (strcmp(key, "hpf") == 0)         return snprintf(buf, buf_len, "%.0f", inst->hpf);
    if (strcmp(key, "lpf") == 0)         return snprintf(buf, buf_len, "%.0f", inst->lpf);
    if (strcmp(key, "limiter") == 0)
        return snprintf(buf, buf_len, "%s", LIMITER_NAMES[clampi(inst->limiter, 0, 1)]);

    /* ── knob_N_name ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_name")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8 && KNOB_MAP[idx].label)
            return snprintf(buf, buf_len, "%s", KNOB_MAP[idx].label);
        return 0;
    }

    /* ── knob_N_value ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_value")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8 && KNOB_MAP[idx].key) {
            if (KNOB_MAP[idx].is_enum) {
                return get_param(instance, KNOB_MAP[idx].key, buf, buf_len);
            } else {
                float *p = get_float_param_ptr(inst, idx);
                if (p) return snprintf(buf, buf_len, "%d%%", (int)(*p * 100));
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
            "pre_gain=%.6f;post_gain=%.6f;hpf=%.6f;lpf=%.6f;limiter=%d",
            inst->onset, inst->frequency, inst->brightness, inst->timbre, inst->decay,
            inst->root_note, inst->scale, inst->mix,
            inst->chord_drift, inst->resonators_idx, inst->polyphony_idx, inst->octave_range_idx,
            inst->pre_gain, inst->post_gain, inst->hpf, inst->lpf, inst->limiter);
    }

    return -1; /* unknown key */
}

/* ── Audio processing ────────────────────────────────────────────────────────── */

static void process_block(void *instance, int16_t *audio_buf, int frames) {
    plugin_instance_t *inst = (plugin_instance_t *)instance;
    if (!inst || frames <= 0) return;
    if (frames > 128) frames = 128;

    /* Derived params */
    float pre_gain_lin = powf(10.0f, inst->pre_gain / 20.0f);
    float post_gain_lin = powf(10.0f, inst->post_gain / 20.0f);
    int num_voices = POLYPHONY_OPTIONS[clampi(inst->polyphony_idx, 0, 2)];
    int res_count = RESONATOR_COUNTS[clampi(inst->resonators_idx, 0, 2)];
    int oct_range = clampi(inst->octave_range_idx + 1, 1, 4);

    float decay_seconds = 0.05f * powf(200.0f, inst->decay);
    float decay_rate = expf(-1.0f / (decay_seconds * (float)SAMPLE_RATE));
    float onset_threshold = inst->onset * 50.0f + 0.5f;
    float yin_threshold = 0.02f + (1.0f - inst->frequency) * 0.48f;

    /* Precompute once per block — NOT per sample */
    float mix_angle = inst->mix * 1.5707963f;
    float dry_gain = cosf(mix_angle);
    float wet_gain = sinf(mix_angle);
    onepole_set_freq(&inst->hp_l, inst->hpf);
    onepole_set_freq(&inst->hp_r, inst->hpf);
    onepole_set_freq(&inst->lp_l, inst->lpf);
    onepole_set_freq(&inst->lp_r, inst->lpf);

    /* --- Convert input to mono float and feed analysis buffers --- */
    float mono[BLOCK_SIZE];
    for (int i = 0; i < frames; i++) {
        float l = (float)audio_buf[i * 2] / 32768.0f * pre_gain_lin;
        float r = (float)audio_buf[i * 2 + 1] / 32768.0f * pre_gain_lin;
        mono[i] = (l + r) * 0.5f;
    }

    /* Feed YIN buffer */
    for (int i = 0; i < frames; i++) {
        inst->yin.buffer[inst->yin.write_pos] = mono[i];
        inst->yin.write_pos = (inst->yin.write_pos + 1) % YIN_BUFFER_SIZE;
    }

    /* Feed FFT buffer */
    for (int i = 0; i < frames; i++) {
        inst->mono_buffer[inst->mono_write_pos] = mono[i];
        inst->mono_write_pos++;
        if (inst->mono_write_pos >= FFT_SIZE) {
            inst->mono_write_pos = 0;
            inst->fft_ready = 1;
        }
    }

    /* --- Run analysis when FFT buffer is full --- */
    inst->onset_detected = 0;

    if (inst->fft_ready) {
        inst->fft_ready = 0;

        /* Window and load FFT buffers */
        for (int i = 0; i < FFT_SIZE; i++) {
            inst->fft_re[i] = inst->mono_buffer[i] * inst->fft_window[i];
            inst->fft_im[i] = 0.0f;
        }

        fft_forward(inst->fft_re, inst->fft_im, FFT_SIZE);

        /* Compute magnitude spectrum */
        float magnitude[FFT_SIZE / 2];
        for (int i = 0; i < FFT_SIZE / 2; i++) {
            magnitude[i] = sqrtf(inst->fft_re[i] * inst->fft_re[i] +
                                  inst->fft_im[i] * inst->fft_im[i]);
        }

        /* --- Spectral flux onset detection --- */
        float flux = 0.0f;
        for (int i = 0; i < FFT_SIZE / 2; i++) {
            float diff = magnitude[i] - inst->prev_magnitude[i];
            if (diff > 0.0f) flux += diff;
        }
        if (flux > onset_threshold) {
            inst->onset_detected = 1;
        }

        /* Save magnitudes for next frame */
        memcpy(inst->prev_magnitude, magnitude, sizeof(float) * FFT_SIZE / 2);

        /* --- Spectral flatness --- */
        float log_sum = 0.0f;
        float arith_sum = 0.0f;
        int count = 0;
        for (int i = 1; i < FFT_SIZE / 2; i++) { /* skip DC */
            if (magnitude[i] > 1e-10f) {
                log_sum += logf(magnitude[i]);
                arith_sum += magnitude[i];
                count++;
            }
        }
        if (count > 0 && arith_sum > 1e-10f) {
            float geo_mean = expf(log_sum / (float)count);
            float arith_mean = arith_sum / (float)count;
            inst->spectral_flatness = clampf(geo_mean / arith_mean, 0.0f, 1.0f);
        } else {
            inst->spectral_flatness = 0.0f;
        }
    }

    /* --- YIN pitch detection (throttled — expensive ~1M ops) --- */
    inst->yin_blocks_since_last++;
    if (inst->yin_blocks_since_last >= 8) {
        inst->yin_blocks_since_last = 0;
        /* Reorder circular buffer to contiguous for analysis */
        int wp = inst->yin.write_pos;
        for (int i = 0; i < YIN_BUFFER_SIZE; i++) {
            inst->yin_temp[i] = inst->yin.buffer[(wp + i) % YIN_BUFFER_SIZE];
        }
        memcpy(inst->yin.buffer, inst->yin_temp, sizeof(inst->yin_temp));
        inst->yin.write_pos = 0;
        yin_detect(&inst->yin, yin_threshold, &inst->detected_freq, &inst->detected_confidence);
    }

    /* --- Compute final brightness from flatness + manual offset --- */
    /* Inverse flatness: tonal (flatness=0) → bright, noisy (flatness=1) → dull */
    float auto_brightness = 1.0f - inst->spectral_flatness;
    float brightness_final = clampf(auto_brightness + (inst->brightness - 0.5f), 0.0f, 1.0f);
    /* Map to Q factor: bright (1.0) → high Q (~50), dull (0.0) → low Q (~2) */
    float q_value = 2.0f + brightness_final * 48.0f;

    /* --- Voice allocation on onset --- */
    if (inst->onset_detected && inst->detected_freq > 20.0f && inst->detected_confidence > 0.1f) {
        float midi_note = freq_to_midi(inst->detected_freq);
        float quantized = quantize_to_scale(midi_note, inst->root_note, inst->scale);
        quantized = apply_chord_drift(quantized, inst->chord_drift, &inst->rng_state);

        /* Allocate voice (round-robin with steal) */
        int vi = inst->next_voice % num_voices;
        inst->next_voice = (inst->next_voice + 1) % num_voices;

        voice_t *v = &inst->voices[vi];
        v->active = 1;
        v->midi_note = quantized;
        v->amplitude = 1.0f;
        v->decay_rate = decay_rate;

        /* Tune resonator bank (precomputes SVF coefficients) */
        tune_resonator_bank(v, quantized, inst->root_note, inst->scale,
                            res_count, oct_range, inst->timbre, q_value);

        /* Clear filter state for new note to avoid clicks */
        for (int r = 0; r < v->num_resonators; r++) {
            v->resonators[r].lp = 0.0f;
            v->resonators[r].bp = 0.0f;
        }
    }

    /* --- Process audio through resonator bank --- */
    float wet_l_buf[BLOCK_SIZE];
    float wet_r_buf[BLOCK_SIZE];
    memset(wet_l_buf, 0, sizeof(wet_l_buf));
    memset(wet_r_buf, 0, sizeof(wet_r_buf));

    for (int vi = 0; vi < num_voices; vi++) {
        voice_t *v = &inst->voices[vi];
        if (!v->active) continue;

        /* Precompute per-voice constants outside sample loop */
        float pan = 0.5f + (float)(vi - num_voices / 2) * 0.15f;
        pan = clampf(pan, 0.0f, 1.0f);
        float pan_l = 1.0f - pan;
        float pan_r = pan;
        float inv_res = (v->num_resonators > 0) ? 1.0f / (float)v->num_resonators : 0.0f;

        for (int i = 0; i < frames; i++) {
            float input_sample = mono[i];
            float resonator_sum = 0.0f;

            for (int r = 0; r < v->num_resonators; r++) {
                if (v->resonators[r].fc > 0.0f) {
                    resonator_sum += svf_process_bp(&v->resonators[r], input_sample);
                }
            }

            resonator_sum *= inv_res;
            float voiced = resonator_sum * v->amplitude;
            v->amplitude *= v->decay_rate;

            wet_l_buf[i] += voiced * pan_l;
            wet_r_buf[i] += voiced * pan_r;
        }

        /* Deactivate if envelope has decayed — flush denormals */
        if (v->amplitude < 0.0001f) {
            v->active = 0;
            v->amplitude = 0.0f;
        }
    }

    /* --- Output chain: HPF → LPF → Limiter → Post-gain → Mix --- */
    for (int i = 0; i < frames; i++) {
        float dry_l = (float)audio_buf[i * 2] / 32768.0f;
        float dry_r = (float)audio_buf[i * 2 + 1] / 32768.0f;

        float wl = wet_l_buf[i];
        float wr = wet_r_buf[i];

        /* HPF */
        if (inst->hpf > 20.5f) {
            wl = onepole_hp(&inst->hp_l, wl);
            wr = onepole_hp(&inst->hp_r, wr);
        }

        /* LPF */
        if (inst->lpf < 19500.0f) {
            wl = onepole_lp(&inst->lp_l, wl);
            wr = onepole_lp(&inst->lp_r, wr);
        }

        /* Limiter */
        if (inst->limiter) {
            wl = tape_soft_clip(wl);
            wr = tape_soft_clip(wr);
        }

        /* Post-gain */
        wl *= post_gain_lin;
        wr *= post_gain_lin;

        /* Equal-power dry/wet crossfade */
        float out_l = dry_gain * dry_l + wet_gain * wl;
        float out_r = dry_gain * dry_r + wet_gain * wr;

        /* Clamp and write back */
        int32_t il = (int32_t)(out_l * 32767.0f);
        int32_t ir = (int32_t)(out_r * 32767.0f);
        if (il >  32767) il =  32767; if (il < -32768) il = -32768;
        if (ir >  32767) ir =  32767; if (ir < -32768) ir = -32768;
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
