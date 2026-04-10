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

/* Arp pattern names */
#define NUM_ARP_PATTERNS 5
static const char *ARP_PATTERN_NAMES[NUM_ARP_PATTERNS] = { "Off", "Up", "Down", "Random", "Chord" };

/* Arp sync mode names */
#define NUM_ARP_SYNC 2
static const char *ARP_SYNC_NAMES[NUM_ARP_SYNC] = { "Free", "Sync" };

/* Motion LFO shape names */
#define NUM_MOTION_SHAPES 4
static const char *MOTION_SHAPE_NAMES[NUM_MOTION_SHAPES] = { "Sine", "Triangle", "Square", "S&H" };

/* ── Preset structure ─────────────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    float onset, frequency, brightness, timbre, decay;
    int root_note, scale;
    float mix;
    float chord_drift;
    int resonators_idx, polyphony_idx, octave_range_idx;
    float pre_gain, post_gain;
    int hpf, lpf, limiter, compress;
    float arp_rate;
    int arp_pattern, arp_sync;
    float drift;
    float motion_rate, motion_depth;
    int motion_shape;
    float scoop;
} spectra_preset_t;

#define NUM_PRESETS 30

/* 30 curated presets */
static const spectra_preset_t PRESETS[NUM_PRESETS] = {
    /* 0: Bell resonator */
    { "Bell", 0.9f, 0.7f, 0.8f, 0.0f, 0.3f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 1: Pad resonator */
    { "Pad", 0.5f, 0.5f, 0.6f, 1.0f, 0.8f, 0, 8, 0.8f, 0.0f, 1, 2, 3, -3.0f, 0.0f, 40, 10000, 1, 0.3f, 0.3f, 0, 0.0f, 0.2f, 0.5f, 0, 0.0f },
    /* 2: Arp up */
    { "Arp Up", 0.7f, 0.4f, 0.5f, 0.0f, 0.2f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.6f, 1, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 3: Arp random */
    { "Arp Rnd", 0.8f, 0.3f, 0.6f, 0.0f, 0.15f, 0, 11, 1.0f, 0.0f, 2, 2, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.7f, 3, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 4: Drifter */
    { "Drifter", 0.6f, 0.5f, 0.4f, 0.5f, 0.6f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.2f, 0.5f, 0, 0.0f, 0.6f, 0.0f, 0, 0.0f },
    /* 5: Shimmer */
    { "Shimmer", 0.3f, 0.8f, 0.9f, 0.8f, 0.9f, 9, 4, 0.9f, 0.0f, 1, 2, 3, 3.0f, 0.0f, 100, 18000, 1, 0.05f, 0.3f, 0, 0.0f, 0.0f, 0.7f, 1, 0.3f },
    /* 6: Drone bass */
    { "Drone Bass", 0.4f, 0.2f, 0.3f, 0.0f, 0.95f, 0, 0, 0.7f, 0.0f, 2, 1, 1, -6.0f, 0.0f, 20, 800, 1, 0.2f, 0.4f, 4, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 7: Tremolo */
    { "Tremolo", 0.7f, 0.5f, 0.5f, 0.0f, 0.4f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.5f, 0, 0.0f, 0.7f, 0.8f, 0, 0.0f },
    /* 8: Bells chord */
    { "Bells", 0.8f, 0.6f, 0.7f, 0.0f, 0.5f, 5, 0, 1.0f, 0.0f, 1, 2, 2, 2.0f, 0.0f, 20, 20000, 1, 0.15f, 0.5f, 4, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 9: Scooped */
    { "Scooped", 0.7f, 0.5f, 0.4f, 0.0f, 0.3f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.8f },
    /* 10: Evolving */
    { "Evolving", 0.5f, 0.5f, 0.5f, 0.5f, 0.7f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.2f, 0.5f, 0, 0.0f, 0.3f, 0.4f, 1, 0.2f },
    /* 11: Stutter */
    { "Stutter", 0.9f, 0.4f, 0.6f, 0.0f, 0.1f, 0, 11, 1.0f, 0.0f, 2, 3, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.8f, 2, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 12: Resonant */
    { "Resonant", 0.6f, 0.6f, 0.9f, 0.0f, 0.4f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 13: Wavy */
    { "Wavy", 0.6f, 0.5f, 0.5f, 0.0f, 0.5f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.2f, 0.4f, 0, 0.3f, 0.5f, 0.4f, 1, 0.0f },
    /* 14: Pulsing */
    { "Pulsing", 0.7f, 0.5f, 0.5f, 0.0f, 0.3f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.6f, 0, 0.0f, 0.5f, 1.0f, 0, 0.0f },
    /* 15: Minimal */
    { "Minimal", 0.8f, 0.3f, 0.3f, 0.0f, 0.5f, 0, 0, 0.5f, 0.0f, 0, 1, 0, 0.0f, 0.0f, 80, 5000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 16: Lush */
    { "Lush", 0.5f, 0.6f, 0.7f, 1.0f, 0.8f, 0, 7, 1.0f, 0.0f, 2, 2, 3, 0.0f, 2.0f, 20, 20000, 1, 0.3f, 0.3f, 0, 0.0f, 0.2f, 0.3f, 0, 0.1f },
    /* 17: Crisp */
    { "Crisp", 0.85f, 0.7f, 0.8f, 0.0f, 0.2f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 3.0f, 0.0f, 200, 15000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 18: Glitch */
    { "Glitch", 0.9f, 0.4f, 0.5f, 0.0f, 0.05f, 0, 11, 1.0f, 0.2f, 2, 3, 1, 0.0f, -3.0f, 20, 20000, 1, 0.1f, 0.7f, 3, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 19: Warm */
    { "Warm", 0.6f, 0.4f, 0.4f, 0.8f, 0.6f, 0, 0, 1.0f, 0.0f, 2, 2, 2, -3.0f, 0.0f, 20, 10000, 1, 0.25f, 0.4f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 20: Bright */
    { "Bright", 0.7f, 0.7f, 1.0f, 0.0f, 0.3f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 3.0f, 0.0f, 100, 20000, 1, 0.1f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 21: Deep */
    { "Deep", 0.5f, 0.2f, 0.3f, 0.5f, 0.85f, 0, 9, 0.8f, 0.0f, 1, 1, 2, -6.0f, 0.0f, 20, 2000, 1, 0.3f, 0.4f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 22: Airy */
    { "Airy", 0.3f, 0.8f, 0.8f, 0.9f, 0.8f, 9, 0, 1.0f, 0.0f, 2, 2, 3, 6.0f, 0.0f, 500, 20000, 1, 0.05f, 0.3f, 0, 0.0f, 0.1f, 0.2f, 1, 0.0f },
    /* 23: Metallic */
    { "Metallic", 0.85f, 0.6f, 0.8f, 0.0f, 0.25f, 0, 0, 1.0f, 0.0f, 2, 2, 1, 2.0f, 0.0f, 40, 20000, 1, 0.1f, 0.6f, 0, 0.0f, 0.0f, 0.0f, 0, 0.2f },
    /* 24: Modulating */
    { "Modulating", 0.6f, 0.5f, 0.5f, 0.0f, 0.5f, 0, 0, 1.0f, 0.0f, 2, 2, 2, 0.0f, 0.0f, 20, 20000, 1, 0.2f, 0.5f, 0, 0.0f, 0.4f, 0.6f, 0, 0.0f },
    /* 25: Sparse */
    { "Sparse", 0.95f, 0.3f, 0.4f, 0.0f, 0.5f, 0, 0, 0.4f, 0.0f, 0, 1, 1, 0.0f, 0.0f, 40, 15000, 1, 0.15f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 26: Liquid */
    { "Liquid", 0.5f, 0.5f, 0.6f, 0.7f, 0.75f, 0, 7, 0.9f, 0.0f, 1, 2, 2, -2.0f, 1.0f, 20, 12000, 1, 0.25f, 0.4f, 0, 0.0f, 0.15f, 0.4f, 0, 0.05f },
    /* 27: Rhythmic */
    { "Rhythmic", 0.8f, 0.5f, 0.5f, 0.0f, 0.15f, 0, 11, 1.0f, 0.0f, 2, 3, 1, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.7f, 1, 0.0f, 0.0f, 0.0f, 0, 0.0f },
    /* 28: Spacious */
    { "Spacious", 0.4f, 0.5f, 0.5f, 1.0f, 0.9f, 0, 0, 1.0f, 0.0f, 2, 2, 3, 0.0f, 3.0f, 20, 20000, 1, 0.2f, 0.3f, 0, 0.0f, 0.2f, 0.3f, 1, 0.15f },
    /* 29: Chaos */
    { "Chaos", 0.9f, 0.5f, 0.6f, 0.0f, 0.2f, 0, 11, 1.0f, 0.5f, 2, 3, 2, 0.0f, 0.0f, 20, 20000, 1, 0.1f, 0.8f, 3, 0.8f, 0.0f, 0.0f, 0, 0.3f },
};

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

/* Page 3 knob mapping (Motion page)
 * 1=Arp Rate, 2=Arp Pattern, 3=Arp Sync, 4=Drift,
 * 5=Motion Rate, 6=Motion Depth, 7=Motion Shape, 8=Scoop */
static const knob_def_t KNOB_MAP_P3[8] = {
    { "arp_rate",     "Arp Rate",   0, 1, 0.01f, 0 },
    { "arp_pattern",  "Arp Ptrn",   0, 4, 1.0f,  1 },
    { "arp_sync",     "Arp Sync",   0, 1, 1.0f,  1 },
    { "drift",        "Drift",      0, 1, 0.01f, 0 },
    { "motion_rate",  "Mot Rate",   0, 1, 0.01f, 0 },
    { "motion_depth", "Mot Depth",  0, 1, 0.01f, 0 },
    { "motion_shape", "Mot Shape",  0, 3, 1.0f,  1 },
    { "scoop",        "Scoop",      0, 1, 0.01f, 0 },
};

/* Page 4 knob mapping (Patch page)
 * 1=Preset, 2=Rnd Preset, 3=Rnd Spectra, 4=Rnd Motion,
 * 5=Rnd Pan, 6=Stereo Width */
static const knob_def_t KNOB_MAP_P4[8] = {
    { "preset",       "Preset",     0, 29, 1.0f, 1 },
    { "rnd_preset",   "Rnd Prs",    0, 1, 1.0f, 0 },
    { "rnd_spectra",  "Rnd Spc",    0, 1, 1.0f, 0 },
    { "rnd_motion",   "Rnd Mot",    0, 1, 1.0f, 0 },
    { "rnd_pan",      "Rnd Pan",    0, 1, 1.0f, 0 },
    { "stereo_width", "Stereo",     0, 1, 0.01f, 0 },
    { "", "", 0, 1, 0.01f, 0 },
    { "", "", 0, 1, 0.01f, 0 },
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
    float excitation_gain_smooth; /* smoothed excitation multiplier (prevents volume jumps) */

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

    /* ── Motion page state ── */
    float arp_rate;           /* 0–1 → mapped to Hz or clock div */
    int   arp_pattern;        /* 0=Off, 1=Up, 2=Down, 3=Random, 4=Chord */
    int   arp_sync;           /* 0=Free, 1=Sync */
    float arp_phase;          /* 0–1, wraps */
    int   arp_current_step;   /* which resonator is currently active */
    uint32_t arp_rng;         /* xorshift state for Random mode */
    float arp_gate[MAX_RESONATORS]; /* per-resonator gate mask */

    float drift;              /* 0–1 amount */
    float drift_noise[MAX_RESONATORS]; /* per-resonator smoothed noise */
    uint32_t drift_rng;       /* independent RNG */
    float drift_ratio[MAX_RESONATORS]; /* per-resonator pitch ratio */

    float motion_rate;        /* 0–1 → Hz */
    float motion_depth;       /* 0–1 */
    int   motion_shape;       /* 0=Sine, 1=Tri, 2=Square, 3=S&H */
    float motion_phase;       /* 0–1, wraps */
    float motion_sh_value;    /* held value for S&H */
    uint32_t motion_sh_rng;   /* RNG for S&H */

    float scoop;              /* 0–1 → notch depth */
    float scoop_z1[MAX_RESONATORS]; /* per-resonator scoop notch state */
    float scoop_z2[MAX_RESONATORS];

    /* ── Patch page state ── */
    int current_preset;       /* 0-29 */
    float stereo_width;       /* 0-1 (0=mono, 1=full stereo) */
    float pan_spread[MAX_VOICES]; /* per-voice pan override (0-1) */

    /* Rnd button display counters (show 1 briefly, then decay to 0) */
    int rnd_preset_counter;    /* countdown: display 1 while > 0, else 0 */
    int rnd_spectra_counter;
    int rnd_motion_counter;
    int rnd_pan_counter;

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

/* Page 3 (Motion) float param pointers */
static float *get_float_param_ptr_p3(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 0: return &inst->arp_rate;
        case 3: return &inst->drift;
        case 4: return &inst->motion_rate;
        case 5: return &inst->motion_depth;
        case 7: return &inst->scoop;
        default: return NULL;
    }
}

/* Page 3 (Motion) int param pointers */
static int *get_int_param_ptr_p3(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 1: return &inst->arp_pattern;
        case 2: return &inst->arp_sync;
        case 6: return &inst->motion_shape;
        default: return NULL;
    }
}

/* Page 4 (Patch) float param pointers */
static float *get_float_param_ptr_p4(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 5: return &inst->stereo_width;
        default: return NULL;
    }
}

/* Page 4 (Patch) int param pointers */
static int *get_int_param_ptr_p4(plugin_instance_t *inst, int idx) {
    switch (idx) {
        case 0: return &inst->current_preset;
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

/* ── Motion page engines ─────────────────────────────────────────────────────── */

/* Arp rate mapping: 0→0.1 Hz, 1→20 Hz exponential */
static inline float arp_rate_to_hz(float knob) {
    return 0.1f * powf(200.0f, knob);
}

/* Arp: compute per-resonator gate mask */
static void arp_tick(plugin_instance_t *inst, int num_resonators, int frames) {
    if (inst->arp_pattern == 0) {
        /* Off — all resonators active */
        for (int i = 0; i < num_resonators; i++)
            inst->arp_gate[i] = 1.0f;
        return;
    }

    float hz = arp_rate_to_hz(inst->arp_rate);
    float phase_inc = hz * (float)frames / (float)SAMPLE_RATE;
    inst->arp_phase += phase_inc;

    int stepped = 0;
    while (inst->arp_phase >= 1.0f) {
        inst->arp_phase -= 1.0f;
        stepped = 1;
    }

    if (stepped) {
        switch (inst->arp_pattern) {
        case 1: /* Up */
            inst->arp_current_step = (inst->arp_current_step + 1) % num_resonators;
            break;
        case 2: /* Down */
            inst->arp_current_step = (inst->arp_current_step - 1 + num_resonators) % num_resonators;
            break;
        case 3: /* Random */
            inst->arp_rng ^= inst->arp_rng << 13;
            inst->arp_rng ^= inst->arp_rng >> 17;
            inst->arp_rng ^= inst->arp_rng << 5;
            inst->arp_current_step = (int)(inst->arp_rng % (uint32_t)num_resonators);
            break;
        case 4: /* Chord — all pulse together */
            break;
        }
    }

    /* Set gate mask */
    for (int i = 0; i < num_resonators; i++) {
        if (inst->arp_pattern == 4) {
            /* Chord: all resonators follow a single envelope */
            float env = (inst->arp_phase < 0.5f) ? 1.0f : 0.0f;
            inst->arp_gate[i] = env;
        } else {
            /* Single-note arp: smooth gate with short fade */
            inst->arp_gate[i] = (i == inst->arp_current_step) ? 1.0f : 0.02f;
        }
    }
}

/* Drift: perturb resonator center frequencies */
static void drift_tick(plugin_instance_t *inst, int num_resonators) {
    if (inst->drift < 0.001f) {
        for (int i = 0; i < num_resonators; i++)
            inst->drift_ratio[i] = 1.0f;
        return;
    }

    float max_cents = 50.0f * inst->drift; /* 0–50 cents */
    float smooth = 0.995f; /* ~4.5 Hz @ 44100/128 blocks/sec */

    for (int i = 0; i < num_resonators; i++) {
        /* White noise via xorshift */
        inst->drift_rng ^= inst->drift_rng << 13;
        inst->drift_rng ^= inst->drift_rng >> 17;
        inst->drift_rng ^= inst->drift_rng << 5;
        float noise = ((float)(inst->drift_rng & 0xFFFF) / 32768.0f) - 1.0f; /* -1..+1 */

        /* Smooth random walk */
        inst->drift_noise[i] = smooth * inst->drift_noise[i] + (1.0f - smooth) * noise;

        /* Apply as pitch ratio: cents → ratio */
        float cents = inst->drift_noise[i] * max_cents;
        inst->drift_ratio[i] = powf(2.0f, cents / 1200.0f);
    }
}

/* Motion LFO: per-sample, per-resonator amplitude modulation */
static float motion_lfo_sample(plugin_instance_t *inst) {
    float rate_hz = 0.1f * powf(200.0f, inst->motion_rate);
    float phase_inc = rate_hz / (float)SAMPLE_RATE;

    inst->motion_phase += phase_inc;
    int wrapped = 0;
    if (inst->motion_phase >= 1.0f) {
        inst->motion_phase -= 1.0f;
        wrapped = 1;
    }

    float p = inst->motion_phase;
    float val;
    switch (inst->motion_shape) {
    case 0: /* Sine */
        val = sinf(p * 2.0f * M_PI);
        break;
    case 1: /* Triangle */
        val = (p < 0.5f) ? (4.0f * p - 1.0f) : (3.0f - 4.0f * p);
        break;
    case 2: /* Square */
        val = (p < 0.5f) ? 1.0f : -1.0f;
        break;
    case 3: /* S&H (Sample & Hold) */
        if (wrapped) {
            inst->motion_sh_rng ^= inst->motion_sh_rng << 13;
            inst->motion_sh_rng ^= inst->motion_sh_rng >> 17;
            inst->motion_sh_rng ^= inst->motion_sh_rng << 5;
            inst->motion_sh_value = ((float)(inst->motion_sh_rng & 0xFFFF) / 32768.0f) - 1.0f;
        }
        val = inst->motion_sh_value;
        break;
    default:
        val = 0.0f;
    }

    /* Unipolar 0–1 for amplitude modulation */
    return 1.0f - inst->motion_depth * (0.5f + 0.5f * val);
}

/* Scoop notch filter: per-resonator, per-sample */
static inline float scoop_process(float in, float freq_hz, float scoop_amount,
                                   float *z1, float *z2) {
    if (scoop_amount < 0.001f) return in;

    float w0 = 2.0f * M_PI * freq_hz / (float)SAMPLE_RATE;
    float Q = 10.0f;
    float alpha = sinf(w0) / (2.0f * Q);

    /* Notch coefficients */
    float b0 = 1.0f;
    float b1 = -2.0f * cosf(w0);
    float b2 = 1.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cosf(w0);
    float a2 = 1.0f - alpha;

    /* Normalize */
    b0 /= a0; b1 /= a0; b2 /= a0;
    a1 /= a0; a2 /= a0;

    float out = b0 * in + *z1 + DENORMAL_GUARD;
    *z1 = b1 * in - a1 * out + *z2;
    *z2 = b2 * in - a2 * out;

    /* Crossfade between dry and notched by scoop amount */
    return in + scoop_amount * (out - in);
}

/* ── Randomization functions ────────────────────────────────────────────────── */

static void apply_preset(plugin_instance_t *inst, int idx) {
    if (idx < 0 || idx >= NUM_PRESETS) return;
    const spectra_preset_t *p = &PRESETS[idx];

    inst->onset = p->onset;
    inst->frequency = p->frequency;
    inst->brightness = p->brightness;
    inst->timbre = p->timbre;
    inst->decay = p->decay;
    inst->root_note = p->root_note;
    inst->scale = p->scale;
    inst->mix = p->mix;
    inst->chord_drift = p->chord_drift;
    inst->resonators_idx = p->resonators_idx;
    inst->polyphony_idx = p->polyphony_idx;
    inst->octave_range_idx = p->octave_range_idx;
    inst->pre_gain = p->pre_gain;
    inst->post_gain = p->post_gain;
    inst->hpf = p->hpf;
    inst->lpf = p->lpf;
    inst->limiter = p->limiter;
    inst->compress = p->compress;
    inst->arp_rate = p->arp_rate;
    inst->arp_pattern = p->arp_pattern;
    inst->arp_sync = p->arp_sync;
    inst->drift = p->drift;
    inst->motion_rate = p->motion_rate;
    inst->motion_depth = p->motion_depth;
    inst->motion_shape = p->motion_shape;
    inst->scoop = p->scoop;
    inst->current_preset = idx;
}

static void randomize_preset(plugin_instance_t *inst) {
    inst->onset = rand_float(&inst->rng_state);
    inst->frequency = rand_float(&inst->rng_state);
    inst->brightness = rand_float(&inst->rng_state);
    inst->timbre = rand_float(&inst->rng_state);
    inst->decay = 0.05f + rand_float(&inst->rng_state) * 0.95f;
    inst->root_note = (int)(rand_float(&inst->rng_state) * NUM_ROOT_NOTES);
    inst->scale = (int)(rand_float(&inst->rng_state) * NUM_SCALES);
    inst->mix = 0.5f + rand_float(&inst->rng_state) * 0.5f;
    inst->chord_drift = rand_float(&inst->rng_state) * 0.5f;
    inst->resonators_idx = (int)(rand_float(&inst->rng_state) * 3);
    inst->polyphony_idx = (int)(rand_float(&inst->rng_state) * 3);
    inst->octave_range_idx = (int)(rand_float(&inst->rng_state) * 4);
    inst->pre_gain = -6.0f + rand_float(&inst->rng_state) * 12.0f;
    inst->post_gain = -3.0f + rand_float(&inst->rng_state) * 6.0f;
    inst->hpf = 20 + (int)(rand_float(&inst->rng_state) * 200);
    inst->lpf = 5000 + (int)(rand_float(&inst->rng_state) * 15000);
    inst->limiter = (rand_float(&inst->rng_state) > 0.3f) ? 1 : 0;
    inst->compress = rand_float(&inst->rng_state) * 0.5f;
    inst->arp_rate = rand_float(&inst->rng_state);
    inst->arp_pattern = (int)(rand_float(&inst->rng_state) * 5);
    inst->arp_sync = (rand_float(&inst->rng_state) > 0.7f) ? 1 : 0;
    inst->drift = rand_float(&inst->rng_state) * 0.7f;
    inst->motion_rate = rand_float(&inst->rng_state);
    inst->motion_depth = rand_float(&inst->rng_state) * 0.8f;
    inst->motion_shape = (int)(rand_float(&inst->rng_state) * NUM_MOTION_SHAPES);
    inst->scoop = rand_float(&inst->rng_state) * 0.5f;
    inst->current_preset = -1; /* no preset */
}

static void randomize_spectra_page(plugin_instance_t *inst) {
    inst->onset = rand_float(&inst->rng_state);
    inst->frequency = rand_float(&inst->rng_state);
    inst->brightness = rand_float(&inst->rng_state);
    inst->timbre = rand_float(&inst->rng_state);
    inst->decay = 0.05f + rand_float(&inst->rng_state) * 0.95f;
    inst->root_note = (int)(rand_float(&inst->rng_state) * NUM_ROOT_NOTES);
    inst->scale = (int)(rand_float(&inst->rng_state) * NUM_SCALES);
    inst->mix = 0.5f + rand_float(&inst->rng_state) * 0.5f;
}

static void randomize_motion_page(plugin_instance_t *inst) {
    inst->arp_rate = rand_float(&inst->rng_state);
    inst->arp_pattern = (int)(rand_float(&inst->rng_state) * 5);
    inst->arp_sync = (rand_float(&inst->rng_state) > 0.7f) ? 1 : 0;
    inst->drift = rand_float(&inst->rng_state) * 0.7f;
    inst->motion_rate = rand_float(&inst->rng_state);
    inst->motion_depth = rand_float(&inst->rng_state) * 0.8f;
    inst->motion_shape = (int)(rand_float(&inst->rng_state) * NUM_MOTION_SHAPES);
    inst->scoop = rand_float(&inst->rng_state) * 0.5f;
}

static void randomize_panning(plugin_instance_t *inst, int num_voices) {
    for (int i = 0; i < num_voices; i++) {
        inst->pan_spread[i] = rand_float(&inst->rng_state);
    }
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

    /* Page 3 (Motion) defaults */
    inst->arp_rate = 0.5f;
    inst->arp_pattern = 0;  /* Off */
    inst->arp_sync = 0;     /* Free */
    inst->arp_phase = 0.0f;
    inst->arp_current_step = 0;
    inst->arp_rng = 0xCAFEBABE;  /* Different seed from main RNG */
    inst->drift = 0.0f;
    inst->drift_rng = 0xDEAFDEAF;
    inst->motion_rate = 0.3f;
    inst->motion_depth = 0.0f;
    inst->motion_shape = 0;  /* Sine */
    inst->motion_phase = 0.0f;
    inst->motion_sh_value = 0.0f;
    inst->motion_sh_rng = 0xFEEDBEEF;
    inst->scoop = 0.0f;

    /* Initialize per-resonator Motion state */
    for (int i = 0; i < MAX_RESONATORS; i++) {
        inst->arp_gate[i] = 1.0f;
        inst->drift_noise[i] = 0.0f;
        inst->drift_ratio[i] = 1.0f;
        inst->scoop_z1[i] = 0.0f;
        inst->scoop_z2[i] = 0.0f;
    }

    /* Patch page defaults */
    inst->current_preset = 0;
    inst->stereo_width = 1.0f;
    for (int i = 0; i < MAX_VOICES; i++) {
        inst->pan_spread[i] = (float)i / (float)(MAX_VOICES - 1);
    }

    /* Initialize excitation gain smoothing (prevents volume jumps on input transients) */
    inst->input_rms = 0.0f;
    inst->excitation_gain_smooth = 1.0f;

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
        if (strcmp(val, "Motion") == 0) inst->current_page = 2;
        else if (strcmp(val, "Control") == 0) inst->current_page = 1;
        else if (strcmp(val, "Patch") == 0) inst->current_page = 3;
        else inst->current_page = 0; /* "Spectra", "Main", or root */
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

        if (inst->current_page == 3) {
            /* Patch page */
            k = &KNOB_MAP_P4[idx];
            /* Rnd buttons: trigger on knob turn and set display counter */
            if (idx == 1) { randomize_preset(inst); inst->rnd_preset_counter = 3; return; }
            if (idx == 2) { randomize_spectra_page(inst); inst->rnd_spectra_counter = 3; return; }
            if (idx == 3) { randomize_motion_page(inst); inst->rnd_motion_counter = 3; return; }
            if (idx == 4) { randomize_panning(inst, MAX_VOICES); inst->rnd_pan_counter = 3; return; }
            if (k->is_enum) ip = get_int_param_ptr_p4(inst, idx);
            else fp = get_float_param_ptr_p4(inst, idx);
        } else if (inst->current_page == 2) {
            /* Motion page */
            k = &KNOB_MAP_P3[idx];
            if (k->is_enum) ip = get_int_param_ptr_p3(inst, idx);
            else fp = get_float_param_ptr_p3(inst, idx);
        } else if (inst->current_page == 1) {
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
    /* ── Page 3 (Motion) params ── */
    else if (strcmp(key, "arp_rate") == 0)    inst->arp_rate = clampf(atof(val), 0, 1);
    else if (strcmp(key, "arp_pattern") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_ARP_PATTERNS; i++) {
            if (strcmp(val, ARP_PATTERN_NAMES[i]) == 0) { inst->arp_pattern = i; found = 1; break; }
        }
        if (!found) inst->arp_pattern = clampi(atoi(val), 0, NUM_ARP_PATTERNS - 1);
    }
    else if (strcmp(key, "arp_sync") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_ARP_SYNC; i++) {
            if (strcmp(val, ARP_SYNC_NAMES[i]) == 0) { inst->arp_sync = i; found = 1; break; }
        }
        if (!found) inst->arp_sync = clampi(atoi(val), 0, NUM_ARP_SYNC - 1);
    }
    else if (strcmp(key, "drift") == 0)       inst->drift = clampf(atof(val), 0, 1);
    else if (strcmp(key, "motion_rate") == 0) inst->motion_rate = clampf(atof(val), 0, 1);
    else if (strcmp(key, "motion_depth") == 0) inst->motion_depth = clampf(atof(val), 0, 1);
    else if (strcmp(key, "motion_shape") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_MOTION_SHAPES; i++) {
            if (strcmp(val, MOTION_SHAPE_NAMES[i]) == 0) { inst->motion_shape = i; found = 1; break; }
        }
        if (!found) inst->motion_shape = clampi(atoi(val), 0, NUM_MOTION_SHAPES - 1);
    }
    else if (strcmp(key, "scoop") == 0)       inst->scoop = clampf(atof(val), 0, 1);
    /* ── Patch page params ── */
    else if (strcmp(key, "preset") == 0) {
        int found = 0;
        for (int i = 0; i < NUM_PRESETS; i++) {
            if (strcmp(val, PRESETS[i].name) == 0) { apply_preset(inst, i); found = 1; break; }
        }
        if (!found) {
            int idx = clampi(atoi(val), 0, NUM_PRESETS - 1);
            apply_preset(inst, idx);
        }
    }
    else if (strcmp(key, "rnd_preset") == 0 && atof(val) > 0.5f) {
        randomize_preset(inst);
    }
    else if (strcmp(key, "rnd_spectra") == 0 && atof(val) > 0.5f) {
        randomize_spectra_page(inst);
    }
    else if (strcmp(key, "rnd_motion") == 0 && atof(val) > 0.5f) {
        randomize_motion_page(inst);
    }
    else if (strcmp(key, "rnd_pan") == 0 && atof(val) > 0.5f) {
        randomize_panning(inst, MAX_VOICES);
    }
    else if (strcmp(key, "stereo_width") == 0) {
        inst->stereo_width = clampf(atof(val), 0, 1);
    }
    /* ── State restore ── */
    else if (strcmp(key, "state") == 0) {
        sscanf(val,
            "onset=%f;frequency=%f;brightness=%f;timbre=%f;decay=%f;"
            "root_note=%d;scale=%d;mix=%f;"
            "chord_drift=%f;resonators=%d;polyphony=%d;octave_range=%d;"
            "pre_gain=%f;post_gain=%f;hpf=%d;lpf=%d;limiter=%d;compress=%f;"
            "arp_rate=%f;arp_pattern=%d;arp_sync=%d;drift=%f;"
            "motion_rate=%f;motion_depth=%f;motion_shape=%d;scoop=%f;"
            "current_preset=%d;stereo_width=%f",
            &inst->onset, &inst->frequency, &inst->brightness, &inst->timbre, &inst->decay,
            &inst->root_note, &inst->scale, &inst->mix,
            &inst->chord_drift, &inst->resonators_idx, &inst->polyphony_idx, &inst->octave_range_idx,
            &inst->pre_gain, &inst->post_gain, &inst->hpf, &inst->lpf, &inst->limiter, &inst->compress,
            &inst->arp_rate, &inst->arp_pattern, &inst->arp_sync, &inst->drift,
            &inst->motion_rate, &inst->motion_depth, &inst->motion_shape, &inst->scoop,
            &inst->current_preset, &inst->stereo_width);
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
            "{\"key\":\"limiter\",\"name\":\"Limiter\",\"type\":\"enum\",\"options\":[\"Off\",\"On\"]},"
            "{\"key\":\"arp_rate\",\"name\":\"Arp Rate\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"arp_pattern\",\"name\":\"Arp Ptrn\",\"type\":\"enum\",\"options\":[\"Off\",\"Up\",\"Down\",\"Random\",\"Chord\"]},"
            "{\"key\":\"arp_sync\",\"name\":\"Arp Sync\",\"type\":\"enum\",\"options\":[\"Free\",\"Sync\"]},"
            "{\"key\":\"drift\",\"name\":\"Drift\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"motion_rate\",\"name\":\"Mot Rate\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"motion_depth\",\"name\":\"Mot Depth\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"motion_shape\",\"name\":\"Mot Shape\",\"type\":\"enum\",\"options\":[\"Sine\",\"Triangle\",\"Square\",\"S&H\"]},"
            "{\"key\":\"scoop\",\"name\":\"Scoop\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
            "{\"key\":\"preset\",\"name\":\"Preset\",\"type\":\"enum\",\"options\":[\"Bell\",\"Pad\",\"Arp Up\",\"Arp Rnd\",\"Drifter\",\"Shimmer\",\"Drone Bass\",\"Tremolo\",\"Bells\",\"Scooped\",\"Evolving\",\"Stutter\",\"Resonant\",\"Wavy\",\"Pulsing\",\"Minimal\",\"Lush\",\"Crisp\",\"Glitch\",\"Warm\",\"Bright\",\"Deep\",\"Airy\",\"Metallic\",\"Modulating\",\"Sparse\",\"Liquid\",\"Rhythmic\",\"Spacious\",\"Chaos\"]},"
            "{\"key\":\"rnd_preset\",\"name\":\"Rnd Preset\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_spectra\",\"name\":\"Rnd Spectra\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_motion\",\"name\":\"Rnd Motion\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"rnd_pan\",\"name\":\"Rnd Pan\",\"type\":\"int\",\"min\":0,\"max\":1,\"step\":1},"
            "{\"key\":\"stereo_width\",\"name\":\"Stereo Width\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}"
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

    /* Page 3 (Motion) */
    if (strcmp(key, "arp_rate") == 0)   return snprintf(buf, buf_len, "%.4f", inst->arp_rate);
    if (strcmp(key, "arp_pattern") == 0)
        return snprintf(buf, buf_len, "%s", ARP_PATTERN_NAMES[clampi(inst->arp_pattern, 0, NUM_ARP_PATTERNS - 1)]);
    if (strcmp(key, "arp_sync") == 0)
        return snprintf(buf, buf_len, "%s", ARP_SYNC_NAMES[clampi(inst->arp_sync, 0, NUM_ARP_SYNC - 1)]);
    if (strcmp(key, "drift") == 0)      return snprintf(buf, buf_len, "%.4f", inst->drift);
    if (strcmp(key, "motion_rate") == 0) return snprintf(buf, buf_len, "%.4f", inst->motion_rate);
    if (strcmp(key, "motion_depth") == 0) return snprintf(buf, buf_len, "%.4f", inst->motion_depth);
    if (strcmp(key, "motion_shape") == 0)
        return snprintf(buf, buf_len, "%s", MOTION_SHAPE_NAMES[clampi(inst->motion_shape, 0, NUM_MOTION_SHAPES - 1)]);
    if (strcmp(key, "scoop") == 0)      return snprintf(buf, buf_len, "%.4f", inst->scoop);

    /* Page 4 (Patch) */
    if (strcmp(key, "preset") == 0) {
        int idx = clampi(inst->current_preset, 0, NUM_PRESETS - 1);
        return snprintf(buf, buf_len, "%s", PRESETS[idx].name);
    }
    if (strcmp(key, "rnd_preset") == 0)   return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_spectra") == 0)  return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_motion") == 0)   return snprintf(buf, buf_len, "0");
    if (strcmp(key, "rnd_pan") == 0)      return snprintf(buf, buf_len, "0");
    if (strcmp(key, "stereo_width") == 0) return snprintf(buf, buf_len, "%.0f%%", inst->stereo_width * 100.0f);

    /* ── knob_N_name (page-aware) ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_name")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8) {
            const knob_def_t *k;
            if (inst->current_page == 3) k = &KNOB_MAP_P4[idx];
            else if (inst->current_page == 2) k = &KNOB_MAP_P3[idx];
            else if (inst->current_page == 1) k = &KNOB_MAP_P2[idx];
            else k = &KNOB_MAP[idx];
            if (k->label) return snprintf(buf, buf_len, "%s", k->label);
        }
        return 0;
    }

    /* ── knob_N_value (page-aware) ── */
    if (strncmp(key, "knob_", 5) == 0 && strstr(key, "_value")) {
        int idx = atoi(key + 5) - 1;
        if (idx >= 0 && idx < 8) {
            const knob_def_t *k;
            if (inst->current_page == 3) {
                /* Patch page */
                k = &KNOB_MAP_P4[idx];
                if (k->is_enum) {
                    return get_param(instance, k->key, buf, buf_len);
                } else if (idx >= 1 && idx <= 4) {
                    /* Rnd buttons: always display 0 */
                    return snprintf(buf, buf_len, "0");
                } else {
                    float *p = get_float_param_ptr_p4(inst, idx);
                    if (p) {
                        if (idx == 5) return snprintf(buf, buf_len, "%.0f%%", *p * 100);
                        return snprintf(buf, buf_len, "%.0f", *p);
                    }
                }
            } else if (inst->current_page == 2) {
                /* Motion page */
                k = &KNOB_MAP_P3[idx];
                if (k->is_enum) {
                    return get_param(instance, k->key, buf, buf_len);
                } else {
                    float *p = get_float_param_ptr_p3(inst, idx);
                    if (p) return snprintf(buf, buf_len, "%d%%", (int)(*p * 100));
                }
            } else if (inst->current_page == 1) {
                /* Control page */
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
                /* Main page */
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
            "pre_gain=%.6f;post_gain=%.6f;hpf=%d;lpf=%d;limiter=%d;compress=%.6f;"
            "arp_rate=%.6f;arp_pattern=%d;arp_sync=%d;drift=%.6f;"
            "motion_rate=%.6f;motion_depth=%.6f;motion_shape=%d;scoop=%.6f;"
            "current_preset=%d;stereo_width=%.6f",
            inst->onset, inst->frequency, inst->brightness, inst->timbre, inst->decay,
            inst->root_note, inst->scale, inst->mix,
            inst->chord_drift, inst->resonators_idx, inst->polyphony_idx, inst->octave_range_idx,
            inst->pre_gain, inst->post_gain, inst->hpf, inst->lpf, inst->limiter, inst->compress,
            inst->arp_rate, inst->arp_pattern, inst->arp_sync, inst->drift,
            inst->motion_rate, inst->motion_depth, inst->motion_shape, inst->scoop,
            inst->current_preset, inst->stereo_width);
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

    /* ── Smooth excitation gain (prevents volume jumps on input transients) ── */
    float target_excitation = 1.0f + inst->input_rms * 2.0f;
    inst->excitation_gain_smooth += 0.01f * (target_excitation - inst->excitation_gain_smooth);

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

    /* ── Motion page engines ──────────────────────────────────────────────── */
    /* Arp and Drift work per-block and per-resonator respectively. */
    arp_tick(inst, res_count, frames);
    drift_tick(inst, res_count);

    /* ── Process audio through resonator bank ───────────────────────────── */
    float wet_l_buf[BLOCK_SIZE];
    float wet_r_buf[BLOCK_SIZE];
    memset(wet_l_buf, 0, sizeof(wet_l_buf));
    memset(wet_r_buf, 0, sizeof(wet_r_buf));

    for (int vi = 0; vi < num_voices; vi++) {
        voice_t *v = &inst->voices[vi];
        if (!v->active) continue;

        /* Full stereo spread: each voice pans across the field */
        /* Pan with stereo width control (0=mono at center, 1=full stereo spread) */
        float center_pan = 0.5f + (float)(vi - num_voices / 2) * 0.35f;
        center_pan = clampf(center_pan, 0.0f, 1.0f);
        float pan = 0.5f + (center_pan - 0.5f) * inst->stereo_width;
        /* Apply per-voice pan spread override */
        pan = 0.5f + (inst->pan_spread[vi] - 0.5f) * inst->stereo_width;
        pan = clampf(pan, 0.0f, 1.0f);
        float pan_l = 1.0f - pan;
        float pan_r = pan;
        /* With Q-normalized SVF, output level is stable regardless of Q.
         * Scale by num_resonators and apply brightness gain for tonal control. */
        float res_gain = (v->num_resonators > 0) ? 3.0f / sqrtf((float)v->num_resonators) : 0.0f;
        res_gain *= brightness_gain;

        for (int i = 0; i < frames; i++) {
            float input_sample = mono[i] * inst->excitation_gain_smooth;
            float resonator_sum = 0.0f;

            /* Compute motion LFO for this sample (per-sample) */
            float motion_gain = motion_lfo_sample(inst);

            for (int r = 0; r < v->num_resonators; r++) {
                if (v->resonators[r].fc > 0.0f) {
                    /* Apply drift-perturbed frequency to resonator */
                    float freq_with_drift = v->resonators[r].freq * inst->drift_ratio[r];
                    v->resonators[r].freq = freq_with_drift;
                    svf_update_coeff(&v->resonators[r]);

                    /* Process through SVF */
                    float res_out = svf_process_bp(&v->resonators[r], input_sample);

                    /* Apply scoop notch filter */
                    res_out = scoop_process(res_out, v->resonators[r].freq, inst->scoop,
                                           &inst->scoop_z1[r], &inst->scoop_z2[r]);

                    /* Apply arp gate and motion LFO */
                    res_out *= inst->arp_gate[r] * motion_gain;

                    resonator_sum += res_out;
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
