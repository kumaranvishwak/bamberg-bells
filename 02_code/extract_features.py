"""
Bamberg Church Bells - Acoustic Feature Extraction
====================================================
For every individual bell-strike clip, this script:

  1. Computes a high-resolution FFT and finds spectral peaks
  2. Identifies the five classical bell partials (hum, prime, tierce,
     quint, nominal) by their approximate frequency ratio to the
     strongest low-frequency peak (the "prime")
  3. Classifies tierce type (minor ~300 cents / major ~400 cents
     above the prime) -> pre-/post-Hemony-reform tuning style
  4. Estimates T60 decay time from the energy envelope
  5. Computes standard spectral descriptors (centroid, bandwidth,
     rolloff, flatness) as additional ML features

Output: one CSV row per strike -> 04_results/all_strikes_features.csv
"""
import os
import json
import numpy as np
import librosa
import pandas as pd

SR = 48000
N_FFT = 8192        # safe FFT size (avoids the 65536 OOM bug from earlier)
HOP_LENGTH = 1024

# Classical relative partial ratios (relative to the prime/fundamental = 1.0)
PARTIAL_RATIOS = {
    "hum":     0.50,
    "prime":   1.00,
    "tierce":  1.20,   # nominal placeholder; refined per-bell below
    "quint":   1.50,
    "nominal": 2.00,
}
SEARCH_WINDOW = 0.12   # +/- fractional search window around each expected ratio


def find_spectral_peaks(y, sr, n_fft=N_FFT, n_peaks=25):
    """Return (freqs, mags) of the strongest spectral peaks of a clip."""
    window = np.hanning(len(y)) if len(y) < n_fft else np.hanning(n_fft)
    if len(y) < n_fft:
        y_padded = np.pad(y, (0, n_fft - len(y)))
        spec = np.abs(np.fft.rfft(y_padded * window))
    else:
        spec = np.abs(np.fft.rfft(y[:n_fft] * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # restrict to plausible bell partial range (40 Hz - 8000 Hz)
    mask = (freqs > 40) & (freqs < 8000)
    freqs, spec = freqs[mask], spec[mask]

    # simple local-maxima peak picking
    peak_idx = []
    for i in range(2, len(spec) - 2):
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > spec[i-2] and spec[i] > spec[i+2]:
            peak_idx.append(i)
    peak_idx = sorted(peak_idx, key=lambda i: -spec[i])[:n_peaks]
    peak_freqs = freqs[peak_idx]
    peak_mags = spec[peak_idx]
    order = np.argsort(peak_freqs)
    return peak_freqs[order], peak_mags[order]


def identify_partials(peak_freqs, peak_mags):
    """
    Identify the prime as the strongest peak below 600 Hz, then locate
    hum/tierce/quint/nominal as the strongest peaks near their expected
    ratio to the prime.
    """
    if len(peak_freqs) == 0:
        return {}, None

    low_mask = peak_freqs < 600
    if low_mask.sum() == 0:
        prime_idx = np.argmax(peak_mags)
    else:
        candidates = np.where(low_mask)[0]
        prime_idx = candidates[np.argmax(peak_mags[candidates])]
    prime_freq = peak_freqs[prime_idx]

    found = {"prime": float(prime_freq)}
    for name, ratio in PARTIAL_RATIOS.items():
        if name == "prime":
            continue
        target = prime_freq * ratio
        lo, hi = target * (1 - SEARCH_WINDOW), target * (1 + SEARCH_WINDOW)
        in_range = np.where((peak_freqs >= lo) & (peak_freqs <= hi))[0]
        if len(in_range) > 0:
            best = in_range[np.argmax(peak_mags[in_range])]
            found[name] = float(peak_freqs[best])
    return found, prime_freq


def classify_tierce(prime_freq, tierce_freq):
    if prime_freq is None or tierce_freq is None:
        return None, None
    cents = 1200 * np.log2(tierce_freq / prime_freq)
    # minor third ~ 300 cents (pre-Hemony "old" tuning)
    # major third ~ 400 cents (post-Hemony "modern" tuning)
    tierce_type = "minor" if abs(cents - 300) < abs(cents - 400) else "major"
    return tierce_type, float(cents)


def estimate_t60(y, sr, frame_length=2048, hop_length=512):
    """Estimate T60 via linear regression on the dB energy decay curve."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    peak_idx = int(np.argmax(rms_db))
    decay = rms_db[peak_idx:]
    if len(decay) < 5:
        return None
    times = np.arange(len(decay)) * hop_length / sr
    # use the segment from -5dB to -25dB below peak (classic T-x method)
    rel = decay - decay[0]
    mask = (rel <= -5) & (rel >= -30)
    if mask.sum() < 4:
        mask = np.ones_like(rel, dtype=bool)
    slope, intercept = np.polyfit(times[mask], rel[mask], 1)
    if slope >= 0:
        return None
    t60 = -60.0 / slope
    return float(np.clip(t60, 0.1, 30.0))


def spectral_descriptors(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=2048)[0]
    return {
        "spectral_centroid": float(np.mean(cent)),
        "spectral_bandwidth": float(np.mean(bw)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)),
    }


def process_strike(path, bell_id, strike_num):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < int(0.2 * sr):
        return None

    peak_freqs, peak_mags = find_spectral_peaks(y, sr)
    partials, prime_freq = identify_partials(peak_freqs, peak_mags)
    tierce_type, tierce_cents = classify_tierce(partials.get("prime"), partials.get("tierce"))
    t60 = estimate_t60(y, sr)
    spec_feats = spectral_descriptors(y, sr)

    row = {
        "bell_id": bell_id,
        "strike_num": strike_num,
        "file": os.path.basename(path),
        "f_hum": partials.get("hum"),
        "f_prime": partials.get("prime"),
        "f_tierce": partials.get("tierce"),
        "f_quint": partials.get("quint"),
        "f_nominal": partials.get("nominal"),
        "tierce_type": tierce_type,
        "tierce_cents": tierce_cents,
        "t60": t60,
    }
    row.update(spec_feats)
    return row


def main():
    ROOT = "/home/claude/project/COMPLETE_PROJECT"
    STRIKES_DIR = os.path.join(ROOT, "01_data/strikes_segmented")
    OUT_DIR = os.path.join(ROOT, "04_results")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    bell_dirs = sorted([d for d in os.listdir(STRIKES_DIR)
                         if os.path.isdir(os.path.join(STRIKES_DIR, d))])
    for bell_id in bell_dirs:
        bell_dir = os.path.join(STRIKES_DIR, bell_id)
        files = sorted([f for f in os.listdir(bell_dir) if f.endswith((".wav", ".flac"))])
        print(f"Processing {bell_id}: {len(files)} strikes")
        for f in files:
            try:
                row = process_strike(os.path.join(bell_dir, f), bell_id, f)
                if row:
                    rows.append(row)
            except Exception as e:
                print(f"  ! failed on {f}: {e}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "all_strikes_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} strike rows -> {out_path}")


if __name__ == "__main__":
    main()
