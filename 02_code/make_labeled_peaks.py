"""
makes the labeled peaks figure (waveform + FFT with partials marked)
for one church, or all of them with --all. also saves a short audio
clip of the strike for presenting

usage: python3 make_labeled_peaks.py [church_name | --all]
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # parent of 02_code = COMPLETE_PROJECT
sys.path.insert(0, SCRIPT_DIR)
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extract_features import find_spectral_peaks, identify_partials, N_FFT

STRIKES_DIR = os.path.join(ROOT, "01_data/strikes_segmented")
OUT_DIR = os.path.join(ROOT, "03_spectrograms/demo_peaks")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 48000
DISPLAY_DUR = 1.2  # seconds of a single clean strike to show/export

# A representative, known-good strike file per church. These were
# picked as clean, well-isolated single strikes. Set to None to just
# auto-pick the first usable clip in that church's folder.
DEFAULT_STRIKE = {
    "dormplatz":          "dormplatz_strike003.flac",
    "bamberg_cathedral":  None,
    "evangelical_church": None,
    "gruner_markt":       None,
    "karmelite_kloster":  None,
    "obere_pfarre":       None,
    "pestallozistrasse":  None,
    "st_heinrich":        None,
    "st_jakob":           None,
    "st_michael":         None,
    "st_nicholas_church": None,
    "st_stephan":         None,
    "st_stephan_2":       None,
}


def pick_strike_file(church):
    forced = DEFAULT_STRIKE.get(church)
    church_dir = os.path.join(STRIKES_DIR, church)
    if forced:
        return os.path.join(church_dir, forced)
    files = sorted(f for f in os.listdir(church_dir) if f.endswith((".wav", ".flac")))
    for f in files:
        full = os.path.join(church_dir, f)
        if os.path.getsize(full) > 1000:
            return full
    raise FileNotFoundError(f"No usable strike file found for {church}")


def make_demo(church):
    path = pick_strike_file(church)
    y, sr = librosa.load(path, sr=SR, mono=True)

    peak_freqs, peak_mags = find_spectral_peaks(y, sr)
    partials, prime_freq = identify_partials(peak_freqs, peak_mags)
    print(f"[{church}] file={os.path.basename(path)} detected partials:", partials)

    window = np.hanning(min(len(y), N_FFT))
    if len(y) < N_FFT:
        y_padded = np.pad(y, (0, N_FFT - len(y)))
        spec = np.abs(np.fft.rfft(y_padded * window))
    else:
        spec = np.abs(np.fft.rfft(y[:N_FFT] * window))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    spec_db = 20 * np.log10(np.maximum(spec, 1e-6))
    spec_db -= spec_db.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    display_dur = min(DISPLAY_DUR, len(y) / sr)
    n_display = int(display_dur * sr)
    t = np.arange(n_display) / sr
    ax1.plot(t, y[:n_display], color="#1E2761", linewidth=0.8)
    ax1.set_xlim(0, display_dur)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Trimmed individual strike -- {church} (first {display_dur:.1f}s shown)")

    mask = freqs < 2000
    ax2.plot(freqs[mask], spec_db[mask], color="#1E2761", linewidth=1)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB, normalised)")
    ax2.set_title("FFT spectrum -- partials labeled")
    ax2.set_ylim(-60, 14)

    colors = {"hum": "#D4A017", "prime": "#990011", "tierce": "#028090",
              "quint": "#2C5F2D", "nominal": "#6D2E46"}
    order = sorted(partials.items(), key=lambda kv: kv[1])
    for i, (name, freq) in enumerate(order):
        y_text = 4 + (i % 3) * 4.5
        ax2.axvline(freq, color=colors.get(name, "gray"), linestyle="--", linewidth=1.3)
        ax2.annotate(f"{name}: {freq:.0f} Hz", xy=(freq, y_text), xytext=(freq, y_text),
                     ha="center", fontsize=8.5, color=colors.get(name, "gray"), fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"{church}_labeled_peaks.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Also export the trimmed audio itself, playable directly instead of
    # scrubbing through the original multi-second/minute recording.
    clip_path = os.path.join(OUT_DIR, f"{church}_trimmed_strike.wav")
    sf.write(clip_path, y[:n_display], sr)

    print(f"  -> {fig_path}")
    print(f"  -> {clip_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        for church in DEFAULT_STRIKE:
            try:
                make_demo(church)
            except Exception as e:
                print(f"[{church}] skipped: {e}")
    else:
        church = sys.argv[1] if len(sys.argv) > 1 else "dormplatz"
        make_demo(church)
