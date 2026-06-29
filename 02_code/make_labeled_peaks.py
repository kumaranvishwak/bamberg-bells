"""
Builds a clear, labeled "peaks of an individual strike" figure for the
presentation: trimmed waveform on top, FFT magnitude spectrum below with
the five detected partials marked and labeled (hum, prime, tierce,
quint, nominal) -- exactly what was requested: clearly visible peaks
of one individual, trimmed bell strike, not a generic spectrogram.
"""
import sys
sys.path.insert(0, "/home/claude/FINAL/code")
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extract_features import find_spectral_peaks, identify_partials, N_FFT

PATH = "/home/claude/project/COMPLETE_PROJECT/01_data/strikes_segmented/dormplatz/dormplatz_strike003.flac"
OUT = "/home/claude/FINAL/term_paper/figures/labeled_peaks_dormplatz.png"
OUT2 = "/home/claude/project/COMPLETE_PROJECT/03_spectrograms/labeled_peaks_dormplatz.png"

SR = 48000
y, sr = librosa.load(PATH, sr=SR, mono=True)

peak_freqs, peak_mags = find_spectral_peaks(y, sr)
partials, prime_freq = identify_partials(peak_freqs, peak_mags)
print("Detected partials:", partials)

# full FFT for plotting (same window as feature extraction)
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

display_dur = min(1.2, len(y) / sr)
n_display = int(display_dur * sr)
t = np.arange(n_display) / sr
ax1.plot(t, y[:n_display], color="#1E2761", linewidth=0.8)
ax1.set_xlim(0, display_dur)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Trimmed individual strike -- Dormplatz (first 1.2s shown)")

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
plt.savefig(OUT, dpi=150)
plt.savefig(OUT2, dpi=150)
print(f"Saved -> {OUT}")
