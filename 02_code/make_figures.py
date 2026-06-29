"""
Bamberg Church Bells - Figure Generation
==========================================
Produces the figures used in the term paper:
  - spectrograms_grid.png : one log-frequency spectrogram per church
  - results_dashboard.png : 4-panel summary of the statistical results
  - material_vs_deviation.png : tuning deviation grouped by material
"""
import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/claude/project/COMPLETE_PROJECT"
RESULTS_DIR = os.path.join(ROOT, "04_results")
FIG_DIR = os.path.join(ROOT, "03_spectrograms")
PAPER_FIG_DIR = "/home/claude/FINAL/term_paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

STRIKES_DIR = os.path.join(ROOT, "01_data/strikes_segmented")


def representative_strike(bell_id):
    bell_dir = os.path.join(STRIKES_DIR, bell_id)
    files = sorted([f for f in os.listdir(bell_dir) if f.endswith((".wav", ".flac"))])
    for f in files:
        full = os.path.join(bell_dir, f)
        if os.path.getsize(full) > 1000:  # skip empty/corrupt clips
            return full
    return None


def plot_individual_spectrograms():
    """One spectrogram PNG per church, saved separately for map embedding."""
    merged = pd.read_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"))
    out_dir = os.path.join(FIG_DIR, "per_church")
    os.makedirs(out_dir, exist_ok=True)

    for bell_id in sorted(merged["bell_id"].tolist()):
        path = representative_strike(bell_id)
        if path is None:
            continue
        y, sr = librosa.load(path, sr=48000, mono=True)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
        fig, ax = plt.subplots(figsize=(4, 3))
        librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="log",
                                  ax=ax, cmap="gray_r")
        ax.set_ylim(50, 6000)
        ax.set_title(bell_id, fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{bell_id}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
    print(f"Saved per-church spectrograms -> {out_dir}")


def plot_spectrogram_grid():
    merged = pd.read_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"))
    bell_ids = sorted(merged["bell_id"].tolist())
    n = len(bell_ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    axes = axes.flatten()

    for i, bell_id in enumerate(bell_ids):
        path = representative_strike(bell_id)
        ax = axes[i]
        if path is None:
            ax.set_visible(False)
            continue
        y, sr = librosa.load(path, sr=48000, mono=True)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="log",
                                  ax=ax, cmap="gray_r")
        ax.set_title(bell_id, fontsize=9)
        ax.set_ylim(50, 6000)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "spectrograms_grid.png")
    plt.savefig(out_path, dpi=130)
    plt.savefig(os.path.join(PAPER_FIG_DIR, "spectrograms_grid.png"), dpi=130)
    plt.close()
    print(f"Saved {out_path}")


def plot_dashboard():
    merged = pd.read_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"))
    valid = merged.dropna(subset=["casting_year"])
    valid = valid[~valid["certainty"].str.contains("DATA QUALITY ISSUE", na=False)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.scatter(valid["casting_year"], valid["tuning_deviation_cents"])
    for _, r in valid.iterrows():
        ax.annotate(r["bell_id"], (r["casting_year"], r["tuning_deviation_cents"]), fontsize=6)
    ax.set_xlabel("Casting year")
    ax.set_ylabel("Tuning deviation (cents)")
    ax.set_title("Tuning deviation vs. casting year")

    ax = axes[0, 1]
    order = merged.sort_values("t60")
    ax.barh(order["bell_id"], order["t60"])
    ax.set_xlabel("T60 (s)")
    ax.set_title("Decay time (T60) by church")
    ax.tick_params(axis="y", labelsize=7)

    ax = axes[1, 0]
    order2 = merged.sort_values("n_strikes_used", ascending=False)
    ax.bar(order2["bell_id"], order2["n_strikes_used"])
    ax.set_ylabel("Strikes used (capped)")
    ax.set_title("Samples per church (after capping)")
    ax.tick_params(axis="x", rotation=75, labelsize=7)

    ax = axes[1, 1]
    ax.scatter(valid["casting_year"], valid["f_nominal"])
    ax.set_xlabel("Casting year")
    ax.set_ylabel("Nominal frequency (Hz)")
    ax.set_title("Nominal pitch vs. casting year")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "results_dashboard.png")
    plt.savefig(out_path, dpi=130)
    plt.savefig(os.path.join(PAPER_FIG_DIR, "results_dashboard.png"), dpi=130)
    plt.close()
    print(f"Saved {out_path}")


def plot_material_vs_deviation():
    merged = pd.read_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"))
    grouped = merged.groupby("material")["tuning_deviation_cents"].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grouped.index, grouped["mean"])
    for i, (mat, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["mean"] + 1, f"n={int(row['count'])}", ha="center", fontsize=9)
    ax.set_ylabel("Mean tuning deviation (cents)")
    ax.set_title("Tuning deviation by bell material")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "material_vs_deviation.png")
    plt.savefig(out_path, dpi=130)
    plt.savefig(os.path.join(PAPER_FIG_DIR, "material_vs_deviation.png"), dpi=130)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_individual_spectrograms()
    plot_spectrogram_grid()
    plot_dashboard()
    plot_material_vs_deviation()
