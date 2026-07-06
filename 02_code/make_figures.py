"""
Generate the paper figures from the final bell-level results.

The dashboard uses readable church names. Spectrogram generation is
optional because the segmented audio is not included in the public repo.
"""

import os

import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(ROOT, "04_results")
FIG_DIR = os.path.join(ROOT, "03_spectrograms")
PAPER_FIG_DIR = os.path.join(ROOT, "06_term_paper", "figures")
STRIKES_DIR = os.path.join(ROOT, "01_data", "strikes_segmented")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


PLOT_LABELS = {
    "bamberg_cathedral": "Bamberg Cathedral",
    "dormplatz": "Domplatz",
    "evangelical_church": "Erlöserkirche",
    "gruner_markt": "St. Martin",
    "karmelite_kloster": "Karmelitenkirche",
    "obere_pfarre": "Obere Pfarrkirche",
    "pestallozistrasse": "Auferstehungskirche",
    "st_heinrich": "St. Heinrich",
    "st_jakob": "St. Jakob",
    "st_michael": "St. Michael",
    "st_stephan": "St. Stephan",
}


def add_plot_labels(df):
    """Return a copy with readable names for charts."""
    output = df.copy()

    output["plot_label"] = (
        output["bell_id"]
        .map(PLOT_LABELS)
        .fillna(output["church"])
    )

    return output


def representative_strike(bell_id):
    """Return the first valid strike file for one bell."""
    bell_directory = os.path.join(STRIKES_DIR, bell_id)

    files = sorted(
        filename
        for filename in os.listdir(bell_directory)
        if filename.lower().endswith((".wav", ".flac"))
    )

    for filename in files:
        full_path = os.path.join(bell_directory, filename)

        if os.path.getsize(full_path) > 1000:
            return full_path

    return None


def plot_individual_spectrograms():
    """Create one spectrogram per retained bell-level observation."""
    merged = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "bells_aggregated_with_material.csv",
        )
    )

    merged = add_plot_labels(merged)

    label_lookup = dict(
        zip(
            merged["bell_id"],
            merged["plot_label"],
        )
    )

    output_directory = os.path.join(
        FIG_DIR,
        "per_church",
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    for bell_id in sorted(merged["bell_id"].tolist()):
        strike_path = representative_strike(bell_id)

        if strike_path is None:
            continue

        audio, sample_rate = librosa.load(
            strike_path,
            sr=48000,
            mono=True,
        )

        spectrum = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(
                    audio,
                    n_fft=2048,
                    hop_length=256,
                )
            ),
            ref=np.max,
        )

        figure, axis = plt.subplots(figsize=(4, 3))

        librosa.display.specshow(
            spectrum,
            sr=sample_rate,
            hop_length=256,
            x_axis="time",
            y_axis="log",
            ax=axis,
            cmap="gray_r",
        )

        axis.set_ylim(50, 6000)
        axis.set_title(
            label_lookup.get(bell_id, bell_id),
            fontsize=9,
        )

        figure.tight_layout()

        figure.savefig(
            os.path.join(
                output_directory,
                f"{bell_id}.png",
            ),
            dpi=100,
        )

        plt.close(figure)

    print(
        f"Saved per-church spectrograms -> "
        f"{output_directory}"
    )


def plot_spectrogram_grid():
    """Create a grid containing one spectrogram per observation."""
    merged = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "bells_aggregated_with_material.csv",
        )
    )

    merged = add_plot_labels(merged)

    label_lookup = dict(
        zip(
            merged["bell_id"],
            merged["plot_label"],
        )
    )

    bell_ids = sorted(
        merged["bell_id"].tolist()
    )

    columns = 4
    rows = int(
        np.ceil(
            len(bell_ids) / columns
        )
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.2 * rows),
    )

    axes = np.atleast_1d(
        axes
    ).flatten()

    used_axes = 0

    for index, bell_id in enumerate(bell_ids):
        axis = axes[index]
        strike_path = representative_strike(bell_id)

        if strike_path is None:
            axis.set_visible(False)
            continue

        audio, sample_rate = librosa.load(
            strike_path,
            sr=48000,
            mono=True,
        )

        spectrum = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(
                    audio,
                    n_fft=2048,
                    hop_length=256,
                )
            ),
            ref=np.max,
        )

        librosa.display.specshow(
            spectrum,
            sr=sample_rate,
            hop_length=256,
            x_axis="time",
            y_axis="log",
            ax=axis,
            cmap="gray_r",
        )

        axis.set_title(
            label_lookup.get(bell_id, bell_id),
            fontsize=9,
        )

        axis.set_ylim(50, 6000)
        used_axes = index + 1

    for axis in axes[used_axes:]:
        axis.set_visible(False)

    figure.tight_layout()

    output_path = os.path.join(
        FIG_DIR,
        "spectrograms_grid.png",
    )

    figure.savefig(
        output_path,
        dpi=130,
    )

    figure.savefig(
        os.path.join(
            PAPER_FIG_DIR,
            "spectrograms_grid.png",
        ),
        dpi=130,
    )

    plt.close(figure)

    print(f"Saved {output_path}")


def plot_dashboard():
    """Create the four-panel results dashboard."""
    merged = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "bells_aggregated_with_material.csv",
        )
    )

    merged = add_plot_labels(merged)

    valid = merged.dropna(
        subset=["casting_year"]
    ).copy()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    # Tuning deviation against casting year
    axis = axes[0, 0]

    axis.scatter(
        valid["casting_year"],
        valid["tuning_deviation_cents"],
    )

    for _, row in valid.iterrows():
        axis.annotate(
            row["plot_label"],
            (
                row["casting_year"],
                row["tuning_deviation_cents"],
            ),
            fontsize=6,
        )

    axis.set_xlabel("Casting year")
    axis.set_ylabel("Tuning deviation (cents)")
    axis.set_title(
        "Tuning deviation vs. casting year"
    )

    # T60 by church
    axis = axes[0, 1]

    decay_order = merged.sort_values("t60")

    axis.barh(
        decay_order["plot_label"],
        decay_order["t60"],
    )

    axis.set_xlabel("T60 (s)")
    axis.set_title(
        "Decay time (T60) by church"
    )

    axis.tick_params(
        axis="y",
        labelsize=7,
    )

    # Number of strikes used
    axis = axes[1, 0]

    strike_order = merged.sort_values(
        "n_strikes_used",
        ascending=False,
    )

    axis.bar(
        strike_order["plot_label"],
        strike_order["n_strikes_used"],
    )

    axis.set_ylabel(
        "Strikes used (capped)"
    )

    axis.set_title(
        "Samples per church (after capping)"
    )

    axis.tick_params(
        axis="x",
        rotation=75,
        labelsize=7,
    )

    # Nominal frequency against casting year
    axis = axes[1, 1]

    axis.scatter(
        valid["casting_year"],
        valid["f_nominal"],
    )

    axis.set_xlabel("Casting year")
    axis.set_ylabel(
        "Nominal frequency (Hz)"
    )

    axis.set_title(
        "Nominal pitch vs. casting year"
    )

    figure.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "results_dashboard.png",
    )

    figure.savefig(
        output_path,
        dpi=130,
    )

    figure.savefig(
        os.path.join(
            PAPER_FIG_DIR,
            "results_dashboard.png",
        ),
        dpi=130,
    )

    plt.close(figure)

    print(f"Saved {output_path}")


def plot_material_vs_deviation():
    """Compare descriptive mean tuning deviation by material."""
    merged = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "bells_aggregated_with_material.csv",
        )
    )

    grouped = (
        merged
        .groupby("material")[
            "tuning_deviation_cents"
        ]
        .agg(["mean", "count"])
        .reindex(
            [
                "bronze",
                "cast steel",
            ]
        )
        .dropna()
    )

    figure, axis = plt.subplots(
        figsize=(6, 4)
    )

    axis.bar(
        grouped.index,
        grouped["mean"],
    )

    for index, (_, row) in enumerate(
        grouped.iterrows()
    ):
        axis.text(
            index,
            row["mean"] + 1,
            f"n={int(row['count'])}",
            ha="center",
            fontsize=9,
        )

    axis.set_ylabel(
        "Mean tuning deviation (cents)"
    )

    axis.set_title(
        "Tuning deviation by bell material"
    )

    figure.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "material_vs_deviation.png",
    )

    figure.savefig(
        output_path,
        dpi=130,
    )

    figure.savefig(
        os.path.join(
            PAPER_FIG_DIR,
            "material_vs_deviation.png",
        ),
        dpi=130,
    )

    plt.close(figure)

    print(f"Saved {output_path}")


def main():
    try:
        plot_individual_spectrograms()
        plot_spectrogram_grid()

    except FileNotFoundError as error:
        print(
            "Skipping spectrogram generation, "
            f"segmented audio not found: {error}"
        )

        print(
            "Dashboard and material comparison figures "
            "do not need audio and will still be generated."
        )

    plot_dashboard()
    plot_material_vs_deviation()


if __name__ == "__main__":
    main()