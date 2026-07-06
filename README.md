# Bamberg Church Bells

Machine Listening project (CH-Proj-M), SS 2026, University of
Bamberg.

**Authors:** Kumaran Vasu and Srujan Mahajan  
**Supervisor:** Prof. Abesser

This project contains field recordings and acoustic analysis of church
bells in Bamberg. It includes FFT-based bell-partial extraction,
estimated decay time, spectral descriptors, KNN and MLP classification,
figures, and an interactive audio-linked map.

After removing one unverified recording, the final dataset contains:

- 12 retained recording entries
- 11 bell-level observations
- 11 known-year observations used for correlation and classification
- 11 markers on the interactive map

The two St. Stephan recording sessions are merged into one bell-level
observation and one map marker.

## Folders

- `02_code/` – analysis and map-generation scripts
- `03_spectrograms/` – generated spectrogram images and demonstration
  material
- `04_results/` – aggregated CSV files, figures, and
  `model_results.json`
- `05_map/` – generated Leaflet map
- `docs/` – public GitHub Pages version of the interactive map
- `06_term_paper/` – term-paper source files, when included

Raw and segmented audio files are not included because of their size.
The file `04_results/all_strikes_features.csv` contains the extracted
per-strike features, so the statistical analysis, figures, and map can
be reproduced without the original audio.

## Installation

Create and activate a Python virtual environment, then install the
dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproducing the final results

Run these commands from the repository root:

```bash
python 02_code/run_analysis.py
python 02_code/make_figures.py
python 02_code/build_map.py
```

These commands regenerate:

- `04_results/bells_aggregated_with_material.csv`
- `04_results/capped_strikes_features.csv`
- `04_results/model_results.json`
- `04_results/results_dashboard.png`
- `04_results/material_vs_deviation.png`
- `05_map/bamberg_bells_map.html`

The final dataset contains twelve retained recording entries, eleven
bell-level observations, and eleven map markers. The two St. Stephan
sessions are merged before aggregation.

The recording previously labelled `st_nicholas_church` is retained
only in some historical working files for provenance. It is filtered
out before final aggregation and does not appear in the final
statistics, figures, table, or map.

`make_figures.py` also attempts to regenerate spectrograms. The
segmented audio is not included in the repository, so this optional
step is skipped when the audio folders are unavailable. The dashboard
and material-comparison figure are still generated.

## Interactive map

The generated map is available at:

```text
05_map/bamberg_bells_map.html
```

The GitHub Pages copy is:

```text
docs/index.html
```

The map contains embedded metadata and audio samples. Leaflet and
OpenStreetMap resources are loaded from the internet.

## Important data-quality corrections

- St. Martin was re-recorded after its original file was found to be
  identical to the Bamberg Cathedral recording.
- `st_heinrich` was corrected from St. Michael to the independent
  parish church St. Heinrich.
- `evangelical_church` was identified as Erlöserkirche Bamberg.
- Incorrect church coordinates were corrected.
- The strongest-strike selection was fixed to match filenames by stem
  instead of assuming a `.wav` extension.
- The unverified St. Nikolaus recording was excluded from the final
  analysis and map.

## Limitations

The final sample contains only eleven bell-level observations.
Correlation and classification results should therefore be treated as
exploratory. Recording distance, background noise, surrounding
buildings, and short decay windows may affect the extracted features.
