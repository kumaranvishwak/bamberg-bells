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