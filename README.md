# Recording and Spatio-Temporal Mapping of Bamberg's Church Bells

Master Project Machine Listening (CH-Proj-M), SS 2026 — Otto-Friedrich-Universität Bamberg
Authors: Kumaran Vasu, Srujan Mahajan
Supervisor: Prof. Dr.-Ing. Jakob Abeßer

## What this is

Field recordings of 13 church bells across Bamberg, an acoustic feature
extraction pipeline (FFT partials, T60, spectral descriptors), a classical
ML + small neural network (deep learning extension) comparison for a
pre-/post-1900 tuning-era classification task, an interactive spatio-temporal
map, and the term paper write-up.

**Read `06_term_paper/main.pdf` first** — it documents the methodology,
results, and (importantly) the limitations and a data-quality issue found
during this submission, in full.

## Folder structure

```
01_data/
  recordings_original/     13 raw field recordings (not in this git repo —
                            see .gitignore; included in the full project zip)
  strikes_segmented/       per-church segmented strike clips + manifest.json
02_code/
  segment_strikes.py       onset detection -> per-strike WAV clips
  extract_features.py      FFT partial detection, T60, spectral descriptors
  run_analysis.py          aggregation, Pearson correlation, KNN + MLP
                            classification (Leave-One-Out CV)
  make_figures.py          spectrograms grid + results dashboard figures
  build_map.py             builds the self-contained Leaflet map
03_spectrograms/           spectrogram PNGs
04_results/                CSVs, model_results.json, dashboard figures
05_map/                    bamberg_bells_map.html + embedded audio samples
06_term_paper/             main.tex, references.bib, figures/, main.pdf
Vasu_Mahajan_BambergBells_CH_Proj_M_SS_2026.pdf   <- submission file 1
Vasu_Mahajan_BambergBells_CH_Proj_M_SS_2026.zip   <- submission file 2 (LaTeX source)
```

## How to reproduce

```bash
pip install librosa soundfile pandas numpy scipy scikit-learn matplotlib --break-system-packages
python3 02_code/segment_strikes.py     # only needed if re-segmenting from raw audio
python3 02_code/extract_features.py
python3 02_code/run_analysis.py
python3 02_code/make_figures.py
python3 02_code/build_map.py
cd 06_term_paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Known issues — please read before presenting

1. **St. Martin (Gruner_Markt.flac) was a duplicate** of the Bamberg
   Cathedral recording (identical MD5 checksum) — discovered during the
   original submission prep. **This has since been fixed**: the church
   was re-recorded (205.7s, 144 strikes detected) and the corrected data
   is now included in all statistics and the table.
2. **St. Nikolaus has no verified casting year** — flagged as provisional
   in `04_results/bells_aggregated_with_material.csv`.
3. **The two St. Stephan recording sessions are assumed to be the same
   bell** — not independently confirmed.
4. **T60 estimates saturate near the 30 s cap for several churches** —
   the 4-second post-onset capture window was too short to observe true
   decay for some bronze bells. Treat T60 values as lower bounds, not
   precise measurements.
5. Sample size (12 churches with a known, valid casting year) is still
   small — correlation and classification results are not statistically
   conclusive. Note: fixing St. Martin changed which classifier looked
   best (KNN now ties the majority baseline; previously it trailed it
   and the MLP led) — a concrete example of how unstable results are at
   this sample size.

## AI Transparency

See the "AI Transparency" section in the term paper for the full,
required disclosure of how Claude (Anthropic, Sonnet 4.6) was used in
this project's coding, analysis, and writing stages.
