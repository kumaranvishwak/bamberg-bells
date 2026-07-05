# Bamberg Church Bells

Machine Listening project (CH-Proj-M), SS 2026, Uni Bamberg.
Kumaran Vasu, Srujan Mahajan. Supervisor: Prof. Abesser.

Field recordings of 13 church bells in Bamberg, feature extraction
(FFT partials, T60, spectral stuff), KNN vs small MLP for pre/post
1900 classification, and an interactive map. Paper has the actual
writeup and results, this is just the code.

## folders

- `02_code/` - the scripts, in order: segment_strikes -> extract_features
  -> run_analysis -> make_figures / build_map
- `03_spectrograms/` - spectrogram pngs, plus `demo_peaks/` (labeled
  peaks + short audio clips per church for presenting)
- `04_results/` - csvs and model_results.json
- `05_map/` - the leaflet map html

raw audio isn't in git (too big, see .gitignore).

## running it

```
pip install librosa soundfile pandas numpy scipy scikit-learn matplotlib --break-system-packages
python3 02_code/extract_features.py
python3 02_code/run_analysis.py
python3 02_code/make_figures.py
python3 02_code/build_map.py
```

## known issues

- St. Martin recording was originally a duplicate of the cathedral one
  (same file basically), had to re-record it
- St. Nikolaus has no confirmed casting year
- St. Stephan's two recording sessions are the same bell, merged
  before doing any stats on them
- there was also a mixup where st_heinrich got filed under St. Michael
  in the metadata, it's actually its own church, fixed now
- 11 churches with a confirmed year is still a small sample so don't
  read too much into the classification numbers

## AI use

Used Claude for the pipeline code, debugging, and drafting/editing the
paper. Full disclosure is in the paper's AI Transparency section.
