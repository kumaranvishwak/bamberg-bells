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
- `05_map/` - the leaflet map html, 12 bell-level markers (the two
  St. Stephan recording sessions are merged into one marker, so 13
  recording entries become 12 markers)

raw and segmented audio are not in this repo (too big, see
.gitignore). `04_results/all_strikes_features.csv` already has the
extracted per-strike features computed from that audio, so the
statistics and figures can be reproduced without the audio itself.

## running it

Install dependencies first:
```
pip install -r requirements.txt --break-system-packages
```

**To reproduce the statistics and figures** (this works with just
what's in this repo, no audio needed):
```
python3 02_code/run_analysis.py
python3 02_code/make_figures.py
python3 02_code/build_map.py
```

**To reproduce the full audio pipeline from scratch** (segmentation
and feature extraction), you need the original and segmented audio
files, which are not included here:
```
python3 02_code/segment_strikes.py
python3 02_code/extract_features.py
```

## known issues

- St. Martin recording was originally a duplicate of the cathedral one
  (same file basically), had to re-record it
- St. Nikolaus has no confirmed casting year
- St. Stephan's two recording sessions are the same bell, merged
  before doing any stats on them
- there was also a mixup where st_heinrich got filed under St. Michael
  in the metadata, it's actually its own church, fixed now
- found a real bug in the strength-based strike capping: the manifest
  matching code was stripping ".wav" but the manifest files are
  ".flac", so it silently never matched and fell back to picking the
  first 15 strikes instead of the 15 strongest for St. Martin and
  St. Nikolaus. Fixed by matching on the filename stem regardless of
  extension. This changed the aggregated features for both churches
  and, through them, the classification and correlation numbers, see
  the paper's Critical Discussion for the actual before/after
- 11 known-year observations is still a small sample so don't read
  too much into the classification numbers, they moved twice already
  from fixing real bugs

## AI use

Claude (Sonnet 4.6, Anthropic) was used for the pipeline code,
debugging, code review, and drafting/revising the paper. ChatGPT
(GPT-5.5 Thinking, OpenAI) was used separately for methodological
review, consistency checking across the paper, and catching two of
the bugs listed above. Full disclosure with representative prompts is
in the paper's AI Transparency section; this should match that
section, not add to it.
