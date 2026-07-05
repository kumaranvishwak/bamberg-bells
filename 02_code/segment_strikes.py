"""
onset detection on the RMS envelope to cut individual strikes out of
the long recordings, exports each as its own wav clip

only needed for churches that weren't already segmented:
obere_pfarre, st_nicholas_church, karmelite_kloster, st__Heinrich_,
st_stephan (original session)
"""
import os
import json
import numpy as np
import librosa
import soundfile as sf

SR = 48000
PRE_PAD = 0.10     # seconds before onset
POST_WINDOW = 4.0  # seconds after onset to capture decay
MIN_GAP = 0.8      # minimum seconds between detected strikes

def detect_strikes(y, sr):
    """Onset detection via librosa, backed by an RMS-based peak check."""
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units='time',
        backtrack=True, pre_max=20, post_max=20,
        pre_avg=50, post_avg=50, delta=0.07, wait=int(MIN_GAP * sr / 512)
    )
    # strength = RMS in a short window right after onset
    strikes = []
    for t in onset_frames:
        start_sample = int(t * sr)
        end_sample = min(len(y), start_sample + int(0.3 * sr))
        if end_sample <= start_sample:
            continue
        rms = float(np.sqrt(np.mean(y[start_sample:end_sample] ** 2)))
        strikes.append((float(t), rms))
    return strikes

def segment_file(in_path, bell_id, out_dir):
    y, sr = librosa.load(in_path, sr=SR, mono=True)
    duration = len(y) / sr
    strikes = detect_strikes(y, sr)
    if not strikes:
        return []

    max_rms = max(s[1] for s in strikes) or 1.0
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []
    for i, (t, rms) in enumerate(strikes, start=1):
        start_t = max(0.0, t - PRE_PAD)
        end_t = min(duration, t + POST_WINDOW)
        if end_t - start_t < 0.5:
            continue
        start_s = int(start_t * sr)
        end_s = int(end_t * sr)
        clip = y[start_s:end_s]
        fname = f"{bell_id}_strike{i:03d}.wav"
        sf.write(os.path.join(out_dir, fname), clip, sr)
        manifest_rows.append({
            "bell_id": bell_id,
            "strike_num": i,
            "file": fname,
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "strength": round(rms / max_rms, 3)
        })
    return manifest_rows

TARGETS = {
    "obere_pfarre": "01_data/recordings_original/obere_pfarre.flac",
    "st_nicholas_church": "01_data/recordings_original/st_nicholas_church.flac",
    "karmelite_kloster": "01_data/recordings_original/karmelite_kloster.flac",
    "st_heinrich": "01_data/recordings_original/st__Heinrich_.flac",
    "st_stephan": "01_data/recordings_original/st_stephan.flac",
}

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(SCRIPT_DIR)  # parent of 02_code = COMPLETE_PROJECT
    OUT_BASE = os.path.join(ROOT, "01_data/strikes_segmented")
    all_new = []
    for bell_id, rel_path in TARGETS.items():
        in_path = os.path.join(ROOT, rel_path)
        out_dir = os.path.join(OUT_BASE, bell_id)
        rows = segment_file(in_path, bell_id, out_dir)
        print(f"{bell_id}: {len(rows)} strikes detected -> {out_dir}")
        all_new.extend(rows)

    # merge with existing manifest
    manifest_path = os.path.join(OUT_BASE, "manifest.json")
    with open(manifest_path) as f:
        existing = json.load(f)
    existing.extend(all_new)
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nTotal manifest entries now: {len(existing)}")
