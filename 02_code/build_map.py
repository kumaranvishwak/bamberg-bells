"""
Bamberg Church Bells - Interactive Map Builder
=================================================
Builds a single, self-contained HTML file with a Leaflet map. Each
church is a marker; clicking it opens a popup with metadata and an
embedded, playable audio sample (base64-encoded mp3, so the file
works offline except for the OpenStreetMap basemap tiles).
"""
import os
import json
import base64
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # parent of 02_code = COMPLETE_PROJECT
RESULTS_DIR = os.path.join(ROOT, "04_results")
AUDIO_DIR = os.path.join(ROOT, "05_map/map_audio")
SPECTROGRAM_DIR = os.path.join(ROOT, "03_spectrograms/per_church")
OUT_PATH = os.path.join(ROOT, "05_map/bamberg_bells_map.html")


def b64_audio(bell_id):
    path = os.path.join(AUDIO_DIR, f"{bell_id}_sample.mp3")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def format_partials(row):
    names = [("f_hum", "Hum"), ("f_prime", "Prime"), ("f_tierce", "Tierce"),
             ("f_quint", "Quint"), ("f_nominal", "Nominal")]
    parts = []
    for col, label in names:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            parts.append(f"{label} {val:.0f} Hz")
        else:
            parts.append(f"{label} n/a")
    return " &middot; ".join(parts)


def build_markers_js(merged):
    markers = []
    for _, row in merged.iterrows():
        audio_b64 = b64_audio(row["bell_id"])
        audio_html = (
            f'<audio controls style="width:200px"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mpeg"></audio>'
            if audio_b64 else "<i>no audio sample</i>"
        )
        spec_html = (
            f'<br><span style="font-size:11px;color:#444">{format_partials(row)}</span>'
        )
        tuning_dev = "n/a" if pd.isna(row.get("tuning_deviation_cents")) else f"{row['tuning_deviation_cents']:.1f}&#162;"
        year = "unknown / provisional" if pd.isna(row["casting_year"]) else int(row["casting_year"])
        nominal = "n/a" if pd.isna(row["f_nominal"]) else f"{row['f_nominal']:.0f} Hz"
        t60 = "n/a" if pd.isna(row["t60"]) else f"{row['t60']:.1f} s"
        popup = (
            f"<b>{row['church']}</b><br>"
            f"Bell: {row['bell_name']}<br>"
            f"Casting year: {year} ({row['certainty']})<br>"
            f"Material: {row['material']}, founder: {row['founder']}<br>"
            f"Nominal: {nominal} | T60: {t60} | Tuning dev.: {tuning_dev}<br>"
            f"{audio_html}"
            f"{spec_html}"
        ).replace("'", "\\'").replace("\n", "")
        markers.append(
            f"L.marker([{row['lat']}, {row['lon']}]).addTo(map)"
            f".bindPopup('{popup}', {{maxWidth: 260}});"
        )
    return "\n".join(markers)


def main():
    merged = pd.read_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"))
    markers_js = build_markers_js(merged)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bamberg Church Bells - Spatio-Temporal Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  .legend {{ position: absolute; bottom: 20px; left: 20px; z-index: 1000;
             background: white; padding: 10px; border-radius: 6px;
             font-family: sans-serif; font-size: 13px; max-width: 260px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  <b>Bamberg Church Bells</b><br>
  Click a marker to see casting year, material, and hear the recorded strike.<br>
  <i>Provisional/unverified entries are marked in the popup.</i>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  var map = L.map('map').setView([49.892, 10.886], 14);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  {markers_js}
</script>
</body>
</html>
"""
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
