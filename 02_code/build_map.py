"""
Build the Leaflet map from the corrected bell-level results.
Each bell-level observation gets one marker with metadata and an
embedded audio sample. The map automatically fits all valid locations.
"""
import base64
import html
import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT, "04_results")
AUDIO_DIR = os.path.join(ROOT, "05_map", "map_audio")
OUT_PATH = os.path.join(ROOT, "05_map", "bamberg_bells_map.html")


def b64_audio(bell_id):
    path = os.path.join(AUDIO_DIR, f"{bell_id}_sample.mp3")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def clean(value, fallback="unknown"):
    if value is None or pd.isna(value):
        return fallback
    return html.escape(str(value))


def format_partials(row):
    names = [
        ("f_hum", "Hum"),
        ("f_prime", "Prime"),
        ("f_tierce", "Tierce"),
        ("f_quint", "Quint"),
        ("f_nominal", "Nominal"),
    ]
    parts = []
    for col, label in names:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            parts.append(f"{label} {value:.0f} Hz")
        else:
            parts.append(f"{label} n/a")
    return " &middot; ".join(parts)


def marker_style(material):
    value = str(material).strip().lower()
    if value == "cast steel":
        return "#d4a017", "#8a6700"
    if value == "bronze":
        return "#2b7bba", "#175783"
    return "#777777", "#444444"


def build_markers_js(mapped):
    markers = []

    for _, row in mapped.iterrows():
        bell_id = str(row["bell_id"])
        audio_b64 = b64_audio(bell_id)
        audio_html = (
            '<audio controls preload="none" style="width:230px">'
            f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mpeg">'
            "</audio>"
            if audio_b64
            else "<i>No audio sample included</i>"
        )

        year = (
            "unknown / provisional"
            if pd.isna(row.get("casting_year"))
            else str(int(row["casting_year"]))
        )
        nominal = (
            "n/a" if pd.isna(row.get("f_nominal"))
            else f"{row['f_nominal']:.0f} Hz"
        )
        t60 = (
            "n/a" if pd.isna(row.get("t60"))
            else f"{row['t60']:.1f} s"
        )
        tuning_dev = (
            "n/a" if pd.isna(row.get("tuning_deviation_cents"))
            else f"{row['tuning_deviation_cents']:.1f} cents"
        )

        popup = (
            f"<b>{clean(row.get('church'))}</b><br>"
            f"Bell: {clean(row.get('bell_name'))}<br>"
            f"Casting year: {year}<br>"
            f"Material: {clean(row.get('material'))}<br>"
            f"Founder: {clean(row.get('founder'))}<br>"
            f"Status: {clean(row.get('certainty'))}<br>"
            f"Nominal: {nominal} | T60: {t60}<br>"
            f"Tuning deviation: {tuning_dev}<br>"
            f"{audio_html}<br>"
            f'<span style="font-size:11px;color:#444">{format_partials(row)}</span>'
        )

        fill_color, border_color = marker_style(row.get("material"))
        tooltip = clean(row.get("church"))

        markers.append(
            "L.circleMarker("
            f"[{float(row['lat'])}, {float(row['lon'])}], "
            "{radius: 8, weight: 2, opacity: 1, fillOpacity: 0.9, "
            f"color: {json.dumps(border_color)}, fillColor: {json.dumps(fill_color)}}}"
            ").addTo(markerGroup)"
            f".bindPopup({json.dumps(popup)}, {{maxWidth: 300}})"
            f".bindTooltip({json.dumps(tooltip)}, "
            "{direction: 'top', offset: [0, -8], sticky: true});"
        )

    return "\n".join(markers)


def main():
    results_path = os.path.join(
        RESULTS_DIR, "bells_aggregated_with_material.csv"
    )
    merged = pd.read_csv(results_path)

    mapped = merged.dropna(subset=["lat", "lon"]).copy()
    if mapped.empty:
        raise RuntimeError("No valid latitude/longitude values found.")

    markers_js = build_markers_js(mapped)
    total_records = int(merged["n_strikes_total"].notna().sum())
    marker_count = len(mapped)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bamberg Church Bells - Spatio-Temporal Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  .legend {{
    position: absolute;
    bottom: 20px;
    left: 20px;
    z-index: 1000;
    background: rgba(255,255,255,0.96);
    padding: 11px 13px;
    border-radius: 7px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.25);
    font-family: sans-serif;
    font-size: 13px;
    line-height: 1.4;
    max-width: 290px;
  }}
  .dot {{
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: -1px;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  <b>Bamberg Church Bells</b><br>
  {marker_count} bell-level markers derived from 12 retained recording entries.<br>
  Hover for the church name; click for metadata and audio.<br>
  <span class="dot" style="background:#2b7bba"></span>Bronze&nbsp;&nbsp;
  <span class="dot" style="background:#d4a017"></span>Cast steel<br>
  <i>Provisional entries are identified in the popup.</i>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map');
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }}).addTo(map);

  const markerGroup = L.featureGroup().addTo(map);
  {markers_js}

  map.fitBounds(markerGroup.getBounds().pad(0.12));
</script>
</body>
</html>
"""

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(
        f"Wrote {OUT_PATH} "
        f"({os.path.getsize(OUT_PATH) / 1024:.0f} KB, "
        f"{marker_count} markers)"
    )


if __name__ == "__main__":
    main()
