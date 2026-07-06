"""
Build the final Leaflet map from the corrected bell-level results.

The generated HTML is written both to:
- 05_map/bamberg_bells_map.html
- docs/index.html for GitHub Pages
"""

import base64
import html
import json
import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(
    ROOT,
    "04_results",
)

AUDIO_DIR = os.path.join(
    ROOT,
    "05_map",
    "map_audio",
)

OUT_PATH = os.path.join(
    ROOT,
    "05_map",
    "bamberg_bells_map.html",
)

DOCS_DIR = os.path.join(
    ROOT,
    "docs",
)

DOCS_PATH = os.path.join(
    DOCS_DIR,
    "index.html",
)

RETAINED_RECORDING_ENTRIES = 12


def b64_audio(bell_id):
    """Return one MP3 sample encoded for embedding in HTML."""
    audio_path = os.path.join(
        AUDIO_DIR,
        f"{bell_id}_sample.mp3",
    )

    if not os.path.exists(audio_path):
        return None

    with open(
        audio_path,
        "rb",
    ) as audio_file:
        return base64.b64encode(
            audio_file.read()
        ).decode("ascii")


def clean(value, fallback="unknown"):
    """Safely display metadata inside HTML."""
    if value is None or pd.isna(value):
        return fallback

    return html.escape(str(value))


def format_partials(row):
    """Format the five classical partial frequencies."""
    partials = [
        ("f_hum", "Hum"),
        ("f_prime", "Prime"),
        ("f_tierce", "Tierce"),
        ("f_quint", "Quint"),
        ("f_nominal", "Nominal"),
    ]

    output = []

    for column, label in partials:
        value = row.get(column)

        if value is not None and not pd.isna(value):
            output.append(
                f"{label} {value:.0f} Hz"
            )
        else:
            output.append(
                f"{label} n/a"
            )

    return " &middot; ".join(output)


def marker_style(material):
    """Return marker fill and border colours."""
    material_value = str(
        material
    ).strip().lower()

    if material_value == "cast steel":
        return "#d4a017", "#8a6700"

    if material_value == "bronze":
        return "#2b7bba", "#175783"

    return "#777777", "#444444"


def build_markers_js(mapped):
    """Build the Leaflet JavaScript for all map markers."""
    markers = []

    for _, row in mapped.iterrows():
        bell_id = str(row["bell_id"])
        audio_data = b64_audio(bell_id)

        if audio_data:
            audio_html = (
                '<audio controls preload="none" '
                'style="width:230px">'
                f'<source src="data:audio/mp3;base64,'
                f'{audio_data}" type="audio/mpeg">'
                "</audio>"
            )
        else:
            audio_html = (
                "<i>No audio sample included</i>"
            )

        if pd.isna(row.get("casting_year")):
            casting_year = (
                "unknown / provisional"
            )
        else:
            casting_year = str(
                int(row["casting_year"])
            )

        if pd.isna(row.get("f_nominal")):
            nominal = "n/a"
        else:
            nominal = (
                f"{row['f_nominal']:.0f} Hz"
            )

        if pd.isna(row.get("t60")):
            t60 = "n/a"
        else:
            t60 = (
                f"{row['t60']:.1f} s"
            )

        if pd.isna(
            row.get(
                "tuning_deviation_cents"
            )
        ):
            tuning_deviation = "n/a"
        else:
            tuning_deviation = (
                f"{row['tuning_deviation_cents']:.1f} cents"
            )

        popup = (
            f"<b>{clean(row.get('church'))}</b><br>"
            f"Bell: {clean(row.get('bell_name'))}<br>"
            f"Casting year: {casting_year}<br>"
            f"Material: {clean(row.get('material'))}<br>"
            f"Founder: {clean(row.get('founder'))}<br>"
            f"Status: {clean(row.get('certainty'))}<br>"
            f"Nominal: {nominal} | T60: {t60}<br>"
            f"Tuning deviation: {tuning_deviation}<br>"
            f"{audio_html}<br>"
            '<span style="font-size:11px;color:#444">'
            f"{format_partials(row)}"
            "</span>"
        )

        fill_color, border_color = marker_style(
            row.get("material")
        )

        tooltip = clean(
            row.get("church")
        )

        marker_javascript = (
            "L.circleMarker("
            f"[{float(row['lat'])}, "
            f"{float(row['lon'])}], "
            "{"
            "radius: 8, "
            "weight: 2, "
            "opacity: 1, "
            "fillOpacity: 0.9, "
            f"color: {json.dumps(border_color)}, "
            f"fillColor: {json.dumps(fill_color)}"
            "}"
            ").addTo(markerGroup)"
            f".bindPopup("
            f"{json.dumps(popup)}, "
            "{maxWidth: 300}"
            ")"
            f".bindTooltip("
            f"{json.dumps(tooltip)}, "
            "{"
            "direction: 'top', "
            "offset: [0, -8], "
            "sticky: true"
            "}"
            ");"
        )

        markers.append(
            marker_javascript
        )

    return "\n".join(markers)


def main():
    results_path = os.path.join(
        RESULTS_DIR,
        "bells_aggregated_with_material.csv",
    )

    merged = pd.read_csv(
        results_path
    )

    mapped = merged.dropna(
        subset=["lat", "lon"]
    ).copy()

    if mapped.empty:
        raise RuntimeError(
            "No valid latitude/longitude values found."
        )

    marker_count = len(mapped)
    markers_javascript = build_markers_js(
        mapped
    )

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Bamberg Church Bells - Spatio-Temporal Map
</title>

<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

<style>
html,
body,
#map {{
    height: 100%;
    margin: 0;
}}

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

    {marker_count} bell-level markers derived from
    {RETAINED_RECORDING_ENTRIES} retained recording entries.<br>

    Hover for the church name; click for metadata and audio.<br>

    <span class="dot"
          style="background:#2b7bba"></span>
    Bronze&nbsp;&nbsp;

    <span class="dot"
          style="background:#d4a017"></span>
    Cast steel<br>

    <i>
    Provisional entries are identified in the popup.
    </i>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<script>
const map = L.map('map');

L.tileLayer(
    'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }}
).addTo(map);

const markerGroup = L.featureGroup().addTo(map);

{markers_javascript}

map.fitBounds(
    markerGroup.getBounds().pad(0.12)
);
</script>

</body>
</html>
"""

    os.makedirs(
        os.path.dirname(OUT_PATH),
        exist_ok=True,
    )

    with open(
        OUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            html_document
        )

    os.makedirs(
        DOCS_DIR,
        exist_ok=True,
    )

    with open(
        DOCS_PATH,
        "w",
        encoding="utf-8",
    ) as docs_file:
        docs_file.write(
            html_document
        )

    nojekyll_path = os.path.join(
        DOCS_DIR,
        ".nojekyll",
    )

    open(
        nojekyll_path,
        "a",
        encoding="utf-8",
    ).close()

    size_kilobytes = (
        os.path.getsize(OUT_PATH) / 1024
    )

    print(
        f"Wrote {OUT_PATH} "
        f"({size_kilobytes:.0f} KB, "
        f"{marker_count} markers)"
    )

    print(
        f"Updated GitHub Pages copy: "
        f"{DOCS_PATH}"
    )


if __name__ == "__main__":
    main()