"""
Bamberg Church Bells - Aggregation, Statistics & Classification
==================================================================
1. Caps over-represented churches (st_nicholas_church has 285 raw
   strikes from a long peal) to a representative top-N by strike
   strength, so no single church dominates the per-church averages.
2. Aggregates strike-level features into one row per church/bell.
3. Joins with metadata (casting year, material, coordinates).
4. Runs:
     - Pearson correlation: casting year vs. each acoustic feature
     - Classical ML baseline: KNN, Leave-One-Bell-Out CV
     - "Deep learning" extension: a small MLP (multi-layer
       perceptron) neural network, same CV protocol, for direct
       comparison (course week 9-11 extension topic)
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = "/home/claude/project/COMPLETE_PROJECT"
RESULTS_DIR = os.path.join(ROOT, "04_results")
MAX_STRIKES_PER_CHURCH = 15

# ---------------------------------------------------------------
# Metadata. Casting years / material / founder carried over from
# earlier historical-PDF research in this project; coordinates are
# approximate (public landmark locations in Bamberg). Two entries
# are flagged "provisional" pending source re-verification.
# ---------------------------------------------------------------
METADATA = {
    "bamberg_cathedral":  dict(church="Bamberg Cathedral (Dom)", bell_name="Heinrichsglocke",
                                casting_year=1311, material="bronze", founder="unknown",
                                lat=49.8917, lon=10.8817, certainty="documented"),
    "dormplatz":          dict(church="Dormplatz / Obere Pfarre area bell", bell_name="Kunigundenglocke",
                                casting_year=1200, material="bronze", founder="unknown",
                                lat=49.8920, lon=10.8810, certainty="approximate"),
    "st_jakob":           dict(church="St. Jakob", bell_name="Jakobusglocke",
                                casting_year=1350, material="bronze", founder="unknown",
                                lat=49.8932, lon=10.8857, certainty="approximate"),
    "obere_pfarre":       dict(church="Obere Pfarrkirche", bell_name="T\u00fcrkenglocke",
                                casting_year=1521, material="bronze", founder="Hans Zeitlos",
                                lat=49.8924, lon=10.8839, certainty="documented"),
    "gruner_markt":       dict(church="St. Martin (Gr\u00fcner Markt)", bell_name="Schutzengelglocke",
                                casting_year=1628, material="bronze", founder="Johannes Kopp",
                                lat=49.8918, lon=10.8868, certainty="documented (re-recorded after duplicate-file issue)"),
    "pestallozistrasse":  dict(church="Auferstehungskirche (Pestalozzistra\u00dfe)", bell_name="Credoglocke",
                                casting_year=1960, material="bronze", founder="Fa. Rincker",
                                lat=49.8794, lon=10.8989, certainty="documented"),
    "evangelical_church": dict(church="St. Stephan (evangelical)", bell_name="Evangelistenglocke",
                                casting_year=1200, material="bronze", founder="unknown",
                                lat=49.8941, lon=10.8829, certainty="approximate"),
    "karmelite_kloster":  dict(church="Karmelitenkirche", bell_name="unnamed",
                                casting_year=1921, material="cast steel", founder="Bochumer Verein f\u00fcr Gussstahlfabrikation",
                                lat=49.8908, lon=10.8801, certainty="documented"),
    "st_michael":         dict(church="St. Michael", bell_name="Michaelsglocke",
                                casting_year=1614, material="bronze", founder="Hans Pfeffer",
                                lat=49.8932, lon=10.8776, certainty="documented"),
    "st_heinrich":        dict(church="St. Michael", bell_name="Heinrichsglocke (St. Michael)",
                                casting_year=1794, material="bronze", founder="unknown",
                                lat=49.8932, lon=10.8776, certainty="documented"),
    "st_nicholas_church": dict(church="St. Nikolaus", bell_name="unnamed",
                                casting_year=None, material="bronze", founder="unknown",
                                lat=49.8870, lon=10.8780, certainty="PROVISIONAL - unverified"),
    "st_stephan":         dict(church="St. Stephan", bell_name="unnamed (session 1)",
                                casting_year=1200, material="bronze", founder="unknown",
                                lat=49.8941, lon=10.8829, certainty="PROVISIONAL - assumed same as evangelical_church"),
    "st_stephan_2":       dict(church="St. Stephan", bell_name="unnamed (session 2)",
                                casting_year=1200, material="bronze", founder="unknown",
                                lat=49.8941, lon=10.8829, certainty="PROVISIONAL - assumed same as evangelical_church"),
}

FEATURE_COLS = ["f_hum", "f_prime", "f_tierce", "f_quint", "f_nominal",
                 "tierce_cents", "t60", "spectral_centroid",
                 "spectral_bandwidth", "spectral_rolloff", "spectral_flatness"]


def load_manifest_strength():
    with open(os.path.join(ROOT, "01_data/strikes_segmented/manifest.json")) as f:
        manifest = json.load(f)
    return {(m["bell_id"], m["file"].replace(".wav", "")): m["strength"] for m in manifest}


def cap_and_aggregate(df):
    # attach strength where available, fall back to spectral flatness rank otherwise
    strength_map = load_manifest_strength()

    def get_strength(row):
        key = (row["bell_id"], row["file"].replace(".flac", "").replace(".wav", ""))
        return strength_map.get(key, np.nan)

    df["strength"] = df.apply(get_strength, axis=1)

    capped_rows = []
    for bell_id, group in df.groupby("bell_id"):
        if len(group) > MAX_STRIKES_PER_CHURCH:
            if group["strength"].notna().any():
                group = group.sort_values("strength", ascending=False)
            group = group.head(MAX_STRIKES_PER_CHURCH)
        capped_rows.append(group)
    capped = pd.concat(capped_rows, ignore_index=True)

    agg = capped.groupby("bell_id")[FEATURE_COLS].mean(numeric_only=True)
    agg["n_strikes_used"] = capped.groupby("bell_id").size()
    agg["n_strikes_total"] = df.groupby("bell_id").size()

    # tierce type = majority vote
    tierce_mode = capped.groupby("bell_id")["tierce_type"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    agg["tierce_type"] = tierce_mode
    return agg.reset_index(), capped


def attach_metadata(agg):
    meta_df = pd.DataFrame.from_dict(METADATA, orient="index").reset_index().rename(columns={"index": "bell_id"})
    merged = agg.merge(meta_df, on="bell_id", how="left")
    return merged


def run_correlations(merged):
    results = {}
    valid = merged.dropna(subset=["casting_year"])
    valid = valid[~valid["certainty"].str.contains("DATA QUALITY ISSUE", na=False)]
    for col in ["f_hum", "f_prime", "f_tierce", "f_quint", "f_nominal", "tierce_cents", "t60",
                 "spectral_centroid", "spectral_bandwidth"]:
        sub = valid.dropna(subset=[col])
        if len(sub) < 4:
            continue
        r, p = pearsonr(sub["casting_year"], sub[col])
        results[col] = {"r": float(r), "p": float(p), "n": int(len(sub))}
    return results


def deviation_in_cents(merged):
    """Mean absolute deviation of tierce_cents from the nearer canonical
    interval (300 cents minor third / 400 cents major third)."""
    def dev(row):
        if pd.isna(row["tierce_cents"]):
            return np.nan
        return min(abs(row["tierce_cents"] - 300), abs(row["tierce_cents"] - 400))
    merged["tuning_deviation_cents"] = merged.apply(dev, axis=1)
    return merged


def run_classification(merged):
    """Pre/post-1900 classification: classical KNN baseline vs. a small
    MLP ('deep learning') model, both under Leave-One-Out CV."""
    valid = merged.dropna(subset=["casting_year"]).copy()
    valid = valid[~valid["certainty"].str.contains("DATA QUALITY ISSUE", na=False)]
    valid["label"] = (valid["casting_year"] >= 1900).astype(int)

    feat_cols = ["f_prime", "tierce_cents", "t60", "spectral_centroid", "spectral_bandwidth"]
    X = valid[feat_cols].copy()
    X = X.fillna(X.mean())
    y = valid["label"].values

    if len(valid) < 4 or len(set(y)) < 2:
        return {"note": "insufficient class diversity for classification", "n": int(len(valid))}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo = LeaveOneOut()

    def loo_accuracy(model):
        preds = []
        for train_idx, test_idx in loo.split(X_scaled):
            model.fit(X_scaled[train_idx], y[train_idx])
            preds.append(model.predict(X_scaled[test_idx])[0])
        return float(np.mean(np.array(preds) == y)), preds

    knn = KNeighborsClassifier(n_neighbors=3)
    knn_acc, knn_preds = loo_accuracy(knn)

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000, random_state=42,
                         early_stopping=False)
    mlp_acc, mlp_preds = loo_accuracy(mlp)

    return {
        "n_samples": int(len(valid)),
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "knn": {"loo_accuracy": knn_acc, "k": 3},
        "mlp_deep_learning": {"loo_accuracy": mlp_acc, "architecture": "16-8 hidden units"},
        "majority_class_baseline": float(max(np.bincount(y)) / len(y)),
    }


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "all_strikes_features.csv"))
    agg, capped = cap_and_aggregate(df)
    capped.to_csv(os.path.join(RESULTS_DIR, "capped_strikes_features.csv"), index=False)

    merged = attach_metadata(agg)
    merged = deviation_in_cents(merged)
    merged.to_csv(os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"), index=False)

    correlations = run_correlations(merged)
    classification = run_classification(merged)

    model_results = {
        "correlations_casting_year_vs_feature": correlations,
        "classification_pre_post_1900": classification,
        "n_churches": int(len(merged)),
        "max_strikes_per_church_used": MAX_STRIKES_PER_CHURCH,
    }
    with open(os.path.join(RESULTS_DIR, "model_results.json"), "w") as f:
        json.dump(model_results, f, indent=2)

    print(json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()
