"""
aggregates strike features per church, joins with the metadata,
runs the correlation + KNN/MLP classification.

note: st_martin and st_nicholas have way more raw strikes than the
rest so they get capped to the top 15 by strength. st_stephan's two
recording sessions get merged since they're the same bell (avoids
counting it twice / leaking across CV folds). imputation + scaling
are fit inside each LOOCV fold, not on the whole dataset beforehand.
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # parent of 02_code = COMPLETE_PROJECT
RESULTS_DIR = os.path.join(ROOT, "04_results")
MAX_STRIKES_PER_CHURCH = 15

# casting year / material / founder from the Kunstdenkmaeler von Bayern
# research, coordinates are approximate landmark locations
METADATA = {
    "bamberg_cathedral": dict(
        church="Bamberg Cathedral (Dom)",
        bell_name="Heinrichsglocke",
        casting_year=1311,
        material="bronze",
        founder="unknown",
        lat=49.8907306,
        lon=10.8825203,
        certainty="documented church location"
    ),
    "dormplatz": dict(
        church="Domplatz recording location (exact bell unverified)",
        bell_name="Kunigundenglocke",
        casting_year=1200,
        material="bronze",
        founder="unknown",
        lat=49.8920,
        lon=10.8810,
        certainty="field-recording location approximate; exact bell attribution unverified"
    ),
    "st_jakob": dict(
        church="St. Jakob",
        bell_name="Jakobusglocke",
        casting_year=1350,
        material="bronze",
        founder="unknown",
        lat=49.8909361,
        lon=10.8769569,
        certainty="documented church location"
    ),
    "obere_pfarre": dict(
        church="Obere Pfarrkirche",
        bell_name="Türkenglocke",
        casting_year=1521,
        material="bronze",
        founder="Hans Zeitlos",
        lat=49.8893442,
        lon=10.8844125,
        certainty="documented church location"
    ),
    "gruner_markt": dict(
        church="St. Martin (Grüner Markt)",
        bell_name="Schutzengelglocke",
        casting_year=1628,
        material="bronze",
        founder="Johannes Kopp",
        lat=49.8935464,
        lon=10.8882511,
        certainty="documented church location; re-recorded after duplicate-file issue"
    ),
    "pestallozistrasse": dict(
        church="Auferstehungskirche (Pestalozzistraße)",
        bell_name="Credoglocke",
        casting_year=1960,
        material="bronze",
        founder="Fa. Rincker",
        lat=49.9107283,
        lon=10.9152134,
        certainty="documented location: Pestalozzistraße 25/27"
    ),
    "evangelical_church": dict(
        church="Erlöserkirche Bamberg",
        bell_name="recorded peal (individual bell not isolated)",
        casting_year=1958,
        material="bronze",
        founder="Friedrich Wilhelm Schilling",
        lat=49.89479,
        lon=10.89696,
        certainty="documented church location and four-bell peal cast in 1958"
    ),
    "karmelite_kloster": dict(
        church="Karmelitenkirche St. Maria und St. Theodor",
        bell_name="unnamed",
        casting_year=1921,
        material="cast steel",
        founder="Bochumer Verein für Gussstahlfabrikation",
        lat=49.8883239,
        lon=10.8809350,
        certainty="documented church location"
    ),
    "st_michael": dict(
        church="St. Michael",
        bell_name="Michaelsglocke",
        casting_year=1614,
        material="bronze",
        founder="Hans Pfeffer",
        lat=49.8936011,
        lon=10.8774400,
        certainty="documented church location"
    ),
    "st_heinrich": dict(
        church="St. Heinrich",
        bell_name="unnamed",
        casting_year=1956,
        material="cast steel",
        founder="Bochumer Verein für Gussstahlfabrikation",
        lat=49.904081,
        lon=10.909031,
        certainty="documented independent parish church at Eugen-Pacelli-Platz 1; corrected from St. Michael attribution"
    ),
    "st_stephan": dict(
        church="St. Stephan",
        bell_name="recorded bell/peal (session 1)",
        casting_year=1200,
        material="bronze",
        founder="unknown",
        lat=49.8883481,
        lon=10.8861836,
        certainty="documented church location; two sessions merged for analysis"
    ),
    "st_stephan_2": dict(
        church="St. Stephan",
        bell_name="recorded bell/peal (session 2)",
        casting_year=1200,
        material="bronze",
        founder="unknown",
        lat=49.8883481,
        lon=10.8861836,
        certainty="documented church location; two sessions merged for analysis"
    ),
}

FEATURE_COLS = ["f_hum", "f_prime", "f_tierce", "f_quint", "f_nominal",
                 "tierce_cents", "t60", "spectral_centroid",
                 "spectral_bandwidth", "spectral_rolloff", "spectral_flatness",
                 "tuning_deviation_cents"]

# The two St. Stephan recording sessions are the same physical bell
# (see Section III, Data Quality Correction). They must be merged into
# one bell-level observation before any statistics are computed,
# otherwise the same bell is counted twice and can leak across the
# train/test split in Leave-One-Out CV.
BELL_ID_MERGE = {"st_stephan_2": "st_stephan"}


def add_per_strike_tuning_deviation(df):
    """Tuning deviation must be computed per strike, then averaged,
    not averaged first and then converted to a deviation. Computing it
    on the already-averaged tierce_cents would let strikes that are
    off in opposite directions cancel out in the mean."""
    def dev(row):
        if pd.isna(row["tierce_cents"]):
            return np.nan
        return min(abs(row["tierce_cents"] - 300), abs(row["tierce_cents"] - 400))
    df["tuning_deviation_cents"] = df.apply(dev, axis=1)
    return df


def load_manifest_strength():
    with open(os.path.join(ROOT, "01_data/strikes_segmented/manifest.json")) as f:
        manifest = json.load(f)
    return {(m["bell_id"], os.path.splitext(os.path.basename(m["file"]))[0]): m["strength"] for m in manifest}


def cap_and_aggregate(df):
    df = df.copy()

    # attach strength using the ORIGINAL bell_id (manifest.json keys are
    # per recording session, e.g. "st_stephan_2", not per physical bell)
    strength_map = load_manifest_strength()

    def get_strength(row):
        key = (row["bell_id"], os.path.splitext(os.path.basename(row["file"]))[0])
        return strength_map.get(key, np.nan)

    df["strength"] = df.apply(get_strength, axis=1)

    # Now merge the two St. Stephan sessions into one bell-level id,
    # so they are grouped and capped together as a single observation
    # rather than counted as two independent ones.
    df["bell_id"] = df["bell_id"].replace(BELL_ID_MERGE)

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


def run_classification(merged):
    """Pre/post-1900 classification: classical KNN baseline vs. a small
    MLP ('deep learning') model, both under Leave-One-Out CV."""
    valid = merged.dropna(subset=["casting_year"]).copy()
    valid = valid[~valid["certainty"].str.contains("DATA QUALITY ISSUE", na=False)]
    valid["label"] = (valid["casting_year"] >= 1900).astype(int)

    feat_cols = ["f_prime", "tierce_cents", "t60", "spectral_centroid", "spectral_bandwidth"]
    X = valid[feat_cols].copy().values
    y = valid["label"].values

    if len(valid) < 4 or len(set(y)) < 2:
        return {"note": "insufficient class diversity for classification", "n": int(len(valid))}

    loo = LeaveOneOut()

    def loo_accuracy(make_model):
        # Imputation and standardisation are fit ONLY on the training
        # fold each time, inside a Pipeline, then applied to the held
        # out test point. Fitting the scaler on the full dataset before
        # the loop (as an earlier version of this script did) leaks
        # information about the test point into training.
        preds = []
        for train_idx, test_idx in loo.split(X):
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("clf", make_model()),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            preds.append(pipe.predict(X[test_idx])[0])
        return np.array(preds)

    knn_preds = loo_accuracy(lambda: KNeighborsClassifier(n_neighbors=3))
    mlp_preds = loo_accuracy(lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000,
                                                     random_state=42, early_stopping=False))

    def summarize(preds):
        acc = float(np.mean(preds == y))
        cm = confusion_matrix(y, preds, labels=[0, 1])
        post1900_recall = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() > 0 else float("nan")
        bal_acc = float(balanced_accuracy_score(y, preds))
        return {"loo_accuracy": acc, "balanced_accuracy": bal_acc,
                "post1900_recall": post1900_recall,
                "confusion_matrix": cm.tolist()}

    knn_summary = summarize(knn_preds)
    knn_summary["k"] = 3
    mlp_summary = summarize(mlp_preds)
    mlp_summary["architecture"] = "16-8 hidden units"

    return {
        "n_samples": int(len(valid)),
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "knn": knn_summary,
        "mlp_deep_learning": mlp_summary,
        "majority_class_baseline": float(max(np.bincount(y)) / len(y)),
    }


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "all_strikes_features.csv"))

    # Remove the unverified St. Nikolaus recording completely.
    df = df[df["bell_id"] != "st_nicholas_church"].copy()

    df = add_per_strike_tuning_deviation(df)
    agg, capped = cap_and_aggregate(df)
    capped.to_csv(
        os.path.join(RESULTS_DIR, "capped_strikes_features.csv"),
        index=False
    )

    merged = attach_metadata(agg)
    merged.to_csv(
        os.path.join(RESULTS_DIR, "bells_aggregated_with_material.csv"),
        index=False
    )

    correlations = run_correlations(merged)
    classification = run_classification(merged)

    model_results = {
        "correlations_casting_year_vs_feature": correlations,
        "classification_pre_post_1900": classification,
        "n_churches": int(len(merged)),
        "max_strikes_per_church_used": MAX_STRIKES_PER_CHURCH,
    }

    with open(
        os.path.join(RESULTS_DIR, "model_results.json"),
        "w"
    ) as f:
        json.dump(model_results, f, indent=2)

    print(json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()

