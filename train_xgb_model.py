import glob
import json
import os
import pandas as pd
import requests
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GroupKFold, KFold, cross_val_score, RandomizedSearchCV

KOI_LABELS_CSV = "koi_labels.csv"
KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,koi_disposition+from+cumulative&format=csv"
)


def load_confirmed_kic_ids():
    """Real CONFIRMED-planet KIC IDs from the NASA Exoplanet Archive KOI table."""
    if not os.path.exists(KOI_LABELS_CSV):
        print("Fetching KOI dispositions from NASA Exoplanet Archive...")
        try:
            response = requests.get(KOI_TAP_URL, timeout=120)
            response.raise_for_status()
            with open(KOI_LABELS_CSV, "w") as handle:
                handle.write(response.text)
        except Exception as exc:
            print(f"Could not fetch KOI labels ({exc}). Falling back to score-based labeling.")
            return set()

    koi = pd.read_csv(KOI_LABELS_CSV)
    confirmed = koi.loc[koi["koi_disposition"] == "CONFIRMED", "kepid"]
    return set(confirmed.astype(str).str.zfill(9))


confirmed_kic_ids = load_confirmed_kic_ids()
print(f"Loaded {len(confirmed_kic_ids)} confirmed-planet KIC IDs")


def fetch_labelled_lightcurves(n_per_class, download_dir, files_per_kic=1):
    """Download light curves for known CONFIRMED and FALSE POSITIVE KICs."""
    import random
    import re
    from web.pipeline import BASE_URL, download_file, fetch_links

    koi = pd.read_csv(KOI_LABELS_CSV)
    koi["kic"] = koi["kepid"].astype(str).str.zfill(9)

    rng = random.Random(42)
    rows = []
    for disposition in ("CONFIRMED", "FALSE POSITIVE"):
        kics = sorted(set(koi.loc[koi["koi_disposition"] == disposition, "kic"]))
        rng.shuffle(kics)

        collected = 0
        for kic in kics:
            if collected >= n_per_class:
                break
            target_url = f"{BASE_URL}{kic[:4]}/{kic}/"
            try:
                urls = [u for u in fetch_links(target_url) if u.lower().endswith("_llc.fits")]
            except Exception:
                continue
            if not urls:
                continue

            for file_url in urls[:files_per_kic]:
                fname = os.path.basename(file_url)
                local_path = os.path.join(download_dir, fname)
                try:
                    if not os.path.exists(local_path):
                        download_file(file_url, local_path)
                    rows.append({"filename": fname, "local_path": os.path.abspath(local_path),
                                 "source_url": file_url, "disposition": disposition})
                    collected += 1
                    print(f"  [{disposition}] {collected}/{n_per_class}: {fname}")
                except Exception as exc:
                    print(f"  failed {fname}: {exc}")

    return pd.DataFrame(rows)


GENERATE_DATA = True
TARGETED_LABELS = True
N_PER_CLASS = 200

if GENERATE_DATA:
    print("GENERATING TRAINING DATA FROM KEPLER ARCHIVE")

    from web.pipeline import run_pipeline, fetch_kepler_llc_from_archive
    import os

    DOWNLOAD_DIR = "kepler_llc_downloads/"
    OUTPUT_CSV = "transit_results_labelled_targets.csv" if TARGETED_LABELS else "transit_results_100targets.csv"
    TOP_CANDIDATES_CSV = "top_candidates_100targets.csv"
    CAUTION_CANDIDATES_CSV = "caution_candidates_100targets.csv"

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if TARGETED_LABELS:
        print(f"\nDownloading {N_PER_CLASS} confirmed + {N_PER_CLASS} false-positive light curves...\n")
        downloads_df = fetch_labelled_lightcurves(N_PER_CLASS, DOWNLOAD_DIR)
    else:
        print("\nDownloading random Kepler light curves from archive...")
        print("(This may take 10-30 minutes depending on network speed)\n")
        downloads_df = fetch_kepler_llc_from_archive(
            target_count=100,
            download_dir=DOWNLOAD_DIR,
            max_buckets=50,
            randomize=True,
            random_seed=42
        )
    
    if len(downloads_df) == 0:
        print("No files downloaded. Check network connection.")
        GENERATE_DATA = False
    else:
        print(f"\nDownloaded {len(downloads_df)} light curves")
        
        targets = downloads_df['filename'].tolist()
        
        print(f"\nRunning pipeline on {len(targets)} targets...")
        print("(This may take 20-60 minutes depending on CPU)\n")
        
        results_df, failures, summary = run_pipeline(
            targets=targets,
            data_dir=DOWNLOAD_DIR,
            export_csv=True,
            output_csv=OUTPUT_CSV,
            show_plot=False,
            kernel_size=101,
            prominence=0.0002,
            top_candidates_csv=TOP_CANDIDATES_CSV,
            caution_candidates_csv=CAUTION_CANDIDATES_CSV,
            review_threshold=70.0,
            review_now_threshold=75.0,
            min_peaks=20,
            max_peaks=400,
            downloads_df=downloads_df
        )
        
        print(f"TRAINING DATA GENERATION COMPLETE")
        print(f"Processed: {summary['processed']} targets")
        print(f"Succeeded: {summary['succeeded']} targets")
        print(f"Failed: {summary['failed']} targets")
        print(f"Total candidate dips: {summary['total_candidate_dips']}")
        print(f"\nOutput files:")
        print(f"  - {OUTPUT_CSV} ({len(results_df)} rows)")
        print(f"  - {TOP_CANDIDATES_CSV}")
        print(f"  - {CAUTION_CANDIDATES_CSV}")
        print(f"  - Downloads in: {DOWNLOAD_DIR}")
        print(f"\nReady for training. Training will use this generated data.\n")
   
files = glob.glob("data/results/results_*.csv")
files += glob.glob("transit_results.csv")
files += glob.glob("transit_results_*targets.csv")
files = list(dict.fromkeys(files)) 
if not files:
    raise FileNotFoundError("No results CSV files were found.")

frames = []
for path in files:
    frame = pd.read_csv(path)
    print(path, frame.shape)
    frames.append(frame)

data = pd.concat(frames, ignore_index=True, sort=False)
data = data.drop_duplicates(subset="target", keep="last").reset_index(drop=True)
print(f"Combined dataset: {data.shape[0]} unique targets from {len(files)} files")

if "review_status" not in data.columns:
    raise ValueError("Missing review_status column for initial labeling.")

if "kic_id" not in data.columns:
    print("Extracting kic_id from target column...")
    data["kic_id"] = data["target"].astype(str).str.extract(r"kplr(\d+)")[0]

data["is_confirmed_planet"] = data["kic_id"].isin(
    confirmed_kic_ids
).astype(int)
print(f"Real-label matches: {int(data['is_confirmed_planet'].sum())} confirmed of {len(data)} targets")

if data["is_confirmed_planet"].nunique() < 2:
    if "final_ranking_score" not in data.columns:
        raise ValueError(
            "Only one label class from review_status and no final_ranking_score for fallback labeling."
        )

    data["final_ranking_score"] = pd.to_numeric(
        data["final_ranking_score"], errors="coerce"
    )
    data = data.dropna(subset=["final_ranking_score"]).copy()

    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 rows to create two classes.")

    k = max(1, int(round(0.30 * n)))
    top_idx = data["final_ranking_score"].sort_values(ascending=False).index[:k]

    data["is_confirmed_planet"] = 0
    data.loc[top_idx, "is_confirmed_planet"] = 1

    print(f"Applied forced fallback labeling: positives={k}, negatives={n-k}")
    
print("Label counts:")
print(data["is_confirmed_planet"].value_counts(dropna=False))

print("\nAvailable Columns")
print(data.columns.tolist())

FEATURES = [
    "num_peaks",
    "estimated_period_days",
    "period_stability_cv",
    "mean_transit_depth",
    "median_transit_depth",
    "max_transit_depth",
    "bls_period_days",
    "bls_duration_days",
    "bls_depth",
    "bls_power",
    "mean_detrended_flux",
    "std_detrended_flux",
    "kernel_size",
    "prominence",
    "transit_snr",
    "depth_consistency_cv",
    "valid_fraction",
    "observation_baseline_days",
    "period_agreement",
]


missing_columns = [col for col in FEATURES if col not in data.columns]
available_features = [col for col in FEATURES if col in data.columns]

if missing_columns:
    print(f"\nMissing columns (will fill with default -1): {missing_columns}")
    for col in missing_columns:
        data[col] = -1.0

print(f"Using {len(available_features)} available features out of {len(FEATURES)} total")

if len(available_features) < 5:
    raise ValueError(f"Too few available features: {available_features}. Need at least 5.")

X = data[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(-1)
y = data["is_confirmed_planet"]

if "target" not in data.columns:
    raise ValueError("Missing target column for group split.")

groups = data["target"].astype(str).str.extract(r"kplr(\d+)")[0]
groups = groups.fillna(data["target"].astype(str))

if groups.nunique() < 2:
    print("Only one KIC group found. Falling back to per-target grouping.")
    groups = data["target"].astype(str)

split_found = False
for seed in range(42, 142):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups))

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    if y_train.nunique() == 2 and y_test.nunique() == 2:
        split_found = True
        print("Using group split seed:", seed)
        break

if not split_found:
    print("Could not find valid group split. Falling back to stratified split.")
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=1,
    random_state=42,
)

model.fit(
    X.iloc[train_indices],
    y.iloc[train_indices],
)

probabilities = model.predict_proba(
    X.iloc[test_indices]
)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y.iloc[test_indices],
    probabilities
)

f1 = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = f1.argmax()
best_threshold = thresholds[best_idx - 1] if best_idx > 0 else 0.5
best_threshold = min(max(float(best_threshold), 0.41), 0.99)

print("Best threshold:", best_threshold)

print(classification_report(
    y.iloc[test_indices],
    probabilities >= best_threshold,
    zero_division=0,
))

uncertain_candidates = data.iloc[test_indices].copy()
uncertain_candidates["ml_probability"] = probabilities

uncertain_candidates = uncertain_candidates[
    uncertain_candidates["ml_probability"].between(0.35, 0.55)
].sort_values("ml_probability")

uncertain_columns = [
    "target",
    "ml_probability",
    "final_ranking_score",
    "transit_snr",
    "period_agreement",
]

available_uncertain_columns = [
    column for column in uncertain_columns
    if column in uncertain_candidates.columns
]

uncertain_candidates[available_uncertain_columns].to_csv(
    "uncertain_candidates.csv",
    index=False,
)

print(
    f"Saved {len(uncertain_candidates)} uncertain candidates "
    "to uncertain_candidates.csv"
)

n_unique_groups = groups.nunique()
print(f"\nNumber of unique groups: {n_unique_groups}")

if n_unique_groups >= 5:
    cv = GroupKFold(n_splits=5)
    print("Using GroupKFold with n_splits=5")
elif n_unique_groups >= 3:
    n_splits = n_unique_groups
    cv = GroupKFold(n_splits=n_splits)
    print(f"Using GroupKFold with n_splits={n_splits} (limited by group count)")
else:
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=min(5, len(X) // 3), random_state=42, shuffle=True)
    print(f"Using KFold with n_splits={cv.n_splits} (too few groups for GroupKFold)")

parameters = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    model,
    parameters,
    n_iter=20,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=1,
)

if isinstance(cv, GroupKFold):
    search.fit(X, y, groups=groups)
else:
    search.fit(X, y)

model = search.best_estimator_
print("Best parameters:", search.best_params_)

print("\nCross-Validation Results")
if isinstance(cv, GroupKFold):
    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="roc_auc",
    )
else:
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,   
}).sort_values("importance", ascending=False)

importance_df.to_csv("transit_xgb_feature_importance.csv", index=False)
print(importance_df)

print("Fold ROC-AUC:", scores)
print("Mean ROC-AUC:", scores.mean())
print("Std ROC-AUC:", scores.std())
test_roc_auc = roc_auc_score(y.iloc[test_indices], probabilities)
print("Test set ROC-AUC:", test_roc_auc)

model.save_model("transit_xgb_model.json")

metadata = {
    "features": FEATURES,
    "best_params": search.best_params_,
    "best_threshold": float(best_threshold),
    "mean_cv_roc_auc": float(scores.mean()),
    "std_cv_roc_auc": float(scores.std()),
    "test_roc_auc": float(test_roc_auc),
    "label_counts": {str(key): int(value) for key, value in y.value_counts().items()},
    "training_rows": int(len(data)),
    "real_label_matches": int(data["is_confirmed_planet"].sum()),
}
with open("transit_xgb_metadata.json", "w", encoding="utf-8") as metadata_file:
    json.dump(metadata, metadata_file, indent=2)

print("\nModel saved successfully to transit_xgb_model.json")
print("Metadata saved successfully to transit_xgb_metadata.json")