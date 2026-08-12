"""
evaluator.py
============

Evaluation for the IBM Quantum Qubit Viability experiment.

IMPORTANT:
This evaluator keeps the SAME:
- label thresholds
- 17 historical features
- shift(1) + expanding historical-feature logic
- Random Forest input schema

Current experiment:
    Model A: historical Dec/Jan model
    Model B: trained Jul 13-Jul 19
    Model C: trained Jul 13-Aug 11

CURRENT TEMPORAL COMPARISON:
    Test window = Jul 20-Aug 11

    Model A -> can be evaluated on this window.
    Model B -> can be evaluated on this window because training ended Jul 19.
    Model C -> MUST NOT be evaluated on this window because it was trained
               using data through Aug 11.

FINAL EXPERIMENT:
After new data exists after Aug 11, change FINAL_TEST_START/END to the
same future window and this script will evaluate all three models on that
common unseen period.

Run:
    python evaluator.py
"""

import os
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# If you keep the CSV inside notebooks/ instead, change this line.
CAL_CSV = os.path.join(
    PROJECT_ROOT,
    "backup_calibration_history_20260811_1706.csv"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT WINDOWS
# ============================================================

# Current valid temporal comparison:
# Model B was trained through Jul 19.
CURRENT_TEST_START = "2026-07-20"
CURRENT_TEST_END   = "2026-08-11"

# Model C was trained through Aug 11.
MODEL_C_TRAIN_END = "2026-08-11"

# Later, when the final future collection exists, change these.
# Example:
# FINAL_TEST_START = "2026-08-12"
# FINAL_TEST_END   = "2026-09-10"
#
# Leave None for now so the script does not accidentally evaluate
# Model C on its own training data.
FINAL_TEST_START = None
FINAL_TEST_END = None


# ============================================================
# LABEL THRESHOLDS — EXACTLY THE SAME AS TRAINING SCRIPT
# ============================================================

T1_THRESH = 100.0
T2_THRESH = 50.0
RE_THRESH = 0.05


# ============================================================
# FEATURES — EXACTLY THE SAME AS TRAINING SCRIPT
# ============================================================

HIST_FEATURES = [
    "hist_T1_mean",
    "hist_T1_std",
    "hist_T1_min",
    "hist_T1_max",

    "hist_T2_mean",
    "hist_T2_std",
    "hist_T2_min",
    "hist_T2_max",

    "hist_RE_mean",
    "hist_RE_std",
    "hist_RE_min",
    "hist_RE_max",

    "prev_T1",
    "prev_T2",
    "prev_RE",

    "hist_coherence_product",
    "hist_t2_t1_ratio",
]


# ============================================================
# LOAD CSV
# ============================================================

def load_calibration(csv_path):

    print("=" * 70)
    print("LOADING CALIBRATION CSV")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Calibration CSV not found:\n{csv_path}\n\n"
            "Update CAL_CSV at the top of evaluator.py."
        )

    df = pd.read_csv(csv_path)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    df["T1_us"] = pd.to_numeric(
        df["T1_us"],
        errors="coerce"
    )

    df["T2_us"] = pd.to_numeric(
        df["T2_us"],
        errors="coerce"
    )

    df["readout_error"] = pd.to_numeric(
        df["readout_error"],
        errors="coerce"
    )

    df["qubit"] = pd.to_numeric(
        df["qubit"],
        errors="coerce"
    ).astype(int)

    df = df.dropna(
        subset=[
            "T1_us",
            "T2_us",
            "readout_error"
        ]
    )

    df = df.sort_values(
        [
            "backend",
            "qubit",
            "snapshot_date"
        ]
    ).reset_index(drop=True)

    print(f"Rows: {len(df):,}")
    print(
        f"Date range: "
        f"{df['snapshot_date'].min().date()} "
        f"→ "
        f"{df['snapshot_date'].max().date()}"
    )

    print(
        f"Unique calendar dates: "
        f"{df['snapshot_date'].nunique()}"
    )

    print(
        f"Backends: "
        f"{sorted(df['backend'].unique())}"
    )

    print(
        f"T1 mean: "
        f"{df['T1_us'].mean():.1f} μs"
    )

    return df


# ============================================================
# LABEL — EXACTLY THE SAME
# ============================================================

def add_label(df):

    df = df.copy()

    df["label"] = (
        (df["T1_us"] > T1_THRESH)
        &
        (df["T2_us"] > T2_THRESH)
        &
        (df["readout_error"] < RE_THRESH)
    ).astype(int)

    return df


# ============================================================
# HISTORICAL FEATURES — EXACTLY THE SAME
# ============================================================

def add_hist_features(df):

    """
    SAME LOGIC AS train_from_csv.py.

    Shift + expanding window.

    Current-session T1/T2/RE NEVER appears in any feature.
    """

    df = df.sort_values(
        [
            "backend",
            "qubit",
            "snapshot_date"
        ]
    ).copy()

    df["snapshot_id"] = (
        df.groupby(
            [
                "backend",
                "qubit"
            ]
        )["snapshot_date"]
        .transform(
            lambda x: pd.factorize(x)[0]
        )
    )

    for col, alias in [
        ("T1_us", "T1"),
        ("T2_us", "T2"),
        ("readout_error", "RE")
    ]:

        g = df.groupby(
            [
                "backend",
                "qubit"
            ]
        )[col]

        df[f"hist_{alias}_mean"] = (
            g.transform(
                lambda x:
                x.shift(1).expanding().mean()
            )
        )

        df[f"hist_{alias}_std"] = (
            g.transform(
                lambda x:
                x.shift(1).expanding().std()
            )
        )

        df[f"hist_{alias}_min"] = (
            g.transform(
                lambda x:
                x.shift(1).expanding().min()
            )
        )

        df[f"hist_{alias}_max"] = (
            g.transform(
                lambda x:
                x.shift(1).expanding().max()
            )
        )

        df[f"prev_{alias}"] = (
            g.transform(
                lambda x:
                x.shift(1)
            )
        )

    df["hist_coherence_product"] = (
        df["hist_T1_mean"]
        *
        df["hist_T2_mean"]
    )

    df["hist_t2_t1_ratio"] = (
        df["hist_T2_mean"]
        /
        (
            df["hist_T1_mean"]
            +
            1e-9
        )
    )

    return df


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def leakage_audit(df):

    snap0 = df[
        df["snapshot_id"] == 0
    ]

    nan_rates = (
        snap0[HIST_FEATURES]
        .isnull()
        .mean()
    )

    if nan_rates.min() != 1.0:
        raise RuntimeError(
            "LEAKAGE DETECTED: snapshot 0 has "
            "non-NaN historical features."
        )

    print(
        "Leakage audit PASSED — "
        "snapshot 0 is 100% NaN"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(filename):

    path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(
        f"Loaded: {filename}"
    )

    return model


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model,
    df,
    model_name,
    test_start,
    test_end
):

    print()
    print("=" * 70)
    print(
        f"{model_name} — TEMPORAL HOLDOUT"
    )
    print("=" * 70)

    test = df[
        (
            df["snapshot_date"]
            >=
            pd.Timestamp(test_start)
        )
        &
        (
            df["snapshot_date"]
            <=
            pd.Timestamp(test_end)
        )
    ].copy()

    test = test.dropna(
        subset=HIST_FEATURES
    )

    print(
        f"Test window: "
        f"{test_start} → {test_end}"
    )

    print(
        f"Test dates: "
        f"{test['snapshot_date'].nunique()}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    if len(test) == 0:
        print(
            "NO TEST DATA"
        )
        return None

    if test["label"].nunique() < 2:
        print(
            "ERROR: Test set has only one class."
        )
        return None

    X = test[
        HIST_FEATURES
    ]

    y = test["label"]

    probability = (
        model
        .predict_proba(X)[:, 1]
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y,
        probability
    )

    bal_acc = (
        balanced_accuracy_score(
            y,
            prediction
        )
    )

    f1 = f1_score(
        y,
        prediction
    )

    mcc = matthews_corrcoef(
        y,
        prediction
    )

    accuracy = accuracy_score(
        y,
        prediction
    )

    cm = confusion_matrix(
        y,
        prediction
    )

    print()
    print(
        f"AUC:                {auc:.4f}"
    )

    print(
        f"Balanced Accuracy:  {bal_acc:.4f}"
    )

    print(
        f"F1:                 {f1:.4f}"
    )

    print(
        f"MCC:                {mcc:.4f}"
    )

    print(
        f"Accuracy:           {accuracy:.4f}"
    )

    print()
    print(
        f"Actual viable rate: "
        f"{y.mean():.4f}"
    )

    print(
        f"Predicted viable:   "
        f"{prediction.mean():.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # PER-BACKEND
    # --------------------------------------------------------

    backend_results = []

    print()
    print("-" * 70)
    print(
        f"{model_name} — PER BACKEND"
    )
    print("-" * 70)

    for backend in sorted(
        test["backend"].unique()
    ):

        sub = test[
            test["backend"] == backend
        ]

        if sub["label"].nunique() < 2:
            print(
                f"{backend}: skipped — "
                f"only one class"
            )
            continue

        bp = (
            model
            .predict_proba(
                sub[HIST_FEATURES]
            )[:, 1]
        )

        bpred = (
            bp >= 0.5
        ).astype(int)

        row = {
            "backend": backend,
            "rows": int(len(sub)),
            "AUC": float(
                roc_auc_score(
                    sub["label"],
                    bp
                )
            ),
            "BalAcc": float(
                balanced_accuracy_score(
                    sub["label"],
                    bpred
                )
            ),
            "F1": float(
                f1_score(
                    sub["label"],
                    bpred
                )
            ),
            "MCC": float(
                matthews_corrcoef(
                    sub["label"],
                    bpred
                )
            )
        }

        backend_results.append(row)

        print(
            f"{backend:20s} "
            f"rows={row['rows']:5d} "
            f"AUC={row['AUC']:.4f} "
            f"BalAcc={row['BalAcc']:.4f} "
            f"F1={row['F1']:.4f} "
            f"MCC={row['MCC']:.4f}"
        )

    return {
        "model": model_name,
        "test_start": test_start,
        "test_end": test_end,
        "test_rows": int(len(test)),
        "test_calendar_dates": int(
            test["snapshot_date"].nunique()
        ),
        "AUC": float(auc),
        "BalancedAccuracy": float(bal_acc),
        "F1": float(f1),
        "MCC": float(mcc),
        "Accuracy": float(accuracy),
        "actual_viable_rate": float(
            y.mean()
        ),
        "predicted_viable_rate": float(
            prediction.mean()
        ),
        "confusion_matrix": cm.tolist(),
        "backend_results": backend_results
    }


# ============================================================
# PRINT LOBO METADATA
# ============================================================

def show_training_metadata(filename):

    metadata_file = filename.replace(
        ".pkl",
        "_metadata.json"
    )

    path = os.path.join(
        RESULTS_DIR,
        metadata_file
    )

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        meta = json.load(f)

    print()
    print(
        f"{meta.get('model_name', filename)} "
        "TRAINING / LOBO METADATA"
    )

    print(
        f"Training: "
        f"{meta.get('training_start')} → "
        f"{meta.get('training_end')}"
    )

    print(
        f"Training rows: "
        f"{meta.get('training_rows')}"
    )

    print(
        f"LOBO Mean AUC: "
        f"{meta.get('lobo_mean_auc')}"
    )

    if meta.get("lobo_results"):
        for row in meta["lobo_results"]:
            print(
                f"  {row.get('test_backend')}: "
                f"AUC={row.get('AUC')} "
                f"BalAcc={row.get('BalAcc')} "
                f"MCC={row.get('MCC')}"
            )

    return meta


# ============================================================
# CURRENT COMPARISON
# ============================================================

def current_comparison(df):

    print()
    print("=" * 70)
    print(
        "CURRENT TEMPORAL COMPARISON"
    )
    print("=" * 70)

    print(
        f"Common test window: "
        f"{CURRENT_TEST_START} → "
        f"{CURRENT_TEST_END}"
    )

    # Model A
    model_a = load_model(
        "qubit_model_v2.pkl"
    )

    result_a = evaluate_model(
        model_a,
        df,
        "MODEL A — HISTORICAL",
        CURRENT_TEST_START,
        CURRENT_TEST_END
    )

    # Model B
    model_b = load_model(
        "model_b_7day.pkl"
    )

    result_b = evaluate_model(
        model_b,
        df,
        "MODEL B — 7 DAY",
        CURRENT_TEST_START,
        CURRENT_TEST_END
    )

    # Model C deliberately excluded.
    print()
    print(
        "MODEL C — NOT SCORED"
    )
    print(
        "Reason: Model C was trained through "
        f"{MODEL_C_TRAIN_END}."
    )
    print(
        "The current test window ends on the "
        "same date, so scoring it here would "
        "not be an unseen-data evaluation."
    )

    return result_a, result_b


# ============================================================
# FINAL THREE-MODEL COMPARISON
# ============================================================

def final_three_model_comparison(
    df,
    test_start,
    test_end
):

    print()
    print("=" * 70)
    print(
        "FINAL THREE-MODEL UNSEEN TEST"
    )
    print("=" * 70)

    if pd.Timestamp(test_start) <= pd.Timestamp(
        MODEL_C_TRAIN_END
    ):
        raise ValueError(
            "Final test must start AFTER "
            f"{MODEL_C_TRAIN_END} so Model C "
            "is evaluated on genuinely unseen data."
        )

    results = []

    models = [
        (
            "MODEL A — HISTORICAL",
            "qubit_model_v2.pkl"
        ),
        (
            "MODEL B — 7 DAY",
            "model_b_7day.pkl"
        ),
        (
            "MODEL C — 30 DAY",
            "model_c_30day.pkl"
        )
    ]

    for name, filename in models:

        model = load_model(filename)

        result = evaluate_model(
            model,
            df,
            name,
            test_start,
            test_end
        )

        if result:
            results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "IBM QUANTUM — MODEL EVALUATOR"
    )
    print("=" * 70)

    df = load_calibration(
        CAL_CSV
    )

    # Build labels/features ONCE over the entire chronological CSV.
    # This preserves the exact shift+expanding logic.
    df = add_label(df)
    df = add_hist_features(df)

    leakage_audit(df)

    # --------------------------------------------------------
    # CURRENT STATUS
    # --------------------------------------------------------

    result_a, result_b = (
        current_comparison(df)
    )

    # --------------------------------------------------------
    # SHOW THE LOBO RESULTS GENERATED BY train_from_csv.py
    # These are useful diagnostics, but NOT the same as
    # temporal future-holdout performance.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXISTING TRAINING / LOBO RESULTS"
    )
    print("=" * 70)

    show_training_metadata(
        "model_b_7day.pkl"
    )

    show_training_metadata(
        "model_c_30day.pkl"
    )

    # --------------------------------------------------------
    # OPTIONAL FINAL EXPERIMENT
    # --------------------------------------------------------

    final_results = None

    if (
        FINAL_TEST_START is not None
        and FINAL_TEST_END is not None
    ):

        final_results = (
            final_three_model_comparison(
                df,
                FINAL_TEST_START,
                FINAL_TEST_END
            )
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    report = {
        "current_test_window": {
            "start": CURRENT_TEST_START,
            "end": CURRENT_TEST_END
        },
        "model_a_current_temporal": result_a,
        "model_b_current_temporal": result_b,
        "model_c_current_temporal": None,
        "model_c_status":
            "Not evaluated because current data "
            "does not contain observations after "
            "its training end date.",
        "final_three_model_results":
            final_results
    }

    output_path = os.path.join(
        RESULTS_DIR,
        "model_evaluation_report.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CURRENT A vs B COMPARISON"
    )
    print("=" * 70)

    print(
        f"{'Model':25s} "
        f"{'AUC':>8s} "
        f"{'BalAcc':>8s} "
        f"{'F1':>8s} "
        f"{'MCC':>8s}"
    )

    print("-" * 65)

    for result in [
        result_a,
        result_b
    ]:

        if result is None:
            continue

        print(
            f"{result['model']:25s} "
            f"{result['AUC']:8.4f} "
            f"{result['BalancedAccuracy']:8.4f} "
            f"{result['F1']:8.4f} "
            f"{result['MCC']:8.4f}"
        )

    print()
    print(
        "Model C: pending genuine future data."
    )

    print()
    print(
        "Evaluation report saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
