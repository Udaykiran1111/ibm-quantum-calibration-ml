"""
collector.py — Final three-model live IBM Quantum collector.

Model A: XGBoost historical baseline
Model B: Random Forest, Jul 13-Jul 19 training
Model C: Random Forest, Jul 13-Aug 11 training

CRITICAL INFERENCE RULE:
    Today's calibration is NEVER part of today's historical features.

Order:
    1. Fetch today's calibration.
    2. Load DB history strictly before today.
    3. Build the same 17 historical features used in training.
    4. Score today's qubits through A/B/C.
    5. Save A/B/C rankings.
    6. Save today's raw calibration.

This makes the live pipeline causal and prevents current-session leakage.
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

T1_THRESH = 100.0
T2_THRESH = 50.0
RE_THRESH = 0.05

HIST_FEATURES = [
    "hist_T1_mean", "hist_T1_std", "hist_T1_min", "hist_T1_max",
    "hist_T2_mean", "hist_T2_std", "hist_T2_min", "hist_T2_max",
    "hist_RE_mean", "hist_RE_std", "hist_RE_min", "hist_RE_max",
    "prev_T1", "prev_T2", "prev_RE",
    "hist_coherence_product", "hist_t2_t1_ratio",
]

MODEL_REGISTRY = [
    {
        "model_name": "model_a",
        "file": "qubit_model_v2.pkl",
        "label": "Model A — Historical (Dec28-Jan02)",
        "algorithm": "XGBoost",
        "required": True,
    },
    {
        "model_name": "model_b",
        "file": "model_b_7day.pkl",
        "label": "Model B — 7-Day (Jul13-Jul19)",
        "algorithm": "Random Forest",
        "required": True,
    },
    {
        "model_name": "model_c",
        "file": "model_c_30day.pkl",
        "label": "Model C — 30-Day (Jul13-Aug11)",
        "algorithm": "Random Forest",
        "required": True,
    },
]


def load_models():
    loaded = []
    for entry in MODEL_REGISTRY:
        path = os.path.join(MODELS_DIR, entry["file"])
        if not os.path.exists(path):
            if entry["required"]:
                raise FileNotFoundError(
                    f"Required model missing: {path}"
                )
            continue

        with open(path, "rb") as f:
            model = pickle.load(f)

        loaded.append({**entry, "model": model})
        print(
            f"  Loaded {entry['model_name']}: "
            f"{entry['file']} [{entry['algorithm']}]"
        )

    if len(loaded) != 3:
        raise RuntimeError(
            f"Expected all 3 models, loaded {len(loaded)}."
        )

    return loaded


def fetch_calibration(token):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed."
        )

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        "Connecting to IBM Quantum..."
    )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
    )

    backends = service.backends(
        simulator=False,
        operational=True,
    )

    if not backends:
        raise RuntimeError("No operational IBM Quantum backends found.")

    print(
        f"  Discovered {len(backends)} operational backends: "
        + ", ".join(b.name for b in backends)
    )

    rows = []

    for backend in backends:
        bname = backend.name
        print(f"  Fetching {bname}...")

        try:
            props = backend.properties()
            if props is None:
                print(f"    No calibration data; skipping {bname}.")
                continue
            backend_ts = props.last_update_date
        except Exception as exc:
            print(f"    Could not fetch {bname}: {exc}")
            continue

        for qubit, qprops in enumerate(props.qubits):
            values = {p.name: p.value for p in qprops}

            t1 = values.get("T1")
            t2 = values.get("T2")
            re = values.get("readout_error")

            if t1 is None or t2 is None or re is None:
                continue

            rows.append(
                {
                    "backend": bname,
                    "qubit": int(qubit),
                    "snapshot_date": date.today(),
                    "backend_ts": backend_ts,
                    "T1_us": float(t1),
                    "T2_us": float(t2),
                    "readout_error": float(re),
                }
            )

    if not rows:
        raise RuntimeError("No usable calibration observations were fetched.")

    df = pd.DataFrame(rows)

    mean_t1 = df["T1_us"].mean()
    print(
        f"  Fetched {len(df):,} rows across "
        f"{df['backend'].nunique()} backends."
    )
    print(f"  T1 mean = {mean_t1:.1f} us")

    if mean_t1 > 10000:
        raise RuntimeError(
            "T1 unit sanity check failed. Mean T1 is > 10,000 us."
        )

    # Enforce one row per backend/qubit/date before DB insertion.
    df = (
        df.sort_values(["backend", "qubit", "backend_ts"])
        .drop_duplicates(
            subset=["backend", "qubit", "snapshot_date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def compute_features(today_df, history_df, prediction_date):
    """
    Build exactly the 17 historical features.

    history_df MUST contain only dates < prediction_date.
    The assertion below makes that invariant executable.
    """
    if not history_df.empty:
        history_df = history_df.copy()
        history_df["snapshot_date"] = pd.to_datetime(
            history_df["snapshot_date"]
        ).dt.date

        bad = history_df[
            history_df["snapshot_date"] >= prediction_date
        ]
        if not bad.empty:
            raise RuntimeError(
                "LEAKAGE GUARD FAILED: history contains current/future date."
            )

    rows = []

    for backend in sorted(today_df["backend"].unique()):
        today_b = today_df[
            today_df["backend"] == backend
        ].copy()

        hist_b = (
            history_df[
                history_df["backend"] == backend
            ].copy()
            if not history_df.empty
            else pd.DataFrame()
        )

        for _, current in today_b.iterrows():
            q = int(current["qubit"])

            q_hist = (
                hist_b[hist_b["qubit"] == q]
                .sort_values("snapshot_date")
            )

            row = current.to_dict()

            if q_hist.empty:
                row["can_predict"] = False
                for feature in HIST_FEATURES:
                    row[feature] = np.nan
                rows.append(row)
                continue

            t1 = q_hist["T1_us"].to_numpy(dtype=float)
            t2 = q_hist["T2_us"].to_numpy(dtype=float)
            re = q_hist["readout_error"].to_numpy(dtype=float)

            row["hist_T1_mean"] = np.mean(t1)
            row["hist_T1_std"] = np.std(t1) if len(t1) > 1 else 0.0
            row["hist_T1_min"] = np.min(t1)
            row["hist_T1_max"] = np.max(t1)

            row["hist_T2_mean"] = np.mean(t2)
            row["hist_T2_std"] = np.std(t2) if len(t2) > 1 else 0.0
            row["hist_T2_min"] = np.min(t2)
            row["hist_T2_max"] = np.max(t2)

            row["hist_RE_mean"] = np.mean(re)
            row["hist_RE_std"] = np.std(re) if len(re) > 1 else 0.0
            row["hist_RE_min"] = np.min(re)
            row["hist_RE_max"] = np.max(re)

            row["prev_T1"] = t1[-1]
            row["prev_T2"] = t2[-1]
            row["prev_RE"] = re[-1]

            row["hist_coherence_product"] = (
                row["hist_T1_mean"] * row["hist_T2_mean"]
            )
            row["hist_t2_t1_ratio"] = (
                row["hist_T2_mean"]
                / (row["hist_T1_mean"] + 1e-9)
            )

            row["can_predict"] = True
            rows.append(row)

    return pd.DataFrame(rows)


def score_with_model(feat_df, model):
    predictable = feat_df[
        feat_df["can_predict"] == True
    ].copy()

    if predictable.empty:
        return pd.DataFrame()

    # Current-session label is stored only as the observed outcome.
    # It is NEVER supplied to the model.
    predictable["label"] = (
        (predictable["T1_us"] > T1_THRESH)
        & (predictable["T2_us"] > T2_THRESH)
        & (predictable["readout_error"] < RE_THRESH)
    ).astype(int)

    X = predictable[HIST_FEATURES]

    if X.isna().any().any():
        raise RuntimeError(
            "NaN historical features reached prediction stage."
        )

    predictable["viability_score"] = model.predict_proba(X)[:, 1]

    predictable["viability_rank"] = (
        predictable.groupby("backend")["viability_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return predictable


def main():
    from database import (
        setup_database,
        insert_calibration_rows,
        insert_ranking_rows,
        get_history_for_backend,
    )

    print("=" * 72)
    print(f"QUBITTELEMETRY FINAL COLLECTOR — {date.today()}")
    print("=" * 72)

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError("IBM_QUANTUM_TOKEN is not set.")

    models = load_models()
    setup_database()

    # ------------------------------------------------------------
    # 1. Fetch today's observations.
    # ------------------------------------------------------------
    today = date.today()
    today_df = fetch_calibration(token)

    if today_df["snapshot_date"].nunique() != 1:
        raise RuntimeError("Fetched data contains multiple snapshot dates.")

    # ------------------------------------------------------------
    # 2. IMPORTANT: load history BEFORE inserting today's rows.
    #    Also enforce date < today in the DB query.
    # ------------------------------------------------------------
    hist_frames = []

    for backend in today_df["backend"].unique():
        history = get_history_for_backend(
            backend,
            before_date=today,
        )
        if not history.empty:
            hist_frames.append(history)

    history_df = (
        pd.concat(hist_frames, ignore_index=True)
        if hist_frames
        else pd.DataFrame()
    )

    print(
        f"[DB] Prior-history rows loaded: {len(history_df):,}"
    )

    # Defense-in-depth.
    if not history_df.empty:
        history_df["snapshot_date"] = pd.to_datetime(
            history_df["snapshot_date"]
        ).dt.date

        if (history_df["snapshot_date"] >= today).any():
            raise RuntimeError(
                "ABORT: current/future calibration leaked into history."
            )

    # ------------------------------------------------------------
    # 3. Build features once.
    # ------------------------------------------------------------
    feat_df = compute_features(
        today_df=today_df,
        history_df=history_df,
        prediction_date=today,
    )

    n_predictable = int(feat_df["can_predict"].sum())
    print(f"[FEATURES] Predictable qubits: {n_predictable:,}")

    # ------------------------------------------------------------
    # 4. Score ALL THREE models using the SAME feature dataframe.
    # ------------------------------------------------------------
    if n_predictable:
        for entry in models:
            ranked = score_with_model(
                feat_df,
                entry["model"],
            )

            ranking_rows = []
            for _, row in ranked.iterrows():
                ranking_rows.append(
                    {
                        "backend": row["backend"],
                        "qubit": int(row["qubit"]),
                        "snapshot_date": row["snapshot_date"],
                        "viability_score": round(
                            float(row["viability_score"]), 6
                        ),
                        "viability_rank": int(row["viability_rank"]),
                        "label": int(row["label"]),
                        "T1_us": round(float(row["T1_us"]), 3),
                        "T2_us": round(float(row["T2_us"]), 3),
                        "readout_error": round(
                            float(row["readout_error"]), 6
                        ),
                    }
                )

            inserted = insert_ranking_rows(
                ranking_rows,
                model_name=entry["model_name"],
            )

            print(
                f"[DB] {entry['model_name']} "
                f"({entry['algorithm']}): "
                f"{inserted:,} ranking rows inserted."
            )
    else:
        print(
            "[PREDICTION] No qubit has prior history. "
            "Predictions skipped."
        )

    # ------------------------------------------------------------
    # 5. Only AFTER prediction, persist today's calibration.
    # ------------------------------------------------------------
    inserted = insert_calibration_rows(
        today_df.to_dict("records")
    )

    print(
        f"[DB] Today's calibration rows inserted: {inserted:,}"
    )

    print("=" * 72)
    print("COLLECTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()