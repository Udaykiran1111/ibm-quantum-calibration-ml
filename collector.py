"""
collector.py — Daily IBM Quantum Calibration Collector + Predictor

What this script does every time it runs:
    1. Connect to IBM Quantum and auto-discover available operational backends
    2. Fetch today's calibration for all backends
    3. Deduplicate to session level (one row per backend, qubit)
    4. Save raw calibration to MySQL (calibration_history table)
    5. Load historical data from MySQL to compute 17 historical features
    6. Run qubit_model_v2.pkl to score each qubit's viability
    7. Rank qubits per backend by viability score
    8. Save rankings to MySQL (qubit_rankings table)

Run manually: python collector.py
Run via cron: GitHub Actions fires this at 6AM UTC daily
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

# Load environment variables from.env (local) or GitHub Secrets (CI)
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "qubit_model_v2.pkl")

# ── Label thresholds (must match training notebook exactly) ───────────────────
T1_THRESH = 100.0 # microseconds
T2_THRESH = 50.0 # microseconds
RE_THRESH = 0.05 # 5%

# ── Feature list (must match training notebook exactly) ───────────────────────
HIST_FEATURES = [
    'hist_T1_mean', 'hist_T1_std', 'hist_T1_min', 'hist_T1_max',
    'hist_T2_mean', 'hist_T2_std', 'hist_T2_min', 'hist_T2_max',
    'hist_RE_mean', 'hist_RE_std', 'hist_RE_min', 'hist_RE_max',
    'prev_T1', 'prev_T2', 'prev_RE',
    'hist_coherence_product', 'hist_t2_t1_ratio'
]


# ── Step 1: Fetch calibration from IBM Quantum API ────────────────────────────
def fetch_calibration(token: str) -> pd.DataFrame:
    """
    Connect to IBM Quantum and auto-discover operational backends.
    Returns a session-level DataFrame: one row per (backend, qubit).
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        print("Run: pip install qiskit-ibm-runtime")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    print(" Discovering available backends...")
    all_backends = service.backends(simulator=False, operational=True)
    
    if not all_backends:
        print("ERROR: No operational backends found for your account.")
        sys.exit(1)
    
    backend_names = [b.name for b in all_backends]
    print(f" Found {len(backend_names)} operational backends: {', '.join(backend_names)}")

    rows = []
    for backend in all_backends:
        backend_name = backend.name
        print(f" Fetching {backend_name}...")
        try:
            props = backend.properties()
            if props is None:
                print(f" WARNING: {backend_name} has no calibration data. Skipping.")
                continue
            backend_ts = props.last_update_date
        except Exception as e:
            print(f" WARNING: Could not fetch {backend_name}: {e}")
            continue

        for qubit_idx, qubit_props in enumerate(props.qubits):
            prop_dict = {p.name: p.value for p in qubit_props}
            T1 = prop_dict.get('T1', None)
            T2 = prop_dict.get('T2', None)
            RE = prop_dict.get('readout_error', None)

            if T1 is None or T2 is None or RE is None:
                continue

            # DEBUG: Uncomment to check units
            # if qubit_idx == 0:
            #     print(f"    DEBUG {backend_name} Q0: T1={T1}, T2={T2}, RE={RE}")

            rows.append({
                'backend': backend_name,
                'qubit': qubit_idx,
                'snapshot_date': date.today(),
                'backend_ts': backend_ts,
                'T1_us': T1,   # IBM API now returns microseconds directly
                'T2_us': T2,   # IBM API now returns microseconds directly
                'readout_error': RE,
            })

    if not rows:
        print("ERROR: No calibration data fetched from any backend.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f" Fetched {len(df)} qubit observations across {df['backend'].nunique()} backends.")
    return df


# ── Step 2: Compute historical features from MySQL history ────────────────────
def compute_features(today_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 17 historical features for today's qubits using stored history.

    Design:
        - today_df = current calibration (T1_us, T2_us, readout_error per qubit)
        - history_df = all prior days from MySQL (calibration_history table)

    Features are computed from history ONLY — today's values are the prediction target.
    This is the same design as the training notebook (no leakage).

    If a qubit has no history (Day 1), features are NaN → no prediction made.
    From Day 2 onward, all qubits get predictions.
    """
    all_rows = []

    for backend in today_df['backend'].unique():
        today_b = today_df[today_df['backend'] == backend].copy()
        hist_b = history_df[history_df['backend'] == backend].copy() if not history_df.empty else pd.DataFrame()

        if hist_b.empty:
            print(f" {backend}: No history in DB yet — Day 1. Predictions skipped.")
            # Still save calibration but skip scoring
            today_b['can_predict'] = False
            for col in HIST_FEATURES:
                today_b[col] = np.nan
            all_rows.append(today_b)
            continue

        # Sort history by date
        hist_b = hist_b.sort_values(['qubit', 'snapshot_date'])

        # For each qubit, compute rolling stats from all prior observations
        feat_rows = []
        for qubit in today_b['qubit'].unique():
            q_hist = hist_b[hist_b['qubit'] == qubit].sort_values('snapshot_date')
            q_today = today_b[today_b['qubit'] == qubit].iloc[0].to_dict()

            if len(q_hist) == 0:
                # No history for this qubit
                for col in HIST_FEATURES:
                    q_today[col] = np.nan
                q_today['can_predict'] = False
            else:
                t1_hist = q_hist['T1_us'].values
                t2_hist = q_hist['T2_us'].values
                re_hist = q_hist['readout_error'].values

                q_today['hist_T1_mean'] = np.mean(t1_hist)
                q_today['hist_T1_std'] = np.std(t1_hist) if len(t1_hist) > 1 else 0.0
                q_today['hist_T1_min'] = np.min(t1_hist)
                q_today['hist_T1_max'] = np.max(t1_hist)

                q_today['hist_T2_mean'] = np.mean(t2_hist)
                q_today['hist_T2_std'] = np.std(t2_hist) if len(t2_hist) > 1 else 0.0
                q_today['hist_T2_min'] = np.min(t2_hist)
                q_today['hist_T2_max'] = np.max(t2_hist)

                q_today['hist_RE_mean'] = np.mean(re_hist)
                q_today['hist_RE_std'] = np.std(re_hist) if len(re_hist) > 1 else 0.0
                q_today['hist_RE_min'] = np.min(re_hist)
                q_today['hist_RE_max'] = np.max(re_hist)

                # Previous snapshot (most recent historical value)
                q_today['prev_T1'] = t1_hist[-1]
                q_today['prev_T2'] = t2_hist[-1]
                q_today['prev_RE'] = re_hist[-1]

                # Engineered features
                q_today['hist_coherence_product'] = q_today['hist_T1_mean'] * q_today['hist_T2_mean']
                q_today['hist_t2_t1_ratio'] = q_today['hist_T2_mean'] / (q_today['hist_T1_mean'] + 1e-9)

                q_today['can_predict'] = True

            feat_rows.append(q_today)

        all_rows.append(pd.DataFrame(feat_rows))

    return pd.concat(all_rows, ignore_index=True)


# ── Step 3: Score and rank qubits ─────────────────────────────────────────────
def score_qubits(feat_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Run the trained Random Forest model to score each qubit.
    Computes physics-grounded label and viability rank per backend.
    """
    predictable = feat_df[feat_df['can_predict'] == True].copy()

    if predictable.empty:
        print(" No qubits have enough history for prediction yet (Day 1).")
        return pd.DataFrame()

    # Compute label from CURRENT session values
    predictable['label'] = (
        (predictable['T1_us'] > T1_THRESH) &
        (predictable['T2_us'] > T2_THRESH) &
        (predictable['readout_error'] < RE_THRESH)
    ).astype(int)

    # Score
    X = predictable[HIST_FEATURES].fillna(0)
    predictable['viability_score'] = model.predict_proba(X)[:, 1]

    # Rank within backend (1 = best)
    predictable['viability_rank'] = predictable.groupby('backend')['viability_score'].rank(
        ascending=False, method='min'
    ).astype(int)

    return predictable


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from database import (
        setup_database, insert_calibration_rows,
        insert_ranking_rows, get_history_for_backend
    )

    print("=" * 60)
    print(f"IBM QUANTUM COLLECTOR — {date.today()}")
    print("=" * 60)

    # ── Check token ───────────────────────────────────────────────────────────
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: IBM_QUANTUM_TOKEN not set.")
        print("Add it to your.env file: IBM_QUANTUM_TOKEN=your_token_here")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Copy qubit_model_v2.pkl to the models/ folder.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded: {MODEL_PATH}")

    # ── Setup DB ──────────────────────────────────────────────────────────────
    setup_database()

    # ── Fetch calibration ─────────────────────────────────────────────────────
    today_df = fetch_calibration(token)

    # ── Save raw calibration to DB ────────────────────────────────────────────
    calib_rows = today_df.to_dict('records')
    inserted = insert_calibration_rows(calib_rows)
    print(f"[DB] Calibration rows inserted: {inserted}")

    # ── Load history from DB ──────────────────────────────────────────────────
    hist_frames = []
    for backend in today_df['backend'].unique(): # Use discovered backends
        h = get_history_for_backend(backend)
        if not h.empty:
            hist_frames.append(h)
    history_df = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    print(f"[DB] Historical rows loaded: {len(history_df)}")

    # ── Compute features ──────────────────────────────────────────────────────
    feat_df = compute_features(today_df, history_df)

    # ── Score qubits ──────────────────────────────────────────────────────────
    ranked_df = score_qubits(feat_df, model)

    if ranked_df.empty:
        print("No rankings to save (Day 1 — no history yet).")
        print("Run again tomorrow to get predictions.")
        return

    # ── Save rankings to DB ───────────────────────────────────────────────────
    ranking_rows = []
    for _, row in ranked_df.iterrows():
        ranking_rows.append({
            'backend': row['backend'], # Backend stored with qubit
            'qubit': int(row['qubit']),
            'snapshot_date': row['snapshot_date'],
            'viability_score':round(float(row['viability_score']), 6),
            'viability_rank': int(row['viability_rank']),
            'label': int(row['label']),
            'T1_us': round(float(row['T1_us']), 3),
            'T2_us': round(float(row['T2_us']), 3),
            'readout_error': round(float(row['readout_error']), 6),
        })

    inserted = insert_ranking_rows(ranking_rows)
    print(f"[DB] Ranking rows inserted: {inserted}")

    # ── Print top 5 per backend ───────────────────────────────────────────────
    print()
    print("TOP 5 VIABLE QUBITS PER BACKEND:")
    print("-" * 60)
    for backend in sorted(ranked_df['backend'].unique()): # Use discovered backends
        sub = ranked_df[ranked_df['backend'] == backend].nsmallest(5, 'viability_rank')
        print(f"\n{backend}:")
        for _, r in sub.iterrows():
            status = "✓" if r['label'] == 1 else "✗"
            print(f" Rank {int(r['viability_rank']):3d} | Q{int(r['qubit']):3d} | "
                  f"score={r['viability_score']:.3f} | "
                  f"T1={r['T1_us']:.0f}μs T2={r['T2_us']:.0f}μs RE={r['readout_error']:.3f} {status}")

    print()
    print(f"Collection complete at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()