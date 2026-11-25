"""
train_from_csv.py
=================
Trains Model B (7-day) and Model C (30-day) directly from backup CSV files.
Use this if MySQL is unavailable or expired.

Run:
    python train_from_csv.py

Saves:
    models/model_b_7day.pkl
    models/model_c_30day.pkl
    results/model_b_metadata.json
    results/model_c_metadata.json
    results/training_summary.txt
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef
)

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point these to your backup CSV files
CAL_CSV = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "backup_calibration_history_20260811_1706.csv"
)
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR= os.path.join(SCRIPT_DIR, "results")

os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR,  exist_ok=True)

# ── Thresholds — must match Model A exactly ───────────────────────────────────
T1_THRESH = 100.0
T2_THRESH = 50.0
RE_THRESH = 0.05

# ── Feature list — must match Model A exactly ─────────────────────────────────
HIST_FEATURES = [
    'hist_T1_mean', 'hist_T1_std', 'hist_T1_min', 'hist_T1_max',
    'hist_T2_mean', 'hist_T2_std', 'hist_T2_min', 'hist_T2_max',
    'hist_RE_mean', 'hist_RE_std', 'hist_RE_min', 'hist_RE_max',
    'prev_T1',      'prev_T2',     'prev_RE',
    'hist_coherence_product', 'hist_t2_t1_ratio'
]

RF_PARAMS = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)

# ── Training windows ──────────────────────────────────────────────────────────
MODEL_B_START = '2026-07-13'
MODEL_B_END   = '2026-07-19'   # 7 days
MODEL_C_START = '2026-07-13'
MODEL_C_END   = '2026-08-11'   # 30 days (today = last collected day)


def load_calibration(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['T1_us'] = pd.to_numeric(df['T1_us'], errors='coerce')
    df['T2_us'] = pd.to_numeric(df['T2_us'], errors='coerce')
    df['readout_error'] = pd.to_numeric(df['readout_error'], errors='coerce')
    df['qubit'] = pd.to_numeric(df['qubit'], errors='coerce').astype(int)
    df = df.dropna(subset=['T1_us', 'T2_us', 'readout_error'])
    return df.sort_values(['backend', 'qubit', 'snapshot_date']).reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['label'] = (
        (df['T1_us']        > T1_THRESH) &
        (df['T2_us']        > T2_THRESH) &
        (df['readout_error'] < RE_THRESH)
    ).astype(int)
    return df


def add_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift + expanding window — identical to Model A.
    Current session T1/T2/RE NEVER appears in any feature.
    """
    df = df.sort_values(['backend', 'qubit', 'snapshot_date']).copy()
    df['snapshot_id'] = df.groupby(['backend', 'qubit'])['snapshot_date'].transform(
        lambda x: pd.factorize(x)[0]
    )
    for col, alias in [('T1_us','T1'), ('T2_us','T2'), ('readout_error','RE')]:
        g = df.groupby(['backend', 'qubit'])[col]
        df[f'hist_{alias}_mean'] = g.transform(lambda x: x.shift(1).expanding().mean())
        df[f'hist_{alias}_std']  = g.transform(lambda x: x.shift(1).expanding().std())
        df[f'hist_{alias}_min']  = g.transform(lambda x: x.shift(1).expanding().min())
        df[f'hist_{alias}_max']  = g.transform(lambda x: x.shift(1).expanding().max())
        df[f'prev_{alias}']      = g.transform(lambda x: x.shift(1))
    df['hist_coherence_product'] = df['hist_T1_mean'] * df['hist_T2_mean']
    df['hist_t2_t1_ratio']       = df['hist_T2_mean'] / (df['hist_T1_mean'] + 1e-9)
    return df


def leakage_audit(df: pd.DataFrame, name: str):
    snap0     = df[df['snapshot_id'] == 0]
    nan_rates = snap0[HIST_FEATURES].isnull().mean()
    assert nan_rates.min() == 1.0, f"LEAKAGE in {name}: snapshot 0 has non-NaN features"
    print(f"  [{name}] Leakage audit PASSED")


def build_dataset(raw: pd.DataFrame, start: str, end: str, name: str) -> pd.DataFrame:
    window = raw[
        (raw['snapshot_date'] >= pd.Timestamp(start)) &
        (raw['snapshot_date'] <= pd.Timestamp(end))
    ].copy()
    print(f"\n  [{name}] Window: {start} → {end}")
    print(f"  [{name}] Dates:  {window['snapshot_date'].nunique()} unique")
    print(f"  [{name}] Backends: {sorted(window['backend'].unique())}")
    window = add_label(window)
    window = add_hist_features(window)
    leakage_audit(window, name)
    model_df = window[window['snapshot_id'] >= 1].dropna(subset=HIST_FEATURES).copy()
    print(f"  [{name}] Model rows: {len(model_df):,}")
    print(f"  [{name}] Viable rate: {model_df['label'].mean():.3f}")
    print(f"  [{name}] Per backend:")
    for b in sorted(model_df['backend'].unique()):
        sub = model_df[model_df['backend']==b]
        print(f"           {b}: {len(sub):,} rows, {sub['label'].mean():.3f} viable")
    return model_df


def lobo_eval(model_df: pd.DataFrame, model, name: str) -> dict:
    """Leave-One-Backend-Out evaluation within training data."""
    backends = sorted(model_df['backend'].unique())
    results  = []
    for test_b in backends:
        tr = model_df[model_df['backend'] != test_b]
        te = model_df[model_df['backend'] == test_b]
        if te['label'].nunique() < 2:
            continue
        m2 = RandomForestClassifier(**RF_PARAMS)
        m2.fit(tr[HIST_FEATURES], tr['label'])
        yp = m2.predict_proba(te[HIST_FEATURES])[:, 1]
        yc = m2.predict(te[HIST_FEATURES])
        results.append({
            'test_backend': test_b,
            'AUC':          round(roc_auc_score(te['label'], yp), 4),
            'BalAcc':       round(balanced_accuracy_score(te['label'], yc), 4),
            'MCC':          round(matthews_corrcoef(te['label'], yc), 4),
        })
    return results


def train_and_save(model_df, name, pkl_file, start, end) -> object:
    print(f"\n  [{name}] Training Random Forest ({RF_PARAMS['n_estimators']} trees)...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(model_df[HIST_FEATURES], model_df['label'])

    # LOBO within training window
    print(f"  [{name}] Running LOBO within training window...")
    lobo = lobo_eval(model_df, model, name)
    mean_auc = round(sum(r['AUC'] for r in lobo) / len(lobo), 4) if lobo else None
    print(f"  [{name}] LOBO Results:")
    for r in lobo:
        print(f"           {r['test_backend']}: AUC={r['AUC']} BalAcc={r['BalAcc']} MCC={r['MCC']}")
    if mean_auc:
        print(f"           Mean AUC: {mean_auc}")

    # Save model
    pkl_path = os.path.join(MODELS_DIR, pkl_file)
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [{name}] Saved: {pkl_path}")

    # Save metadata
    meta = {
        'model_name':        name,
        'model_file':        pkl_file,
        'model_type':        'RandomForestClassifier',
        'training_start':    start,
        'training_end':      end,
        'training_rows':     len(model_df),
        'n_features':        len(HIST_FEATURES),
        'features':          HIST_FEATURES,
        'label_thresholds':  {'T1_us': T1_THRESH, 'T2_us': T2_THRESH, 'RE': RE_THRESH},
        'label_source':      'current_session',
        'feature_source':    'prior_sessions_only_shift_expand',
        'leakage_status':    'NONE_VERIFIED_BY_AUDIT',
        'rf_params':         RF_PARAMS,
        'lobo_results':      lobo,
        'lobo_mean_auc':     mean_auc,
        'trained_on_date':   str(date.today()),
        'backends':          sorted(model_df['backend'].unique().tolist()),
        'n_training_days':   int(model_df['snapshot_date'].nunique())
                             if 'snapshot_date' in model_df.columns else 'unknown',
    }
    meta_path = os.path.join(RESULTS_DIR, pkl_file.replace('.pkl', '_metadata.json'))
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  [{name}] Metadata: {meta_path}")
    return model, meta


def main():
    print("=" * 65)
    print(f"TRAINING MODEL B AND MODEL C — {date.today()}")
    print("=" * 65)

    # Load calibration CSV
    print(f"\nLoading calibration from: {CAL_CSV}")
    if not os.path.exists(CAL_CSV):
        print(f"ERROR: File not found: {CAL_CSV}")
        print("Update CAL_CSV path at the top of this script.")
        return

    raw = load_calibration(CAL_CSV)
    print(f"Loaded: {len(raw):,} rows")
    print(f"Date range: {raw['snapshot_date'].min().date()} → {raw['snapshot_date'].max().date()}")
    print(f"Unique dates: {raw['snapshot_date'].nunique()}")
    print(f"Backends: {sorted(raw['backend'].unique())}")
    print(f"T1 mean: {raw['T1_us'].mean():.1f} μs")

    if raw['T1_us'].mean() > 10000:
        print("ERROR: T1 values look like unit bug (millions). Check CSV.")
        return

    summaries = []

    # ── MODEL B — 7 days ──────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("MODEL B  — 7-Day (July 13–19)")
    print("─" * 65)
    df_b = build_dataset(raw, MODEL_B_START, MODEL_B_END, "Model_B")
    model_b, meta_b = train_and_save(
        df_b, "Model_B", "model_b_7day.pkl", MODEL_B_START, MODEL_B_END
    )
    summaries.append(('Model B', meta_b))

    # ── MODEL C — 30 days ─────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("MODEL C  — 30-Day (July 13 – Aug 11)")
    print("─" * 65)
    df_c = build_dataset(raw, MODEL_C_START, MODEL_C_END, "Model_C")
    model_c, meta_c = train_and_save(
        df_c, "Model_C", "model_c_30day.pkl", MODEL_C_START, MODEL_C_END
    )
    summaries.append(('Model C', meta_c))

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    summary_lines = []
    summary_lines.append(f"Training completed: {date.today()}")
    summary_lines.append("")

    for mname, meta in summaries:
        line = (f"{mname}: {meta['training_start']} → {meta['training_end']} | "
                f"{meta['training_rows']:,} rows | "
                f"LOBO Mean AUC: {meta.get('lobo_mean_auc', 'N/A')}")
        print(f"  {line}")
        summary_lines.append(line)

    summary_lines.append("")
    summary_lines.append("All three models ready:")
    summary_lines.append("  models/qubit_model_v2.pkl  — Model A (Dec28-Jan02)")
    summary_lines.append("  models/model_b_7day.pkl    — Model B (Jul13-Jul19)")
    summary_lines.append("  models/model_c_30day.pkl   — Model C (Jul13-Aug11)")
    summary_lines.append("")
    summary_lines.append("Next:")
    summary_lines.append("  1. Push all 3 pkl files to GitHub")
    summary_lines.append("  2. collector.py will auto-detect and use all 3")
    summary_lines.append("  3. Dashboard will show all 3 models")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print()
    for line in summary_lines[-7:]:
        print(f"  {line}")
    print("=" * 65)


if __name__ == "__main__":
    main()