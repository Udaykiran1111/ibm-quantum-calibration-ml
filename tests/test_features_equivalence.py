"""The live feature builder must equal the training-notebook feature logic."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import HIST_FEATURES, build_next_session_features

CSV = ROOT / "data/raw/railway_backups/backup_calibration_history_20260811_1706.csv"


def training_style(df):
    """Verbatim logic of the training notebooks (shift + expanding)."""
    df = df.sort_values(["backend", "qubit", "snapshot_date"]).copy()
    for col, a in [("T1_us", "T1"), ("T2_us", "T2"), ("readout_error", "RE")]:
        g = df.groupby(["backend", "qubit"])[col]
        for fn in ["mean", "std", "min", "max"]:
            df[f"hist_{a}_{fn}"] = g.transform(lambda x: getattr(x.shift(1).expanding(), fn)())
        df[f"prev_{a}"] = g.transform(lambda x: x.shift(1))
    df["hist_coherence_product"] = df["hist_T1_mean"] * df["hist_T2_mean"]
    df["hist_t2_t1_ratio"] = df["hist_T2_mean"] / (df["hist_T1_mean"] + 1e-9)
    return df


def test_equivalence():
    df = pd.read_csv(CSV, parse_dates=["snapshot_date"])
    ref = training_style(df)
    for target in ["2026-07-20", "2026-08-05", "2026-08-11"]:
        new = build_next_session_features(df, target)
        old = ref[ref.snapshot_date == target].dropna(subset=HIST_FEATURES)
        m = new.merge(old, on=["backend", "qubit"], suffixes=("_new", "_old"))
        assert len(m) == len(new) == len(old) > 400, (target, len(m), len(new), len(old))
        for f in HIST_FEATURES:
            assert np.allclose(m[f + "_new"], m[f + "_old"], rtol=1e-9, atol=1e-9), (target, f)
        print(f"  {target}: {len(new)} qubits, all 17 features identical to training logic")


def test_no_future_rows_used():
    df = pd.read_csv(CSV, parse_dates=["snapshot_date"])
    a = build_next_session_features(df[df.snapshot_date < "2026-07-20"], "2026-07-20")
    b = build_next_session_features(df, "2026-07-20")          # includes later dates
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    print("  rows dated on/after the target date are ignored")


if __name__ == "__main__":
    test_equivalence(); test_no_future_rows_used(); print("features: PASS")
