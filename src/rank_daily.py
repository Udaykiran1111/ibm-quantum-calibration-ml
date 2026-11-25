"""
rank_daily.py - rank qubits for TARGET_DATE using ONLY earlier sessions.

Frozen models (no retraining): Model B (primary) and Model C (secondary).
Writes WRITE-ONCE: predictions/daily/YYYY-MM-DD_model_b.csv / _model_c.csv
Needs >= 2 earlier sessions; otherwise it says so and exits cleanly.
"""
import argparse, hashlib, pickle, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import HIST_FEATURES, build_next_session_features

MODELS = [("model_b", "model_b_7day.pkl"), ("model_c", "model_c_30day.pkl")]


def load_history(daily_dir: Path) -> pd.DataFrame:
    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--daily-dir", default="data/daily/calibration")
    ap.add_argument("--models-dir", default="models/production")
    ap.add_argument("--out-dir", default="predictions/daily")
    a = ap.parse_args()

    hist = load_history(Path(a.daily_dir))
    feat = build_next_session_features(hist, a.target_date) if not hist.empty else hist
    if feat.empty:
        print(f"Not enough history before {a.target_date} (need >= 2 earlier sessions). No ranking written.")
        return 0

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, fname in MODELS:
        out = out_dir / f"{a.target_date}_{name}.csv"
        if out.exists():
            print(f"{out} exists - write-once, skipping."); continue
        mpath = Path(a.models_dir) / fname
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pickle.load(open(mpath, "rb"))
        assert list(model.classes_) == [0, 1]
        score = model.predict_proba(feat[HIST_FEATURES])[:, 1]
        df = feat[["backend", "qubit", "n_sessions"]].copy()
        df["viability_score"] = score
        df["viability_rank"] = df.groupby("backend")["viability_score"].rank(ascending=False, method="first").astype(int)
        df["target_date"] = a.target_date
        df["model"] = name
        df["model_sha256"] = hashlib.sha256(mpath.read_bytes()).hexdigest()
        df.sort_values(["backend", "viability_rank"]).to_csv(out, index=False)
        print(f"Wrote {out} ({len(df)} qubits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
