"""
10_rolling_prospective_baselines.py
===================================
Rolling-origin ("walk-forward") prospective evaluation of qubit-viability prediction
on the 30 surviving days of live data (13 Jul - 11 Aug 2026).

For every test day D the models are fitted ONLY on rows from days before D,
then scored on day D. Nothing from day D (or later) is used for fitting.

Predictors compared
  persistence     score = was the qubit viable in its previous session (0/1)
  recent3_mean    score = share of viable sessions among the previous 3 sessions
  history_mean    score = share of viable sessions among ALL previous sessions
  logistic        logistic regression on the same 17 features (standardised)
  rf              Random Forest on the same 17 features (200 trees, depth 8, seed 42)

Two protocols
  temporal        train on earlier days of all three backends, test on day D (all backends)
  temporal_lobo   train on earlier days of the OTHER two backends, test on the held-out backend

Training windows (rf / logistic): the last W days before D, W in {1, 3, 7, 14}.
The features themselves always use ALL earlier history (as in the production models);
W only controls how many recent days of rows are used to fit the classifier.

Metric: ROC-AUC computed per test day (pooled over backends in `temporal`, averaged over
backends in `temporal_lobo`), then averaged over test days. Test days are the unit of
analysis, because qubits on the same day share one calibration and are not independent.
Confidence intervals: moving-block bootstrap over test days (block length 3, 2000 draws).
Paired differences (model minus persistence) use the same resampled days.

Inputs : data/raw/railway_backups/backup_calibration_history_20260811_1706.csv
Outputs: results/reproduction/rolling_*.csv and rolling_summary.txt

Run (from the repository root):
    python analysis/scripts/10_rolling_prospective_baselines.py
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features import HIST_FEATURES, add_label, build_next_session_features  # noqa: E402

WINDOWS = [1, 3, 7, 14]
RF_PARAMS = dict(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
SEED = 42


# ---------------------------------------------------------------- data + features
def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for c in ["T1_us", "T2_us", "readout_error"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["qubit"] = pd.to_numeric(df["qubit"], errors="coerce").astype(int)
    df = df.dropna(subset=["T1_us", "T2_us", "readout_error"])
    df = df.drop_duplicates(["backend", "qubit", "snapshot_date"], keep="last")
    return df.sort_values(["backend", "qubit", "snapshot_date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shift + expanding window, identical to the training notebooks."""
    df = add_label(df).sort_values(["backend", "qubit", "snapshot_date"]).copy()
    df["session_idx"] = df.groupby(["backend", "qubit"]).cumcount()
    for col, alias in [("T1_us", "T1"), ("T2_us", "T2"), ("readout_error", "RE")]:
        g = df.groupby(["backend", "qubit"])[col]
        df[f"hist_{alias}_mean"] = g.transform(lambda x: x.shift(1).expanding().mean())
        df[f"hist_{alias}_std"] = g.transform(lambda x: x.shift(1).expanding().std())
        df[f"hist_{alias}_min"] = g.transform(lambda x: x.shift(1).expanding().min())
        df[f"hist_{alias}_max"] = g.transform(lambda x: x.shift(1).expanding().max())
        df[f"prev_{alias}"] = g.transform(lambda x: x.shift(1))
    df["hist_coherence_product"] = df["hist_T1_mean"] * df["hist_T2_mean"]
    df["hist_t2_t1_ratio"] = df["hist_T2_mean"] / (df["hist_T1_mean"] + 1e-9)
    # label-history baselines (previous sessions only)
    gl = df.groupby(["backend", "qubit"])["label"]
    df["b_persistence"] = gl.transform(lambda x: x.shift(1))
    df["b_recent3_mean"] = gl.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["b_history_mean"] = gl.transform(lambda x: x.shift(1).expanding().mean())
    return df


def self_check(feat: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Rolling features must equal the live feature builder used by the daily pipeline."""
    target = feat["snapshot_date"].max()
    live = build_next_session_features(raw[raw["snapshot_date"] < target], target)
    mine = feat[feat["snapshot_date"] == target].dropna(subset=HIST_FEATURES)
    m = live.merge(mine, on=["backend", "qubit"], suffixes=("_live", "_mine"))
    assert len(m) > 400, f"self-check: only {len(m)} rows matched"
    for f in HIST_FEATURES:
        a, b = m[f + "_live"].to_numpy(float), m[f + "_mine"].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), f"feature mismatch: {f}"
    print(f"[self-check] rolling features == live builder on {len(m)} rows ({target.date()})")


# ---------------------------------------------------------------- evaluation
def auc_or_nan(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def make_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(**RF_PARAMS)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def fit_predict(kind, train, test):
    if train["label"].nunique() < 2:
        return np.full(len(test), np.nan)
    m = make_model(kind)
    m.fit(train[HIST_FEATURES], train["label"])
    return m.predict_proba(test[HIST_FEATURES])[:, 1]


def run(feat: pd.DataFrame, first_test_idx: int) -> pd.DataFrame:
    dates = sorted(feat["snapshot_date"].unique())
    usable = feat.dropna(subset=HIST_FEATURES).copy()  # rows that have all 17 features
    usable_dates = sorted(usable["snapshot_date"].unique())
    backends = sorted(feat["backend"].unique())
    rows = []
    for i in range(first_test_idx, len(dates)):
        D = dates[i]
        test_all = usable[usable["snapshot_date"] == D]
        test_all = test_all.dropna(subset=["b_persistence"])
        prior = [d for d in usable_dates if d < D]
        for protocol in ["temporal", "temporal_lobo"]:
            splits = [(None, test_all)] if protocol == "temporal" else [
                (b, test_all[test_all["backend"] == b]) for b in backends]
            per_b = {}
            for held, test in splits:
                if test.empty:
                    continue
                res = {}
                for base in ["persistence", "recent3_mean", "history_mean"]:
                    res[base] = auc_or_nan(test["label"], test[f"b_{base}"])
                for W in WINDOWS:
                    wd = prior[-W:]
                    if len(wd) < W:
                        continue
                    tr = usable[usable["snapshot_date"].isin(wd)]
                    if held is not None:
                        tr = tr[tr["backend"] != held]
                    for kind in ["logistic", "rf"]:
                        res[f"{kind}_w{W}"] = auc_or_nan(test["label"], fit_predict(kind, tr, test))
                for k, v in res.items():
                    per_b.setdefault(k, []).append(v)
            for k, vs in per_b.items():
                rows.append(dict(date=pd.Timestamp(D).date(), protocol=protocol, predictor=k,
                                 auc=float(np.nanmean(vs)), n_test=int(len(test_all))))
        print(f"  tested {pd.Timestamp(D).date()}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
def block_bootstrap_means(x: np.ndarray, block=3, n=2000, seed=SEED):
    """Moving-block bootstrap of the mean of a time series of per-day values."""
    rng = np.random.default_rng(seed)
    T = len(x)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    idx_all = []
    for b in range(n):
        starts = rng.integers(0, T - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        idx_all.append(idx)
        out[b] = np.mean(x[idx])
    return out, np.array(idx_all)


def summarise(per_date: pd.DataFrame, common_dates) -> pd.DataFrame:
    out = []
    for protocol, g in per_date.groupby("protocol"):
        wide = g[g["date"].isin(common_dates)].pivot(index="date", columns="predictor", values="auc").sort_index()
        wide = wide.dropna(axis=1)
        base = wide["persistence"].to_numpy()
        for pred in wide.columns:
            x = wide[pred].to_numpy()
            diff = x - base
            _, idx = block_bootstrap_means(x)
            means = np.array([x[i].mean() for i in idx])
            dmeans = np.array([diff[i].mean() for i in idx])
            out.append(dict(
                protocol=protocol, predictor=pred, n_test_days=len(x),
                mean_auc=x.mean(), auc_lo=np.percentile(means, 2.5), auc_hi=np.percentile(means, 97.5),
                diff_vs_persistence=diff.mean(),
                diff_lo=np.percentile(dmeans, 2.5), diff_hi=np.percentile(dmeans, 97.5),
                share_days_better=float((diff > 0).mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/raw/railway_backups/backup_calibration_history_20260811_1706.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/reproduction"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load(Path(a.csv))
    feat = add_features(raw)
    self_check(feat, raw)
    dates = sorted(feat["snapshot_date"].unique())
    print(f"data: {len(raw):,} rows, {len(dates)} dates, backends {sorted(raw['backend'].unique())}")

    # first test day: the largest window (14) needs 14 earlier days that already have all 17 features
    # (features exist from the 3rd session onward) -> index 2 + 14 = 16
    first_idx = 2 + max(WINDOWS)
    per_date = run(feat, first_idx)
    per_date.to_csv(out / "rolling_per_date_auc.csv", index=False)

    common = sorted(per_date["date"].unique())
    summ = summarise(per_date, common)
    summ.to_csv(out / "rolling_summary.csv", index=False)

    with open(out / "rolling_summary.txt", "w") as f:
        f.write(f"Rolling prospective evaluation, {len(common)} test days "
                f"({common[0]} to {common[-1]}), unit = test day, block bootstrap CI (block 3, 2000 draws)\n\n")
        for protocol, g in summ.groupby("protocol"):
            f.write(f"== protocol: {protocol}\n")
            f.write(g.sort_values("mean_auc", ascending=False).round(4).to_string(index=False) + "\n\n")
    print((out / "rolling_summary.txt").read_text())


if __name__ == "__main__":
    main()
