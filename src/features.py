"""
Shared feature / label logic.

This MUST stay identical to the training notebooks
(analysis/notebooks/01_model_b_7day_train_test.ipynb and
 02_model_c_30_observed_dates_train.ipynb): 17 features built from PRIOR
sessions only (expanding window over earlier sessions, ddof=1 std).
tests/test_features_equivalence.py proves the equivalence on real data.
"""
import pandas as pd

T1_THRESH = 100.0   # us
T2_THRESH = 50.0    # us
RE_THRESH = 0.05

HIST_FEATURES = [
    "hist_T1_mean", "hist_T1_std", "hist_T1_min", "hist_T1_max",
    "hist_T2_mean", "hist_T2_std", "hist_T2_min", "hist_T2_max",
    "hist_RE_mean", "hist_RE_std", "hist_RE_min", "hist_RE_max",
    "prev_T1", "prev_T2", "prev_RE",
    "hist_coherence_product", "hist_t2_t1_ratio",
]


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (
        (df["T1_us"] > T1_THRESH)
        & (df["T2_us"] > T2_THRESH)
        & (df["readout_error"] < RE_THRESH)
    ).astype(int)
    return df


def build_next_session_features(history: pd.DataFrame, target_date) -> pd.DataFrame:
    """
    Features for the session on `target_date`, using ONLY sessions strictly
    before `target_date`. One row per (backend, qubit) that has >= 2 prior
    sessions (the std feature needs two). Returns columns:
    backend, qubit, n_sessions + the 17 HIST_FEATURES.
    """
    h = history.copy()
    h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
    td = pd.Timestamp(target_date)
    h = h[h["snapshot_date"] < td]
    if h.empty:
        return pd.DataFrame(columns=["backend", "qubit", "n_sessions"] + HIST_FEATURES)
    assert h["snapshot_date"].max() < td, "LEAKAGE GUARD: history reaches target date"

    h = (h.sort_values(["backend", "qubit", "snapshot_date"])
          .drop_duplicates(["backend", "qubit", "snapshot_date"], keep="last"))
    g = h.groupby(["backend", "qubit"])
    out = g.agg(
        n_sessions=("snapshot_date", "nunique"),
        hist_T1_mean=("T1_us", "mean"), hist_T1_std=("T1_us", "std"),
        hist_T1_min=("T1_us", "min"), hist_T1_max=("T1_us", "max"),
        hist_T2_mean=("T2_us", "mean"), hist_T2_std=("T2_us", "std"),
        hist_T2_min=("T2_us", "min"), hist_T2_max=("T2_us", "max"),
        hist_RE_mean=("readout_error", "mean"), hist_RE_std=("readout_error", "std"),
        hist_RE_min=("readout_error", "min"), hist_RE_max=("readout_error", "max"),
        prev_T1=("T1_us", "last"), prev_T2=("T2_us", "last"),
        prev_RE=("readout_error", "last"),
    ).reset_index()
    out["hist_coherence_product"] = out["hist_T1_mean"] * out["hist_T2_mean"]
    out["hist_t2_t1_ratio"] = out["hist_T2_mean"] / (out["hist_T1_mean"] + 1e-9)
    out = out[out["n_sessions"] >= 2].reset_index(drop=True)
    return out[["backend", "qubit", "n_sessions"] + HIST_FEATURES]
