"""Offline end-to-end check using simulated IBM backends and real July data."""
import subprocess, sys, tempfile, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/railway_backups/backup_calibration_history_20260811_1706.csv"
PY = sys.executable


def run(args, cwd):
    return subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True)


def test_collect_fake_and_write_once():
    with tempfile.TemporaryDirectory() as tmp:
        r = run([str(ROOT / "src/collect_daily.py"), "--fake", "--date", "2026-11-18"], tmp)
        assert r.returncode == 0, r.stderr
        cal = Path(tmp) / "_fake_out/calibration/2026-11-18.csv"
        cou = Path(tmp) / "_fake_out/couplers/2026-11-18.csv"
        meta = Path(tmp) / "_fake_out/meta/2026-11-18.json"
        df = pd.read_csv(cal); cdf = pd.read_csv(cou)
        assert set(df.backend) == {"ibm_fez", "ibm_kingston", "ibm_marrakesh"}
        assert 400 < len(df) <= 468, len(df)
        assert {"T1_us", "T2_us", "readout_error", "snapshot_date"} <= set(df.columns)
        assert 10 < df.T1_us.mean() < 1000
        assert set(cdf.gate) <= {"cz", "rzz"} and (cdf.gate == "cz").sum() >= 3 * 300 and cdf.gate_error.between(0, 1).all()  # baseline uses cz only
        assert meta.exists()
        h1 = hashlib.sha256(cal.read_bytes()).hexdigest()
        r2 = run([str(ROOT / "src/collect_daily.py"), "--fake", "--date", "2026-11-18"], tmp)
        assert "write-once" in r2.stdout
        assert hashlib.sha256(cal.read_bytes()).hexdigest() == h1
    print("  collect (simulated backends): schema, counts, couplers, write-once OK")


def test_rank_causal_and_graceful():
    full = pd.read_csv(CSV)
    cols = ["snapshot_date", "backend", "qubit", "T1_us", "T2_us", "readout_error"]
    with tempfile.TemporaryDirectory() as tmp:
        daily = Path(tmp) / "cal"; daily.mkdir()
        for d in ["2026-07-13", "2026-07-14", "2026-07-15", "2026-07-20"]:  # 07-20 = future file
            full[full.snapshot_date == d][cols].to_csv(daily / f"{d}.csv", index=False)
        common = ["--daily-dir", str(daily), "--models-dir", str(ROOT / "models/production"),
                  "--out-dir", str(Path(tmp) / "pred")]
        r = run([str(ROOT / "src/rank_daily.py"), "--target-date", "2026-07-14", *common], ROOT)
        assert r.returncode == 0 and "Not enough history" in r.stdout        # only 1 earlier session
        r = run([str(ROOT / "src/rank_daily.py"), "--target-date", "2026-07-16", *common], ROOT)
        assert r.returncode == 0, r.stderr
        for m in ("model_b", "model_c"):
            p = pd.read_csv(Path(tmp) / f"pred/2026-07-16_{m}.csv")
            assert (p.n_sessions == 3).all()                                  # 07-20 file ignored
            assert p.viability_score.between(0, 1).all()
            for _, g in p.groupby("backend"):
                assert sorted(g.viability_rank) == list(range(1, len(g) + 1))
        r = run([str(ROOT / "src/rank_daily.py"), "--target-date", "2026-07-16", *common], ROOT)
        assert "write-once" in r.stdout
    print("  rank: needs >=2 sessions, ignores later files, scores/ranks valid, write-once OK")


if __name__ == "__main__":
    test_collect_fake_and_write_once(); test_rank_causal_and_graceful(); print("pipeline: PASS")
