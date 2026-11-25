# QubitTelemetry

Can historical calibration data predict which physical qubits on IBM Heron
processors will perform well, and does choosing qubits this way help real circuits?

**Status: research in progress.** Read these two files first:

- [`docs/EVIDENCE_LEDGER.md`](docs/EVIDENCE_LEDGER.md) - every result, its value, and the file it comes from
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) - mistakes and weak points found in the earlier write-ups
- [`docs/DATA_GAPS.md`](docs/DATA_GAPS.md) - what was lost when the Railway database trial expired

## What exists

| Part | Where |
|---|---|
| Audited ML evaluation (leakage + duplicate-row fix, leave-one-backend-out, future-date tests) | `analysis/notebooks/`, `results/` |
| Frozen models A (XGBoost), B and C (Random Forest), temporal 1/3/7/30-day models | `models/` |
| Hardware pilot 0 / 1 / 2 (IBM Quantum, GHZ circuits vs random chains) | `hardware/` |
| 7-day confirmatory round (day 1 = the first snapshot): daily data, rankings, hardware | `data/daily/`, `predictions/daily/`, `hardware/confirmatory_7day/` |
| Everything superseded (Railway-era code, old scripts) - kept, never deleted | `archive/` |

## Daily pipeline (no database)

`.github/workflows/collect_daily.yml` runs every day at 06:00 UTC:

1. `src/collect_daily.py` reads IBM's calibration data and writes plain CSV files (write-once).
2. `src/rank_daily.py` ranks qubits with the frozen Models B and C using **only earlier days**.
3. The new files are committed to this repository, so the git history is the audit trail.

## Tests (no IBM account needed)

```bash
pip install -r requirements.txt
python tests/test_features_equivalence.py   # live features == training features
python tests/test_pipeline_offline.py       # collector + ranking on simulated backends
```

## Where did every old file go?

`docs/FILE_MAP.csv` lists the original path, the new path and a SHA-256 for every
file from the two original project folders. `MANIFEST.sha256` covers this tree.

MIT License.
