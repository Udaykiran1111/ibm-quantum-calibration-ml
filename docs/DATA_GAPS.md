# Data gaps

| Missing | Why | Impact |
|---|---|---|
| Calibration history 2026-08-12 -> 2026-09-07 (27 days, about 12,600 rows) | Stored only in a Railway MySQL database; the free trial expired | The temporal 1/3/7/30-day notebooks cannot be re-run from raw data |
| Live Model A/B/C rankings for the same period | Same database | Regenerable only if the calibration history is recovered |
| Calibration after 2026-09-07 (collector ran to about 2026-09-12) | Same database | None of the saved analyses used it |
| Aug 23 experiment manifest | Overwritten by a Sep 8 notebook re-run | Executed qubits are still recorded in the results file |
| Results of resubmitted Kingston job `dafpntd1ierc738nnmpg` | Job never returned; the IBM account was locked | None - the completed earlier Kingston job is available |

Still available:

- Calibration backup 2026-07-13 -> 2026-08-11 (30 days, 13,980 rows): `data/raw/railway_backups/`
- Model A raw training data (Dec 28 2025 -> Jan 3 2026): `data/raw/model_a_training/`
- Predictions and actual T1/T2/readout values for 2026-09-04, 05 and 06:
  `results/temporal_sweet_spot/temporal_*_predictions_2026-09-0*.csv`
- All trained models and every hardware result file with job IDs and raw counts.
