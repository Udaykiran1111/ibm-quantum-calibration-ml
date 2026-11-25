# Evidence ledger

Status: **file** = read from a saved result file; **reproduced** = recomputed from raw data in the audit;
**audit** = computed in the Nov 2026 audit, reproduction script still to be added to `analysis/scripts/`.

## Machine learning

| Result | Value | Source | Status |
|---|---|---|---|
| First model (rejected: leakage + duplicate rows) | AUC 0.9993 | `docs/README_original_ibm_quantum.md` | author-reported |
| Model A leave-one-backend-out (XGBoost) | 0.8747 / 0.9435 / 0.8939, mean 0.9040 | `results/ml_evaluation/lobo_results.csv` | file |
| Model A future-date holdout (3 Jan 2026) | 0.8292 / 0.9159 / 0.8487 | `results/ml_evaluation/jan03_holdout_results.csv` | file |
| Model B / C leave-one-backend-out (Random Forest) | 0.8853 / 0.8838 | `results/ml_evaluation/model_b_c_mean_auc_comparison.csv` | file; C reproduced |
| Model B temporal holdout (Jul 20 - Aug 11) | AUC 0.8957 | `analysis/notebooks/01_model_b_7day_train_test.ipynb` | reproduced |
| Temporal windows 1/3/7/30 d, LOBO mean AUC | 0.8918 / 0.8766 / 0.8701 / 0.8939 | `results/temporal_sweet_spot/temporal_sweet_spot_selected_window_summary.csv` | file |
| Prospective AUC, Sep 4-6, windows 1/3/7/30 d | 0.888 / 0.895 / 0.899 / 0.900 | `results/temporal_sweet_spot/temporal_sweet_spot_all_prospective_predictions.csv` | audit |
| Top-10 precision on Sep 4-6 (base rate 0.62) | 0.91 - 0.96 | same file | audit |
| Persistence baseline (yesterday's label) | AUC 0.84 - 0.87 | `data/raw/railway_backups/` and the Sep 4-6 files | audit |
| Model A live, Jul 13-18 | AUC about 0.864 | `results/ml_evaluation/live_validation.csv` | audit |

## Hardware (IBM Quantum, 3-qubit GHZ circuits)

| Experiment | Design | Source | Status |
|---|---|---|---|
| Pilot 0 (Fez, Model A vs random) | 1 model run, 10 random | `hardware/pilot0_fez_model_a/` | file |
| Pilot 1 (Aug 23): Models A, B, C vs 3 random controls, 3 backends | 256 shots, one run per cell | `hardware/pilot1_abc_aug23/final_hardware_results.csv` | file |
| Pilot 2 (Sep 8-9): 30-day model chain vs 10 random chains, 3 backends | 11 circuits per backend, 256 shots | `hardware/pilot2_temporal_sep8/evaluation_original/hardware_all_loaded.csv` | file |
| Pilot 2 outcome | Sweet-spot ranked 3rd, 1st, 5th of 11 (Fez, Kingston, Marrakesh) | same file | audit |
| Predicted viability vs measured fidelity (33 circuits) | within-backend Spearman rho = +0.39, p = 0.025 (exploratory) | same file + Sep 6 predictions | audit |

Not claimed: that any model beats random selection with statistical confidence, or that any
training window is optimal. See `KNOWN_ISSUES.md`.
