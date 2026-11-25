# Known issues in the earlier write-ups (found during the Nov 2026 audit)

These are recorded, not hidden. The corrected analysis supersedes the originals.

1. **Algorithm label.** The README/article table "Final corrected Random Forest results"
   (mean AUC 0.9040) is the **XGBoost** result. `results/ml_evaluation/lobo_results.csv`
   shows XGBoost 0.8747 / 0.9435 / 0.8939 (mean 0.9040) and Random Forest
   0.9178 / 0.9457 / 0.8825 (mean 0.9153). Model A's metadata says XGBoost.
2. **"7-day Model B won on hardware" is within noise.** On ibm_fez Models A and B ran the
   *same* qubits [21,22,23] (0.957 vs 0.973). On ibm_kingston Models B and C ran the *same*
   chain [149,150,151] (0.980 vs 0.969). Pilot 1 used 3 random controls per backend and one run per cell.
3. **"Window" means training window.** In the 1/3/7/30-day experiment the features always use
   all earlier history; only the number of recent days used to train the classifier changes.
4. **No detectable best window on future dates.** Leave-one-backend-out favoured 30 days
   (0.8939 vs 0.8701 for 7 days) but the confidence intervals overlap, and on the Sep 4-6
   prospective test all windows land at mean AUC 0.888-0.900 (audit-computed).
5. **A simple baseline is strong.** Yesterday's label alone gives AUC 0.84-0.87 on the same data;
   the models add about +0.04 to +0.05 (audit-computed). Model A, trained in Dec-Jan, scored about
   0.864 on the first six live July days versus about 0.878 for persistence on the same days.
6. **Kingston.** `hardware/pilot2_temporal_sep8/final_analysis_original/` lists three primary
   backends but computes on two (22 observations). The completed Kingston job
   `dafgfgl1ierc738nbdd0` (status DONE) was dropped in favour of a resubmitted job
   `dafpntd1ierc738nnmpg` that never returned. The corrected analysis must include the completed job.
7. **Invalid statistics in the original hardware analysis:** a "bootstrap interval" of
   [0.0570, 0.0613] (just the two backend values), a Pearson r = -1.0 on two points, and a sign test
   with p = 0.5 on two points. Do not cite them.
8. **Timing.** Pilot 2 predictions are for Sep 6, but the jobs ran Sep 8-9 (job_state file).
9. **Manifest overwritten.** `hardware/pilot1_abc_aug23/final_experiment_manifest.*` is dated
   2026-09-08 while the results are from Aug 23. The executed qubits are in
   `hardware/pilot1_abc_aug23/final_hardware_results.csv`; the Aug 23 manifest itself is lost.
10. **Level-1 Aer simulations** (`hardware/level1_simulation/`) build their noise model from the same
    calibration data the models learn from, so they are not independent evidence.
11. **Layout orientation.** Old scripts passed the sorted physical qubit numbers as the layout, so
    some random chains needed swaps (depth 17-18 vs 12). The new round orients the circuit along the
    physical path.
12. **Absolute paths.** Some saved CSV/JSON files contain the author's local paths
    (`D:\Downloads\...`). They are provenance only.
