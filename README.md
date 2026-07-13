# IBM Quantum Calibration Profiling — Cross-Backend ML Stability Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live%20Collection%20Running-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Can a model trained on one IBM quantum computer predict qubit stability on a completely different one — including across hardware generations?**

</div>

---

## Overview

This project collects real calibration data from three IBM Heron-generation quantum backends and classifies qubit **circuit-viability** using machine learning. The core contribution is a **Leave-One-Backend-Out (LOBO)** evaluation framework — train on two backends, test on a third never seen during training — including cross-generation transfer (Heron r1 → r2).

### What makes this different from prior work

| Prior Work | This Project |
|---|---|
| Single backend only | 3 backends, cross-generation transfer |
| Random 80/20 split | Leave-One-Backend-Out (LOBO) evaluation |
| No explainability | SHAP per feature, per backend |
| Percentile-based labels (circular) | Physics-grounded label with temporal separation |
| Static snapshot | Live 7-day collection running now |

---

## Key Results (Final Model — v2, Leak-Proof)

### Leave-One-Backend-Out Classification (Random Forest, 17 Historical Features)

| Test Backend | Transfer | AUC | Balanced Accuracy | MCC |
|---|---|---|---|---|
| ibm_fez | r1+r2 → r2 | **0.8747** | 0.8428 | 0.688 |
| ibm_torino | r2+r2 → r1 | **0.9435** | 0.9023 | 0.803 |
| ibm_marrakesh | r1+r2 → r2 | **0.8939** | 0.8115 | 0.672 |
| **Mean** | | **0.9040** | **0.852** | **0.721** |

### Bootstrap 95% Confidence Intervals (2000 iterations, cluster by qubit)

| Backend | AUC | 95% CI |
|---|---|---|
| ibm_fez | 0.8747 | [0.822, 0.921] |
| ibm_torino | 0.9435 | [0.911, 0.971] |
| ibm_marrakesh | 0.8939 | [0.853, 0.928] |

All CI lower bounds above 0.82 — results are statistically stable.

### Jan03 Next-Day Holdout (Genuine Production Test)

| Backend | AUC | Balanced Accuracy |
|---|---|---|
| ibm_fez | 0.8292 | 0.7602 |
| ibm_torino | 0.9159 | 0.8699 |
| ibm_marrakesh | 0.8487 | 0.7890 |
| **Mean** | **0.8646** | **0.806** |

Model trained on Dec 28–Jan 02, tested on Jan 03 data it never saw. Mean AUC 0.865 on a future date.

### Top SHAP Features (Statistical Utility — Not Physical Causality)

| Rank | Feature | Importance |
|---|---|---|
| 1 | hist_coherence_product (T1×T2) | 15.9% |
| 2 | hist_RE_mean | 11.4% |
| 3 | hist_RE_min | 9.5% |
| 4 | hist_t2_t1_ratio | 7.7% |
| 5 | hist_T2_mean | 7.1% |

Coherence and readout quality dominate — consistent with quantum decoherence theory.

---

## Scientific Integrity — Three Flaws Fixed

An earlier version of this project reported AUC 0.9993. Peer review identified three flaws. Both the flawed and corrected versions are preserved in this repository.

| Flaw | Where it was | How it was fixed |
|---|---|---|
| **Label leakage** — model learned its own formula | `01_exploratory_analysis.ipynb` | Label from current session; features from prior sessions only (temporal separation) |
| **Pseudoreplication** — 53,400 rows = 24× copies of 2,225 real observations | Original notebook | Deduplicated to session level before any modelling |
| **SHAP overclaim** — said model "learned quantum physics" | Original README | Corrected to "consistent with decoherence theory" |

---

## Backends Studied

| Backend | Generation | Qubits | Mean T1 | Mean RE |
|---|---|---|---|---|
| ibm_fez | Heron r2 | 156 | 144.5 μs | 1.88% |
| ibm_torino | Heron r1 | 133 | 176.2 μs | 4.35% |
| ibm_marrakesh | Heron r2 | 156 | 195.2 μs | 3.29% |

---

## Methodology

### Physics-Grounded Label (No Circular Self-Labelling)

A qubit is **circuit-viable (label=1)** if ALL three conditions hold in the current calibration session:

```
T1 > 100 μs   — minimum relaxation for ~10-gate circuits
T2 >  50 μs   — minimum dephasing coherence
RE <   5%     — below practical QEC threshold
```

### Feature Engineering (17 Historical Features — Zero Leakage)

Features are computed exclusively from **prior calibration sessions** using a shift+expanding window. The current session's values are never used as features — only as the prediction target. Verified by a 3-test mandatory audit before every model training run.

```
hist_T1_mean/std/min/max      — T1 rolling statistics from prior days
hist_T2_mean/std/min/max      — T2 rolling statistics from prior days
hist_RE_mean/std/min/max      — Readout error rolling statistics
prev_T1, prev_T2, prev_RE     — Most recent prior snapshot values
hist_coherence_product         — hist_T1_mean × hist_T2_mean
hist_t2_t1_ratio               — hist_T2_mean / hist_T1_mean
```

### LOBO Evaluation

```
Round 1:  Train fez + marrakesh  →  Test torino     (r2+r2 → r1)
Round 2:  Train torino + fez     →  Test marrakesh  (r1+r2 → r2)
Round 3:  Train torino + marrakesh →  Test fez      (r1+r2 → r2)
```

---

## Repository Structure

```
ibm-quantum-calibration-ml/
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   ← Original pipeline (preserved, flaws documented)
│   └── 02_final_model.ipynb            ← Corrected leak-proof pipeline (current)
│
├── figures/
│   ├── fig_roc_lobo.png                ← LOBO ROC curves with bootstrap CI
│   ├── fig_bootstrap_ci.png            ← AUC confidence interval plot
│   ├── fig_shap.png                    ← SHAP feature importance
│   └── (fig1–fig10 from exploratory analysis)
│
├── results/
│   ├── lobo_results.csv                ← Per-backend LOBO metrics
│   ├── bootstrap_ci.csv                ← 95% CIs
│   ├── jan03_holdout_results.csv       ← Next-day production test
│   └── shap_importance.csv            ← Feature importance table
│
├── data/
│   └── .gitkeep                        ← CSVs not included (see below)
│
├── models/
│   └── .gitkeep                        ← qubit_model_v2.pkl not pushed (binary)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

Data collected directly from IBM Quantum Platform using `backend.properties()` API.

> **Raw CSV files are not included** — they contain IBM Quantum telemetry data.  
> To reproduce: create an IBM Quantum account at quantum.ibm.com, collect calibration snapshots from ibm_fez, ibm_torino, ibm_marrakesh over multiple consecutive days, and place the CSV in `data/`.

**Training window:** Dec 28, 2025 → Jan 02, 2026 (5 consecutive days, no gaps)  
**Holdout:** Jan 03, 2026 (next-day prediction test)  
**Live collection:** Running daily — July 2026 onward

---

## Setup

```bash
git clone https://github.com/Udaykiran1111/ibm-quantum-calibration-ml.git
cd ibm-quantum-calibration-ml
pip install -r requirements.txt
jupyter notebook notebooks/02_final_model.ipynb
```

---

## Publication Status

📄 **Manuscript in preparation** — targeting IEEE Quantum Week (QCE 2026) or arXiv (quant-ph).

---

## Author

**Vattikuti Uday Kiran**  
B.Tech CSE (Data Science & ML), Lovely Professional University, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-uday--kiran--vattikuti-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/uday-kiran-vattikuti)
[![GitHub](https://img.shields.io/badge/GitHub-Udaykiran1111-black?style=flat-square&logo=github)](https://github.com/Udaykiran1111)

---

## Citation

```bibtex
@misc{udaykiran2025ibmquantum,
  author = {Vattikuti Uday Kiran},
  title  = {Cross-Backend Calibration Profiling of IBM Heron Quantum Processors
            Using Leak-Proof Machine Learning},
  year   = {2025},
  note   = {Manuscript in preparation},
  url    = {https://github.com/Udaykiran1111/ibm-quantum-calibration-ml}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.