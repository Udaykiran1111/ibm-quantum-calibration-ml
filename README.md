<<<<<<< HEAD
# IBM Quantum Calibration Profiling — ML Stability Classification

**Status:** Research in progress · Manuscript in preparation

## Overview

This project analyzes real IBM Quantum backend calibration data across 
three Heron-generation processors — ibm_fez (r2), ibm_torino (r1), 
and ibm_marrakesh (r2) — to classify qubit circuit-viability using 
machine learning.

The key contribution is a Leave-One-Backend-Out (LOBO) evaluation 
framework that tests whether stability patterns learned on one quantum 
backend generalise to a completely unseen backend — including across 
hardware generations (Heron r1 → r2).

## Key Results

| Test Backend   | Transfer      | AUC    | F1     |
|----------------|---------------|--------|--------|
| ibm_fez        | r1+r2 → r2    | 0.9984 | 0.9860 |
| ibm_torino     | r2+r2 → r1    | 0.9998 | 0.9987 |
| ibm_marrakesh  | r1+r2 → r2    | 0.9996 | 0.9910 |

Mean LOBO AUC: **0.9993**

## Dataset

Data collected directly from IBM Quantum Platform using the Qiskit 
Runtime API. The dataset covers:
- 3 backends: ibm_fez, ibm_torino, ibm_marrakesh
- 5 daily snapshots per backend
- 53,400 records total (T1, T2, readout error per qubit)

> The raw CSV is not included in this repository. 
> To reproduce: collect calibration data from 
> [IBM Quantum Platform](https://quantum.ibm.com) 
> using the backend.properties() API.

## Methodology

- **Label:** Physics-grounded circuit-viability 
  (T1 > 100μs AND T2 > 50μs AND readout error < 5%)
- **Model:** XGBoost with 9 engineered features 
  (coherence product, T2/T1 ratio, per-qubit medians)
- **Evaluation:** Leave-One-Backend-Out cross-validation
- **Explainability:** SHAP feature importance analysis

## Repository Structure

## Setup

```bash
pip install -r requirements.txt
```

Then open `notebooks/quantum_calibration_analysis.ipynb` 
and update `DATA_PATH` to your local CSV file.

## Author

Vattikuti Uday Kiran  
B.Tech CSE (Data Science & ML), Lovely Professional University  
[LinkedIn](https://linkedin.com/in/udaykiran1111) · 
[GitHub](https://github.com/Udaykiran1111)

## Citation

> Manuscript in preparation. Please check back for arXiv link.
=======
# ibm-quantum-calibration-ml
Cross-backend calibration profiling of IBM Heron quantum processors using XGBoost and SHAP — research in progress
>>>>>>> f36dc9e38070d81daea1e00b1b9b06d2fca56f04
