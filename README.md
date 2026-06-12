# IBM Quantum Calibration Profiling — Cross-Backend ML Stability Classification
  
## Overview
 
This project analyzes real calibration data from three IBM Heron-generation quantum backends to classify qubit **circuit-viability** using machine learning. The core novelty is a **Leave-One-Backend-Out (LOBO)** evaluation framework — the model is trained on two backends and tested on a third it has never seen, including across hardware generations (Heron r1 → r2).
 
Unlike prior work that uses synthetic or simulated data, this pipeline operates entirely on live IBM Quantum Platform calibration snapshots, with physics-grounded labels and full SHAP explainability.
 
---
 
## Backends Studied
 
| Backend | Generation | Qubits | Median T1 | Readout Error |
|---|---|---|---|---|
| ibm_fez | Heron r2 | 156 | 146.3 μs | 1.88% |
| ibm_torino | Heron r1 | 133 | 176.2 μs | 4.35% |
| ibm_marrakesh | Heron r2 | 156 | 193.1 μs | 3.29% |
 
---
 
## Key Results
 
### Leave-One-Backend-Out (LOBO) Classification
 
| Test Backend | Transfer Direction | AUC | Accuracy | F1 |
|---|---|---|---|---|
| ibm_fez | r1 + r2 → r2 | **0.9984** | 98.21% | 0.9860 |
| ibm_torino | r2 + r2 → r1 | **0.9998** | 99.85% | 0.9987 |
| ibm_marrakesh | r1 + r2 → r2 | **0.9996** | 98.85% | 0.9910 |
 
> **Mean LOBO AUC: 0.9993** — XGBoost generalises across Heron r1 and r2 generations with AUC > 0.998
 
### SHAP Feature Importance
 
| Rank | Feature | Importance |
|---|---|---|
| 1 | readout_error | 29.2% |
| 2 | coherence_product (T1 × T2) | 23.1% |
| 3 | T2_us | 20.3% |
| 4 | T1_us | 16.8% |
| 5 | qubit_RE_median | 6.4% |
 
> SHAP confirms the model learned real quantum physics — readout error and coherence dominate, matching established theory.
 
---
 
## Methodology
 
### 1. Data Collection
Calibration data collected directly from IBM Quantum Platform across 5 daily snapshots per backend, covering T1 relaxation time, T2 coherence time, readout error, and gate error per qubit.
 
### 2. Physics-Grounded Label
A qubit is labelled **circuit-viable (1)** if all three conditions hold:
 
```
T1  > 100 μs   — minimum coherence for multi-gate circuits
T2  >  50 μs   — minimum phase coherence
RE  <   5%     — below practical QEC threshold
```
 
This avoids the circular self-labelling problem common in prior work (percentile-based instability scores).
 
### 3. Feature Engineering
Nine features were engineered from raw calibration parameters:
 
```
T1_us, T2_us, readout_error          — raw calibration parameters
coherence_product  (T1 × T2)         — joint coherence quality metric
t2_t1_ratio        (T2 / T1)         — proximity to 2×T1 physical limit
qubit_T1_median                       — per-qubit historical stability
qubit_RE_median                       — per-qubit historical readout quality
qubit, snapshot_id                    — structural identifiers
```
 
### 4. Leave-One-Backend-Out (LOBO) Evaluation
```
Round 1:  Train on fez + marrakesh   →  Test on torino     (r2 → r1)
Round 2:  Train on torino + fez      →  Test on marrakesh  (r1+r2 → r2)
Round 3:  Train on torino + marrakesh →  Test on fez       (r1+r2 → r2)
```
 
This rigorously tests cross-backend generalisability — the hardest and most meaningful evaluation possible with this data.
 
### 5. Explainability
SHAP TreeExplainer applied globally and per-backend to validate that learned feature importance aligns with quantum physics theory.
 
---
 
## Statistical Analysis
 
Cross-backend differences validated using Mann-Whitney U tests (non-parametric, no normality assumption). All backend pairs differ significantly on T1, T2, and readout error at **p < 0.001**.
 
Qubit quality heterogeneity quantified using the **Gini coefficient of T1** — a novel metric applied to quantum calibration data showing within-backend coherence inequality.
 
---
 
## Repository Structure
 
```
ibm-quantum-calibration-ml/
│
├── notebooks/
│   └── quantum_calibration_analysis.ipynb   ← Full pipeline
│
├── figures/
│   ├── fig1_T1_distribution.png             ← T1 violin plots per backend
│   ├── fig2_T1_T2_scatter.png               ← T1 vs T2 with readout error
│   ├── fig3_drift.png                       ← Day-to-day median drift
│   ├── fig4_T1_variance.png                 ← Heron r1 vs r2 stability
│   ├── fig5_readout_heatmap.png             ← Readout error heatmap
│   ├── fig6_label_distribution.png          ← Circuit-viable qubit fraction
│   ├── fig7_roc_cm.png                      ← ROC curves + confusion matrices
│   ├── fig8_shap.png                        ← SHAP global importance
│   ├── fig9_shap_per_backend.png            ← SHAP per backend comparison
│   └── fig10_qubit_correlation.png          ← Inter-qubit T1 correlation
│
├── data/
│   └── .gitkeep                             ← CSV not included (see below)
│
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Dataset
 
Data collected from [IBM Quantum Platform](https://quantum.ibm.com) using the Qiskit Runtime API (`backend.properties()`).
 
> **The raw CSV is not included in this repository.**  
> To reproduce the dataset: access IBM Quantum Platform, select the target backends (ibm_fez, ibm_torino, ibm_marrakesh), and collect calibration snapshots using `backend.properties()` over multiple days.
 
Place your CSV in the `data/` folder and update `DATA_PATH` in the notebook before running.
 
---
 
## Setup & Reproduction
 
```bash
# Clone the repo
git clone https://github.com/Udaykiran1111/ibm-quantum-calibration-ml.git
cd ibm-quantum-calibration-ml
 
# Install dependencies
pip install -r requirements.txt
 
# Open the notebook
jupyter notebook notebooks/quantum_calibration_analysis.ipynb
```
 
Update the `DATA_PATH` variable in Section 1 of the notebook to point to your local CSV file.
 
---
 
## Output Figures
 
All 10 figures are pre-generated in the `figures/` folder and embedded in the notebook output. No re-running required to view results.
 
---
 
## Publication Status
 
📄 **Manuscript in preparation** — targeting IEEE Quantum Week (QCE) or an arXiv preprint (quant-ph).
 
---
 
## Author
 
**Vattikuti Uday Kiran**  
B.Tech Computer Science & Engineering (Data Science & ML Minor)  
Lovely Professional University, Jalandhar, India  
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-udaykiran1111-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/uday-kiran-vattikuti)
[![GitHub](https://img.shields.io/badge/GitHub-Udaykiran1111-black?style=flat-square&logo=github)](https://github.com/Udaykiran1111)
 
---
 

---
 
## License
 
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
