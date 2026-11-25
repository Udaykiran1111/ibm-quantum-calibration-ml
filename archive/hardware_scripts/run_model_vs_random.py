# """
# QuantumCircuitBench
# ===================

# LEVEL 1 — MODEL VS RANDOM EXPERIMENT

# Research question
# -----------------
# Do qubits selected by the QubitTelemetry viability models produce
# better simulated quantum-circuit performance than randomly selected
# qubits when the same real IBM calibration data is used to construct
# the noise model?

# Groups
# ------
# 1. Model A TOP
# 2. Model B TOP
# 3. Model C TOP
# 4. RANDOM CONTROL
# 5. WORST CONTROL

# Backends
# --------
# - ibm_fez
# - ibm_kingston
# - ibm_marrakesh

# Circuit
# -------
# 3-qubit GHZ state.

# Noise
# -----
# The noise model is constructed from the real T1, T2 and readout-error
# values stored in the QubitTelemetry Railway database.

# IMPORTANT
# ---------
# This is a LOCAL Qiskit Aer simulation.

# No IBM QPU time is consumed.

# This experiment does NOT claim to measure real hardware performance.
# It tests whether the ML-selected qubits have better predicted
# calibration-driven circuit performance than controls.
# """

# from __future__ import annotations

# import json
# import random
# from datetime import date
# from pathlib import Path

# import numpy as np
# import pandas as pd

# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import (
#     NoiseModel,
#     thermal_relaxation_error,
#     ReadoutError,
# )


# # ============================================================
# # CONFIGURATION
# # ============================================================

# SHOTS = 4096
# N_QUBITS = 3

# # Fixed seed makes the random control reproducible.
# RANDOM_SEED = 20260822

# # Number of random control groups used for the experiment.
# # We use multiple random trials so one lucky/unlucky random
# # selection does not dominate the result.
# RANDOM_TRIALS = 500

# OUTPUT_DIR = Path("results")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RESULT_JSON = OUTPUT_DIR / "model_vs_random_results.json"
# RESULT_CSV = OUTPUT_DIR / "model_vs_random_results.csv"


# # Approximate gate durations used ONLY for the Level-1
# # calibration-driven simulation.
# GATE_TIME_1Q = 50e-9
# GATE_TIME_2Q = 300e-9

# DEFAULT_T1_US = 100.0
# DEFAULT_T2_US = 50.0
# DEFAULT_READOUT_ERROR = 0.05


# # ============================================================
# # DATABASE
# # ============================================================

# def load_database():
#     """
#     Import the existing project's database connector.

#     The experiment is read-only.
#     """
#     from railway_db import get_connection

#     return get_connection()


# def get_latest_snapshot_date(conn) -> str:
#     sql = """
#         SELECT MAX(snapshot_date)
#         FROM calibration_history
#     """

#     cur = conn.cursor()
#     cur.execute(sql)

#     value = cur.fetchone()[0]
#     cur.close()

#     if value is None:
#         raise RuntimeError("No calibration data found in Railway database.")

#     return str(value)


# def load_calibration(conn, snapshot_date: str) -> pd.DataFrame:
#     """
#     Load real calibration values for one snapshot.
#     """

#     sql = """
#         SELECT
#             backend,
#             qubit,
#             T1_us,
#             T2_us,
#             readout_error,
#             snapshot_date
#         FROM calibration_history
#         WHERE snapshot_date = %s
#         ORDER BY backend, qubit
#     """

#     return pd.read_sql(sql, conn, params=[snapshot_date])


# def load_rankings(conn, snapshot_date: str) -> pd.DataFrame:
#     """
#     Load model rankings for the same snapshot.
#     """

#     sql = """
#         SELECT
#             model_name,
#             backend,
#             qubit,
#             snapshot_date,
#             viability_score,
#             viability_rank,
#             label
#         FROM qubit_rankings
#         WHERE snapshot_date = %s
#         ORDER BY model_name, backend, viability_rank
#     """

#     return pd.read_sql(sql, conn, params=[snapshot_date])


# # ============================================================
# # DATA VALIDATION
# # ============================================================

# def validate_data(calibration: pd.DataFrame, rankings: pd.DataFrame):
#     required_cal = {
#         "backend",
#         "qubit",
#         "T1_us",
#         "T2_us",
#         "readout_error",
#         "snapshot_date",
#     }

#     required_rank = {
#         "model_name",
#         "backend",
#         "qubit",
#         "snapshot_date",
#         "viability_score",
#         "viability_rank",
#     }

#     missing_cal = required_cal - set(calibration.columns)
#     missing_rank = required_rank - set(rankings.columns)

#     if missing_cal:
#         raise RuntimeError(
#             f"Calibration data missing columns: {sorted(missing_cal)}"
#         )

#     if missing_rank:
#         raise RuntimeError(
#             f"Ranking data missing columns: {sorted(missing_rank)}"
#         )

#     if calibration.empty:
#         raise RuntimeError("Calibration dataframe is empty.")

#     if rankings.empty:
#         raise RuntimeError("Ranking dataframe is empty.")

#     print()
#     print("DATA VALIDATION")
#     print("-" * 60)
#     print(f"Calibration rows : {len(calibration):,}")
#     print(f"Ranking rows     : {len(rankings):,}")
#     print(
#         "Backends         :",
#         sorted(calibration["backend"].dropna().unique().tolist())
#     )
#     print(
#         "Models           :",
#         sorted(rankings["model_name"].dropna().unique().tolist())
#     )

#     print("PASS")


# # ============================================================
# # QUANTUM CIRCUIT
# # ============================================================

# def build_ghz_circuit(n_qubits: int = 3) -> QuantumCircuit:
#     """
#     Build:

#         |000> + |111>
#         ----------------
#               sqrt(2)

#     """

#     qc = QuantumCircuit(n_qubits, n_qubits)

#     qc.h(0)

#     for i in range(n_qubits - 1):
#         qc.cx(i, i + 1)

#     qc.measure(range(n_qubits), range(n_qubits))

#     return qc


# def ideal_success_probability(counts: dict, shots: int) -> float:
#     """
#     For a GHZ circuit the valid results are only:

#         000
#         111
#     """

#     correct = counts.get("000", 0) + counts.get("111", 0)

#     return correct / shots


# # ============================================================
# # NOISE MODEL
# # ============================================================

# def clean_value(value, default):
#     if value is None:
#         return default

#     try:
#         value = float(value)

#         if not np.isfinite(value):
#             return default

#         return value

#     except Exception:
#         return default


# def build_noise_model(qubits: list[dict]) -> NoiseModel:
#     """
#     Build a per-qubit calibration-driven noise model.

#     q0, q1, q2 correspond to the physical qubits selected
#     by the experiment.
#     """

#     noise_model = NoiseModel()

#     for physical_position, q in enumerate(qubits):

#         T1_us = clean_value(q["T1_us"], DEFAULT_T1_US)
#         T2_us = clean_value(q["T2_us"], DEFAULT_T2_US)
#         readout = clean_value(
#             q["readout_error"],
#             DEFAULT_READOUT_ERROR,
#         )

#         # Convert microseconds -> seconds.
#         T1 = T1_us * 1e-6
#         T2 = T2_us * 1e-6

#         # Physical constraint.
#         T2 = min(T2, 2.0 * T1)

#         readout = min(max(readout, 0.0), 0.5)

#         # ----------------------------------------------------
#         # Single-qubit relaxation/dephasing
#         # ----------------------------------------------------

#         one_q_error = thermal_relaxation_error(
#             T1,
#             T2,
#             GATE_TIME_1Q,
#         )

#         noise_model.add_quantum_error(
#             one_q_error,
#             ["h", "x", "rx", "ry", "rz"],
#             [physical_position],
#         )

#         # ----------------------------------------------------
#         # Readout
#         # ----------------------------------------------------

#         readout_error = ReadoutError(
#             [
#                 [1 - readout, readout],
#                 [readout, 1 - readout],
#             ]
#         )

#         noise_model.add_readout_error(
#             readout_error,
#             [physical_position],
#         )

#     # --------------------------------------------------------
#     # Two-qubit gates
#     #
#     # NOTE:
#     # This Level-1 simulation uses a generic linear topology.
#     # It is NOT claiming to reproduce the exact IBM coupling map.
#     # --------------------------------------------------------

#     for i in range(len(qubits) - 1):

#         q1 = qubits[i]
#         q2 = qubits[i + 1]

#         T1_1 = clean_value(q1["T1_us"], DEFAULT_T1_US) * 1e-6
#         T2_1 = clean_value(q1["T2_us"], DEFAULT_T2_US) * 1e-6

#         T1_2 = clean_value(q2["T1_us"], DEFAULT_T1_US) * 1e-6
#         T2_2 = clean_value(q2["T2_us"], DEFAULT_T2_US) * 1e-6

#         T2_1 = min(T2_1, 2.0 * T1_1)
#         T2_2 = min(T2_2, 2.0 * T1_2)

#         error_1 = thermal_relaxation_error(
#             T1_1,
#             T2_1,
#             GATE_TIME_2Q,
#         )

#         error_2 = thermal_relaxation_error(
#             T1_2,
#             T2_2,
#             GATE_TIME_2Q,
#         )

#         two_q_error = error_1.tensor(error_2)

#         noise_model.add_quantum_error(
#             two_q_error,
#             ["cx"],
#             [i, i + 1],
#         )

#     return noise_model


# # ============================================================
# # SIMULATION
# # ============================================================

# def simulate_group(
#     qubits: list[dict],
#     shots: int = SHOTS,
# ) -> dict:

#     if len(qubits) != N_QUBITS:
#         raise ValueError(
#             f"Exactly {N_QUBITS} qubits are required."
#         )

#     circuit = build_ghz_circuit(N_QUBITS)

#     noise_model = build_noise_model(qubits)

#     simulator = AerSimulator(
#         noise_model=noise_model
#     )

#     transpiled = transpile(
#         circuit,
#         simulator,
#         optimization_level=0,
#     )

#     job = simulator.run(
#         transpiled,
#         shots=shots,
#     )

#     counts = job.result().get_counts()

#     fidelity = ideal_success_probability(
#         counts,
#         shots,
#     )

#     forbidden_probability = 1.0 - fidelity

#     return {
#         "fidelity": round(fidelity, 6),
#         "forbidden_probability": round(
#             forbidden_probability,
#             6,
#         ),
#         "counts": counts,
#     }


# # ============================================================
# # QUbit SELECTION
# # ============================================================

# def select_top_qubits(
#     rankings_backend: pd.DataFrame,
#     calibration_backend: pd.DataFrame,
#     n: int,
# ) -> list[dict]:

#     ranked = rankings_backend.sort_values(
#         ["viability_rank", "viability_score"],
#         ascending=[True, False],
#     ).head(n)

#     return merge_ranking_with_calibration(
#         ranked,
#         calibration_backend,
#     )


# def select_worst_qubits(
#     rankings_backend: pd.DataFrame,
#     calibration_backend: pd.DataFrame,
#     n: int,
# ) -> list[dict]:

#     ranked = rankings_backend.sort_values(
#         ["viability_rank", "viability_score"],
#         ascending=[False, False],
#     ).head(n)

#     return merge_ranking_with_calibration(
#         ranked,
#         calibration_backend,
#     )


# def merge_ranking_with_calibration(
#     ranked: pd.DataFrame,
#     calibration_backend: pd.DataFrame,
# ) -> list[dict]:

#     merged = ranked.merge(
#         calibration_backend[
#             [
#                 "qubit",
#                 "T1_us",
#                 "T2_us",
#                 "readout_error",
#             ]
#         ],
#         on="qubit",
#         how="inner",
#         validate="one_to_one",
#     )

#     if len(merged) != len(ranked):
#         missing = set(ranked["qubit"]) - set(merged["qubit"])

#         raise RuntimeError(
#             "Ranking/calibration mismatch. "
#             f"Missing qubits: {sorted(missing)}"
#         )

#     return merged.to_dict("records")


# def select_random_qubits(
#     calibration_backend: pd.DataFrame,
#     excluded_qubits: set,
#     rng: random.Random,
#     n: int,
# ) -> list[dict]:

#     available = calibration_backend[
#         ~calibration_backend["qubit"].isin(excluded_qubits)
#     ].copy()

#     if len(available) < n:
#         raise RuntimeError(
#             "Not enough qubits for random control."
#         )

#     indices = rng.sample(
#         list(available.index),
#         n,
#     )

#     return available.loc[indices].to_dict("records")


# # ============================================================
# # SUMMARY
# # ============================================================

# def summarize_group(qubits: list[dict]) -> dict:

#     return {
#         "qubits": [
#             int(q["qubit"])
#             for q in qubits
#         ],
#         "mean_T1_us": round(
#             float(
#                 np.mean(
#                     [
#                         clean_value(
#                             q["T1_us"],
#                             DEFAULT_T1_US,
#                         )
#                         for q in qubits
#                     ]
#                 )
#             ),
#             3,
#         ),
#         "mean_T2_us": round(
#             float(
#                 np.mean(
#                     [
#                         clean_value(
#                             q["T2_us"],
#                             DEFAULT_T2_US,
#                         )
#                         for q in qubits
#                     ]
#                 )
#             ),
#             3,
#         ),
#         "mean_readout_error": round(
#             float(
#                 np.mean(
#                     [
#                         clean_value(
#                             q["readout_error"],
#                             DEFAULT_READOUT_ERROR,
#                         )
#                         for q in qubits
#                     ]
#                 )
#             ),
#             6,
#         ),
#     }


# # ============================================================
# # MAIN EXPERIMENT
# # ============================================================

# def main():

#     print("=" * 72)
#     print("QUANTUM CIRCUIT BENCH")
#     print("LEVEL 1 — MODEL VS RANDOM")
#     print("=" * 72)

#     print(f"Shots          : {SHOTS}")
#     print(f"Circuit        : {N_QUBITS}-qubit GHZ")
#     print(f"Random trials  : {RANDOM_TRIALS}")
#     print(f"Random seed    : {RANDOM_SEED}")

#     conn = load_database()

#     try:

#         snapshot = get_latest_snapshot_date(conn)

#         print()
#         print(f"Latest Railway snapshot: {snapshot}")

#         calibration = load_calibration(
#             conn,
#             snapshot,
#         )

#         rankings = load_rankings(
#             conn,
#             snapshot,
#         )

#     finally:
#         conn.close()

#     validate_data(
#         calibration,
#         rankings,
#     )

#     models = sorted(
#         rankings["model_name"]
#         .dropna()
#         .unique()
#         .tolist()
#     )

#     backends = sorted(
#         calibration["backend"]
#         .dropna()
#         .unique()
#         .tolist()
#     )

#     all_results = []

#     # ========================================================
#     # BACKEND LOOP
#     # ========================================================

#     for backend in backends:

#         print()
#         print("=" * 72)
#         print(f"BACKEND: {backend}")
#         print("=" * 72)

#         cal_backend = calibration[
#             calibration["backend"] == backend
#         ].copy()

#         rank_backend = rankings[
#             rankings["backend"] == backend
#         ].copy()

#         print(
#             f"Available physical qubits: {len(cal_backend)}"
#         )

#         # ----------------------------------------------------
#         # Model experiments
#         # ----------------------------------------------------

#         model_groups = {}

#         for model in models:

#             model_rank = rank_backend[
#                 rank_backend["model_name"] == model
#             ].copy()

#             if len(model_rank) < N_QUBITS:
#                 print(
#                     f"Skipping {model}: "
#                     f"only {len(model_rank)} ranking rows."
#                 )
#                 continue

#             top_qubits = select_top_qubits(
#                 model_rank,
#                 cal_backend,
#                 N_QUBITS,
#             )

#             print()
#             print(
#                 f"{model.upper()} TOP: "
#                 f"{[q['qubit'] for q in top_qubits]}"
#             )

#             result = simulate_group(
#                 top_qubits,
#                 SHOTS,
#             )

#             stats = summarize_group(
#                 top_qubits,
#             )

#             print(
#                 f"  Fidelity       : {result['fidelity']:.4f}"
#             )

#             print(
#                 f"  Error          : "
#                 f"{result['forbidden_probability']:.4f}"
#             )

#             print(
#                 f"  Mean T1        : "
#                 f"{stats['mean_T1_us']:.2f} us"
#             )

#             print(
#                 f"  Mean T2        : "
#                 f"{stats['mean_T2_us']:.2f} us"
#             )

#             print(
#                 f"  Mean readout   : "
#                 f"{stats['mean_readout_error']:.4f}"
#             )

#             model_groups[model] = top_qubits

#             all_results.append(
#                 {
#                     "snapshot_date": snapshot,
#                     "backend": backend,
#                     "group": "top",
#                     "model": model,
#                     "trial": 0,
#                     "fidelity": result["fidelity"],
#                     "forbidden_probability":
#                         result["forbidden_probability"],
#                     **stats,
#                 }
#             )

#         # ----------------------------------------------------
#         # WORST MODEL-A/B/C control
#         #
#         # We use the worst-ranked qubits from each model.
#         # ----------------------------------------------------

#         for model in models:

#             model_rank = rank_backend[
#                 rank_backend["model_name"] == model
#             ].copy()

#             if len(model_rank) < N_QUBITS:
#                 continue

#             worst_qubits = select_worst_qubits(
#                 model_rank,
#                 cal_backend,
#                 N_QUBITS,
#             )

#             result = simulate_group(
#                 worst_qubits,
#                 SHOTS,
#             )

#             stats = summarize_group(
#                 worst_qubits,
#             )

#             print()
#             print(
#                 f"{model.upper()} WORST: "
#                 f"{[q['qubit'] for q in worst_qubits]}"
#             )

#             print(
#                 f"  Fidelity: {result['fidelity']:.4f}"
#             )

#             all_results.append(
#                 {
#                     "snapshot_date": snapshot,
#                     "backend": backend,
#                     "group": "worst",
#                     "model": model,
#                     "trial": 0,
#                     "fidelity": result["fidelity"],
#                     "forbidden_probability":
#                         result["forbidden_probability"],
#                     **stats,
#                 }
#             )

#         # ----------------------------------------------------
#         # RANDOM CONTROL
#         #
#         # Random trials are independent and reproducible.
#         # ----------------------------------------------------

#         print()
#         print("RANDOM CONTROL")

#         for trial in range(1, RANDOM_TRIALS + 1):

#             rng = random.Random(
#                 RANDOM_SEED
#                 + hash(backend) % 100000
#                 + trial
#             )

#             random_qubits = select_random_qubits(
#                 cal_backend,
#                 excluded_qubits=set(),
#                 rng=rng,
#                 n=N_QUBITS,
#             )

#             result = simulate_group(
#                 random_qubits,
#                 SHOTS,
#             )

#             stats = summarize_group(
#                 random_qubits,
#             )

#             print(
#                 f"  Trial {trial:02d}: "
#                 f"qubits={[q['qubit'] for q in random_qubits]} "
#                 f"fidelity={result['fidelity']:.4f}"
#             )

#             all_results.append(
#                 {
#                     "snapshot_date": snapshot,
#                     "backend": backend,
#                     "group": "random",
#                     "model": "random_control",
#                     "trial": trial,
#                     "fidelity": result["fidelity"],
#                     "forbidden_probability":
#                         result["forbidden_probability"],
#                     **stats,
#                 }
#             )

#     # ========================================================
#     # DATAFRAME
#     # ========================================================

#     df = pd.DataFrame(
#         all_results
#     )

#     df.to_csv(
#         RESULT_CSV,
#         index=False,
#     )

#     # ========================================================
#     # SUMMARY
#     # ========================================================

#     random_summary = (
#         df[
#             df["group"] == "random"
#         ]
#         .groupby("backend")
#         .agg(
#             random_mean_fidelity=(
#                 "fidelity",
#                 "mean",
#             ),
#             random_std_fidelity=(
#                 "fidelity",
#                 "std",
#             ),
#         )
#         .reset_index()
#     )

#     top_summary = (
#         df[
#             df["group"] == "top"
#         ]
#         .groupby(
#             ["backend", "model"]
#         )
#         .agg(
#             model_mean_fidelity=(
#                 "fidelity",
#                 "mean",
#             )
#         )
#         .reset_index()
#     )

#     summary = top_summary.merge(
#         random_summary,
#         on="backend",
#         how="left",
#     )

#     summary[
#         "improvement_vs_random"
#     ] = (
#         summary["model_mean_fidelity"]
#         - summary["random_mean_fidelity"]
#     )

#     summary[
#         "relative_improvement_percent"
#     ] = (
#         summary["improvement_vs_random"]
#         / summary["random_mean_fidelity"]
#         * 100.0
#     )

#     # --------------------------------------------------------
#     # Global model comparison
#     # --------------------------------------------------------

#     global_summary = (
#         summary.groupby("model")
#         .agg(
#             mean_model_fidelity=(
#                 "model_mean_fidelity",
#                 "mean",
#             ),
#             mean_random_fidelity=(
#                 "random_mean_fidelity",
#                 "mean",
#             ),
#             mean_improvement=(
#                 "improvement_vs_random",
#                 "mean",
#             ),
#         )
#         .reset_index()
#     )

#     # --------------------------------------------------------
#     # Save everything
#     # --------------------------------------------------------

#     output = {
#         "experiment": "model_vs_random",
#         "level": 1,
#         "snapshot_date": snapshot,
#         "shots": SHOTS,
#         "n_qubits": N_QUBITS,
#         "random_seed": RANDOM_SEED,
#         "random_trials": RANDOM_TRIALS,
#         "circuit": "3-qubit GHZ",
#         "simulation_engine": "Qiskit Aer",
#         "qpu_time_used": 0,
#         "question": (
#             "Do ML-selected qubits produce better "
#             "calibration-driven simulated circuit "
#             "performance than randomly selected qubits?"
#         ),
#         "backend_summary":
#             summary.to_dict("records"),
#         "global_model_summary":
#             global_summary.to_dict("records"),
#         "raw_results":
#             all_results,
#     }

#     with open(
#         RESULT_JSON,
#         "w",
#         encoding="utf-8",
#     ) as f:

#         json.dump(
#             output,
#             f,
#             indent=2,
#             default=str,
#         )

#     # ========================================================
#     # PRINT FINAL TABLE
#     # ========================================================

#     print()
#     print("=" * 72)
#     print("MODEL VS RANDOM — FINAL RESULT")
#     print("=" * 72)

#     display = summary.copy()

#     display[
#         "model_mean_fidelity"
#     ] = display[
#         "model_mean_fidelity"
#     ].round(4)

#     display[
#         "random_mean_fidelity"
#     ] = display[
#         "random_mean_fidelity"
#     ].round(4)

#     display[
#         "improvement_vs_random"
#     ] = display[
#         "improvement_vs_random"
#     ].round(4)

#     display[
#         "relative_improvement_percent"
#     ] = display[
#         "relative_improvement_percent"
#     ].round(2)

#     print(
#         display[
#             [
#                 "backend",
#                 "model",
#                 "model_mean_fidelity",
#                 "random_mean_fidelity",
#                 "improvement_vs_random",
#                 "relative_improvement_percent",
#             ]
#         ].to_string(index=False)
#     )

#     print()
#     print("=" * 72)
#     print("GLOBAL MODEL SUMMARY")
#     print("=" * 72)

#     print(
#         global_summary.round(4).to_string(
#             index=False
#         )
#     )

#     print()
#     print("Results saved:")
#     print(f"  {RESULT_CSV}")
#     print(f"  {RESULT_JSON}")

#     print()
#     print("=" * 72)
#     print("LEVEL 1 MODEL VS RANDOM EXPERIMENT COMPLETE")
#     print("=" * 72)


# if __name__ == "__main__":
#     main()

"""
QuantumCircuitBench
===================

LEVEL 1 — SWEET-SPOT MODEL VS RANDOM

Purpose
-------
Compare the final temporal sweet-spot model-selected qubits against
random topology-valid controls using a local Qiskit Aer simulation.

IMPORTANT
---------
- LOCAL SIMULATION ONLY
- ZERO IBM QPU USAGE
- No hardware jobs are submitted here.
- Real IBM calibration values are used to construct the noise model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    thermal_relaxation_error,
    ReadoutError,
)


# ============================================================
# CONFIGURATION
# ============================================================

SHOTS = 4096
N_QUBITS = 3

RANDOM_SEED = 20260907
RANDOM_TRIALS = 500

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "sweet_spot_vs_random_results.json"
RESULT_CSV = OUTPUT_DIR / "sweet_spot_vs_random_results.csv"

# Transferred Level-2 candidate file.
CANDIDATE_CSV = OUTPUT_DIR / "level2_candidates.csv"

# Approximate gate durations for calibration-driven simulation.
GATE_TIME_1Q = 50e-9
GATE_TIME_2Q = 300e-9

DEFAULT_T1_US = 100.0
DEFAULT_T2_US = 50.0
DEFAULT_READOUT_ERROR = 0.05


# ============================================================
# DATABASE
# ============================================================

def load_database():
    from railway_db import get_connection
    return get_connection()


def get_latest_snapshot_date(conn) -> str:
    sql = """
        SELECT MAX(snapshot_date)
        FROM calibration_history
    """

    cur = conn.cursor()
    cur.execute(sql)

    value = cur.fetchone()[0]
    cur.close()

    if value is None:
        raise RuntimeError(
            "No calibration data found in Railway database."
        )

    return str(value)


def load_calibration(
    conn,
    snapshot_date: str,
) -> pd.DataFrame:

    sql = """
        SELECT
            backend,
            qubit,
            T1_us,
            T2_us,
            readout_error,
            snapshot_date
        FROM calibration_history
        WHERE snapshot_date = %s
        ORDER BY backend, qubit
    """

    return pd.read_sql(
        sql,
        conn,
        params=[snapshot_date],
    )


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_calibration(calibration: pd.DataFrame):

    required = {
        "backend",
        "qubit",
        "T1_us",
        "T2_us",
        "readout_error",
        "snapshot_date",
    }

    missing = required - set(calibration.columns)

    if missing:
        raise RuntimeError(
            f"Calibration data missing columns: {sorted(missing)}"
        )

    if calibration.empty:
        raise RuntimeError(
            "Calibration dataframe is empty."
        )


# ============================================================
# CANDIDATE DATA
# ============================================================

def load_candidates() -> pd.DataFrame:

    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {CANDIDATE_CSV}\n"
            "Run select_level2_candidates.py first."
        )

    df = pd.read_csv(CANDIDATE_CSV)

    required = {
        "backend",
        "selection_type",
        "qubits",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    return df


def parse_qubits(value) -> list[int]:

    if isinstance(value, list):
        return [int(x) for x in value]

    text = str(value).strip()

    # Handles:
    # "[1, 2, 3]"
    # "1,2,3"
    # "(1, 2, 3)"
    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )

    values = [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]

    return [int(x) for x in values]


def get_sweet_spot_candidates(
    candidates: pd.DataFrame,
    backend: str,
) -> pd.DataFrame:

    rows = candidates[
        (candidates["backend"] == backend)
        &
        (
            candidates["selection_type"]
            .astype(str)
            .str.lower()
            .isin(
                {
                    "sweet_spot",
                    "sweet-spot",
                    "sweet spot",
                    "temporal_30day",
                    "temporal_30day_rf",
                }
            )
        )
    ].copy()

    if rows.empty:
        raise RuntimeError(
            f"No Sweet-Spot candidate found for {backend}."
        )

    return rows


def get_random_candidates(
    candidates: pd.DataFrame,
    backend: str,
) -> pd.DataFrame:

    rows = candidates[
        (candidates["backend"] == backend)
        &
        (
            candidates["selection_type"]
            .astype(str)
            .str.lower()
            .str.contains("random")
        )
    ].copy()

    return rows


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def build_ghz_circuit(
    n_qubits: int = 3,
) -> QuantumCircuit:

    qc = QuantumCircuit(
        n_qubits,
        n_qubits,
    )

    qc.h(0)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    qc.measure(
        range(n_qubits),
        range(n_qubits),
    )

    return qc


def ideal_success_probability(
    counts: dict,
    shots: int,
) -> float:

    correct = (
        counts.get("000", 0)
        +
        counts.get("111", 0)
    )

    return correct / shots


# ============================================================
# NOISE MODEL
# ============================================================

def clean_value(
    value,
    default,
):

    if value is None:
        return default

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def build_noise_model(
    qubits: list[dict],
) -> NoiseModel:

    noise_model = NoiseModel()

    for position, q in enumerate(qubits):

        T1_us = clean_value(
            q["T1_us"],
            DEFAULT_T1_US,
        )

        T2_us = clean_value(
            q["T2_us"],
            DEFAULT_T2_US,
        )

        readout = clean_value(
            q["readout_error"],
            DEFAULT_READOUT_ERROR,
        )

        T1 = T1_us * 1e-6
        T2 = T2_us * 1e-6

        # Physical constraint:
        # T2 <= 2*T1
        T2 = min(
            T2,
            2.0 * T1,
        )

        readout = min(
            max(readout, 0.0),
            0.5,
        )

        one_q_error = thermal_relaxation_error(
            T1,
            T2,
            GATE_TIME_1Q,
        )

        noise_model.add_quantum_error(
            one_q_error,
            ["h", "x", "rx", "ry", "rz"],
            [position],
        )

        readout_error = ReadoutError(
            [
                [1 - readout, readout],
                [readout, 1 - readout],
            ]
        )

        noise_model.add_readout_error(
            readout_error,
            [position],
        )

    # Generic linear 3-qubit CX topology.
    # This is a calibration-driven simulation,
    # not an exact IBM coupling-map simulation.

    for i in range(len(qubits) - 1):

        q1 = qubits[i]
        q2 = qubits[i + 1]

        T1_1 = (
            clean_value(
                q1["T1_us"],
                DEFAULT_T1_US,
            )
            * 1e-6
        )

        T2_1 = (
            clean_value(
                q1["T2_us"],
                DEFAULT_T2_US,
            )
            * 1e-6
        )

        T1_2 = (
            clean_value(
                q2["T1_us"],
                DEFAULT_T1_US,
            )
            * 1e-6
        )

        T2_2 = (
            clean_value(
                q2["T2_us"],
                DEFAULT_T2_US,
            )
            * 1e-6
        )

        T2_1 = min(
            T2_1,
            2.0 * T1_1,
        )

        T2_2 = min(
            T2_2,
            2.0 * T1_2,
        )

        error_1 = thermal_relaxation_error(
            T1_1,
            T2_1,
            GATE_TIME_2Q,
        )

        error_2 = thermal_relaxation_error(
            T1_2,
            T2_2,
            GATE_TIME_2Q,
        )

        two_q_error = error_1.tensor(
            error_2
        )

        noise_model.add_quantum_error(
            two_q_error,
            ["cx"],
            [i, i + 1],
        )

    return noise_model


# ============================================================
# SIMULATION
# ============================================================

def simulate_group(
    qubits: list[dict],
    shots: int = SHOTS,
) -> dict:

    if len(qubits) != N_QUBITS:
        raise ValueError(
            f"Exactly {N_QUBITS} qubits are required."
        )

    circuit = build_ghz_circuit(
        N_QUBITS
    )

    noise_model = build_noise_model(
        qubits
    )

    simulator = AerSimulator(
        noise_model=noise_model
    )

    transpiled = transpile(
        circuit,
        simulator,
        optimization_level=0,
    )

    job = simulator.run(
        transpiled,
        shots=shots,
    )

    counts = (
        job
        .result()
        .get_counts()
    )

    fidelity = ideal_success_probability(
        counts,
        shots,
    )

    return {
        "fidelity": round(
            fidelity,
            6,
        ),
        "forbidden_probability": round(
            1.0 - fidelity,
            6,
        ),
        "counts": counts,
    }


# ============================================================
# CALIBRATION LOOKUP
# ============================================================

def get_qubit_records(
    calibration_backend: pd.DataFrame,
    qubits: list[int],
) -> list[dict]:

    lookup = (
        calibration_backend
        .set_index("qubit")
        .to_dict("index")
    )

    records = []

    for qubit in qubits:

        if qubit not in lookup:
            raise RuntimeError(
                f"Qubit {qubit} is missing from "
                "the calibration snapshot."
            )

        row = lookup[qubit].copy()
        row["qubit"] = int(qubit)

        records.append(row)

    return records


# ============================================================
# SUMMARY
# ============================================================

def summarize_group(
    qubits: list[dict],
) -> dict:

    return {
        "qubits": [
            int(q["qubit"])
            for q in qubits
        ],
        "mean_T1_us": round(
            float(
                np.mean(
                    [
                        clean_value(
                            q["T1_us"],
                            DEFAULT_T1_US,
                        )
                        for q in qubits
                    ]
                )
            ),
            3,
        ),
        "mean_T2_us": round(
            float(
                np.mean(
                    [
                        clean_value(
                            q["T2_us"],
                            DEFAULT_T2_US,
                        )
                        for q in qubits
                    ]
                )
            ),
            3,
        ),
        "mean_readout_error": round(
            float(
                np.mean(
                    [
                        clean_value(
                            q["readout_error"],
                            DEFAULT_READOUT_ERROR,
                        )
                        for q in qubits
                    ]
                )
            ),
            6,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("QUANTUM CIRCUIT BENCH")
    print("LEVEL 1 — SWEET-SPOT VS RANDOM")
    print("=" * 72)

    print(f"Shots         : {SHOTS}")
    print(f"Circuit       : {N_QUBITS}-qubit GHZ")
    print(f"Random trials : {RANDOM_TRIALS}")
    print(f"Random seed   : {RANDOM_SEED}")
    print()
    print("QPU USAGE     : 0 seconds")
    print("ENGINE        : Qiskit Aer")
    print()

    # --------------------------------------------------------
    # LOAD RAILWAY DATA
    # --------------------------------------------------------

    conn = load_database()

    try:

        snapshot = get_latest_snapshot_date(
            conn
        )

        print(
            f"Latest Railway snapshot: {snapshot}"
        )

        calibration = load_calibration(
            conn,
            snapshot,
        )

    finally:
        conn.close()

    validate_calibration(
        calibration
    )

    print(
        f"Calibration rows: {len(calibration):,}"
    )

    # --------------------------------------------------------
    # LOAD CANDIDATES
    # --------------------------------------------------------

    candidates = load_candidates()

    print(
        f"Candidate rows  : {len(candidates):,}"
    )

    backends = [
        "ibm_fez",
        "ibm_kingston",
        "ibm_marrakesh",
    ]

    all_results = []

    # --------------------------------------------------------
    # BACKENDS
    # --------------------------------------------------------

    for backend in backends:

        print()
        print("=" * 72)
        print(f"BACKEND: {backend}")
        print("=" * 72)

        cal_backend = calibration[
            calibration["backend"] == backend
        ].copy()

        if cal_backend.empty:
            raise RuntimeError(
                f"No calibration data for {backend}."
            )

        # ----------------------------------------------------
        # SWEET-SPOT
        # ----------------------------------------------------

        sweet_rows = get_sweet_spot_candidates(
            candidates,
            backend,
        )

        sweet_row = sweet_rows.iloc[0]

        sweet_qubits = parse_qubits(
            sweet_row["qubits"]
        )

        if len(sweet_qubits) != N_QUBITS:
            raise RuntimeError(
                f"Sweet-Spot selection for {backend} "
                f"does not contain exactly 3 qubits: "
                f"{sweet_qubits}"
            )

        sweet_records = get_qubit_records(
            cal_backend,
            sweet_qubits,
        )

        print(
            f"Sweet-Spot qubits: {sweet_qubits}"
        )

        sweet_result = simulate_group(
            sweet_records
        )

        sweet_stats = summarize_group(
            sweet_records
        )

        print(
            f"  Fidelity : "
            f"{sweet_result['fidelity']:.4f}"
        )

        print(
            f"  Mean T1  : "
            f"{sweet_stats['mean_T1_us']:.2f} us"
        )

        print(
            f"  Mean T2  : "
            f"{sweet_stats['mean_T2_us']:.2f} us"
        )

        print(
            f"  Mean RE  : "
            f"{sweet_stats['mean_readout_error']:.5f}"
        )

        all_results.append(
            {
                "snapshot_date": snapshot,
                "backend": backend,
                "group": "sweet_spot",
                "model": "sweet_spot",
                "trial": 0,
                "fidelity":
                    sweet_result["fidelity"],
                "forbidden_probability":
                    sweet_result[
                        "forbidden_probability"
                    ],
                **sweet_stats,
            }
        )

        # ----------------------------------------------------
        # RANDOM CONTROLS
        # ----------------------------------------------------

        random_rows = get_random_candidates(
            candidates,
            backend,
        )

        if random_rows.empty:

            print(
                "No random candidates found "
                "in candidate file."
            )

            print(
                "Generating random qubit controls "
                "from current calibration."
            )

            rng = random.Random(
                RANDOM_SEED
                + backends.index(backend)
            )

            for trial in range(
                1,
                RANDOM_TRIALS + 1,
            ):

                indices = rng.sample(
                    list(cal_backend.index),
                    N_QUBITS,
                )

                random_records = (
                    cal_backend
                    .loc[indices]
                    .to_dict("records")
                )

                result = simulate_group(
                    random_records
                )

                stats = summarize_group(
                    random_records
                )

                all_results.append(
                    {
                        "snapshot_date":
                            snapshot,
                        "backend":
                            backend,
                        "group":
                            "random",
                        "model":
                            "random_control",
                        "trial":
                            trial,
                        "fidelity":
                            result["fidelity"],
                        "forbidden_probability":
                            result[
                                "forbidden_probability"
                            ],
                        **stats,
                    }
                )

        else:

            for trial_index, (_, row) in enumerate(
                random_rows.iterrows(),
                start=1,
            ):

                if trial_index > RANDOM_TRIALS:
                    break

                random_qubits = parse_qubits(
                    row["qubits"]
                )

                if len(random_qubits) != N_QUBITS:
                    continue

                random_records = get_qubit_records(
                    cal_backend,
                    random_qubits,
                )

                result = simulate_group(
                    random_records
                )

                stats = summarize_group(
                    random_records
                )

                all_results.append(
                    {
                        "snapshot_date":
                            snapshot,
                        "backend":
                            backend,
                        "group":
                            "random",
                        "model":
                            "random_control",
                        "trial":
                            trial_index,
                        "fidelity":
                            result["fidelity"],
                        "forbidden_probability":
                            result[
                                "forbidden_probability"
                            ],
                        **stats,
                    }
                )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        all_results
    )

    if df.empty:
        raise RuntimeError(
            "No simulation results were produced."
        )

    df.to_csv(
        RESULT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    sweet_summary = (
        df[df["group"] == "sweet_spot"]
        .groupby("backend")
        .agg(
            sweet_spot_fidelity=(
                "fidelity",
                "mean",
            )
        )
        .reset_index()
    )

    random_summary = (
        df[df["group"] == "random"]
        .groupby("backend")
        .agg(
            random_mean_fidelity=(
                "fidelity",
                "mean",
            ),
            random_std_fidelity=(
                "fidelity",
                "std",
            ),
            random_n=(
                "fidelity",
                "count",
            ),
        )
        .reset_index()
    )

    summary = sweet_summary.merge(
        random_summary,
        on="backend",
        how="left",
    )

    summary[
        "improvement_vs_random"
    ] = (
        summary["sweet_spot_fidelity"]
        -
        summary["random_mean_fidelity"]
    )

    summary[
        "relative_improvement_percent"
    ] = (
        summary["improvement_vs_random"]
        /
        summary["random_mean_fidelity"]
        * 100.0
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    global_summary = {
        "mean_sweet_spot_fidelity":
            float(
                summary[
                    "sweet_spot_fidelity"
                ].mean()
            ),
        "mean_random_fidelity":
            float(
                summary[
                    "random_mean_fidelity"
                ].mean()
            ),
        "mean_improvement":
            float(
                summary[
                    "improvement_vs_random"
                ].mean()
            ),
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {
        "experiment":
            "sweet_spot_vs_random",
        "level":
            1,
        "snapshot_date":
            snapshot,
        "shots":
            SHOTS,
        "n_qubits":
            N_QUBITS,
        "random_seed":
            RANDOM_SEED,
        "random_trials":
            RANDOM_TRIALS,
        "circuit":
            "3-qubit GHZ",
        "simulation_engine":
            "Qiskit Aer",
        "qpu_time_used":
            0,
        "model":
            "sweet_spot",
        "control":
            "random topology-valid controls",
        "backend_summary":
            summary.to_dict(
                "records"
            ),
        "global_summary":
            global_summary,
        "raw_results":
            all_results,
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            default=str,
        )

    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print()
    print("=" * 72)
    print("SWEET-SPOT VS RANDOM — FINAL RESULT")
    print("=" * 72)

    display = summary.copy()

    numeric_columns = [
        "sweet_spot_fidelity",
        "random_mean_fidelity",
        "random_std_fidelity",
        "improvement_vs_random",
        "relative_improvement_percent",
    ]

    for column in numeric_columns:
        if column in display.columns:
            display[column] = display[
                column
            ].round(4)

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        f"Global Sweet-Spot fidelity : "
        f"{global_summary['mean_sweet_spot_fidelity']:.4f}"
    )

    print(
        f"Global Random fidelity     : "
        f"{global_summary['mean_random_fidelity']:.4f}"
    )

    print(
        f"Global improvement         : "
        f"{global_summary['mean_improvement']:.4f}"
    )

    print()
    print("Results saved:")
    print(
        f"  {RESULT_CSV}"
    )
    print(
        f"  {RESULT_JSON}"
    )

    print()
    print("=" * 72)
    print(
        "LEVEL 1 SWEET-SPOT EXPERIMENT COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()