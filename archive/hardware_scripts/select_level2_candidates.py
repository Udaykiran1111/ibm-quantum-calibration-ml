"""
LEVEL 2 — PHYSICALLY CONNECTED QUBIT CANDIDATE SELECTION

Purpose
-------
Select physically connected 3-qubit chains for:

    Model A
    Model B
    Model C
    Random controls

Data sources
------------
1. Current calibration/ranking data from Railway MySQL
2. Current IBM Quantum backend topology

IMPORTANT
---------
This script NEVER submits a quantum circuit.
It consumes ZERO QPU runtime.

Selection rule
--------------
For each model:

    score(chain) =
        mean(model viability score of the 3 qubits)

The highest-scoring physically connected chain is selected.

Random controls are selected only from physically valid
3-qubit chains, so the comparison remains topology-valid.
"""

import os
import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_INSTANCE")

if not TOKEN:
    raise RuntimeError("IBM_QUANTUM_TOKEN is missing from .env")

if not INSTANCE:
    raise RuntimeError("IBM_INSTANCE is missing from .env")


BACKENDS = [
    "ibm_fez",
    "ibm_marrakesh",
    "ibm_kingston",
]

MODELS = [
    "model_a",
    "model_b",
    "model_c",
]

RANDOM_SEED = 20260823
RANDOM_TRIALS = 10

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "level2_candidates.json"
OUTPUT_CSV = OUTPUT_DIR / "level2_candidates.csv"


# ============================================================
# DATABASE
# ============================================================

def load_database():
    from railway_db import get_connection
    return get_connection()


def get_latest_snapshot(conn):
    sql = """
        SELECT MAX(snapshot_date)
        FROM calibration_history
    """

    cur = conn.cursor()
    cur.execute(sql)

    row = cur.fetchone()

    if not row or row[0] is None:
        raise RuntimeError("No calibration snapshots found in Railway.")

    return str(row[0])


def load_calibration(conn, snapshot_date):
    sql = """
        SELECT
            backend,
            qubit,
            T1_us,
            T2_us,
            readout_error
        FROM calibration_history
        WHERE snapshot_date = %s
        ORDER BY backend, qubit
    """

    return pd.read_sql(sql, conn, params=[snapshot_date])


def load_rankings(conn, snapshot_date):
    sql = """
        SELECT
            model_name,
            backend,
            qubit,
            viability_score,
            viability_rank
        FROM qubit_rankings
        WHERE snapshot_date = %s
          AND model_name IN ('model_a', 'model_b', 'model_c')
        ORDER BY model_name, backend, viability_rank
    """

    return pd.read_sql(sql, conn, params=[snapshot_date])


# ============================================================
# IBM SERVICE
# ============================================================

def load_ibm_service():

    print("Connecting to IBM Quantum...")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE,
    )

    print("IBM authentication: PASS")

    return service


# ============================================================
# TOPOLOGY
# ============================================================

def get_connected_chains(backend):
    """
    Return all unique connected 3-qubit chains.

    A valid chain is:

        qA ---- qB ---- qC

    where qA-qB and qB-qC are both physical coupling edges.
    """

    config = backend.configuration()

    coupling_map = config.coupling_map

    adjacency = {
        q: set()
        for q in range(backend.num_qubits)
    }

    for a, b in coupling_map:

        adjacency[a].add(b)
        adjacency[b].add(a)

    chains = set()

    for middle in range(backend.num_qubits):

        neighbors = sorted(adjacency[middle])

        for i in range(len(neighbors)):

            for j in range(i + 1, len(neighbors)):

                left = neighbors[i]
                right = neighbors[j]

                chain = tuple(
                    sorted(
                        (
                            left,
                            middle,
                            right,
                        )
                    )
                )

                chains.add(chain)

    return sorted(chains)


# ============================================================
# MODEL CHAIN SELECTION
# ============================================================

def select_best_model_chain(
    rankings_backend,
    chains,
):
    """
    Select the connected 3-qubit chain with the highest
    average model viability score.
    """

    score_lookup = dict(
        zip(
            rankings_backend["qubit"],
            rankings_backend["viability_score"],
        )
    )

    candidates = []

    for chain in chains:

        scores = []

        valid = True

        for qubit in chain:

            if qubit not in score_lookup:
                valid = False
                break

            scores.append(
                float(score_lookup[qubit])
            )

        if not valid:
            continue

        mean_score = sum(scores) / len(scores)

        candidates.append(
            {
                "qubits": list(chain),
                "scores": scores,
                "mean_viability": mean_score,
                "minimum_viability": min(scores),
            }
        )

    if not candidates:
        raise RuntimeError(
            "No valid model-ranked connected chains found."
        )

    candidates.sort(
        key=lambda x: (
            x["mean_viability"],
            x["minimum_viability"],
        ),
        reverse=True,
    )

    return candidates[0], candidates


# ============================================================
# CALIBRATION INFORMATION
# ============================================================

def add_calibration_information(
    candidate,
    calibration_backend,
):
    """
    Add current physical calibration statistics to a candidate.
    """

    calibration_lookup = (
        calibration_backend
        .set_index("qubit")
        .to_dict("index")
    )

    t1_values = []
    t2_values = []
    readout_values = []

    for qubit in candidate["qubits"]:

        data = calibration_lookup.get(qubit)

        if data is None:
            continue

        if pd.notna(data["T1_us"]):
            t1_values.append(
                float(data["T1_us"])
            )

        if pd.notna(data["T2_us"]):
            t2_values.append(
                float(data["T2_us"])
            )

        if pd.notna(data["readout_error"]):
            readout_values.append(
                float(data["readout_error"])
            )

    candidate["mean_T1_us"] = (
        sum(t1_values) / len(t1_values)
        if t1_values else None
    )

    candidate["mean_T2_us"] = (
        sum(t2_values) / len(t2_values)
        if t2_values else None
    )

    candidate["mean_readout_error"] = (
        sum(readout_values) / len(readout_values)
        if readout_values else None
    )

    return candidate


# ============================================================
# RANDOM CONTROL SELECTION
# ============================================================

def select_random_chains(
    chains,
    number,
    seed,
):
    """
    Select random physically connected chains.

    The random control is topology-valid.
    """

    rng = random.Random(seed)

    if len(chains) < number:
        raise RuntimeError(
            f"Only {len(chains)} valid chains available."
        )

    selected = rng.sample(chains, number)

    return [
        {
            "qubits": list(chain)
        }
        for chain in selected
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IBM QUANTUM CIRCUIT BENCH")
    print("LEVEL 2 — PHYSICALLY CONNECTED CANDIDATE SELECTION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("No quantum circuit will be submitted.")
    print("Expected QPU usage: 0 seconds.")
    print()

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    conn = load_database()

    try:

        snapshot_date = get_latest_snapshot(conn)

        print(
            f"Latest Railway snapshot: {snapshot_date}"
        )

        calibration = load_calibration(
            conn,
            snapshot_date,
        )

        rankings = load_rankings(
            conn,
            snapshot_date,
        )

    finally:
        conn.close()

    # --------------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------------

    print()
    print("DATA VALIDATION")
    print("-" * 70)

    print(
        f"Calibration rows : {len(calibration)}"
    )

    print(
        f"Ranking rows     : {len(rankings)}"
    )

    print(
        f"Backends         : "
        f"{sorted(calibration['backend'].unique())}"
    )

    print(
        f"Models           : "
        f"{sorted(rankings['model_name'].unique())}"
    )

    expected_calibration = 466
    expected_rankings = 1398

    if len(calibration) != expected_calibration:
        print(
            f"WARNING: expected approximately "
            f"{expected_calibration} calibration rows."
        )

    if len(rankings) != expected_rankings:
        print(
            f"WARNING: expected approximately "
            f"{expected_rankings} ranking rows."
        )

    for model in MODELS:

        count = len(
            rankings[
                rankings["model_name"] == model
            ]
        )

        if count == 0:
            raise RuntimeError(
                f"No rankings found for {model}"
            )

    print("PASS")

    # --------------------------------------------------------
    # IBM
    # --------------------------------------------------------

    service = load_ibm_service()

    all_results = []

    # --------------------------------------------------------
    # EACH BACKEND
    # --------------------------------------------------------

    for backend_name in BACKENDS:

        print()
        print("=" * 70)
        print(f"BACKEND: {backend_name}")
        print("=" * 70)

        backend = service.backend(backend_name)

        print(
            f"Physical qubits : {backend.num_qubits}"
        )

        # topology
        chains = get_connected_chains(backend)

        print(
            f"Connected chains: {len(chains)}"
        )

        # current calibration
        cal_backend = calibration[
            calibration["backend"]
            == backend_name
        ].copy()

        print(
            f"Calibration rows: {len(cal_backend)}"
        )

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------

        backend_results = {
            "backend": backend_name,
            "snapshot_date": snapshot_date,
            "num_physical_qubits": backend.num_qubits,
            "connected_chain_count": len(chains),
            "models": {},
        }

        for model in MODELS:

            model_backend = rankings[
                (rankings["model_name"] == model)
                &
                (rankings["backend"] == backend_name)
            ].copy()

            candidate, _ = select_best_model_chain(
                model_backend,
                chains,
            )

            candidate = add_calibration_information(
                candidate,
                cal_backend,
            )

            backend_results["models"][model] = candidate

            print()
            print(
                f"{model.upper()} BEST CONNECTED CHAIN"
            )

            print(
                f"  Qubits             : "
                f"{candidate['qubits']}"
            )

            print(
                f"  Individual scores  : "
                f"{[round(x, 4) for x in candidate['scores']]}"
            )

            print(
                f"  Mean viability     : "
                f"{candidate['mean_viability']:.4f}"
            )

            print(
                f"  Minimum viability  : "
                f"{candidate['minimum_viability']:.4f}"
            )

            print(
                f"  Mean T1            : "
                f"{candidate['mean_T1_us']:.2f} us"
            )

            print(
                f"  Mean T2            : "
                f"{candidate['mean_T2_us']:.2f} us"
            )

            print(
                f"  Mean readout error : "
                f"{candidate['mean_readout_error']:.5f}"
            )

        # ----------------------------------------------------
        # RANDOM CONTROLS
        # ----------------------------------------------------

        random_chains = select_random_chains(
            chains,
            RANDOM_TRIALS,
            RANDOM_SEED,
        )

        backend_results["random_controls"] = []

        print()
        print(
            "RANDOM TOPOLOGY-VALID CONTROLS"
        )

        for idx, random_candidate in enumerate(
            random_chains,
            start=1,
        ):

            random_candidate = add_calibration_information(
                random_candidate,
                cal_backend,
            )

            backend_results[
                "random_controls"
            ].append(random_candidate)

            print(
                f"  Trial {idx:02d}: "
                f"{random_candidate['qubits']} "
                f"| T1={random_candidate['mean_T1_us']:.2f} "
                f"| T2={random_candidate['mean_T2_us']:.2f} "
                f"| RE={random_candidate['mean_readout_error']:.5f}"
            )

        all_results.append(backend_results)

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output = {
        "experiment": "Level 2 physically connected candidate selection",
        "snapshot_date": snapshot_date,
        "random_seed": RANDOM_SEED,
        "random_trials": RANDOM_TRIALS,
        "qpu_submission": False,
        "qpu_runtime_used": 0,
        "backends": all_results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    rows = []

    for backend_result in all_results:

        backend_name = backend_result["backend"]

        for model in MODELS:

            candidate = (
                backend_result["models"][model]
            )

            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "backend": backend_name,
                    "selection_type": model,
                    "qubits": str(
                        candidate["qubits"]
                    ),
                    "mean_viability":
                        candidate["mean_viability"],
                    "minimum_viability":
                        candidate["minimum_viability"],
                    "mean_T1_us":
                        candidate["mean_T1_us"],
                    "mean_T2_us":
                        candidate["mean_T2_us"],
                    "mean_readout_error":
                        candidate["mean_readout_error"],
                }
            )

        for idx, candidate in enumerate(
            backend_result["random_controls"],
            start=1,
        ):

            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "backend": backend_name,
                    "selection_type":
                        f"random_{idx:02d}",
                    "qubits":
                        str(candidate["qubits"]),
                    "mean_viability": None,
                    "minimum_viability": None,
                    "mean_T1_us":
                        candidate["mean_T1_us"],
                    "mean_T2_us":
                        candidate["mean_T2_us"],
                    "mean_readout_error":
                        candidate["mean_readout_error"],
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CANDIDATE SELECTION COMPLETE")
    print("=" * 70)

    print()
    print(f"Snapshot used : {snapshot_date}")
    print(f"JSON saved    : {OUTPUT_JSON}")
    print(f"CSV saved     : {OUTPUT_CSV}")
    print()
    print("QPU circuit submitted : NO")
    print("QPU runtime used      : 0 seconds")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()