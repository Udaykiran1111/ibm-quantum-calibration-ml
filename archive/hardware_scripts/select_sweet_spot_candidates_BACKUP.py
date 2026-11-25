"""
IBM CIRCUITS
============

SWEET-SPOT CANDIDATE SELECTION

Purpose
-------
Prepare physically connected 3-qubit candidates for the final
Sweet-Spot experiment.

NO quantum circuit is submitted.

QPU usage: 0 seconds.

Sweet-Spot:
    30-day temporal model
    Sep 6, 2026 ranking

Controls:
    10 random topology-valid 3-qubit chains per backend
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_INSTANCE")

if not TOKEN:
    raise RuntimeError("IBM_QUANTUM_TOKEN missing from .env")

if not INSTANCE:
    raise RuntimeError("IBM_INSTANCE missing from .env")


BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]

SWEET_SPOT_WINDOW = 30
PREDICTION_DATE = "2026-09-06"

RANDOM_SEED = 20260907
RANDOM_TRIALS = 10

RESULTS_DIR = Path("results")
SWEET_SPOT_DIR = RESULTS_DIR / "sweet_spot"

RESULTS_DIR.mkdir(exist_ok=True)
SWEET_SPOT_DIR.mkdir(exist_ok=True)

TOP1_FILE = (
    SWEET_SPOT_DIR
    / "sep6_all_models_top1_per_backend.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "sweet_spot_level2_candidates.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "sweet_spot_level2_candidates.json"
)


# ============================================================
# IBM CONNECTION
# ============================================================

def connect_ibm():
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
    Return unique connected 3-qubit chains.

    A chain is:

        q0 -- q1 -- q2

    The three qubits must therefore be physically connected
    through the middle qubit.
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

        neighbors = sorted(
            adjacency[middle]
        )

        for i in range(len(neighbors)):

            for j in range(
                i + 1,
                len(neighbors)
            ):

                left = neighbors[i]
                right = neighbors[j]

                chain = tuple(
                    sorted(
                        [
                            left,
                            middle,
                            right,
                        ]
                    )
                )

                chains.add(chain)

    return sorted(chains)


# ============================================================
# TRANSFERRED SWEET-SPOT RANKING
# ============================================================

def load_sweet_spot_top1():

    if not TOP1_FILE.exists():

        raise FileNotFoundError(
            f"\nMissing:\n{TOP1_FILE}\n\n"
            "Make sure the transferred file is located at:\n"
            "results/sweet_spot/"
        )

    df = pd.read_csv(
        TOP1_FILE
    )

    required = {
        "model_name",
        "window_days",
        "prediction_date",
        "backend",
        "qubit",
        "viability_score",
        "viability_rank",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Sweet-Spot Top-1 file is missing columns: "
            f"{sorted(missing)}"
        )

    # Only the final 30-day temporal model.
    df = df[
        (df["model_name"] == "temporal_30day")
        &
        (df["window_days"] == SWEET_SPOT_WINDOW)
        &
        (
            df["prediction_date"].astype(str)
            == PREDICTION_DATE
        )
        &
        (df["viability_rank"] == 1)
    ].copy()

    if len(df) != 3:

        raise RuntimeError(
            "Expected exactly 3 Sweet-Spot Top-1 rows "
            f"for {PREDICTION_DATE}; found {len(df)}."
        )

    if set(df["backend"]) != set(BACKENDS):

        raise RuntimeError(
            "Sweet-Spot file does not contain exactly "
            "FEZ, Kingston and Marrakesh."
        )

    return df


# ============================================================
# CHAIN VALIDATION
# ============================================================

def is_connected_chain(
    chain,
    chains,
):

    return tuple(
        sorted(chain)
    ) in set(chains)


# ============================================================
# RANDOM CONTROLS
# ============================================================

def select_random_chains(
    chains,
    excluded_chain,
    number,
    seed,
):

    available = [
        chain
        for chain in chains
        if tuple(sorted(chain))
        != tuple(sorted(excluded_chain))
    ]

    if len(available) < number:

        raise RuntimeError(
            f"Only {len(available)} random chains available."
        )

    rng = random.Random(seed)

    return rng.sample(
        available,
        number,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("IBM CIRCUIT BENCH")
    print("SWEET-SPOT CANDIDATE SELECTION")
    print("=" * 72)

    print()
    print("Experiment model : Temporal 30-day Sweet-Spot")
    print(f"Prediction date  : {PREDICTION_DATE}")
    print(f"Random controls  : {RANDOM_TRIALS}/backend")
    print("QPU execution    : NONE")
    print()

    # --------------------------------------------------------
    # LOAD TRANSFERRED MODEL RESULT
    # --------------------------------------------------------

    top1 = load_sweet_spot_top1()

    print(
        "Transferred Sweet-Spot ranking: PASS"
    )

    # --------------------------------------------------------
    # IBM SERVICE
    # --------------------------------------------------------

    service = connect_ibm()

    all_rows = []
    all_json = []

    # --------------------------------------------------------
    # BACKENDS
    # --------------------------------------------------------

    for backend_name in BACKENDS:

        print()
        print("=" * 72)
        print(
            f"BACKEND: {backend_name}"
        )
        print("=" * 72)

        backend = service.backend(
            backend_name
        )

        chains = get_connected_chains(
            backend
        )

        print(
            f"Physical qubits : {backend.num_qubits}"
        )

        print(
            f"Valid 3Q chains : {len(chains)}"
        )

        # ----------------------------------------------------
        # SWEET-SPOT
        # ----------------------------------------------------

        row = top1[
            top1["backend"]
            == backend_name
        ].iloc[0]

        sweet_qubits = [
            int(row["qubit"])
        ]

        # Top-1 file contains ONE qubit per backend.
        # We therefore recover the connected 3Q chain by
        # choosing the best connected chain containing that
        # Sweet-Spot top-ranked qubit.

        ranking_file = (
            SWEET_SPOT_DIR
            / "sep6_all_models_top10_per_backend.csv"
        )

        if not ranking_file.exists():

            raise FileNotFoundError(
                f"Missing:\n{ranking_file}"
            )

        rankings = pd.read_csv(
            ranking_file
        )

        rankings = rankings[
            (rankings["model_name"] == "temporal_30day")
            &
            (rankings["window_days"] == 30)
            &
            (
                rankings["prediction_date"].astype(str)
                == PREDICTION_DATE
            )
            &
            (
                rankings["backend"]
                == backend_name
            )
        ].copy()

        if rankings.empty:

            raise RuntimeError(
                f"No 30-day Top-10 rankings for "
                f"{backend_name}."
            )

        score_lookup = dict(
            zip(
                rankings["qubit"].astype(int),
                rankings["viability_score"].astype(float),
            )
        )

        candidate_chains = []

        for chain in chains:

            if not any(
                q in score_lookup
                for q in chain
            ):
                continue

            scores = []

            valid = True

            for q in chain:

                if q not in score_lookup:

                    valid = False
                    break

                scores.append(
                    score_lookup[q]
                )

            if not valid:
                continue

            candidate_chains.append(
                {
                    "qubits": list(chain),
                    "scores": scores,
                    "mean_viability": (
                        sum(scores)
                        / len(scores)
                    ),
                    "minimum_viability": min(
                        scores
                    ),
                }
            )

        if not candidate_chains:

            raise RuntimeError(
                f"No valid connected chain could "
                f"be constructed for {backend_name}."
            )

        # Highest mean viability, then strongest
        # minimum qubit score.
        candidate_chains.sort(
            key=lambda x: (
                x["mean_viability"],
                x["minimum_viability"],
            ),
            reverse=True,
        )

        sweet = candidate_chains[0]

        print()
        print("SWEET-SPOT CHAIN")
        print(
            f"  Qubits          : {sweet['qubits']}"
        )
        print(
            f"  Scores          : "
            f"{[round(x, 6) for x in sweet['scores']]}"
        )
        print(
            f"  Mean viability  : "
            f"{sweet['mean_viability']:.6f}"
        )

        all_rows.append(
            {
                "snapshot_date":
                    PREDICTION_DATE,
                "backend":
                    backend_name,
                "selection_type":
                    "sweet_spot",
                "model_name":
                    "temporal_30day",
                "window_days":
                    30,
                "qubits":
                    str(sweet["qubits"]),
                "viability_scores":
                    str(sweet["scores"]),
                "mean_viability":
                    sweet["mean_viability"],
                "minimum_viability":
                    sweet["minimum_viability"],
                "random_trial":
                    0,
            }
        )

        # ----------------------------------------------------
        # RANDOM CONTROLS
        # ----------------------------------------------------

        random_chains = select_random_chains(
            chains,
            sweet["qubits"],
            RANDOM_TRIALS,
            RANDOM_SEED
            + BACKENDS.index(backend_name),
        )

        print()
        print("RANDOM CONTROLS")

        backend_json = {
            "backend":
                backend_name,
            "sweet_spot":
                sweet,
            "random_controls":
                [],
        }

        for index, chain in enumerate(
            random_chains,
            start=1,
        ):

            print(
                f"  Random {index:02d}: "
                f"{list(chain)}"
            )

            record = {
                "snapshot_date":
                    PREDICTION_DATE,
                "backend":
                    backend_name,
                "selection_type":
                    "random_control",
                "model_name":
                    "random",
                "window_days":
                    None,
                "qubits":
                    str(list(chain)),
                "viability_scores":
                    None,
                "mean_viability":
                    None,
                "minimum_viability":
                    None,
                "random_trial":
                    index,
            }

            all_rows.append(record)

            backend_json[
                "random_controls"
            ].append(
                {
                    "trial":
                        index,
                    "qubits":
                        list(chain),
                }
            )

        all_json.append(
            backend_json
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_rows
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    output = {
        "experiment":
            "sweet_spot_vs_random",
        "model":
            "temporal_30day",
        "prediction_date":
            PREDICTION_DATE,
        "random_seed":
            RANDOM_SEED,
        "random_trials_per_backend":
            RANDOM_TRIALS,
        "qpu_submission":
            False,
        "qpu_runtime_seconds":
            0,
        "backends":
            all_json,
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

    print()
    print("=" * 72)
    print("SELECTION COMPLETE")
    print("=" * 72)

    print()
    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print()
    print("QPU usage: 0 seconds")


if __name__ == "__main__":
    main()