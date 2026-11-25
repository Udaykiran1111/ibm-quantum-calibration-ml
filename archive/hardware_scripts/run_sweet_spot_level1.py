"""
IBM CIRCUIT BENCH
=================

LEVEL 1 — SWEET-SPOT VS RANDOM

LOCAL QISKIT AER SIMULATION ONLY.

QPU usage: 0 seconds.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    thermal_relaxation_error,
)

from railway_db import get_connection


# ============================================================
# CONFIG
# ============================================================

SHOTS = 4096

RESULTS_DIR = Path("results")

CANDIDATE_FILE = (
    RESULTS_DIR
    / "sweet_spot_level2_candidates.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "sweet_spot_level1_results.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "sweet_spot_level1_results.json"
)

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]

GATE_1Q_SECONDS = 50e-9
GATE_2Q_SECONDS = 300e-9

DEFAULT_T1_US = 100.0
DEFAULT_T2_US = 50.0
DEFAULT_RE = 0.05


# ============================================================
# UTILITIES
# ============================================================

def safe_float(value, default):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def parse_qubits(value):

    return [
        int(x)
        for x in ast.literal_eval(
            str(value)
        )
    ]


# ============================================================
# GHZ
# ============================================================

def build_ghz():

    qc = QuantumCircuit(
        3,
        3,
    )

    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.measure(
        range(3),
        range(3),
    )

    return qc


# ============================================================
# NOISE
# ============================================================

def build_noise_model(
    records,
):

    noise = NoiseModel()

    for position, row in enumerate(
        records
    ):

        t1 = (
            safe_float(
                row["T1_us"],
                DEFAULT_T1_US,
            )
            * 1e-6
        )

        t2 = (
            safe_float(
                row["T2_us"],
                DEFAULT_T2_US,
            )
            * 1e-6
        )

        t2 = min(
            t2,
            2.0 * t1,
        )

        re = safe_float(
            row["readout_error"],
            DEFAULT_RE,
        )

        re = min(
            max(re, 0.0),
            0.5,
        )

        one_q = thermal_relaxation_error(
            t1,
            t2,
            GATE_1Q_SECONDS,
        )

        noise.add_quantum_error(
            one_q,
            ["h"],
            [position],
        )

        readout = ReadoutError(
            [
                [1 - re, re],
                [re, 1 - re],
            ]
        )

        noise.add_readout_error(
            readout,
            [position],
        )

    # Generic linear CX noise.
    #
    # This is a calibration-driven model,
    # not a complete replica of IBM's proprietary
    # hardware noise model.

    for i in range(2):

        row_a = records[i]
        row_b = records[i + 1]

        t1_a = (
            safe_float(
                row_a["T1_us"],
                DEFAULT_T1_US,
            )
            * 1e-6
        )

        t2_a = (
            safe_float(
                row_a["T2_us"],
                DEFAULT_T2_US,
            )
            * 1e-6
        )

        t1_b = (
            safe_float(
                row_b["T1_us"],
                DEFAULT_T1_US,
            )
            * 1e-6
        )

        t2_b = (
            safe_float(
                row_b["T2_us"],
                DEFAULT_T2_US,
            )
            * 1e-6
        )

        t2_a = min(
            t2_a,
            2 * t1_a,
        )

        t2_b = min(
            t2_b,
            2 * t1_b,
        )

        err_a = thermal_relaxation_error(
            t1_a,
            t2_a,
            GATE_2Q_SECONDS,
        )

        err_b = thermal_relaxation_error(
            t1_b,
            t2_b,
            GATE_2Q_SECONDS,
        )

        two_q = err_a.tensor(
            err_b
        )

        noise.add_quantum_error(
            two_q,
            ["cx"],
            [i, i + 1],
        )

    return noise


# ============================================================
# SIMULATE
# ============================================================

def simulate(records):

    qc = build_ghz()

    noise = build_noise_model(
        records
    )

    simulator = AerSimulator(
        noise_model=noise
    )

    compiled = transpile(
        qc,
        simulator,
        optimization_level=0,
    )

    result = simulator.run(
        compiled,
        shots=SHOTS,
    ).result()

    counts = result.get_counts()

    total = sum(
        counts.values()
    )

    fidelity = (
        counts.get("000", 0)
        +
        counts.get("111", 0)
    ) / total

    return {
        "fidelity":
            float(fidelity),
        "counts":
            counts,
        "depth":
            compiled.depth(),
    }


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration():

    conn = get_connection()

    try:

        sql = """
            SELECT
                backend,
                qubit,
                T1_us,
                T2_us,
                readout_error,
                snapshot_date
            FROM calibration_history
            WHERE snapshot_date = (
                SELECT MAX(snapshot_date)
                FROM calibration_history
            )
            ORDER BY backend, qubit
        """

        return pd.read_sql(
            sql,
            conn,
        )

    finally:

        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("LEVEL 1 — SWEET-SPOT VS RANDOM")
    print("=" * 72)

    print()
    print(
        "Execution engine : Qiskit Aer"
    )
    print(
        f"Shots            : {SHOTS}"
    )
    print(
        "QPU usage        : 0 seconds"
    )
    print()

    if not CANDIDATE_FILE.exists():

        raise FileNotFoundError(
            f"Run select_sweet_spot_candidates.py first:\n"
            f"{CANDIDATE_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    calibration = load_calibration()

    print(
        f"Calibration rows: {len(calibration)}"
    )

    all_results = []

    for backend in BACKENDS:

        print()
        print("=" * 72)
        print(
            f"BACKEND: {backend}"
        )
        print("=" * 72)

        cal = calibration[
            calibration["backend"]
            == backend
        ].copy()

        lookup = (
            cal
            .set_index("qubit")
            .to_dict("index")
        )

        backend_candidates = candidates[
            candidates["backend"]
            == backend
        ]

        # ----------------------------------------------------
        # SWEET-SPOT + RANDOM
        # ----------------------------------------------------

        for _, candidate in (
            backend_candidates.iterrows()
        ):

            qubits = parse_qubits(
                candidate["qubits"]
            )

            records = []

            for q in qubits:

                if q not in lookup:

                    raise RuntimeError(
                        f"{backend} q{q} missing "
                        "from current calibration."
                    )

                records.append(
                    lookup[q]
                )

            result = simulate(
                records
            )

            group = (
                "sweet_spot"
                if candidate[
                    "selection_type"
                ]
                == "sweet_spot"
                else "random_control"
            )

            trial = int(
                candidate[
                    "random_trial"
                ]
            )

            print(
                f"{group:15s} "
                f"trial={trial:02d} "
                f"qubits={qubits} "
                f"fidelity={result['fidelity']:.4f}"
            )

            all_results.append(
                {
                    "backend":
                        backend,
                    "group":
                        group,
                    "trial":
                        trial,
                    "qubits":
                        str(qubits),
                    "fidelity":
                        result["fidelity"],
                    "transpiled_depth":
                        result["depth"],
                    "shots":
                        SHOTS,
                    "counts":
                        json.dumps(
                            result["counts"]
                        ),
                    "snapshot_date":
                        str(
                            calibration[
                                "snapshot_date"
                            ].iloc[0]
                        ),
                }
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    summary = []

    for backend in BACKENDS:

        sub = df[
            df["backend"]
            == backend
        ]

        sweet = sub[
            sub["group"]
            == "sweet_spot"
        ]["fidelity"]

        random = sub[
            sub["group"]
            == "random_control"
        ]["fidelity"]

        summary.append(
            {
                "backend":
                    backend,
                "sweet_spot_fidelity":
                    float(
                        sweet.iloc[0]
                    ),
                "random_mean_fidelity":
                    float(
                        random.mean()
                    ),
                "random_std":
                    float(
                        random.std(
                            ddof=1
                        )
                    ),
                "difference":
                    float(
                        sweet.iloc[0]
                        - random.mean()
                    ),
                "random_n":
                    int(
                        len(random)
                    ),
            }
        )

    output = {
        "experiment":
            "sweet_spot_vs_random",
        "level":
            1,
        "simulation":
            "Qiskit Aer",
        "shots":
            SHOTS,
        "qpu_usage_seconds":
            0,
        "summary":
            summary,
        "raw_results":
            all_results,
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
    print("LEVEL 1 COMPLETE")
    print("=" * 72)

    for row in summary:

        print(
            f"{row['backend']:15s} "
            f"Sweet={row['sweet_spot_fidelity']:.4f} "
            f"Random={row['random_mean_fidelity']:.4f} "
            f"Δ={row['difference']:+.4f}"
        )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print()
    print("QPU usage: 0 seconds")


if __name__ == "__main__":
    main()