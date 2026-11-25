"""
KINGSTON-ONLY — FINAL SWEET-SPOT HARDWARE EXPERIMENT
=====================================================

PURPOSE
-------
Submit ONE FRESH Kingston hardware job for the already-defined
Temporal 30-day Sweet-Spot experiment.

IMPORTANT
---------
This script intentionally DOES NOT reuse the old queued Kingston job:

    dafgfgl1ierc738nbdd0

It submits exactly ONE NEW Kingston job containing:
    1 Sweet-Spot circuit
    10 random-control circuits
    256 shots/circuit
    11 PUBs total

The old queued Kingston job is NOT cancelled and is NOT used for the
official Kingston result.

The new Kingston result is saved to:

    results/sweet_spot_level2_ibm_kingston_result.json

If the FEZ and Marrakesh result files already exist, this script also
updates:

    results/sweet_spot_level2_hardware_results.json

No other backend is submitted by this script.
"""

from __future__ import annotations

import ast
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_INSTANCE")

BACKEND_NAME = "ibm_kingston"

MODEL_NAME = "temporal_30day"
PREDICTION_DATE = "2026-09-06"

SHOTS = 256
RANDOM_CONTROLS = 10
EXPECTED_CIRCUITS = 11

RESULTS_DIR = Path("results")

CANDIDATE_FILE = (
    RESULTS_DIR / "sweet_spot_level2_candidates.csv"
)

KINGSTON_RESULT_FILE = (
    RESULTS_DIR / "sweet_spot_level2_ibm_kingston_result.json"
)

FINAL_RESULT_FILE = (
    RESULTS_DIR / "sweet_spot_level2_hardware_results.json"
)

# The old queued Kingston job is deliberately NOT reused.
OLD_KINGSTON_JOB_ID = "dafgfgl1ierc738nbdd0"

POLL_SECONDS = 5
NETWORK_RETRIES = 5
NETWORK_RETRY_SECONDS = 10


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    temp.replace(path)


# ============================================================
# IBM CONNECTION
# ============================================================

def connect() -> QiskitRuntimeService:
    if not TOKEN:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN missing from .env"
        )

    if not INSTANCE:
        raise RuntimeError(
            "IBM_INSTANCE missing from .env"
        )

    print(
        "Connecting to IBM Quantum..."
    )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE,
    )

    print(
        "IBM authentication: PASS"
    )
    print(
        f"IBM instance: {INSTANCE}"
    )

    return service


# ============================================================
# CIRCUIT
# ============================================================

def build_ghz() -> QuantumCircuit:
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
# CANDIDATE MANIFEST
# ============================================================

def parse_qubits(
    value: Any,
) -> list[int]:
    try:
        qubits = ast.literal_eval(
            str(value)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse qubit list: {value}"
        ) from exc

    if not isinstance(
        qubits,
        (list, tuple),
    ):
        raise RuntimeError(
            f"Invalid qubit list: {value}"
        )

    result = [
        int(q)
        for q in qubits
    ]

    if len(result) != 3:
        raise RuntimeError(
            f"Expected 3 qubits, got {result}"
        )

    if len(set(result)) != 3:
        raise RuntimeError(
            f"Duplicate qubits in {result}"
        )

    return result


def load_and_validate_candidates() -> pd.DataFrame:
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {CANDIDATE_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    required = {
        "prediction_date",
        "backend",
        "selection_type",
        "model_name",
        "qubits",
    }

    missing = (
        required
        - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    sub = candidates[
        candidates["backend"] == BACKEND_NAME
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No candidate rows for {BACKEND_NAME}"
        )

    dates = (
        sub["prediction_date"]
        .astype(str)
        .unique()
    )

    if PREDICTION_DATE not in dates:
        raise RuntimeError(
            f"{BACKEND_NAME}: prediction date "
            f"{PREDICTION_DATE} not found."
        )

    sweet = sub[
        sub["selection_type"]
        == "sweet_spot"
    ]

    randoms = sub[
        sub["selection_type"]
        == "random_control"
    ]

    if len(sweet) != 1:
        raise RuntimeError(
            f"{BACKEND_NAME}: expected exactly "
            f"1 Sweet-Spot row, got {len(sweet)}"
        )

    if len(randoms) < RANDOM_CONTROLS:
        raise RuntimeError(
            f"{BACKEND_NAME}: expected at least "
            f"{RANDOM_CONTROLS} random controls, "
            f"got {len(randoms)}"
        )

    if str(
        sweet.iloc[0]["model_name"]
    ) != MODEL_NAME:
        raise RuntimeError(
            f"Sweet-Spot model mismatch: expected "
            f"{MODEL_NAME}, got "
            f"{sweet.iloc[0]['model_name']}"
        )

    parse_qubits(
        sweet.iloc[0]["qubits"]
    )

    for _, row in randoms.head(
        RANDOM_CONTROLS
    ).iterrows():
        parse_qubits(
            row["qubits"]
        )

    print(
        "Candidate manifest: PASS"
    )

    return candidates


# ============================================================
# PREPARE 11 CIRCUITS
# ============================================================

def prepare_experiment(
    backend: Any,
    candidates: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[QuantumCircuit],
]:
    sub = candidates[
        candidates["backend"]
        == BACKEND_NAME
    ].copy()

    sweet = sub[
        sub["selection_type"]
        == "sweet_spot"
    ].iloc[0]

    randoms = sub[
        sub["selection_type"]
        == "random_control"
    ].head(
        RANDOM_CONTROLS
    )

    groups: list[dict[str, Any]] = [
        {
            "group": "sweet_spot",
            "trial": 0,
            "qubits": parse_qubits(
                sweet["qubits"]
            ),
            "model": MODEL_NAME,
            "mean_viability": (
                float(
                    sweet["mean_viability"]
                )
                if (
                    "mean_viability"
                    in sweet.index
                    and pd.notna(
                        sweet["mean_viability"]
                    )
                )
                else None
            ),
        }
    ]

    for trial, (_, row) in enumerate(
        randoms.iterrows(),
        start=1,
    ):
        groups.append(
            {
                "group": "random_control",
                "trial": trial,
                "qubits": parse_qubits(
                    row["qubits"]
                ),
                "model": "random",
                "mean_viability": None,
            }
        )

    if len(groups) != EXPECTED_CIRCUITS:
        raise RuntimeError(
            f"Expected {EXPECTED_CIRCUITS} circuits, "
            f"got {len(groups)}"
        )

    base = build_ghz()

    metadata = []
    circuits = []

    print()
    print(
        f"Preparing {BACKEND_NAME}"
    )
    print(
        f"Circuits: {len(groups)}"
    )

    for index, item in enumerate(
        groups
    ):
        qubits = item["qubits"]

        compiled = transpile(
            base,
            backend=backend,
            initial_layout=qubits,
            optimization_level=1,
        )

        depth = int(
            compiled.depth()
        )

        print(
            f"  {index + 1:02d}/"
            f"{len(groups):02d} "
            f"{item['group']:15s} "
            f"{qubits} "
            f"| depth={depth}"
        )

        metadata.append(
            {
                "circuit_index": index,
                "group": item["group"],
                "trial": item["trial"],
                "physical_qubits": qubits,
                "model": item["model"],
                "mean_viability": item[
                    "mean_viability"
                ],
                "transpiled_depth": depth,
            }
        )

        circuits.append(
            compiled
        )

    return metadata, circuits


# ============================================================
# COUNTS / FIDELITY
# ============================================================

def extract_counts(
    pub_result: Any,
) -> dict[str, int]:
    data = pub_result.data

    if hasattr(data, "c"):
        counts = data.c.get_counts()

        return {
            str(k): int(v)
            for k, v in counts.items()
        }

    for name in dir(data):
        if name.startswith("_"):
            continue

        try:
            obj = getattr(
                data,
                name,
            )

            if hasattr(
                obj,
                "get_counts",
            ):
                counts = (
                    obj.get_counts()
                )

                return {
                    str(k): int(v)
                    for k, v in counts.items()
                }

        except Exception:
            continue

    raise RuntimeError(
        "Could not extract SamplerV2 measurement counts."
    )


def ghz_fidelity(
    counts: dict[str, int],
) -> float:
    total = sum(
        int(v)
        for v in counts.values()
    )

    if total <= 0:
        raise RuntimeError(
            "No measurement shots returned."
        )

    return (
        int(
            counts.get(
                "000",
                0,
            )
        )
        +
        int(
            counts.get(
                "111",
                0,
            )
        )
    ) / total


# ============================================================
# SUBMIT FRESH KINGSTON JOB
# ============================================================

def submit_fresh_job(
    backend: Any,
    circuits: list[QuantumCircuit],
) -> str:
    print()
    print("=" * 72)
    print(
        "SUBMITTING ONE FRESH KINGSTON JOB"
    )
    print("=" * 72)

    print(
        f"Backend : {BACKEND_NAME}"
    )
    print(
        f"Circuits: {len(circuits)}"
    )
    print(
        f"Shots   : {SHOTS} per circuit"
    )
    print(
        "Mode    : Job"
    )
    print()
    print(
        "OLD Kingston job will NOT be reused:"
    )
    print(
        f"  {OLD_KINGSTON_JOB_ID}"
    )
    print(
        "OLD Kingston job will NOT be cancelled."
    )
    print()

    sampler = SamplerV2(
        mode=backend
    )

    try:
        job = sampler.run(
            circuits,
            shots=SHOTS,
        )
    except Exception as exc:
        raise RuntimeError(
            "Fresh Kingston submission failed "
            "before an IBM job ID was returned. "
            f"Error: {exc}"
        ) from exc

    job_id = str(
        job.job_id()
    )

    print()
    print(
        "NEW KINGSTON JOB CREATED"
    )
    print(
        f"NEW JOB ID: {job_id}"
    )

    # This ID is printed immediately so it can be recovered
    # even if the local process later loses connectivity.
    print(
        "IMPORTANT: SAVE THIS JOB ID:"
    )
    print(
        job_id
    )

    return job_id


# ============================================================
# STATUS / RESULT
# ============================================================

def get_status(
    job: Any,
) -> str | None:
    for attempt in range(
        1,
        NETWORK_RETRIES + 1,
    ):
        try:
            return str(
                job.status()
            )

        except Exception as exc:
            print(
                f"Status query failed "
                f"({attempt}/{NETWORK_RETRIES}): "
                f"{exc}"
            )

            if attempt < NETWORK_RETRIES:
                time.sleep(
                    NETWORK_RETRY_SECONDS
                )

    return None


def get_qpu_usage(
    job: Any,
) -> float | None:
    try:
        usage = job.usage(
            partial=False
        )

        if usage is not None:
            return float(usage)

    except Exception:
        pass

    try:
        metrics = job.metrics()

        usage_data = metrics.get(
            "usage",
            {},
        )

        value = usage_data.get(
            "quantum_seconds"
        )

        if value is not None:
            return float(value)

    except Exception:
        pass

    return None


def process_result(
    result: Any,
    metadata: list[dict[str, Any]],
    job_id: str,
) -> list[dict[str, Any]]:
    if len(result) != len(metadata):
        raise RuntimeError(
            f"Expected {len(metadata)} PUB results, "
            f"got {len(result)}"
        )

    rows = []

    for index, item in enumerate(
        metadata
    ):
        counts = extract_counts(
            result[index]
        )

        fidelity = ghz_fidelity(
            counts
        )

        rows.append(
            {
                "backend": BACKEND_NAME,
                "circuit_index": item[
                    "circuit_index"
                ],
                "group": item["group"],
                "trial": item["trial"],
                "model": item["model"],
                "physical_qubits": item[
                    "physical_qubits"
                ],
                "mean_viability": item[
                    "mean_viability"
                ],
                "shots": SHOTS,
                "job_id": job_id,
                "counts": counts,
                "ghz_fidelity": fidelity,
                "transpiled_depth": item[
                    "transpiled_depth"
                ],
            }
        )

    return rows


def save_kingston_result(
    job_id: str,
    rows: list[dict[str, Any]],
    qpu_usage: float | None,
) -> None:
    payload = {
        "experiment": "sweet_spot_vs_random",
        "level": 2,
        "model": MODEL_NAME,
        "prediction_date": PREDICTION_DATE,
        "shots": SHOTS,
        "random_controls_per_backend": RANDOM_CONTROLS,
        "backend": BACKEND_NAME,
        "job_id": job_id,
        "qpu_usage_seconds": qpu_usage,
        "results": rows,
        "completed_at_utc": utc_now(),
    }

    atomic_write_json(
        KINGSTON_RESULT_FILE,
        payload,
    )

    print()
    print(
        f"Kingston result saved: "
        f"{KINGSTON_RESULT_FILE}"
    )


# ============================================================
# UPDATE COMBINED RESULT
# ============================================================

def update_combined_output() -> None:
    backend_files = {
        "ibm_fez": (
            RESULTS_DIR
            / "sweet_spot_level2_ibm_fez_result.json"
        ),
        "ibm_kingston": KINGSTON_RESULT_FILE,
        "ibm_marrakesh": (
            RESULTS_DIR
            / "sweet_spot_level2_ibm_marrakesh_result.json"
        ),
    }

    all_results = []
    summaries = []

    total_qpu = 0.0
    qpu_known = True

    for backend_name, path in backend_files.items():
        if not path.exists():
            qpu_known = False
            continue

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        all_results.extend(
            data.get(
                "results",
                [],
            )
        )

        usage = data.get(
            "qpu_usage_seconds"
        )

        if usage is None:
            qpu_known = False
        else:
            total_qpu += float(
                usage
            )

        summaries.append(
            {
                "backend": backend_name,
                "job_id": data.get(
                    "job_id"
                ),
                "result_file": str(path),
                "qpu_usage_seconds": usage,
            }
        )

    # Do not pretend the complete experiment is finished unless
    # all three backend result files exist.
    complete = all(
        path.exists()
        for path in backend_files.values()
    )

    payload = {
        "experiment": "sweet_spot_vs_random",
        "level": 2,
        "model": MODEL_NAME,
        "prediction_date": PREDICTION_DATE,
        "shots": SHOTS,
        "random_controls_per_backend": RANDOM_CONTROLS,
        "backends": list(
            backend_files.keys()
        ),
        "hardware_jobs": len(
            summaries
        ),
        "total_hardware_circuits": len(
            all_results
        ),
        "total_qpu_seconds": (
            total_qpu
            if qpu_known
            else None
        ),
        "complete": complete,
        "results": all_results,
        "backend_job_summary": summaries,
        "updated_at_utc": utc_now(),
    }

    atomic_write_json(
        FINAL_RESULT_FILE,
        payload,
    )

    print()
    print(
        f"Combined output updated: "
        f"{FINAL_RESULT_FILE}"
    )

    print(
        f"Complete experiment: "
        f"{'YES' if complete else 'NO'}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print(
        "KINGSTON-ONLY — FRESH SWEET-SPOT HARDWARE RUN"
    )
    print("=" * 72)

    print()
    print(
        "THIS SCRIPT SUBMITS ONE NEW KINGSTON JOB."
    )
    print(
        "It does NOT reuse the old queued Kingston job."
    )
    print(
        f"Old job preserved: {OLD_KINGSTON_JOB_ID}"
    )

    candidates = (
        load_and_validate_candidates()
    )

    service = connect()

    backend = service.backend(
        BACKEND_NAME
    )

    metadata, circuits = (
        prepare_experiment(
            backend,
            candidates,
        )
    )

    # --------------------------------------------------------
    # FRESH SUBMISSION
    # --------------------------------------------------------

    job_id = submit_fresh_job(
        backend,
        circuits,
    )

    # Persist the experiment metadata locally immediately.
    submission_record = {
        "experiment": "sweet_spot_vs_random",
        "backend": BACKEND_NAME,
        "model": MODEL_NAME,
        "prediction_date": PREDICTION_DATE,
        "shots": SHOTS,
        "job_id": job_id,
        "old_queued_job_id": OLD_KINGSTON_JOB_ID,
        "metadata": metadata,
        "submitted_at_utc": utc_now(),
    }

    submission_file = (
        RESULTS_DIR
        / "sweet_spot_level2_ibm_kingston_submission.json"
    )

    atomic_write_json(
        submission_file,
        submission_record,
    )

    print()
    print(
        f"Submission checkpoint: "
        f"{submission_file}"
    )

    # --------------------------------------------------------
    # MONITOR NEW JOB
    # --------------------------------------------------------

    # Retrieve the job by ID. This deliberately uses the NEW ID,
    # not the old queued Kingston job.
    job = service.job(
        job_id
    )

    print()
    print("=" * 72)
    print(
        "MONITORING NEW KINGSTON JOB"
    )
    print("=" * 72)
    print(
        f"Job ID: {job_id}"
    )

    while True:
        status = get_status(
            job
        )

        if status is None:
            print()
            print(
                "IBM status endpoint temporarily unavailable."
            )
            print(
                "The NEW Kingston job remains submitted."
            )
            print(
                f"Recover it later with job ID: {job_id}"
            )
            return

        print(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"NEW KINGSTON: {status}"
        )

        if status == "DONE":
            break

        if status in {
            "ERROR",
            "CANCELLED",
        }:
            try:
                error = job.error_message()
            except Exception:
                error = None

            raise RuntimeError(
                f"NEW Kingston job ended in {status}. "
                f"Job ID={job_id}. "
                f"Error={error}"
            )

        time.sleep(
            POLL_SECONDS
        )

    # --------------------------------------------------------
    # RETRIEVE RESULT
    # --------------------------------------------------------

    print()
    print(
        "NEW Kingston job DONE."
    )
    print(
        "Retrieving 11 PUB results..."
    )

    result = None

    for attempt in range(
        1,
        NETWORK_RETRIES + 1,
    ):
        try:
            result = job.result(
                timeout=60,
                poll_interval=1.0,
            )
            break

        except Exception as exc:
            print(
                f"Result retrieval failed "
                f"({attempt}/{NETWORK_RETRIES}): "
                f"{exc}"
            )

            if attempt < NETWORK_RETRIES:
                time.sleep(
                    NETWORK_RETRY_SECONDS
                )

    if result is None:
        print()
        print(
            "Job is DONE, but result retrieval failed."
        )
        print(
            f"NEW Kingston job ID: {job_id}"
        )
        print(
            "Do NOT resubmit. Retrieve this same job later."
        )
        return

    rows = process_result(
        result,
        metadata,
        job_id,
    )

    qpu_usage = get_qpu_usage(
        job
    )

    save_kingston_result(
        job_id,
        rows,
        qpu_usage,
    )

    print()
    print("=" * 72)
    print(
        "NEW KINGSTON RESULT"
    )
    print("=" * 72)

    for row in rows:
        print()
        print(
            f"{row['group']:15s} "
            f"{row['physical_qubits']} "
            f"| fidelity="
            f"{row['ghz_fidelity']:.6f}"
        )
        print(
            f"  Counts: {row['counts']}"
        )

    if qpu_usage is not None:
        print()
        print(
            f"New Kingston QPU usage: "
            f"{qpu_usage:.6f} seconds"
        )

    # --------------------------------------------------------
    # MERGE WITH FEZ + MARRAKESH
    # --------------------------------------------------------

    update_combined_output()

    print()
    print("=" * 72)
    print(
        "KINGSTON FRESH RUN COMPLETE"
    )
    print("=" * 72)
    print()
    print(
        f"NEW Kingston job : {job_id}"
    )
    print(
        f"Kingston result  : {KINGSTON_RESULT_FILE}"
    )
    print(
        f"Combined result  : {FINAL_RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
