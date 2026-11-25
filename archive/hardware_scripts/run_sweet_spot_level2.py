"""
LEVEL 2 — FINAL SWEET-SPOT HARDWARE EXPERIMENT
================================================

Temporal 30-day Sweet-Spot model-selected 3-qubit chains
versus topology-valid random control chains.

IMPORTANT:
- 3 IBM Quantum backends.
- 1 SamplerV2 job/backend.
- 11 circuits/job: 1 Sweet-Spot + 10 random controls.
- 256 shots/circuit.
- Job mode only: SamplerV2(mode=backend).
- No Session or Batch mode.
- Default is DRY RUN.
- --submit submits only backends that do not already have a recorded job.
- --resubmit-kingston intentionally creates one additional Kingston job;
  the older Kingston job is preserved and may still execute.
- Existing jobs are automatically recovered by job ID.
- Job IDs are checkpointed immediately after submission.
- Results are saved per backend as soon as they are available.
- A network/DNS failure does NOT cause a duplicate submission.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
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

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]

MODEL_NAME = "temporal_30day"
PREDICTION_DATE = "2026-09-06"

SHOTS = 256
RANDOM_CONTROLS_TO_RUN = 10
EXPECTED_CIRCUITS_PER_BACKEND = 1 + RANDOM_CONTROLS_TO_RUN

RESULTS_DIR = Path("results")

CANDIDATE_FILE = RESULTS_DIR / "sweet_spot_level2_candidates.csv"
OUTPUT_FILE = RESULTS_DIR / "sweet_spot_level2_hardware_results.json"
STATE_FILE = RESULTS_DIR / "sweet_spot_level2_job_state.json"

# Jobs already submitted in THIS experiment.
# FEZ completed. Kingston is queued. Marrakesh was not submitted.
#
# If STATE_FILE exists, its job IDs take precedence.
INITIAL_JOB_IDS = {
    "ibm_fez": "dafgfb5nj4cs73agbfo0",
    "ibm_kingston": "dafgfgl1ierc738nbdd0",
    "ibm_marrakesh": None,
}

# Polling is intentionally modest. IBM's scheduler, not this script,
# determines when a queued job runs.
POLL_SECONDS = 8
NETWORK_RETRY_SECONDS = 15
MAX_NETWORK_RETRIES = 5


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Temporal 30-day Sweet-Spot hardware experiment"
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit only missing backends, then recover/monitor all "
            "existing jobs."
        ),
    )

    mode.add_argument(
        "--recover",
        action="store_true",
        help=(
            "Never submit a new job. Recover/monitor jobs already recorded "
            "in the state file or INITIAL_JOB_IDS."
        ),
    )

    mode.add_argument(
        "--status",
        action="store_true",
        help="Only print the current status of all recorded jobs.",
    )

    parser.add_argument(
        "--resubmit-kingston",
        action="store_true",
        help=(
            "Submit ONE fresh Kingston job even if an older Kingston job "
            "already exists. The old job is preserved and never cancelled."
        ),
    )

    return parser.parse_args()


# ============================================================
# TIME / JSON UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    temp_path.replace(path)


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        state = {
            "schema_version": 2,
            "experiment": "sweet_spot_vs_random",
            "level": 2,
            "model": MODEL_NAME,
            "prediction_date": PREDICTION_DATE,
            "shots": SHOTS,
            "random_controls_per_backend": RANDOM_CONTROLS_TO_RUN,
            "backends": {},
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }

        for backend_name in BACKENDS:
            existing_job_id = INITIAL_JOB_IDS.get(backend_name)

            state["backends"][backend_name] = {
                "job_id": existing_job_id,
                "status": "UNKNOWN" if existing_job_id else "NOT_SUBMITTED",
                "submitted_at_utc": None,
                "metadata": None,
                "result_file": None,
                "qpu_usage_seconds": None,
                "last_error": None,
                "superseded_job_ids": [],
                "updated_at_utc": utc_now(),
            }

        atomic_write_json(STATE_FILE, state)
        return state

    with STATE_FILE.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid state file: {STATE_FILE}")

    state.setdefault("backends", {})

    for backend_name in BACKENDS:
        state["backends"].setdefault(
            backend_name,
            {
                "job_id": INITIAL_JOB_IDS.get(backend_name),
                "status": (
                    "UNKNOWN"
                    if INITIAL_JOB_IDS.get(backend_name)
                    else "NOT_SUBMITTED"
                ),
                "submitted_at_utc": None,
                "metadata": None,
                "result_file": None,
                "qpu_usage_seconds": None,
                "last_error": None,
                "superseded_job_ids": [],
                "updated_at_utc": utc_now(),
            },
        )

    for backend_name in BACKENDS:
        state["backends"][backend_name].setdefault(
            "superseded_job_ids",
            [],
        )

    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(STATE_FILE, state)


# ============================================================
# IBM CONNECTION
# ============================================================

def connect() -> QiskitRuntimeService:
    if not TOKEN:
        raise RuntimeError("IBM_QUANTUM_TOKEN missing from .env")

    if not INSTANCE:
        raise RuntimeError("IBM_INSTANCE missing from .env")

    print("Connecting to IBM Quantum...")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE,
    )

    print("IBM authentication: PASS")
    print(f"IBM instance: {INSTANCE}")

    return service


# ============================================================
# GHZ CIRCUIT
# ============================================================

def build_ghz() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)

    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))

    return qc


# ============================================================
# COUNTS / FIDELITY
# ============================================================

def extract_counts(pub_result: Any) -> dict[str, int]:
    data = pub_result.data

    if hasattr(data, "c"):
        counts = data.c.get_counts()
        return {str(k): int(v) for k, v in counts.items()}

    for name in dir(data):
        if name.startswith("_"):
            continue

        try:
            obj = getattr(data, name)

            if hasattr(obj, "get_counts"):
                counts = obj.get_counts()
                return {str(k): int(v) for k, v in counts.items()}

        except Exception:
            continue

    raise RuntimeError("Could not extract SamplerV2 measurement counts.")


def ghz_fidelity(counts: dict[str, int]) -> float:
    total = sum(int(v) for v in counts.values())

    if total <= 0:
        raise RuntimeError("No measurement shots returned.")

    return (
        int(counts.get("000", 0))
        + int(counts.get("111", 0))
    ) / total


# ============================================================
# CANDIDATE MANIFEST
# ============================================================

def load_candidates() -> pd.DataFrame:
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing candidate file:\n"
            f"{CANDIDATE_FILE}\n\n"
            "Run first:\n"
            "python select_sweet_spot_candidates.py"
        )

    return pd.read_csv(CANDIDATE_FILE)


def parse_qubits(value: Any) -> list[int]:
    try:
        qubits = ast.literal_eval(str(value))
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse qubit list: {value}"
        ) from exc

    if not isinstance(qubits, (list, tuple)):
        raise RuntimeError(f"Invalid qubit list: {value}")

    result = [int(q) for q in qubits]

    if len(result) != 3:
        raise RuntimeError(f"Expected 3 physical qubits, got {result}")

    if len(set(result)) != 3:
        raise RuntimeError(f"Duplicate physical qubits in {result}")

    return result


def validate_manifest(candidates: pd.DataFrame) -> None:
    required_columns = {
        "prediction_date",
        "backend",
        "selection_type",
        "model_name",
        "qubits",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise RuntimeError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    dates = candidates["prediction_date"].astype(str).unique()

    if PREDICTION_DATE not in dates:
        raise RuntimeError(
            "Candidate manifest does not contain "
            f"prediction date {PREDICTION_DATE}."
        )

    for backend_name in BACKENDS:
        sub = candidates[
            candidates["backend"] == backend_name
        ].copy()

        sweet = sub[
            sub["selection_type"] == "sweet_spot"
        ]

        randoms = sub[
            sub["selection_type"] == "random_control"
        ]

        if len(sweet) != 1:
            raise RuntimeError(
                f"{backend_name}: expected exactly one "
                f"Sweet-Spot chain; found {len(sweet)}."
            )

        if len(randoms) < RANDOM_CONTROLS_TO_RUN:
            raise RuntimeError(
                f"{backend_name}: expected at least "
                f"{RANDOM_CONTROLS_TO_RUN} random controls; "
                f"found {len(randoms)}."
            )

        sweet_row = sweet.iloc[0]

        if str(sweet_row["model_name"]) != MODEL_NAME:
            raise RuntimeError(
                f"{backend_name}: expected model {MODEL_NAME}, "
                f"got {sweet_row['model_name']}"
            )

        parse_qubits(sweet_row["qubits"])

        for _, row in randoms.head(
            RANDOM_CONTROLS_TO_RUN
        ).iterrows():
            parse_qubits(row["qubits"])

    print("Candidate manifest: PASS")
    print(f"Sweet-Spot chains: {len(BACKENDS)}")
    print(
        f"Random controls : "
        f"{RANDOM_CONTROLS_TO_RUN}/backend"
    )
    print(
        f"Total circuits  : "
        f"{len(BACKENDS) * EXPECTED_CIRCUITS_PER_BACKEND}"
    )


# ============================================================
# PREPARE CIRCUITS
# ============================================================

def prepare_backend_experiment(
    backend: Any,
    candidates: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[QuantumCircuit]]:
    """
    Fixed ordering:
      circuit 0   = Sweet-Spot
      circuits 1-10 = random controls in manifest order

    Metadata is checkpointed so recovered IBM results can never be
    mapped to the wrong experiment row by guesswork.
    """

    sub = candidates[
        candidates["backend"] == backend.name
    ].copy()

    sweet = sub[
        sub["selection_type"] == "sweet_spot"
    ].iloc[0]

    randoms = sub[
        sub["selection_type"] == "random_control"
    ].head(RANDOM_CONTROLS_TO_RUN)

    groups: list[dict[str, Any]] = [
        {
            "group": "sweet_spot",
            "trial": 0,
            "qubits": parse_qubits(sweet["qubits"]),
            "model": MODEL_NAME,
            "mean_viability": (
                float(sweet["mean_viability"])
                if "mean_viability" in sweet.index
                and pd.notna(sweet["mean_viability"])
                else None
            ),
        }
    ]

    for index, (_, row) in enumerate(
        randoms.iterrows(),
        start=1,
    ):
        groups.append(
            {
                "group": "random_control",
                "trial": index,
                "qubits": parse_qubits(row["qubits"]),
                "model": "random",
                "mean_viability": None,
            }
        )

    if len(groups) != EXPECTED_CIRCUITS_PER_BACKEND:
        raise RuntimeError(
            f"{backend.name}: expected "
            f"{EXPECTED_CIRCUITS_PER_BACKEND} circuits, "
            f"got {len(groups)}"
        )

    base_circuit = build_ghz()

    transpiled_circuits: list[QuantumCircuit] = []
    metadata: list[dict[str, Any]] = []

    print()
    print(f"Preparing {backend.name}")
    print(f"Circuits: {len(groups)}")

    for circuit_index, item in enumerate(groups):
        qubits = item["qubits"]

        compiled = transpile(
            base_circuit,
            backend=backend,
            initial_layout=qubits,
            optimization_level=1,
        )

        depth = int(compiled.depth())

        print(
            f"  {circuit_index + 1:02d}/"
            f"{len(groups):02d} "
            f"{item['group']:15s} "
            f"{qubits} | depth={depth}"
        )

        transpiled_circuits.append(compiled)

        metadata.append(
            {
                "circuit_index": circuit_index,
                "group": item["group"],
                "trial": item["trial"],
                "physical_qubits": qubits,
                "model": item["model"],
                "mean_viability": item["mean_viability"],
                "transpiled_depth": depth,
            }
        )

    return metadata, transpiled_circuits


def metadata_json_safe(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return json.loads(json.dumps(metadata))


# ============================================================
# JOB STATE HELPERS
# ============================================================

def state_entry(
    state: dict[str, Any],
    backend_name: str,
) -> dict[str, Any]:
    return state["backends"][backend_name]


def record_metadata(
    state: dict[str, Any],
    backend_name: str,
    metadata: list[dict[str, Any]],
) -> None:
    entry = state_entry(state, backend_name)
    entry["metadata"] = metadata_json_safe(metadata)
    entry["updated_at_utc"] = utc_now()


def existing_job_id(
    state: dict[str, Any],
    backend_name: str,
) -> str | None:
    value = state_entry(state, backend_name).get("job_id")

    if value is None:
        return None

    value = str(value).strip()

    return value if value else None


# ============================================================
# SAFE IBM API OPERATIONS
# ============================================================

def retrieve_job(
    service: QiskitRuntimeService,
    job_id: str,
) -> Any:
    return service.job(job_id)


def get_status_with_retry(job: Any) -> str | None:
    last_error: Exception | None = None

    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            return str(job.status())

        except Exception as exc:
            last_error = exc

            print(
                f"  Status query failed "
                f"(attempt {attempt}/{MAX_NETWORK_RETRIES}): "
                f"{exc}"
            )

            if attempt < MAX_NETWORK_RETRIES:
                time.sleep(NETWORK_RETRY_SECONDS)

    print(
        "  Could not reach IBM Quantum after retries. "
        "The job was NOT resubmitted."
    )

    if last_error is not None:
        print(f"  Last network error: {last_error}")

    return None


def get_usage_safely(job: Any) -> float | None:
    try:
        usage = job.usage(partial=False)

        if usage is not None:
            return float(usage)

    except Exception:
        pass

    try:
        metrics = job.metrics()
        usage_data = metrics.get("usage", {})
        quantum_seconds = usage_data.get("quantum_seconds")

        if quantum_seconds is not None:
            return float(quantum_seconds)

    except Exception:
        pass

    return None


def get_error_message_safely(job: Any) -> str | None:
    try:
        message = job.error_message()

        if message:
            return str(message)

    except Exception:
        pass

    return None


# ============================================================
# RESULT PROCESSING
# ============================================================

def process_result(
    backend_name: str,
    job_id: str,
    result: Any,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(result) != len(metadata):
        raise RuntimeError(
            f"{backend_name}: result count {len(result)} "
            f"does not match circuit count {len(metadata)}."
        )

    backend_results: list[dict[str, Any]] = []

    for index, item in enumerate(metadata):
        pub_result = result[index]
        counts = extract_counts(pub_result)
        fidelity = ghz_fidelity(counts)

        backend_results.append(
            {
                "backend": backend_name,
                "circuit_index": item["circuit_index"],
                "group": item["group"],
                "trial": item["trial"],
                "model": item["model"],
                "physical_qubits": item["physical_qubits"],
                "mean_viability": item["mean_viability"],
                "shots": SHOTS,
                "job_id": job_id,
                "counts": counts,
                "ghz_fidelity": fidelity,
                "transpiled_depth": item["transpiled_depth"],
            }
        )

    return {
        "backend": backend_name,
        "job_id": job_id,
        "circuits": backend_results,
    }


def save_backend_result(
    backend_result: dict[str, Any],
    qpu_usage_seconds: float | None,
) -> Path:
    backend_name = backend_result["backend"]

    output = {
        "experiment": "sweet_spot_vs_random",
        "level": 2,
        "model": MODEL_NAME,
        "prediction_date": PREDICTION_DATE,
        "shots": SHOTS,
        "random_controls_per_backend": RANDOM_CONTROLS_TO_RUN,
        "backend": backend_name,
        "job_id": backend_result["job_id"],
        "total_circuits": len(
            backend_result["circuits"]
        ),
        "qpu_usage_seconds": qpu_usage_seconds,
        "results": backend_result["circuits"],
    }

    path = (
        RESULTS_DIR
        / f"sweet_spot_level2_{backend_name}_result.json"
    )

    atomic_write_json(path, output)

    return path


# ============================================================
# SUBMISSION
# ============================================================

def submit_fresh_backend(
    service: QiskitRuntimeService,
    backend: Any,
    candidates: pd.DataFrame,
    state: dict[str, Any],
) -> str:
    """
    Intentionally submit ONE fresh job for a backend even when an older
    job exists.

    Used only for the explicit --resubmit-kingston action.

    The old job ID is archived in superseded_job_ids and is NOT cancelled.
    The new job becomes the active job whose result is used for the final
    experiment output.
    """

    backend_name = backend.name
    entry = state_entry(state, backend_name)

    old_job_id = existing_job_id(
        state,
        backend_name,
    )

    metadata, circuits = prepare_backend_experiment(
        backend,
        candidates,
    )

    record_metadata(
        state,
        backend_name,
        metadata,
    )

    if old_job_id:
        archived = entry.setdefault(
            "superseded_job_ids",
            [],
        )

        if old_job_id not in archived:
            archived.append(old_job_id)

    # Save metadata before submission.
    save_state(state)

    print()
    print("=" * 72)
    print(f"EXPLICIT FRESH SUBMISSION: {backend_name}")
    print("=" * 72)

    if old_job_id:
        print(
            f"  Existing job preserved : {old_job_id}"
        )
        print(
            "  Existing job action    : NOT CANCELLED"
        )

    print(
        f"  New job circuits       : {len(circuits)}"
    )
    print(
        f"  Shots                  : {SHOTS} per circuit"
    )
    print(
        "  Primitive              : SamplerV2"
    )
    print(
        "  Mode                   : job mode (backend)"
    )

    sampler = SamplerV2(mode=backend)

    try:
        job = sampler.run(
            circuits,
            shots=SHOTS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{backend_name}: fresh submission failed before an IBM "
            f"job ID was returned. Existing job was NOT cancelled. "
            f"Error: {exc}"
        ) from exc

    new_job_id = str(job.job_id())

    # CRITICAL: checkpoint the new ID immediately.
    entry["job_id"] = new_job_id
    entry["status"] = "SUBMITTED"
    entry["submitted_at_utc"] = utc_now()
    entry["last_error"] = None
    entry["result_file"] = None
    entry["qpu_usage_seconds"] = None
    entry["updated_at_utc"] = utc_now()

    save_state(state)

    print()
    print(
        f"{backend_name}: NEW JOB SUBMITTED"
    )
    print(
        f"  New Job ID: {new_job_id}"
    )
    print(
        "  Old job: preserved; it may still run independently."
    )
    print(
        f"  State checkpoint: {STATE_FILE}"
    )

    return new_job_id


def ensure_metadata_for_existing_job(
    backend: Any,
    candidates: pd.DataFrame,
    state: dict[str, Any],
) -> None:
    """
    Existing jobs need the same deterministic circuit ordering metadata
    before their results can be decoded safely.
    """

    backend_name = backend.name
    entry = state_entry(state, backend_name)

    if entry.get("metadata"):
        return

    metadata, _ = prepare_backend_experiment(
        backend,
        candidates,
    )

    record_metadata(
        state,
        backend_name,
        metadata,
    )

    save_state(state)


def submit_backend_if_needed(
    service: QiskitRuntimeService,
    backend: Any,
    candidates: pd.DataFrame,
    state: dict[str, Any],
) -> str:
    backend_name = backend.name

    current_id = existing_job_id(
        state,
        backend_name,
    )

    if current_id:
        print()
        print(
            f"{backend_name}: EXISTING JOB FOUND"
        )
        print(f"  Job ID: {current_id}")
        print("  No duplicate submission will be made.")

        ensure_metadata_for_existing_job(
            backend,
            candidates,
            state,
        )

        return current_id

    metadata, circuits = prepare_backend_experiment(
        backend,
        candidates,
    )

    record_metadata(
        state,
        backend_name,
        metadata,
    )

    # Save circuit ordering BEFORE submission.
    save_state(state)

    print()
    print("-" * 72)
    print(f"SUBMITTING: {backend_name}")
    print("-" * 72)
    print(f"Circuits : {len(circuits)}")
    print(f"Shots    : {SHOTS} per circuit")
    print("Primitive: SamplerV2")
    print("Mode     : job mode (backend)")
    print()
    print(
        "Creating exactly ONE IBM job for this backend..."
    )

    sampler = SamplerV2(mode=backend)

    try:
        job = sampler.run(
            circuits,
            shots=SHOTS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{backend_name}: IBM submission failed before a job ID "
            f"was returned. No recovery ID is available. Error: {exc}"
        ) from exc

    job_id = str(job.job_id())

    # CRITICAL CHECKPOINT:
    # Save the IBM job ID immediately after submission.
    entry = state_entry(
        state,
        backend_name,
    )

    entry["job_id"] = job_id
    entry["status"] = "SUBMITTED"
    entry["submitted_at_utc"] = utc_now()
    entry["last_error"] = None
    entry["updated_at_utc"] = utc_now()

    save_state(state)

    print()
    print(f"{backend_name}: SUBMISSION CONFIRMED")
    print(f"  Job ID: {job_id}")
    print("  State checkpoint: SAVED")

    return job_id


# ============================================================
# WAIT / RECOVER ONE JOB
# ============================================================

def wait_for_job(
    service: QiskitRuntimeService,
    backend_name: str,
    job_id: str,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    entry = state_entry(
        state,
        backend_name,
    )

    print()
    print("=" * 72)
    print(f"MONITORING {backend_name}")
    print("=" * 72)
    print(f"Job ID: {job_id}")

    try:
        job = retrieve_job(
            service,
            job_id,
        )

    except Exception as exc:
        entry["last_error"] = str(exc)
        entry["updated_at_utc"] = utc_now()
        save_state(state)

        print(
            f"{backend_name}: could not retrieve job right now."
        )
        print(f"  {exc}")
        print("  No new job will be submitted.")

        return None

    metadata = entry.get("metadata")

    if not metadata:
        raise RuntimeError(
            f"{backend_name}: job exists but circuit metadata is missing "
            f"from {STATE_FILE}. Refusing to guess circuit ordering."
        )

    while True:
        status = get_status_with_retry(job)

        if status is None:
            entry["status"] = "NETWORK_UNAVAILABLE"
            entry["last_error"] = (
                "IBM status endpoint unreachable after retries."
            )
            entry["updated_at_utc"] = utc_now()
            save_state(state)

            print()
            print(
                f"{backend_name}: monitoring paused because IBM "
                "could not be reached."
            )
            print(
                "The job remains on IBM Quantum. "
                "Rerun with --recover or --submit later."
            )

            return None

        entry["status"] = status
        entry["last_error"] = None
        entry["updated_at_utc"] = utc_now()
        save_state(state)

        print(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"{backend_name}: {status}"
        )

        if status == "DONE":
            break

        if status in {"CANCELLED", "ERROR"}:
            error_message = get_error_message_safely(job)

            entry["last_error"] = error_message
            entry["updated_at_utc"] = utc_now()
            save_state(state)

            raise RuntimeError(
                f"{backend_name}: IBM job ended in {status}. "
                f"Job ID={job_id}. "
                f"Error={error_message}"
            )

        time.sleep(POLL_SECONDS)

    print(
        f"{backend_name}: job DONE. Retrieving result..."
    )

    result = None
    last_error: Exception | None = None

    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            result = job.result(
                timeout=60,
                poll_interval=1.0,
            )
            break

        except Exception as exc:
            last_error = exc

            print(
                f"  Result retrieval failed "
                f"(attempt {attempt}/{MAX_NETWORK_RETRIES}): "
                f"{exc}"
            )

            if attempt < MAX_NETWORK_RETRIES:
                time.sleep(NETWORK_RETRY_SECONDS)

    if result is None:
        entry["status"] = "DONE_RESULT_UNREACHABLE"
        entry["last_error"] = str(last_error)
        entry["updated_at_utc"] = utc_now()
        save_state(state)

        print(
            f"{backend_name}: job is DONE on IBM, but result retrieval "
            "failed from this machine."
        )
        print(
            "The job remains safe. Rerun with --recover to retrieve it."
        )

        return None

    backend_result = process_result(
        backend_name,
        job_id,
        result,
        metadata,
    )

    qpu_usage = get_usage_safely(job)

    result_path = save_backend_result(
        backend_result,
        qpu_usage,
    )

    entry["status"] = "DONE"
    entry["result_file"] = str(result_path)
    entry["qpu_usage_seconds"] = qpu_usage
    entry["last_error"] = None
    entry["updated_at_utc"] = utc_now()

    save_state(state)

    print()
    print(
        f"{backend_name}: RESULT SAVED"
    )
    print(f"  File: {result_path}")

    if qpu_usage is not None:
        print(
            f"  QPU usage: {qpu_usage:.6f} seconds"
        )

    for row in backend_result["circuits"]:
        print()
        print(
            f"  {row['group']:15s} "
            f"{row['physical_qubits']} "
            f"| fidelity={row['ghz_fidelity']:.6f}"
        )
        print(
            f"    Counts: {row['counts']}"
        )

    return backend_result


# ============================================================
# DRY RUN
# ============================================================

def dry_run(
    service: QiskitRuntimeService,
    candidates: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print("DRY RUN — NO HARDWARE SUBMISSION")
    print("=" * 72)

    for backend_name in BACKENDS:
        backend = service.backend(backend_name)

        metadata, _ = prepare_backend_experiment(
            backend,
            candidates,
        )

        depths = [
            int(item["transpiled_depth"])
            for item in metadata
        ]

        print()
        print(
            f"{backend_name}: prepared {len(metadata)} circuits"
        )
        print(f"  depths: {depths}")

    print()
    print("DRY RUN COMPLETE.")
    print("No circuits were submitted.")
    print("QPU usage: 0 seconds")


# ============================================================
# STATUS ONLY
# ============================================================

def print_status(
    service: QiskitRuntimeService,
    state: dict[str, Any],
) -> None:
    print()
    print("=" * 72)
    print("CURRENT EXPERIMENT JOB STATUS")
    print("=" * 72)

    for backend_name in BACKENDS:
        job_id = existing_job_id(
            state,
            backend_name,
        )

        entry = state_entry(
            state,
            backend_name,
        )

        if not job_id:
            print()
            print(
                f"{backend_name}: NOT SUBMITTED"
            )
            continue

        try:
            job = retrieve_job(
                service,
                job_id,
            )

            status = get_status_with_retry(job)

            print()
            print(
                f"{backend_name}: {status}"
            )
            print(
                f"  Job ID: {job_id}"
            )

            usage = get_usage_safely(job)

            if usage is not None:
                print(
                    f"  QPU usage: {usage:.6f} seconds"
                )

        except Exception as exc:
            print()
            print(
                f"{backend_name}: STATUS UNAVAILABLE"
            )
            print(f"  Job ID: {job_id}")
            print(f"  Error: {exc}")

        entry["updated_at_utc"] = utc_now()

    save_state(state)


# ============================================================
# FINAL COMBINED OUTPUT
# ============================================================

def build_final_output(
    all_backend_results: list[dict[str, Any]],
) -> dict[str, Any]:
    flattened_results: list[dict[str, Any]] = []

    total_qpu_seconds = 0.0
    known_usage = True

    backend_job_summary = []

    for backend_result in all_backend_results:
        flattened_results.extend(
            backend_result["circuits"]
        )

        backend_name = backend_result["backend"]

        state_path = (
            RESULTS_DIR
            / f"sweet_spot_level2_{backend_name}_result.json"
        )

        qpu_usage = None

        if state_path.exists():
            try:
                with state_path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    saved = json.load(handle)

                qpu_usage = saved.get(
                    "qpu_usage_seconds"
                )

            except Exception:
                qpu_usage = None

        if qpu_usage is None:
            known_usage = False
        else:
            total_qpu_seconds += float(qpu_usage)

        backend_job_summary.append(
            {
                "backend": backend_name,
                "job_id": backend_result["job_id"],
                "result_file": str(state_path),
                "qpu_usage_seconds": qpu_usage,
            }
        )

    return {
        "experiment": "sweet_spot_vs_random",
        "level": 2,
        "model": MODEL_NAME,
        "prediction_date": PREDICTION_DATE,
        "shots": SHOTS,
        "random_controls_per_backend": RANDOM_CONTROLS_TO_RUN,
        "backends": BACKENDS,
        "hardware_jobs": len(all_backend_results),
        "total_hardware_circuits": len(flattened_results),
        "total_qpu_seconds": (
            total_qpu_seconds if known_usage else None
        ),
        "results": flattened_results,
        "backend_job_summary": backend_job_summary,
        "completed_at_utc": utc_now(),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("LEVEL 2 — FINAL SWEET-SPOT HARDWARE EXPERIMENT")
    print("=" * 72)
    print()
    print(f"Sweet-Spot model : {MODEL_NAME}")
    print(f"Prediction date  : {PREDICTION_DATE}")
    print("Backends         : FEZ / Kingston / Marrakesh")
    print(f"Shots            : {SHOTS}")
    print(
        f"Random controls  : "
        f"{RANDOM_CONTROLS_TO_RUN}/backend"
    )
    print(
        f"Total circuits   : "
        f"{len(BACKENDS) * EXPECTED_CIRCUITS_PER_BACKEND}"
    )
    print(
        f"Hardware jobs    : {len(BACKENDS)} maximum"
    )

    if args.status:
        print()
        print("Mode             : STATUS ONLY")
    elif args.resubmit_kingston:
        print()
        print("Mode             : FRESH KINGSTON + SUBMIT/RECOVER")
        print(
            "Kingston         : ONE NEW JOB WILL BE SUBMITTED"
        )
        print(
            "Old Kingston job : PRESERVED / NOT CANCELLED"
        )
    elif args.recover:
        print()
        print("Mode             : RECOVER ONLY")
        print("New submissions  : NONE")
    elif args.submit:
        print()
        print("Mode             : SUBMIT + RECOVER")
        print(
            "Safety           : existing job IDs will NEVER be resubmitted"
        )
    else:
        print()
        print("Mode             : DRY RUN")
        print("No hardware jobs will be submitted.")

    print()

    candidates = load_candidates()
    validate_manifest(candidates)

    service = connect()

    state = load_state()
    save_state(state)

    # Make sure every already-submitted job has deterministic circuit
    # metadata before any result retrieval is attempted.
    for backend_name in BACKENDS:
        job_id = existing_job_id(
            state,
            backend_name,
        )

        if job_id and not state_entry(
            state,
            backend_name,
        ).get("metadata"):
            backend = service.backend(backend_name)

            ensure_metadata_for_existing_job(
                backend,
                candidates,
                state,
            )

    if args.status:
        print_status(
            service,
            state,
        )
        return

    if (
        not args.submit
        and not args.recover
        and not args.resubmit_kingston
    ):
        dry_run(
            service,
            candidates,
        )
        return

    # --------------------------------------------------------
    # OPTIONAL EXPLICIT FRESH KINGSTON SUBMISSION.
    #
    # This intentionally creates ONE additional Kingston job.
    # The previous Kingston job remains on IBM and may still run.
    # --------------------------------------------------------

    if args.resubmit_kingston:
        backend = service.backend("ibm_kingston")

        submit_fresh_backend(
            service,
            backend,
            candidates,
            state,
        )

    # --------------------------------------------------------
    # SUBMIT ALL MISSING BACKENDS FIRST.
    #
    # This is the major execution improvement:
    # Kingston can remain queued while Marrakesh is submitted.
    # We do NOT block Marrakesh behind Kingston.result().
    # --------------------------------------------------------

    if args.submit:
        print()
        print("=" * 72)
        print("SUBMISSION PHASE")
        print("=" * 72)

        for backend_name in BACKENDS:
            if (
                args.resubmit_kingston
                and backend_name == "ibm_kingston"
            ):
                # Already freshly submitted above.
                continue

            backend = service.backend(backend_name)

            submit_backend_if_needed(
                service,
                backend,
                candidates,
                state,
            )

    # --------------------------------------------------------
    # RECOVERY / MONITORING PHASE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RECOVERY / MONITORING PHASE")
    print("=" * 72)

    for backend_name in BACKENDS:
        job_id = existing_job_id(
            state,
            backend_name,
        )

        if not job_id:
            print()
            print(
                f"{backend_name}: no job ID. Skipping."
            )
            continue

        wait_for_job(
            service,
            backend_name,
            job_id,
            state,
        )

    # --------------------------------------------------------
    # Build final combined output only when all three per-backend
    # result files exist.
    # --------------------------------------------------------

    all_done = True
    all_backend_results: list[dict[str, Any]] = []

    for backend_name in BACKENDS:
        entry = state_entry(
            state,
            backend_name,
        )

        result_file = entry.get(
            "result_file"
        )

        if (
            entry.get("status") == "DONE"
            and result_file
            and Path(result_file).exists()
        ):
            with Path(result_file).open(
                "r",
                encoding="utf-8",
            ) as handle:
                saved = json.load(handle)

            all_backend_results.append(
                {
                    "backend": saved["backend"],
                    "job_id": saved["job_id"],
                    "circuits": saved["results"],
                }
            )

        else:
            all_done = False

    if not all_done:
        print()
        print("=" * 72)
        print("EXPERIMENT NOT YET COMPLETE")
        print("=" * 72)
        print()
        print("No duplicate jobs were created.")
        print(f"Checkpoint: {STATE_FILE}")
        print(
            "Run the same command again later to recover."
        )
        return

    final_output = build_final_output(
        all_backend_results
    )

    atomic_write_json(
        OUTPUT_FILE,
        final_output,
    )

    print()
    print("=" * 72)
    print("LEVEL 2 HARDWARE EXPERIMENT COMPLETE")
    print("=" * 72)
    print()
    print(
        f"Hardware jobs     : "
        f"{final_output['hardware_jobs']}"
    )
    print(
        f"Hardware circuits : "
        f"{final_output['total_hardware_circuits']}"
    )

    if final_output["total_qpu_seconds"] is not None:
        print(
            f"Total QPU seconds : "
            f"{final_output['total_qpu_seconds']:.6f}"
        )
    else:
        print(
            "Total QPU seconds : unavailable"
        )

    print(
        f"Saved             : "
        f"{OUTPUT_FILE}"
    )
    print(
        f"Job state         : "
        f"{STATE_FILE}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        print("No submitted IBM job was cancelled.")
        print(
            f"Recover later with: "
            f"python {Path(sys.argv[0]).name} --recover"
        )
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 72)
        print("EXPERIMENT STOPPED SAFELY")
        print("=" * 72)
        print()
        print(f"Error: {exc}")
        print()
        print("IMPORTANT: existing IBM jobs were not cancelled.")
        print(f"Job state file: {STATE_FILE}")
        print(
            f"Recover later with: "
            f"python {Path(sys.argv[0]).name} --recover"
        )
        sys.exit(1)
