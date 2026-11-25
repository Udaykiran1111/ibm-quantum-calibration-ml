import os
import json
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


load_dotenv()

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_INSTANCE")

BACKEND_NAME = "ibm_fez"
SHOTS = 256

MODEL_QUBITS = [99, 114, 115]

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "level2_pilot.json"


def build_ghz():
    qc = QuantumCircuit(3, 3)

    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.measure(range(3), range(3))

    return qc


def connect():

    if not TOKEN:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN missing from .env"
        )

    if not INSTANCE:
        raise RuntimeError(
            "IBM_INSTANCE missing from .env"
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE,
    )

    return service


def extract_counts(pub_result):

    """
    Extract counts from SamplerV2 result.

    The exact classical-register name can vary, so inspect
    the first available data field.
    """

    data = pub_result.data

    # Our circuit creates a classical register named "c".
    if hasattr(data, "c"):
        return data.c.get_counts()

    # Fallback: find the first object that exposes get_counts().
    for name in dir(data):

        if name.startswith("_"):
            continue

        try:
            obj = getattr(data, name)

            if hasattr(obj, "get_counts"):
                return obj.get_counts()

        except Exception:
            continue

    raise RuntimeError(
        "Could not extract measurement counts from SamplerV2 result."
    )


def calculate_ghz_fidelity(counts):

    total = sum(counts.values())

    if total == 0:
        raise RuntimeError(
            "Sampler returned zero measurement shots."
        )

    correct = (
        counts.get("000", 0)
        +
        counts.get("111", 0)
    )

    return correct / total


def run_experiment(
    backend,
    physical_qubits,
    label,
):

    print()
    print("-" * 70)
    print(f"EXPERIMENT: {label}")
    print("-" * 70)

    print(
        f"Physical qubits : {physical_qubits}"
    )

    print(
        f"Shots           : {SHOTS}"
    )

    circuit = build_ghz()

    transpiled = transpile(
        circuit,
        backend=backend,
        initial_layout=physical_qubits,
        optimization_level=1,
    )

    print(
        f"Transpiled depth: {transpiled.depth()}"
    )

    print()
    print(
        "Creating SamplerV2..."
    )

    sampler = SamplerV2(
        mode=backend
    )

    print(
        "Submitting ONE hardware job..."
    )

    job = sampler.run(
        [transpiled],
        shots=SHOTS,
    )

    job_id = job.job_id()

    print()
    print(
        f"JOB ID: {job_id}"
    )

    print(
        "Waiting for IBM hardware result..."
    )

    result = job.result()

    pub_result = result[0]

    counts = extract_counts(pub_result)

    fidelity = calculate_ghz_fidelity(
        counts
    )

    print()
    print("RESULT")
    print("-" * 70)

    print(
        f"Counts   : {counts}"
    )

    print(
        f"Fidelity : {fidelity:.4f}"
    )

    usage = None

    try:

        metrics = job.metrics()

        usage_data = metrics.get(
            "usage",
            {}
        )

        usage = usage_data.get(
            "quantum_seconds"
        )

    except Exception as exc:

        print(
            f"Could not retrieve QPU usage: {exc}"
        )

    if usage is not None:

        print(
            f"QPU usage: {usage:.6f} seconds"
        )

    return {
        "label": label,
        "backend": backend.name,
        "physical_qubits": physical_qubits,
        "shots": SHOTS,
        "job_id": job_id,
        "counts": counts,
        "ghz_fidelity": fidelity,
        "transpiled_depth": transpiled.depth(),
        "qpu_usage_seconds": usage,
    }


def main():

    print("=" * 70)
    print("QUANTUM CIRCUIT BENCH")
    print("LEVEL 2 — FINAL HARDWARE PILOT")
    print("=" * 70)

    print()
    print("REAL HARDWARE TEST")
    print()
    print(f"Backend : {BACKEND_NAME}")
    print(f"Shots   : {SHOTS}")
    print(f"Jobs    : 1")
    print(f"Qubits  : {MODEL_QUBITS}")
    print()

    service = connect()

    print(
        "IBM authentication: PASS"
    )

    backend = service.backend(
        BACKEND_NAME
    )

    status = backend.status()

    print(
        f"Backend operational: "
        f"{status.operational}"
    )

    print(
        f"Pending jobs: "
        f"{status.pending_jobs}"
    )

    if not status.operational:

        raise RuntimeError(
            f"{BACKEND_NAME} is not operational."
        )

    model_result = run_experiment(
        backend,
        MODEL_QUBITS,
        "RANDOM_CONTROL",
    )

    output = {
        "experiment":
            "Level 2 hardware pilot",

        "backend":
            BACKEND_NAME,

        "shots":
            SHOTS,

        "model":
            "model_a",

        "physical_qubits":
            MODEL_QUBITS,

        "result":
            model_result,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("LEVEL 2 PILOT COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        "ONE hardware job was submitted."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()