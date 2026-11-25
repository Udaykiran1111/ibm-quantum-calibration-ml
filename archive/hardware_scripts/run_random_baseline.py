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

# Random #1 was already executed:
# [99, 114, 115] -> 0.9688
#
# Run the remaining nine.
RANDOM_CHAINS = [
    [136, 143, 144],
    [77, 85, 86],
    [118, 128, 129],
    [21, 22, 36],
    [119, 133, 134],
    [46, 47, 48],
    [8, 9, 10],
    [47, 48, 49],
    [97, 106, 107],
]

OUTPUT = Path("results/random_baseline_fez.json")
OUTPUT.parent.mkdir(exist_ok=True)


def build_ghz():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


def get_counts(pub_result):
    data = pub_result.data

    if hasattr(data, "c"):
        return data.c.get_counts()

    for name in dir(data):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(data, name)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
        except Exception:
            pass

    raise RuntimeError("Could not extract counts.")


def fidelity(counts):
    total = sum(counts.values())
    return (
        counts.get("000", 0) +
        counts.get("111", 0)
    ) / total


def main():

    print("=" * 70)
    print("LEVEL 2 — RANDOM BASELINE")
    print("=" * 70)
    print(f"Backend : {BACKEND_NAME}")
    print(f"Shots   : {SHOTS}")
    print(f"Jobs    : {len(RANDOM_CHAINS)}")
    print()

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
        instance=INSTANCE,
    )

    backend = service.backend(BACKEND_NAME)

    if not backend.status().operational:
        raise RuntimeError("Backend is not operational.")

    sampler = SamplerV2(mode=backend)

    results = []

    for i, qubits in enumerate(RANDOM_CHAINS, start=2):

        print("-" * 70)
        print(f"RANDOM TRIAL {i:02d}")
        print(f"Qubits: {qubits}")

        circuit = build_ghz()

        transpiled = transpile(
            circuit,
            backend=backend,
            initial_layout=qubits,
            optimization_level=1,
        )

        print(f"Depth : {transpiled.depth()}")
        print("Submitting...")

        job = sampler.run(
            [transpiled],
            shots=SHOTS,
        )

        print(f"Job ID: {job.job_id()}")

        result = job.result()
        counts = get_counts(result[0])
        f = fidelity(counts)

        print(f"Counts   : {counts}")
        print(f"Fidelity : {f:.4f}")

        results.append({
            "trial": i,
            "backend": BACKEND_NAME,
            "qubits": qubits,
            "shots": SHOTS,
            "job_id": job.job_id(),
            "transpiled_depth": transpiled.depth(),
            "counts": counts,
            "fidelity": f,
        })

    # Include the already completed Trial 01.
    all_results = [
        {
            "trial": 1,
            "backend": BACKEND_NAME,
            "qubits": [99, 114, 115],
            "shots": 256,
            "job_id": "da5ivq6aa69c739ku3v0",
            "fidelity": 0.9688,
        }
    ] + results

    fidelities = [
        r["fidelity"]
        for r in all_results
    ]

    mean = sum(fidelities) / len(fidelities)

    sorted_values = sorted(fidelities)
    n = len(sorted_values)

    if n % 2:
        median = sorted_values[n // 2]
    else:
        median = (
            sorted_values[n // 2 - 1]
            + sorted_values[n // 2]
        ) / 2

    variance = sum(
        (x - mean) ** 2
        for x in fidelities
    ) / (n - 1)

    std = variance ** 0.5

    summary = {
        "backend": BACKEND_NAME,
        "shots": SHOTS,
        "random_trials": len(all_results),
        "results": all_results,
        "statistics": {
            "mean_fidelity": mean,
            "median_fidelity": median,
            "std_fidelity": std,
            "min_fidelity": min(fidelities),
            "max_fidelity": max(fidelities),
        },
        "model_a_fidelity": 0.9570,
        "model_a_qubits": [21, 22, 23],
        "model_a_job_id": "da5itue1vhnc73fm4t90",
    }

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("RANDOM BASELINE COMPLETE")
    print("=" * 70)

    print(f"Random mean   : {mean:.4f}")
    print(f"Random median : {median:.4f}")
    print(f"Random std    : {std:.4f}")
    print(f"Random min    : {min(fidelities):.4f}")
    print(f"Random max    : {max(fidelities):.4f}")

    print()
    print("MODEL A")
    print(f"Fidelity      : 0.9570")

    print()
    print("MODEL A vs RANDOM MEAN")
    print(
        f"Difference    : "
        f"{0.9570 - mean:+.4f}"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()