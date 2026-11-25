import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()

TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_INSTANCE")

print("=" * 70)
print("IBM QUANTUM — LEVEL 2 CONNECTION TEST")
print("=" * 70)

if not TOKEN:
    raise RuntimeError("IBM_QUANTUM_TOKEN is missing from .env")

if not INSTANCE:
    raise RuntimeError("IBM_INSTANCE is missing from .env")

print("API key: FOUND")
print("Instance: FOUND")
print()

print("Connecting to IBM Quantum Platform...")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=TOKEN,
    instance=INSTANCE,
)

print("Authentication: PASS")
print()
print("Available backends:")
print("-" * 70)

backends = service.backends()

for backend in backends:
    try:
        status = backend.status()

        print(
            f"{backend.name:25s} "
            f"qubits={backend.num_qubits:<4} "
            f"operational={status.operational} "
            f"pending={status.pending_jobs}"
        )

    except Exception as exc:
        print(f"{backend.name:25s} status unavailable: {exc}")

print()
print("=" * 70)
print("CONNECTION TEST COMPLETE")
print("No quantum circuit was submitted.")
print("No QPU runtime was intentionally consumed.")
print("=" * 70)