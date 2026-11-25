import os
from pathlib import Path
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
token = os.getenv("IBM_QUANTUM_TOKEN")
if not token:
    raise SystemExit("IBM_QUANTUM_TOKEN is missing.")

service = QiskitRuntimeService(token=token, instance=os.getenv("IBM_INSTANCE") or None)
print("IBM Quantum connection: OK")
for b in service.backends(simulator=False, operational=True):
    print("-", b.name)
