import os, json
from pathlib import Path
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "level2"
OUT.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

token = os.getenv("IBM_QUANTUM_TOKEN")
if not token:
    raise SystemExit("IBM_QUANTUM_TOKEN is missing.")

service = QiskitRuntimeService(token=token, instance=os.getenv("IBM_INSTANCE") or None)
backend_name = os.getenv("IBM_BACKEND")

if backend_name:
    backend = service.backend(backend_name)
else:
    candidates = service.backends(simulator=False, operational=True)
    if not candidates:
        raise SystemExit("No operational IBM Quantum backend available.")
    backend = candidates[0]

# Deliberately conservative: one job, 256 shots, no loop, no session.
SHOTS = 256

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print(f"Backend: {backend.name}")
print(f"Submitting ONE job with {SHOTS} shots.")
job = Sampler(mode=backend).run([qc], shots=SHOTS)
print("Job ID:", job.job_id())
result = job.result()

record = {
    "level": 2,
    "backend": backend.name,
    "job_id": job.job_id(),
    "shots": SHOTS,
    "result": str(result)
}
try:
    record["metrics"] = job.metrics()
except Exception as exc:
    record["metrics_error"] = str(exc)

(OUT / "level2_result.json").write_text(
    json.dumps(record, indent=2, default=str), encoding="utf-8"
)
print("LEVEL 2 COMPLETE")
print(json.dumps(record, indent=2, default=str))
