import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()

token = os.getenv("IBM_QUANTUM_TOKEN") or os.getenv("QISKIT_IBM_TOKEN")

if not token:
    raise RuntimeError("IBM token not found in .env")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=token,
    instance="open-instance",
)

JOB_ID = "dafgfb5nj4cs73agbfo0"

job = service.job(JOB_ID)

print("Job:", job.job_id())
print("Backend:", job.backend().name)
print("Status:", job.status())
print("Usage:", job.usage())

if str(job.status()) != "DONE":
    raise RuntimeError(f"FEZ job is not completed: {job.status()}")

result = job.result()

records = []

for index, pub_result in enumerate(result):
    counts = pub_result.data.c.get_counts()

    fidelity = (
        counts.get("000", 0)
        + counts.get("111", 0)
    ) / 256

    records.append({
        "backend": "ibm_fez",
        "job_id": JOB_ID,
        "circuit_index": index,
        "counts": dict(counts),
        "shots": 256,
        "ghz_fidelity": fidelity,
    })

output_dir = Path("results/sweet_spot")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "fez_hardware_result.json"

with output_file.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print()
print("FEZ RESULT SAVED")
print(output_file)
print("Circuits:", len(records))
print("QPU usage:", job.usage(), "seconds")