from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN, instance=INSTANCE)
backend = service.backend("ibm_kingston")
status = backend.status()
print("operational:", status.operational)
print("pending_jobs:", status.pending_jobs)