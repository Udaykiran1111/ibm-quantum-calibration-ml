import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.getenv("IBM_QUANTUM_TOKEN"),
    instance=os.getenv("IBM_INSTANCE"),
)

BACKENDS = [
    "ibm_fez",
    "ibm_marrakesh",
    "ibm_kingston",
]

print("=" * 70)
print("IBM QUANTUM — LEVEL 2 TOPOLOGY INSPECTION")
print("=" * 70)

for name in BACKENDS:
    print(f"\n{'=' * 70}")
    print(f"BACKEND: {name}")
    print(f"{'=' * 70}")

    backend = service.backend(name)

    print(f"Qubits: {backend.num_qubits}")

    config = backend.configuration()
    coupling_map = config.coupling_map

    print(f"Coupling edges: {len(coupling_map)}")

    # Build undirected adjacency
    adjacency = {i: set() for i in range(backend.num_qubits)}

    for a, b in coupling_map:
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Find connected 3-qubit chains:
    # a -- b -- c
    chains = set()

    for b in range(backend.num_qubits):
        neighbors = sorted(adjacency[b])

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                c = neighbors[j]

                chains.add(tuple(sorted((a, b, c))))

    chains = sorted(chains)

    print(f"Connected 3-qubit chains: {len(chains)}")

    print("\nFirst 20 valid chains:")
    for chain in chains[:20]:
        print(f"  {chain}")

print("\n" + "=" * 70)
print("TOPOLOGY INSPECTION COMPLETE")
print("No quantum circuit was submitted.")
print("No QPU runtime was intentionally consumed.")
print("=" * 70)