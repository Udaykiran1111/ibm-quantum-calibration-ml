from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def run_ideal(circuit, shots=2048):
    sim = AerSimulator()
    return sim.run(circuit, shots=shots).result().get_counts()

def run_controlled_noise(circuit, shots=2048):
    noise = NoiseModel()
    one_q_error = depolarizing_error(0.01, 1)
    two_q_error = depolarizing_error(0.02, 2)
    readout = ReadoutError([[0.98, 0.02], [0.03, 0.97]])
    noise.add_all_qubit_quantum_error(one_q_error, ["h"])
    noise.add_all_qubit_quantum_error(two_q_error, ["cx"])
    noise.add_all_qubit_readout_error(readout)
    sim = AerSimulator(noise_model=noise)
    return sim.run(circuit, shots=shots).result().get_counts()
