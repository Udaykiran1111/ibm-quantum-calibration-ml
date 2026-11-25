from circuits.bell_state import build_bell_circuit
from simulation.simulate import run_ideal

def test_circuit():
    qc = build_bell_circuit()
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2

def test_ideal_distribution():
    counts = run_ideal(build_bell_circuit(), shots=1024)
    forbidden = counts.get("01", 0) + counts.get("10", 0)
    assert forbidden / 1024 < 0.05
