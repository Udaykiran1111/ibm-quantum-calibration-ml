import json
from pathlib import Path
from circuits.bell_state import build_bell_circuit
from simulation.simulate import run_ideal, run_controlled_noise
from analysis.analyze_counts import summarize_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "level1"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    qc = build_bell_circuit()
    ideal = run_ideal(qc, shots=2048)
    noisy = run_controlled_noise(qc, shots=2048)
    report = {
        "level": 1,
        "experiment": "Bell state",
        "shots": 2048,
        "ideal": summarize_counts(ideal),
        "controlled_noisy": summarize_counts(noisy),
    }
    (OUT / "level1_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("LEVEL 1 COMPLETE")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
