"""Offline tests for the hardware circuit selection and the pure helpers of hardware_run.py."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hardware_select as hs  # noqa: E402
import hardware_run as hr     # noqa: E402

PY = sys.executable
# a small chip: qubits 0..7 on a line 0-1-2-3-4-5-6-7 plus a branch 3-8
EDGES = [(i, i + 1) for i in range(7)] + [(3, 8)]
B = "ibm_fez"


def make_world(tmp: Path, dates=("2026-11-25", "2026-11-26", "2026-11-27")):
    cal, cou, pred = tmp / "cal", tmp / "cou", tmp / "pred"
    for d in (cal, cou, pred):
        d.mkdir()
    rng = np.random.default_rng(1)
    for i, d in enumerate(dates):
        rows = []
        for q in range(9):
            if q == 7 and d == dates[-1]:
                continue  # qubit 7 vanishes on the target day
            rows.append({"backend": B, "qubit": q, "snapshot_date": d, "T1_us": 150 + 10 * q, "T2_us": 90,
                         "readout_error": 0.01 + 0.005 * q + 0.001 * i})
        pd.DataFrame(rows).to_csv(cal / f"{d}.csv", index=False)
        crow = []
        for a, b in EDGES:
            for x, y in ((a, b), (b, a)):
                crow.append({"backend": B, "q0": x, "q1": y, "gate": "cz", "gate_error": 0.002 + 0.001 * (a % 3),
                             "gate_length_ns": 68, "snapshot_date": d, "backend_ts": ""})
                crow.append({"backend": B, "q0": x, "q1": y, "gate": "rzz", "gate_error": 0.9,
                             "gate_length_ns": 68, "snapshot_date": d, "backend_ts": ""})
        pd.DataFrame(crow).to_csv(cou / f"{d}.csv", index=False)
    target = dates[-1]
    # model scores: qubits 1,2,3 are the best
    sc = {q: (0.95 if q in (1, 2, 3) else 0.4) for q in range(9)}
    pd.DataFrame([{"backend": B, "qubit": q, "n_sessions": 2, "viability_score": s} for q, s in sc.items()]
                 ).to_csv(pred / f"{target}_model_b.csv", index=False)
    return cal, cou, pred, target


def run_select(cal, cou, pred, out, target, extra=()):
    return subprocess.run([PY, str(ROOT / "src/hardware_select.py"), "--date", target, "--cal-dir", str(cal),
                           "--cou-dir", str(cou), "--pred-dir", str(pred), "--out-dir", str(out), *extra],
                          capture_output=True, text=True)


def test_chains_are_connected_paths():
    ch = hs.chains_from_edges([hs.edge(a, b) for a, b in EDGES])
    # the middle entry is the center qubit; 3 touches 2, 4 and 8, but 4 does not touch 8
    assert (0, 1, 2) in ch and (2, 3, 8) in ch and (4, 3, 8) in ch and (3, 4, 8) not in ch
    for a, b, c in ch:
        assert hs.edge(a, b) in EDGES and hs.edge(b, c) in EDGES and a < c


def test_selection_rules_and_write_once():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cal, cou, pred, target = make_world(t)
        out = t / "out"
        r = run_select(cal, cou, pred, out, target)
        assert r.returncode == 0, r.stderr
        sel = json.loads((out / f"{target}_selection.json").read_text())
        info = sel["backends"][B]
        by_role = {role: c for c in info["circuits"] for role in c["roles"]}
        # model B: highest mean score among connected chains -> (1,2,3)
        assert by_role["model_b"]["qubits"] == [1, 2, 3]
        assert "model_a" in info["models_missing"] and "model_c" in info["models_missing"]
        # baselines use the cz errors only (rzz = 0.9 must not matter) and readout errors rise with qubit id
        assert by_role["baseline_today_errors"]["qubits"][0] <= 1
        # qubit 7 is absent on the target day -> no chain may contain it
        for c in info["circuits"]:
            assert 7 not in c["qubits"]
        # repeat of model B is a separate circuit with the same qubits
        assert by_role["model_b_repeat"]["qubits"] == by_role["model_b"]["qubits"]
        assert by_role["model_b_repeat"]["circuit_id"] != by_role["model_b"]["circuit_id"]
        # 11 random roles, or fewer only if the chip has fewer eligible chains (this toy chip has 7)
        n_rand = sum(1 for c in info["circuits"] for r_ in c["roles"] if r_.startswith("random_"))
        assert n_rand == min(11, info["n_eligible_chains"]) and info["n_eligible_chains"] == 7
        assert sel["shots"] == 1024
        # write-once and deterministic
        before = (out / f"{target}_selection.json").read_bytes()
        r2 = run_select(cal, cou, pred, out, target)
        assert "write-once" in r2.stdout and (out / f"{target}_selection.json").read_bytes() == before
        out2 = t / "out2"
        run_select(cal, cou, pred, out2, target)
        s2 = json.loads((out2 / f"{target}_selection.json").read_text())
        assert [c["qubits"] for c in s2["backends"][B]["circuits"]] == [c["qubits"] for c in info["circuits"]]


def test_refuses_without_history_or_rankings():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cal, cou, pred, target = make_world(t)
        (pred / f"{target}_model_b.csv").unlink()
        r = run_select(cal, cou, pred, t / "o", target)
        assert r.returncode != 0 and "No ranking files" in (r.stderr + r.stdout)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cal, cou, pred, target = make_world(t, dates=("2026-11-26", "2026-11-27"))
        r = run_select(cal, cou, pred, t / "o", target)
        assert r.returncode != 0 and "earlier sessions" in (r.stderr + r.stdout)


def test_run_helpers():
    assert hr.ghz_fidelity({"000": 500, "111": 468, "010": 56}) == (500 + 468) / 1024
    assert hr.ghz_fidelity({"000": 10}) == 1.0
    try:
        hr.ghz_fidelity({})
        assert False
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x" / "s.json"
        hr.atomic_write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}


if __name__ == "__main__":
    for fn in (test_chains_are_connected_paths, test_selection_rules_and_write_once,
               test_refuses_without_history_or_rankings, test_run_helpers):
        fn()
        print("OK ", fn.__name__)
