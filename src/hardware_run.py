"""
hardware_run.py - submit / collect the confirmatory GHZ round on IBM hardware.

  python src/hardware_run.py dry-run --date D   build + transpile every circuit against the REAL backends,
                                                 check them, submit NOTHING (uses no QPU time)
  python src/hardware_run.py submit  --date D   one SamplerV2 job per backend (job mode), job IDs saved at once
  python src/hardware_run.py collect --date D   fetch finished jobs, write counts + fidelities
  python src/hardware_run.py status  --date D   print job states

Circuits come ONLY from hardware/confirmatory_7day/rankings/D_selection.json, which must be committed
BEFORE `submit` (see hardware_select.py and docs/PREREGISTRATION.md).

Safety rules (learned from the earlier rounds)
  * Job mode only: SamplerV2(mode=backend). No Session, no Batch (not available on the Open Plan).
  * A backend that already has a job ID for date D is NEVER submitted again.
  * The job ID is written to disk immediately after IBM returns it.
  * Every circuit must transpile to exactly 2 CZ gates and no SWAP (so depth does not differ between
    chains); otherwise that backend is not submitted and the problem is printed.
  * Result files are write-once.

Env: IBM_QUANTUM_TOKEN, IBM_INSTANCE (GitHub secrets). Never print them.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL_DIR = ROOT / "hardware/confirmatory_7day/rankings"
RAW_DIR = ROOT / "hardware/confirmatory_7day/raw"
SEED_TRANSPILER = 42


# ------------------------------------------------------------------ pure helpers (unit-tested)
def ghz_fidelity(counts: dict) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        raise RuntimeError("No shots returned.")
    return (int(counts.get("000", 0)) + int(counts.get("111", 0))) / total


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def load_selection(date: str, sel_dir: Path = SEL_DIR) -> dict:
    f = sel_dir / f"{date}_selection.json"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run hardware_select.py and commit it first.")
    return json.loads(f.read_text())


def state_path(date: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{date}_jobs.json"


def load_state(date: str, raw_dir: Path = RAW_DIR) -> dict:
    p = state_path(date, raw_dir)
    return json.loads(p.read_text()) if p.exists() else {"date": date, "backends": {}}


# ------------------------------------------------------------------ IBM / Qiskit parts
def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, instance = os.environ.get("IBM_QUANTUM_TOKEN"), os.environ.get("IBM_INSTANCE")
    if not token:
        raise SystemExit("IBM_QUANTUM_TOKEN is not set.")
    kw = {"channel": "ibm_quantum_platform", "token": token}
    if instance:
        kw["instance"] = instance
    svc = QiskitRuntimeService(**kw)
    print("IBM authentication: OK")
    return svc


def build_and_check(backend, circuits_meta):
    """Transpile each chain onto its exact physical qubits and verify there are no swaps."""
    from qiskit import QuantumCircuit, transpile
    base = QuantumCircuit(3, 3)
    base.h(0)
    base.cx(0, 1)
    base.cx(1, 2)
    base.measure(range(3), range(3))
    compiled, problems = [], []
    for c in circuits_meta:
        q = list(c["qubits"])
        t = transpile(base, backend=backend, initial_layout=q, optimization_level=1, seed_transpiler=SEED_TRANSPILER)
        ops = t.count_ops()
        n_cz, n_swap = int(ops.get("cz", 0)), int(ops.get("swap", 0))
        c["transpiled_depth"] = int(t.depth())
        c["n_cz"] = n_cz
        if n_cz != 2 or n_swap != 0:
            problems.append(f"{c['circuit_id']} {q}: cz={n_cz} swap={n_swap} ops={dict(ops)}")
        compiled.append(t)
    return compiled, problems


def cmd_dry_run(date):
    sel = load_selection(date)
    svc = connect()
    bad = 0
    for name, info in sel["backends"].items():
        backend = svc.backend(name)
        compiled, problems = build_and_check(backend, info["circuits"])
        depths = sorted({c["transpiled_depth"] for c in info["circuits"]})
        print(f"{name}: {len(compiled)} circuits transpiled, depths {depths}, problems: {len(problems)}")
        for p in problems:
            print("   PROBLEM:", p)
        bad += len(problems)
    print("DRY RUN OK - nothing submitted." if not bad else f"DRY RUN FOUND {bad} PROBLEM(S) - do not submit.")
    return 1 if bad else 0


def cmd_submit(date):
    from qiskit_ibm_runtime import SamplerV2
    sel = load_selection(date)
    state = load_state(date)
    svc = connect()
    shots = int(sel["shots"])
    failures = 0
    for name, info in sel["backends"].items():
        if state["backends"].get(name, {}).get("job_id"):
            print(f"{name}: job {state['backends'][name]['job_id']} already recorded - NOT submitting again.")
            continue
        backend = svc.backend(name)
        compiled, problems = build_and_check(backend, info["circuits"])
        if problems:
            print(f"{name}: NOT submitted, circuit problems:")
            for p in problems:
                print("   ", p)
            failures += 1
            continue
        try:
            job = SamplerV2(mode=backend).run(compiled, shots=shots)
        except Exception as exc:  # no job ID exists, nothing to duplicate
            print(f"{name}: submission failed before a job ID was returned: {exc}", file=sys.stderr)
            failures += 1
            continue
        state["backends"][name] = {"job_id": str(job.job_id()), "submitted_utc": utc_now(),
                                   "n_circuits": len(compiled), "shots": shots, "status_at_submit": "SUBMITTED"}
        atomic_write_json(state_path(date), state)  # checkpoint immediately
        print(f"{name}: submitted job {job.job_id()} ({len(compiled)} circuits x {shots} shots)")
    return 1 if failures else 0


def _usage(job):
    for call in (lambda: job.usage(partial=False), lambda: job.usage()):  # first form is the one that worked before
        try:
            u = call()
            if u is not None:
                return float(u)
        except Exception:
            pass
    try:
        return float(job.metrics().get("usage", {}).get("quantum_seconds"))
    except Exception:
        return None


def _counts(pub_result):
    data = pub_result.data
    if hasattr(data, "c"):
        return {str(k): int(v) for k, v in data.c.get_counts().items()}
    for nm in dir(data):
        if nm.startswith("_"):
            continue
        obj = getattr(data, nm, None)
        if hasattr(obj, "get_counts"):
            return {str(k): int(v) for k, v in obj.get_counts().items()}
    raise RuntimeError("Could not read counts from the SamplerV2 result.")


def cmd_collect(date):
    import pandas as pd
    sel = load_selection(date)
    state = load_state(date)
    svc = connect()
    out_csv = RAW_DIR / f"{date}_results.csv"
    done = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv).backend.unique())
    rows, pending = [], 0
    for name, info in sel["backends"].items():
        st = state["backends"].get(name)
        if not st or not st.get("job_id"):
            print(f"{name}: no job recorded.")
            continue
        if name in done:
            print(f"{name}: already collected - skipping (write-once).")
            continue
        job = svc.job(st["job_id"])
        status = str(job.status())
        print(f"{name}: job {st['job_id']} status {status}")
        if status.upper() not in ("DONE", "COMPLETED"):
            pending += 1
            if status.upper() in ("ERROR", "CANCELLED", "FAILED"):
                try:
                    print("   error:", job.error_message())
                except Exception:
                    pass
            continue
        res = job.result()
        usage = _usage(job)
        for c, pub in zip(info["circuits"], res):
            counts = _counts(pub)
            rows.append({"date": date, "backend": name, "circuit_id": c["circuit_id"],
                         "qubits": "-".join(map(str, c["qubits"])), "roles": "|".join(c["roles"]),
                         "shots": sum(counts.values()), "n000": counts.get("000", 0), "n111": counts.get("111", 0),
                         "fidelity": ghz_fidelity(counts), "counts_json": json.dumps(counts, sort_keys=True),
                         "job_id": st["job_id"], "job_usage_seconds": usage, "collected_utc": utc_now()})
        st["status_at_collect"] = status
        st["usage_seconds"] = usage
    if rows:
        new = pd.DataFrame(rows)
        if out_csv.exists():
            new = pd.concat([pd.read_csv(out_csv), new], ignore_index=True)
        new.to_csv(out_csv, index=False)
        atomic_write_json(state_path(date), state)
        print(f"Wrote {out_csv} ({len(new)} rows)")
    if pending:
        print(f"{pending} backend(s) not finished yet - run collect again later.")
    return 0


def cmd_status(date):
    state = load_state(date)
    svc = connect()
    for name, st in state["backends"].items():
        job = svc.job(st["job_id"])
        print(f"{name}: {st['job_id']} -> {job.status()}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dry-run", "submit", "collect", "status"])
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    a = ap.parse_args()
    return {"dry-run": cmd_dry_run, "submit": cmd_submit, "collect": cmd_collect, "status": cmd_status}[a.mode](a.date)


if __name__ == "__main__":
    sys.exit(main())
