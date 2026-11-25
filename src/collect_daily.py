"""
collect_daily.py - save today's IBM calibration data as plain CSV files.

Reads backend.properties() (no QPU time is used). Writes, WRITE-ONCE:
  data/daily/calibration/YYYY-MM-DD.csv   per-qubit T1, T2, readout error
  data/daily/couplers/YYYY-MM-DD.csv      two-qubit gate errors (cz and rzz; the GHZ baseline uses cz)
  data/daily/meta/YYYY-MM-DD.json         provenance + SHA-256 of both files
Existing files are never overwritten; re-running on the same day is harmless.

Env: IBM_QUANTUM_TOKEN (required), IBM_INSTANCE (CRN, optional),
     QT_BACKENDS (comma list, default: ibm_fez,ibm_kingston,ibm_marrakesh)
`--fake` uses local simulated backends for testing and writes ONLY to ./_fake_out
"""
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
FAKES = {"ibm_fez": "FakeFez", "ibm_kingston": "FakeKingston", "ibm_marrakesh": "FakeMarrakesh"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_backend(name, service, fake):
    if fake:
        from qiskit_ibm_runtime import fake_provider
        return getattr(fake_provider, FAKES[name])()
    return service.backend(name)


def extract(backend, name, snapshot_date, collected_at):
    """Same extraction as the original collector, plus coupler errors."""
    props = backend.properties()
    if props is None:
        raise RuntimeError("no calibration properties returned")
    backend_ts = props.last_update_date
    q_rows, c_rows = [], []
    for qubit, qprops in enumerate(props.qubits):
        v = {p.name: p.value for p in qprops}
        if v.get("T1") is None or v.get("T2") is None or v.get("readout_error") is None:
            continue
        q_rows.append({
            "backend": name, "qubit": int(qubit),
            "snapshot_date": snapshot_date, "backend_ts": str(backend_ts),
            "T1_us": float(v["T1"]), "T2_us": float(v["T2"]),
            "readout_error": float(v["readout_error"]),
            "prob_meas0_prep1": v.get("prob_meas0_prep1"),
            "prob_meas1_prep0": v.get("prob_meas1_prep0"),
            "collected_at": collected_at,
        })
    for gate in props.gates:
        if len(gate.qubits) != 2:
            continue
        p = {x.name: x.value for x in gate.parameters}
        if p.get("gate_error") is None:
            continue
        c_rows.append({
            "backend": name, "q0": int(gate.qubits[0]), "q1": int(gate.qubits[1]),
            "gate": gate.gate, "gate_error": float(p["gate_error"]),
            "gate_length_ns": p.get("gate_length"),
            "snapshot_date": snapshot_date, "backend_ts": str(backend_ts),
        })
    return q_rows, c_rows, str(backend_ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--date", default=None, help="UTC date YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    now = datetime.now(timezone.utc)
    day = a.date or now.strftime("%Y-%m-%d")
    root = Path("_fake_out") if a.fake else Path("data/daily")
    cal_f, cou_f, meta_f = (root / "calibration" / f"{day}.csv",
                            root / "couplers" / f"{day}.csv",
                            root / "meta" / f"{day}.json")
    if cal_f.exists():
        print(f"{cal_f} already exists - write-once, nothing to do.")
        return 0
    for p in (cal_f, cou_f, meta_f):
        p.parent.mkdir(parents=True, exist_ok=True)

    names = os.environ.get("QT_BACKENDS", ",".join(DEFAULT_BACKENDS)).split(",")
    service = None
    if not a.fake:
        from qiskit_ibm_runtime import QiskitRuntimeService
        kw = {"channel": "ibm_quantum_platform", "token": os.environ["IBM_QUANTUM_TOKEN"]}
        if os.environ.get("IBM_INSTANCE"):
            kw["instance"] = os.environ["IBM_INSTANCE"]
        service = QiskitRuntimeService(**kw)

    collected_at = now.strftime("%Y-%m-%d %H:%M:%S")
    all_q, all_c, status = [], [], {}
    for name in names:
        try:
            b = get_backend(name.strip(), service, a.fake)
            q, c, ts = extract(b, name.strip(), day, collected_at)
            if not q:
                raise RuntimeError("0 usable qubit rows")
            all_q += q; all_c += c
            status[name] = {"ok": True, "qubit_rows": len(q), "coupler_rows": len(c), "backend_ts": ts}
            print(f"  {name}: {len(q)} qubits, {len(c)} couplers (calibrated {ts})")
        except Exception as exc:  # keep going; record the failure
            status[name] = {"ok": False, "error": str(exc)[:300]}
            print(f"  {name}: FAILED - {exc}", file=sys.stderr)

    if not all_q:
        print("No data collected - nothing written.", file=sys.stderr)
        return 1
    qdf = pd.DataFrame(all_q)
    if qdf["T1_us"].mean() > 10000:
        print("T1 unit sanity check failed (mean > 10,000 us). Nothing written.", file=sys.stderr)
        return 1
    qdf.to_csv(cal_f, index=False)
    pd.DataFrame(all_c).to_csv(cou_f, index=False)
    try:
        import qiskit_ibm_runtime as r
        rv = r.__version__
    except Exception:
        rv = None
    meta = {"snapshot_date": day, "collected_at_utc": collected_at, "fake": a.fake,
            "qiskit_ibm_runtime": rv, "backends": status,
            "files": {"calibration": sha256(cal_f), "couplers": sha256(cou_f)}}
    meta_f.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {cal_f} ({len(qdf)} rows)")
    return 0 if all(s["ok"] for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
