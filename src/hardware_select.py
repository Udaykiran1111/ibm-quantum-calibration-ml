"""
hardware_select.py - choose the circuits for ONE hardware test day.

Pure pandas / numpy: it never talks to IBM. Run it, COMMIT its output, and only then submit.
(docs/PREREGISTRATION.md section 3 defines every rule used here.)

For target date D it reads
  predictions/daily/D_model_a|b|c.csv      frozen-model viability scores (Model A may be missing)
  data/daily/calibration/*.csv             T1, T2, readout error (D = today, earlier dates = history)
  data/daily/couplers/*.csv                two-qubit gate errors (cz only)

and for every backend writes hardware/confirmatory_7day/rankings/D_selection.json (+ .csv) with
a list of 3-qubit chains a-b-c (b is physically connected to both a and c):

  model_a / model_b / model_c   chain with the highest mean viability score of its 3 qubits
  baseline_yesterday_errors     lowest (sum of 3 readout errors + 2 cz errors) from the MOST RECENT
                                EARLIER session (information the model also had)
  baseline_today_errors         same score from today's calibration (information the model lacked)
  baseline_history_viable       highest mean share of viable past sessions of its 3 qubits
                                (ties -> lower yesterday error score)
  random_01 .. random_11        uniformly random eligible chains, seed = sha256(date|backend|random)
  model_b_repeat                the Model B chain run a second time (measures run-to-run noise)

Identical chains are executed once and carry several roles. Output is WRITE-ONCE.

Eligible chain = all 3 qubits are in today's calibration file and both couplers have a cz entry today.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import RE_THRESH, T1_THRESH, T2_THRESH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
MODELS = ["model_a", "model_b", "model_c"]
N_RANDOM = 11
SHOTS = 1024
SHORT = {"ibm_fez": "fez", "ibm_kingston": "kin", "ibm_marrakesh": "mar"}


# ------------------------------------------------------------------ helpers
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edge(a: int, b: int):
    return (a, b) if a < b else (b, a)


def read_cal(cal_dir: Path, date: str) -> pd.DataFrame:
    df = pd.read_csv(cal_dir / f"{date}.csv")
    return df.drop_duplicates(["backend", "qubit"], keep="last")


def read_edges(cou_dir: Path, date: str) -> dict:
    """{backend: {(a,b): cz_error}} (mean over the two directions if both are listed)."""
    df = pd.read_csv(cou_dir / f"{date}.csv")
    df = df[df["gate"] == "cz"]
    out = {}
    for b, g in df.groupby("backend"):
        acc = {}
        for q0, q1, e in zip(g["q0"], g["q1"], g["gate_error"]):
            acc.setdefault(edge(int(q0), int(q1)), []).append(float(e))
        out[b] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out


def chains_from_edges(edge_keys) -> list:
    """All 3-qubit paths (a, b, c), a < c, with b connected to both. Sorted, deterministic."""
    adj = {}
    for a, b in edge_keys:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out = []
    for b in sorted(adj):
        for a, c in combinations(sorted(adj[b]), 2):
            out.append((a, b, c))
    return sorted(out)


def error_score(ch, ro: dict, ed: dict):
    a, b, c = ch
    if not all(q in ro for q in ch):
        return None
    e1, e2 = ed.get(edge(a, b)), ed.get(edge(b, c))
    if e1 is None or e2 is None:
        return None
    return float(ro[a] + ro[b] + ro[c] + e1 + e2)


def pick_max(cands, score_of, tie=None):
    """Highest score; ties broken by `tie` (lower is better) then by chain order. Returns (chain, score)."""
    best = None
    for ch in cands:
        s = score_of(ch)
        if s is None:
            continue
        key = (-s, tie(ch) if tie else 0.0, ch)
        if best is None or key < best[0]:
            best = (key, ch, s)
    return None if best is None else (best[1], float(best[2]))


def pick_min(cands, score_of):
    best = None
    for ch in cands:
        s = score_of(ch)
        if s is None:
            continue
        key = (s, ch)
        if best is None or key < best[0]:
            best = (key, ch, s)
    return None if best is None else (best[1], float(best[2]))


# ------------------------------------------------------------------ main logic
def select_for_backend(backend, date, cal_today, edges_today, past, pred_maps, shots, n_random):
    ro_today = dict(zip(cal_today.qubit.astype(int), cal_today.readout_error.astype(float)))
    ed_today = edges_today
    cands = [ch for ch in chains_from_edges(ed_today.keys()) if all(q in ro_today for q in ch)]

    # yesterday = most recent earlier session
    y_date, y_cal, y_ed = past["last"]
    ro_y = dict(zip(y_cal.qubit.astype(int), y_cal.readout_error.astype(float)))
    # viable share over all earlier sessions
    share = past["share"].get(backend, {})

    entries = {}  # chain -> {"roles": [], "scores": {}}

    def add(ch, role, **scores):
        e = entries.setdefault(ch, {"roles": [], "scores": {}})
        e["roles"].append(role)
        e["scores"].update(scores)

    missing = []
    for m in MODELS:
        pm = pred_maps.get(m, {}).get(backend)
        if not pm:
            missing.append(m)
            continue
        r = pick_max(cands, lambda ch, pm=pm: float(np.mean([pm[q] for q in ch])) if all(q in pm for q in ch) else None)
        if r:
            add(r[0], m, **{f"{m}_mean_viability": r[1]})

    y_score = lambda ch: error_score(ch, ro_y, y_ed.get(backend, {}))  # noqa: E731
    t_score = lambda ch: error_score(ch, ro_today, ed_today)           # noqa: E731
    r = pick_min(cands, y_score)
    if r:
        add(r[0], "baseline_yesterday_errors", yesterday_error_score=r[1])
    r = pick_min(cands, t_score)
    if r:
        add(r[0], "baseline_today_errors", today_error_score=r[1])
    r = pick_max(cands, lambda ch: float(np.mean([share[q] for q in ch])) if all(q in share for q in ch) else None,
                 tie=lambda ch: y_score(ch) if y_score(ch) is not None else 9.0)
    if r:
        add(r[0], "baseline_history_viable", history_viable_share=r[1])

    seed = int(hashlib.sha256(f"{date}|{backend}|random".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cands), size=min(n_random, len(cands)), replace=False)
    for i, k in enumerate(sorted(pick.tolist()), 1):
        add(cands[k], f"random_{i:02d}")

    def viab(ch):
        """Mean predicted viability of this chain under EVERY available model (for the correlation analysis)."""
        out = {}
        for m in MODELS:
            pm = pred_maps.get(m, {}).get(backend)
            if pm and all(q in pm for q in ch):
                out[f"viab_{m}"] = float(np.mean([pm[q] for q in ch]))
        return out

    circuits = []
    for ch, e in entries.items():
        circuits.append({"qubits": list(ch), "roles": e["roles"], "scores": {**e["scores"], **viab(ch)},
                         "today_error_score": t_score(ch), "yesterday_error_score": y_score(ch)})
    # explicit repeat of the Model B chain (separate circuit entry, same qubits)
    b_chain = next((c["qubits"] for c in circuits if "model_b" in c["roles"]), None)
    if b_chain:
        circuits.append({"qubits": b_chain, "roles": ["model_b_repeat"], "scores": viab(tuple(b_chain)),
                         "today_error_score": t_score(tuple(b_chain)), "yesterday_error_score": y_score(tuple(b_chain))})
    for i, c in enumerate(circuits):
        c["circuit_id"] = f"{SHORT[backend]}_{i:02d}"
    return {"circuits": circuits, "n_eligible_chains": len(cands), "random_seed": seed,
            "models_missing": missing, "yesterday_date": y_date, "shots": shots}


def build(date, cal_dir, cou_dir, pred_dir, out_dir, shots=SHOTS, n_random=N_RANDOM):
    dates = sorted(p.stem for p in cal_dir.glob("*.csv"))
    if date not in dates:
        raise SystemExit(f"No calibration file for {date} in {cal_dir} - collect it first.")
    earlier = [d for d in dates if d < date]
    if len(earlier) < 2:
        raise SystemExit(f"Need >= 2 earlier sessions before {date}, found {len(earlier)}.")
    cal_today = read_cal(cal_dir, date)
    edges_today = read_edges(cou_dir, date)

    y = earlier[-1]
    y_cal = read_cal(cal_dir, y)
    y_ed = read_edges(cou_dir, y)
    allcal = pd.concat([read_cal(cal_dir, d) for d in earlier], ignore_index=True)
    allcal["y"] = ((allcal.T1_us > T1_THRESH) & (allcal.T2_us > T2_THRESH) & (allcal.readout_error < RE_THRESH)).astype(float)
    share = {b: g.groupby("qubit").y.mean().to_dict() for b, g in allcal.groupby("backend")}
    past = {"last": (y, y_cal, y_ed), "share": share}

    pred_maps, pred_files = {}, {}
    for m in MODELS:
        f = pred_dir / f"{date}_{m}.csv"
        if f.exists():
            p = pd.read_csv(f)
            pred_maps[m] = {b: dict(zip(g.qubit.astype(int), g.viability_score.astype(float))) for b, g in p.groupby("backend")}
            pred_files[m] = sha256_file(f)
    if not pred_maps:
        raise SystemExit(f"No ranking files for {date} in {pred_dir} - run rank_daily.py first.")

    result = {
        "target_date": date, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "history_dates": earlier, "shots": shots, "n_random": n_random,
        "script_sha256": sha256_file(Path(__file__)),
        "inputs_sha256": {"calibration_today": sha256_file(cal_dir / f"{date}.csv"),
                          "couplers_today": sha256_file(cou_dir / f"{date}.csv"), **{f"ranking_{k}": v for k, v in pred_files.items()}},
        "backends": {},
    }
    for b in BACKENDS:
        if b not in set(cal_today.backend):
            print(f"WARNING: {b} missing from today's calibration - skipped", file=sys.stderr)
            continue
        sub = cal_today[cal_today.backend == b]
        past_b = {"last": (y, y_cal[y_cal.backend == b], y_ed), "share": share}
        result["backends"][b] = select_for_backend(
            b, date, sub, edges_today.get(b, {}), past_b,
            {m: {b: pm.get(b)} for m, pm in pred_maps.items()}, shots, n_random)
        # NOTE: select_for_backend reads yesterday's errors via y_ed[backend]
    return result


def write_outputs(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jf = out_dir / f"{result['target_date']}_selection.json"
    cf = out_dir / f"{result['target_date']}_selection.csv"
    jf.write_text(json.dumps(result, indent=2))
    rows = []
    for b, info in result["backends"].items():
        for c in info["circuits"]:
            rows.append({"date": result["target_date"], "backend": b, "circuit_id": c["circuit_id"],
                         "qubits": "-".join(map(str, c["qubits"])), "roles": "|".join(c["roles"]),
                         "today_error_score": c["today_error_score"], "yesterday_error_score": c["yesterday_error_score"],
                         **c["scores"]})
    pd.DataFrame(rows).to_csv(cf, index=False)
    return jf, cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--cal-dir", default=str(ROOT / "data/daily/calibration"))
    ap.add_argument("--cou-dir", default=str(ROOT / "data/daily/couplers"))
    ap.add_argument("--pred-dir", default=str(ROOT / "predictions/daily"))
    ap.add_argument("--out-dir", default=str(ROOT / "hardware/confirmatory_7day/rankings"))
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    jf = out_dir / f"{a.date}_selection.json"
    if jf.exists():
        print(f"{jf} exists - write-once, nothing changed.")
        return 0
    res = build(a.date, Path(a.cal_dir), Path(a.cou_dir), Path(a.pred_dir), out_dir, a.shots, a.n_random)
    jf, cf = write_outputs(res, out_dir)
    print(f"Wrote {jf}")
    for b, info in res["backends"].items():
        print(f"  {b}: {len(info['circuits'])} circuits, {info['n_eligible_chains']} eligible chains, "
              f"missing models: {info['models_missing'] or 'none'}")
        for c in info["circuits"]:
            if not any(r.startswith("random_") for r in c["roles"]) or len(c["roles"]) > 1:
                print(f"     {c['circuit_id']} {c['qubits']}  {'|'.join(c['roles'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
