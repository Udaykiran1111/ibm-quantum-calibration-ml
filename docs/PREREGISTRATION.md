# Pre-registration: short hardware round, 27-29 Nov 2026

Written on **27 Nov 2026**, before any job of this round was submitted. The git commit time of this file
is the proof of order. Changes after the first submission go ONLY in section 9 (deviations), with a date,
never as edits above it.

Status of the evidence: the earlier hardware runs (pilots 0, 1, 2) stay **pilots**. This round is small
(at most 9 day-backend units), so it is a **pre-planned test with effect sizes and intervals, not proof**.
The main claims of the paper do not depend on it.

## 1. Question

Does a 3-qubit chain chosen by the frozen Model B give a higher GHZ fidelity than a typical random chain,
and how does it compare with chains chosen by simple rules (IBM-reported errors, past viable share)?

## 2. Frozen items (never retrained or edited during the round)

| Item | SHA-256 |
|---|---|
| `models/production/model_b_7day.pkl` (primary model) | `5f20f9f232649748de1ae4a67d1e59ed47c30042e5175deab5a5b03599724125` |
| `models/production/model_c_30day.pkl` | `9ece7588177857dcf473a8d59f6a10608c01013ad2061be5f24e24d58d3886c7` |
| `models/production/qubit_model_v2.pkl` (Model A, XGBoost) | `cca3ccc4c64d6e19af110a03add8f7b51787022a5e2a81bfe1ab5a0cffb2a0bf` |
| `src/features.py` (17 features, label thresholds) | `a53bef48cc9dac99bff41695dbcd2bbe287398ac2ff5aa144c8f65e954e13134` |

The deployed daily pipeline ranks **Model B and Model C only** (`src/rank_daily.py` in the repository does
not include Model A, and `requirements.txt` has no `xgboost`). So no Model A chain is chosen in this round and
Model A is reported as "not part of this round". The Model A file and its hash are listed only because it is
part of the repository; it is not replaced by another model and may not be added during the round.

## 3. Choice of circuits (code: `src/hardware_select.py`)

For each test date D and each backend, using ONLY data from before the run:

- Eligible chain: three qubits a-b-c where b is connected to both a and c by a CZ coupler listed today,
  and all three qubits appear in today's calibration file.
- `model_b`, `model_c` (and `model_a` only if a ranking file for it exists, which is not expected): the
  eligible chain with the highest mean predicted viability of its three qubits, from
  `predictions/daily/D_model_*.csv` (history = all sessions before D).
- `baseline_yesterday_errors`: lowest (sum of the 3 readout errors + the 2 CZ errors) in the most recent
  earlier session. Same information the model had.
- `baseline_today_errors`: same score from today's calibration. More information than the model had.
- `baseline_history_viable`: highest mean share of viable past sessions of the 3 qubits (ties: lower
  yesterday error score). This is the simple history rule from the machine-learning analysis.
- `random_01 .. random_11`: 11 distinct eligible chains drawn uniformly, seed from
  sha256("date|backend|random"). Eligible chains include poor ones: a user who knows nothing would pick them.
- `model_b_repeat`: the Model B chain a second time, to measure run-to-run noise.
- A chain chosen for several roles is run once and shared. At most 18 circuits per backend per day.

The selection file is committed to the repository BEFORE submission (the workflow does this).

## 4. Execution

3-qubit GHZ (H, CX, CX, measure), 1024 shots, one SamplerV2 job per backend per day in job mode,
`ibm_fez`, `ibm_kingston`, `ibm_marrakesh`. The circuit is laid out along the physical path (logical middle
qubit on the middle physical qubit), `optimization_level=1`, `seed_transpiler=42`. A backend/day is not
submitted if any circuit needs a SWAP or does not have exactly 2 CZ gates.
Outcome per circuit: fidelity = (count of 000 + count of 111) / shots.

Test dates (UTC dates of the snapshot): **2026-11-27, 2026-11-28, 2026-11-29**. No jobs after 29 Nov
(30 Nov is a buffer for collecting results and backups). The number of days is fixed and is not extended or
shortened depending on results.

## 5. Unit of analysis

One unit = one (date, backend) pair: up to 3 dates x 3 backends = 9 units. Units from the same date share
the same days of drift and are not fully independent; this is stated wherever p-values are given.

## 6. Hypotheses and tests

**H1 (primary).** For Model B, per unit: D_B = fidelity of the Model B chain - median fidelity of the 11 random
chains. H1: mean D_B > 0.
Test: exact one-sided sign-flip test on the unit values D_B (all 2^n sign assignments, n = number of units).
Report the mean and median of D_B, a 95% percentile bootstrap interval over units, and the same analysis
with the 3 dates as units (average over backends). H1 is called supported only if one-sided p < 0.05
and mean D_B > 0. Otherwise it is reported as not supported.

**H2 (secondary).** Model B chain fidelity minus `baseline_today_errors` chain fidelity, per unit. Reported
as mean difference with 95% interval. "Matches" means the interval's lower end is above -0.02 (two
points, about three shot-noise standard errors, chosen before seeing data).

## 7. Secondary analyses (reported with intervals, no p-value claims)

1. Model B chain minus each of: `baseline_yesterday_errors`, `baseline_history_viable`.
2. Model C vs the random median (same test as H1, labelled secondary).
3. Rank of the Model B chain among all distinct circuits of its unit.
4. Within-unit Spearman correlation between `viab_model_b` and measured fidelity over all circuits,
   averaged over units (the file records `viab_model_b` for every circuit).
5. Run-to-run noise: SD of (model_b - model_b_repeat) across units, compared with the binomial
   standard error sqrt(F(1-F)/1024).
6. Fidelity against the summed-error score (today and yesterday) over all circuits.
7. The median and the mean of the random chains are both shown, because one broken random chain moves the mean
   (as in pilot 2).

## 8. Missing data and exclusions

- A job that is accepted by IBM but fails or does not return is reported as a missing unit. It is NOT
  replaced by a new job once a job ID exists.
- If IBM refuses a submission before any job ID exists, the same date may be retried with the same
  committed selection file.
- No circuit, unit or day is removed after seeing results.
- What this round cannot show: anything about other circuit types, other chips, other weeks, or a
  significant advantage over IBM-error-based selection with only 9 units.

## 9. Deviations (append only; date and reason for every change)

_None yet._
