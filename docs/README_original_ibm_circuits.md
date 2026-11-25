# Quantum Circuit Experiment — Level 1 & Level 2

This is a SEPARATE project from the existing IBM qubit-viability system.

Level 1:
- Bell-state circuit
- local Aer simulation
- ideal and controlled noisy simulation
- result report

Level 2:
- same circuit on one IBM Quantum backend
- one small controlled job
- records job ID and usage metrics when available

Railway is READ ONLY here. The existing qubit-viability models are NOT required
for Levels 1/2. Optional model copies can be placed in models/ for a future
cross-project experiment.

## Setup

Python 3.11 recommended.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Copy `.env.example` to `.env` and fill credentials.

## Level 1

    python run_level1.py

## Railway read-only snapshot

    python railway_snapshot.py

## Level 2

First:

    python hardware/check_ibm.py

Then:

    python hardware/run_level2.py

The hardware runner intentionally submits ONE small job and does not loop or
run a continuous workload. It does not touch the existing project's database.
