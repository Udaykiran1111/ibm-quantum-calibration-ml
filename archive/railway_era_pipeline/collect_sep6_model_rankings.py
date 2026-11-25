"""
collect_sep6_model_rankings.py

Read-only export of Model A/B/C Sep-6 rankings from the existing
Railway MySQL database used by Qubit Telemetry.

Run from the ibm_quantum project root:
    python collect_sep6_model_rankings.py

It does NOT insert/update/delete anything in MySQL.
"""

from pathlib import Path
import sys
import pandas as pd

# Project root = directory containing database.py
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import get_connection

TARGET_DATE = "2026-09-06"
MODELS = ("model_a", "model_b", "model_c")
BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")

OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_sweet_spot" / "hardware_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SQL = """
SELECT
    model_name,
    backend,
    qubit,
    viability_rank,
    viability_score,
    label,
    T1_us,
    T2_us,
    readout_error,
    snapshot_date
FROM qubit_rankings
WHERE snapshot_date = %s
  AND model_name IN (%s, %s, %s)
ORDER BY model_name, backend, viability_rank
"""


def main():
    print("=" * 72)
    print("COLLECTING SEP-6 MODEL A/B/C RANKINGS FROM RAILWAY MYSQL")
    print("=" * 72)
    print(f"Target date : {TARGET_DATE}")
    print(f"Models      : {', '.join(MODELS)}")
    print()

    conn = get_connection()
    try:
        df = pd.read_sql(
            SQL,
            conn,
            params=(TARGET_DATE, *MODELS),
        )
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            f"No Model A/B/C ranking rows found for {TARGET_DATE}. "
            "Check the Railway database and snapshot date."
        )

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date.astype(str)

    print(f"Rows found  : {len(df)}")
    print(f"Models found: {sorted(df['model_name'].unique().tolist())}")
    print(f"Backends    : {sorted(df['backend'].unique().tolist())}")
    print()

    # Basic completeness check.
    for model in MODELS:
        m = df[df["model_name"] == model]
        print(
            f"{model}: {len(m)} rows | "
            f"backends={m['backend'].nunique()} | "
            f"top ranks={m.groupby('backend')['viability_rank'].min().to_dict()}"
        )

    # Save complete Sep-6 A/B/C ranking table.
    all_path = OUTPUT_DIR / "sep6_model_abc_all_rankings.csv"
    df.to_csv(all_path, index=False)

    # Save top-10 and top-1 per backend for each model.
    top10 = (
        df[df["viability_rank"] <= 10]
        .sort_values(["model_name", "backend", "viability_rank"])
        .copy()
    )
    top10_path = OUTPUT_DIR / "sep6_model_abc_top10_per_backend.csv"
    top10.to_csv(top10_path, index=False)

    top1 = (
        df[df["viability_rank"] == 1]
        .sort_values(["model_name", "backend"])
        .copy()
    )
    top1_path = OUTPUT_DIR / "sep6_model_abc_top1_per_backend.csv"
    top1.to_csv(top1_path, index=False)

    # Also save one file per model for easy inspection.
    for model in MODELS:
        model_df = df[df["model_name"] == model].sort_values(
            ["backend", "viability_rank"]
        )
        model_path = OUTPUT_DIR / f"sep6_{model}_rankings.csv"
        model_df.to_csv(model_path, index=False)

    print()
    print("Saved:")
    print(f"  {all_path}")
    print(f"  {top10_path}")
    print(f"  {top1_path}")
    for model in MODELS:
        print(f"  {OUTPUT_DIR / f'sep6_{model}_rankings.csv'}")

    print()
    print("TOP-1 SEP-6 MODEL A/B/C SELECTIONS")
    print("-" * 72)
    print(
        top1[
            [
                "model_name",
                "backend",
                "qubit",
                "viability_rank",
                "viability_score",
                "T1_us",
                "T2_us",
                "readout_error",
                "snapshot_date",
            ]
        ].to_string(index=False)
    )

    print()
    print("READ-ONLY COLLECTION COMPLETE.")


if __name__ == "__main__":
    main()
