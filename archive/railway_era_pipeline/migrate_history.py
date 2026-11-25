"""
migrate_history.py

One-time migration of historical IBM Quantum telemetry
from local CSV backups into the NEW Railway MySQL database.

Historical calibration:
    July 13, 2026 -> August 11, 2026

Historical rankings:
    Produced by Model A (qubit_model_v2.pkl)

IMPORTANT:
    This script does NOT train models.
    This script does NOT modify model files.
    This script does NOT generate new predictions.
    It only restores historical DB state.

After migration:
    NEW DB
        calibration_history
            Jul 13 -> Aug 11

        qubit_rankings
            model_a
            Jul 13 -> Aug 11

Then the GitHub collector can collect Aug 12 and generate
new Model A / Model B / Model C predictions.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CALIBRATION_CSV = (
    PROJECT_ROOT /
    "backup_calibration_history_20260811_1706.csv"
)

# CHANGE THIS ONLY if your actual ranking CSV has another name.
RANKINGS_CSV = (
    PROJECT_ROOT /
    "backup_qubit_rankings_20260811_1706.csv"
)

HISTORICAL_MODEL_NAME = "model_a"

EXPECTED_START_DATE = pd.Timestamp("2026-07-13").date()
EXPECTED_END_DATE = pd.Timestamp("2026-08-11").date()

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# HELPERS
# ============================================================

def fail(message: str):
    print()
    print("=" * 70)
    print("MIGRATION FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def get_connection():
    required = [
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
    ]

    missing = [
        key for key in required
        if not os.environ.get(key)
    ]

    if missing:
        fail(
            "Missing MySQL environment variables:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    try:
        return mysql.connector.connect(
            host=os.environ["MYSQL_HOST"],
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ["MYSQL_USER"],
            password=os.environ["MYSQL_PASSWORD"],
            database=os.environ["MYSQL_DATABASE"],
            connection_timeout=60,
            autocommit=False,
        )

    except Error as e:
        fail(f"Could not connect to MySQL:\n{e}")


def check_file(path: Path, description: str):
    if not path.exists():
        fail(
            f"{description} not found:\n"
            f"{path}"
        )


# ============================================================
# TABLE CREATION
# ============================================================

def create_tables(conn):

    cur = conn.cursor()

    print()
    print("Creating/verifying database tables...")

    # --------------------------------------------------------
    # Calibration history
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS calibration_history (
            id INT AUTO_INCREMENT PRIMARY KEY,

            backend VARCHAR(50) NOT NULL,
            qubit INT NOT NULL,

            snapshot_date DATE NOT NULL,
            backend_ts DATETIME NOT NULL,

            T1_us FLOAT,
            T2_us FLOAT,
            readout_error FLOAT,

            collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE KEY uq_backend_qubit_date
                (backend, qubit, snapshot_date)
        )
    """)

    # --------------------------------------------------------
    # Qubit rankings
    #
    # model_name is part of the unique identity.
    # This allows:
    #
    # model_a + qubit + date
    # model_b + qubit + date
    # model_c + qubit + date
    #
    # for the SAME qubit and date.
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS qubit_rankings (
            id INT AUTO_INCREMENT PRIMARY KEY,

            model_name VARCHAR(20) NOT NULL,

            backend VARCHAR(50) NOT NULL,
            qubit INT NOT NULL,

            snapshot_date DATE NOT NULL,

            viability_score FLOAT NOT NULL,
            viability_rank INT NOT NULL,
            label TINYINT NOT NULL,

            T1_us FLOAT,
            T2_us FLOAT,
            readout_error FLOAT,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE KEY uq_ranking_model_date
                (model_name, backend, qubit, snapshot_date)
        )
    """)

    conn.commit()
    cur.close()

    print("Database tables ready.")


# ============================================================
# LOAD + VALIDATE CALIBRATION CSV
# ============================================================

def load_calibration_csv():

    print()
    print("=" * 70)
    print("CALIBRATION HISTORY")
    print("=" * 70)

    check_file(
        CALIBRATION_CSV,
        "Calibration CSV"
    )

    print(f"File: {CALIBRATION_CSV}")

    df = pd.read_csv(CALIBRATION_CSV)

    print(f"Raw rows: {len(df):,}")

    required = [
        "backend",
        "qubit",
        "snapshot_date",
        "backend_ts",
        "T1_us",
        "T2_us",
        "readout_error",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        fail(
            "Calibration CSV is missing columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    # Keep only DB columns
    df = df[required].copy()

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    ).dt.date

    df["backend_ts"] = pd.to_datetime(
        df["backend_ts"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    if df["snapshot_date"].isna().any():
        fail("Calibration CSV contains invalid snapshot_date values.")

    if df["backend_ts"].isna().any():
        fail("Calibration CSV contains invalid backend_ts values.")

    if df["backend"].isna().any():
        fail("Calibration CSV contains missing backend values.")

    if df["qubit"].isna().any():
        fail("Calibration CSV contains missing qubit values.")

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    min_date = df["snapshot_date"].min()
    max_date = df["snapshot_date"].max()

    unique_dates = df["snapshot_date"].nunique()

    backends = sorted(
        df["backend"].dropna().unique().tolist()
    )

    print(f"Date range: {min_date} -> {max_date}")
    print(f"Unique dates: {unique_dates}")
    print(f"Backends: {backends}")

    # We expect exactly the historical experiment window.
    if min_date != EXPECTED_START_DATE:
        fail(
            f"Unexpected calibration start date.\n"
            f"Expected: {EXPECTED_START_DATE}\n"
            f"Found:    {min_date}"
        )

    if max_date != EXPECTED_END_DATE:
        fail(
            f"Unexpected calibration end date.\n"
            f"Expected: {EXPECTED_END_DATE}\n"
            f"Found:    {max_date}"
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "backend",
            "qubit",
            "snapshot_date",
        ]
    ).sum()

    print(f"Duplicate DB-key rows: {duplicates:,}")

    if duplicates:
        print(
            "WARNING: duplicate backend/qubit/date rows "
            "exist in the CSV."
        )
        print(
            "INSERT IGNORE will protect the database."
        )

    # --------------------------------------------------------
    # T1 sanity
    # --------------------------------------------------------

    t1_mean = df["T1_us"].mean()

    print(f"T1 mean: {t1_mean:.1f} us")

    print()
    print("Calibration validation PASSED.")

    return df


# ============================================================
# IMPORT CALIBRATION
# ============================================================

def import_calibration(conn, df):

    print()
    print("Importing calibration_history...")

    sql = """
        INSERT IGNORE INTO calibration_history
        (
            backend,
            qubit,
            snapshot_date,
            backend_ts,
            T1_us,
            T2_us,
            readout_error
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """

    rows = []

    for _, r in df.iterrows():

        rows.append((
            str(r["backend"]),
            int(r["qubit"]),
            r["snapshot_date"],
            r["backend_ts"].to_pydatetime(),
            None if pd.isna(r["T1_us"]) else float(r["T1_us"]),
            None if pd.isna(r["T2_us"]) else float(r["T2_us"]),
            None if pd.isna(r["readout_error"])
            else float(r["readout_error"]),
        ))

    cur = conn.cursor()

    batch_size = 1000
    inserted_total = 0

    for start in range(
        0,
        len(rows),
        batch_size
    ):

        batch = rows[
            start:start + batch_size
        ]

        cur.executemany(
            sql,
            batch
        )

        conn.commit()

        inserted_total += cur.rowcount

        processed = min(
            start + batch_size,
            len(rows)
        )

        print(
            f"  Progress: "
            f"{processed:,}/{len(rows):,}"
        )

    cur.close()

    print(
        f"Calibration import complete. "
        f"New rows: {inserted_total:,}"
    )


# ============================================================
# LOAD + VALIDATE RANKING CSV
# ============================================================

def load_ranking_csv():

    print()
    print("=" * 70)
    print("HISTORICAL MODEL A RANKINGS")
    print("=" * 70)

    check_file(
        RANKINGS_CSV,
        "Ranking CSV"
    )

    print(f"File: {RANKINGS_CSV}")

    df = pd.read_csv(RANKINGS_CSV)

    print(f"Raw rows: {len(df):,}")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Old ranking CSV did NOT necessarily contain model_name.
    #
    # We assign every historical row to Model A.
    # --------------------------------------------------------

    required = [
        "backend",
        "qubit",
        "snapshot_date",
        "viability_score",
        "viability_rank",
        "label",
        "T1_us",
        "T2_us",
        "readout_error",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        fail(
            "Ranking CSV is missing columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )

    df = df[required].copy()

    # --------------------------------------------------------
    # Assign historical model identity
    # --------------------------------------------------------

    df["model_name"] = HISTORICAL_MODEL_NAME

    # --------------------------------------------------------
    # Normalize dates
    # --------------------------------------------------------

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    ).dt.date

    if df["snapshot_date"].isna().any():
        fail(
            "Ranking CSV contains invalid snapshot_date values."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    min_date = df["snapshot_date"].min()
    max_date = df["snapshot_date"].max()

    unique_dates = df["snapshot_date"].nunique()

    backends = sorted(
        df["backend"].dropna().unique().tolist()
    )

    print(f"Assigned model_name: {HISTORICAL_MODEL_NAME}")
    print(f"Date range: {min_date} -> {max_date}")
    print(f"Unique dates: {unique_dates}")
    print(f"Backends: {backends}")

    if min_date < EXPECTED_START_DATE:
        fail(
            "Ranking CSV contains dates earlier than "
            "the historical experiment start."
        )

    if max_date > EXPECTED_END_DATE:
        fail(
            "Ranking CSV contains dates later than "
            "August 11, 2026."
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "model_name",
            "backend",
            "qubit",
            "snapshot_date",
        ]
    ).sum()

    print(
        f"Duplicate model/backend/qubit/date rows: "
        f"{duplicates:,}"
    )

    if duplicates:
        print(
            "WARNING: duplicate ranking rows detected."
        )
        print(
            "INSERT IGNORE will prevent duplicate DB records."
        )

    print()
    print("Ranking validation PASSED.")

    return df


# ============================================================
# IMPORT RANKINGS
# ============================================================

def import_rankings(conn, df):

    print()
    print("Importing historical Model A rankings...")

    sql = """
        INSERT IGNORE INTO qubit_rankings
        (
            model_name,
            backend,
            qubit,
            snapshot_date,
            viability_score,
            viability_rank,
            label,
            T1_us,
            T2_us,
            readout_error
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """

    rows = []

    for _, r in df.iterrows():

        rows.append((
            HISTORICAL_MODEL_NAME,
            str(r["backend"]),
            int(r["qubit"]),
            r["snapshot_date"],
            float(r["viability_score"]),
            int(r["viability_rank"]),
            int(r["label"]),
            None if pd.isna(r["T1_us"])
            else float(r["T1_us"]),
            None if pd.isna(r["T2_us"])
            else float(r["T2_us"]),
            None if pd.isna(r["readout_error"])
            else float(r["readout_error"]),
        ))

    cur = conn.cursor()

    batch_size = 1000
    inserted_total = 0

    for start in range(
        0,
        len(rows),
        batch_size
    ):

        batch = rows[
            start:start + batch_size
        ]

        cur.executemany(
            sql,
            batch
        )

        conn.commit()

        inserted_total += cur.rowcount

        processed = min(
            start + batch_size,
            len(rows)
        )

        print(
            f"  Progress: "
            f"{processed:,}/{len(rows):,}"
        )

    cur.close()

    print(
        f"Ranking import complete. "
        f"New rows: {inserted_total:,}"
    )


# ============================================================
# DATABASE VERIFICATION
# ============================================================

def verify_database(conn):

    print()
    print("=" * 70)
    print("FINAL DATABASE VERIFICATION")
    print("=" * 70)

    cur = conn.cursor()

    # --------------------------------------------------------
    # Calibration count
    # --------------------------------------------------------

    cur.execute("""
        SELECT COUNT(*)
        FROM calibration_history
    """)

    calibration_count = cur.fetchone()[0]

    # --------------------------------------------------------
    # Calibration date range
    # --------------------------------------------------------

    cur.execute("""
        SELECT
            MIN(snapshot_date),
            MAX(snapshot_date),
            COUNT(DISTINCT snapshot_date)
        FROM calibration_history
    """)

    cal_min, cal_max, cal_days = cur.fetchone()

    # --------------------------------------------------------
    # Calibration backends
    # --------------------------------------------------------

    cur.execute("""
        SELECT
            backend,
            COUNT(*)
        FROM calibration_history
        GROUP BY backend
        ORDER BY backend
    """)

    calibration_backends = cur.fetchall()

    # --------------------------------------------------------
    # Ranking count
    # --------------------------------------------------------

    cur.execute("""
        SELECT COUNT(*)
        FROM qubit_rankings
    """)

    ranking_count = cur.fetchone()[0]

    # --------------------------------------------------------
    # Ranking models
    # --------------------------------------------------------

    cur.execute("""
        SELECT
            model_name,
            COUNT(*),
            MIN(snapshot_date),
            MAX(snapshot_date),
            COUNT(DISTINCT snapshot_date)
        FROM qubit_rankings
        GROUP BY model_name
        ORDER BY model_name
    """)

    ranking_models = cur.fetchall()

    # --------------------------------------------------------
    # Ranking backends
    # --------------------------------------------------------

    cur.execute("""
        SELECT
            model_name,
            backend,
            COUNT(*)
        FROM qubit_rankings
        GROUP BY model_name, backend
        ORDER BY model_name, backend
    """)

    ranking_backends = cur.fetchall()

    cur.close()

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("CALIBRATION HISTORY")
    print("-------------------")
    print(f"Total rows       : {calibration_count:,}")
    print(f"Minimum date     : {cal_min}")
    print(f"Maximum date     : {cal_max}")
    print(f"Unique dates     : {cal_days}")

    print()
    print("Rows by backend:")

    for backend, count in calibration_backends:
        print(
            f"  {backend:20s} {count:,}"
        )

    print()
    print("QUbit rankings")
    print("--------------")
    print(f"Total rows       : {ranking_count:,}")

    print()
    print("Rankings by model:")

    for (
        model,
        count,
        min_date,
        max_date,
        days,
    ) in ranking_models:

        print(
            f"  {model:10s} "
            f"rows={count:,} "
            f"dates={days} "
            f"range={min_date} -> {max_date}"
        )

    print()
    print("Rankings by model/backend:")

    for model, backend, count in ranking_backends:
        print(
            f"  {model:10s} "
            f"{backend:20s} "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if calibration_count == 0:
        fail(
            "Calibration table is empty after migration."
        )

    if cal_min != EXPECTED_START_DATE:
        fail(
            f"Calibration minimum date is wrong: {cal_min}"
        )

    if cal_max != EXPECTED_END_DATE:
        fail(
            f"Calibration maximum date is wrong: {cal_max}"
        )

    if HISTORICAL_MODEL_NAME not in [
        row[0] for row in ranking_models
    ]:
        fail(
            "model_a rankings were not found."
        )

    print()
    print("=" * 70)
    print("MIGRATION VERIFIED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IBM QUANTUM — HISTORICAL DATABASE MIGRATION")
    print("=" * 70)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Calibration:  {CALIBRATION_CSV}")
    print(f"Rankings:     {RANKINGS_CSV}")

    print()
    print(
        "Historical ranking model: "
        f"{HISTORICAL_MODEL_NAME}"
    )

    print()
    print(
        "This script will restore historical data only."
    )

    print(
        "No model training or prediction will occur."
    )

    # --------------------------------------------------------
    # Check files FIRST
    # --------------------------------------------------------

    check_file(
        CALIBRATION_CSV,
        "Calibration CSV"
    )

    check_file(
        RANKINGS_CSV,
        "Ranking CSV"
    )

    # --------------------------------------------------------
    # Load and validate BEFORE connecting
    # --------------------------------------------------------

    calibration_df = load_calibration_csv()
    ranking_df = load_ranking_csv()

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    print()
    print("Connecting to NEW Railway MySQL...")

    conn = get_connection()

    print("MySQL connection successful.")

    try:

        # ----------------------------------------------------
        # Create schema
        # ----------------------------------------------------

        create_tables(conn)

        # ----------------------------------------------------
        # Import calibration
        # ----------------------------------------------------

        import_calibration(
            conn,
            calibration_df
        )

        # ----------------------------------------------------
        # Import Model A rankings
        # ----------------------------------------------------

        import_rankings(
            conn,
            ranking_df
        )

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        verify_database(conn)

    except Exception as e:

        conn.rollback()

        print()
        print("=" * 70)
        print("MIGRATION ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)

        raise

    finally:

        try:
            conn.close()
        except Exception:
            pass

    print()
    print("=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print()
    print("The NEW Railway DB now contains:")
    print("  - historical calibration data")
    print("  - historical Model A rankings")
    print()
    print("Next:")
    print("  1. Verify Railway counts")
    print("  2. Fix collector empty-history handling")
    print("  3. Run GitHub Actions")
    print("  4. Collect Aug 12")
    print("  5. Generate Model A/B/C predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()