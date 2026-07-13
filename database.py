"""
database.py — MySQL connection and schema for IBM Quantum Live Layer
All DB operations go through this file. Collector and dashboard import from here.
"""

import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime


# ── Connection ────────────────────────────────────────────────────────────────
def get_connection():
    """
    Returns a MySQL connection using environment variables.
    Set these in your .env file or GitHub Secrets:
      MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    """
    return mysql.connector.connect(
        host     = os.environ.get("MYSQL_HOST"),
        port     = int(os.environ.get("MYSQL_PORT", 3306)),
        user     = os.environ.get("MYSQL_USER"),
        password = os.environ.get("MYSQL_PASSWORD"),
        database = os.environ.get("MYSQL_DATABASE"),
    )


# ── Schema creation ───────────────────────────────────────────────────────────
CREATE_CALIBRATION_TABLE = """
CREATE TABLE IF NOT EXISTS calibration_history (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    backend          VARCHAR(50)  NOT NULL,
    qubit            INT          NOT NULL,
    snapshot_date    DATE         NOT NULL,
    backend_ts       DATETIME     NOT NULL,
    T1_us            FLOAT,
    T2_us            FLOAT,
    readout_error    FLOAT,
    collected_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_backend_qubit_date (backend, qubit, snapshot_date)
);
"""

CREATE_RANKINGS_TABLE = """
CREATE TABLE IF NOT EXISTS qubit_rankings (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    backend          VARCHAR(50)  NOT NULL,
    qubit            INT          NOT NULL,
    snapshot_date    DATE         NOT NULL,
    viability_score  FLOAT        NOT NULL,
    viability_rank   INT          NOT NULL,
    label            TINYINT      NOT NULL,
    T1_us            FLOAT,
    T2_us            FLOAT,
    readout_error    FLOAT,
    created_at       DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_ranking_date (backend, qubit, snapshot_date)
);
"""


def setup_database():
    """Create tables if they don't exist. Run once on first startup."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(CREATE_CALIBRATION_TABLE)
    cur.execute(CREATE_RANKINGS_TABLE)
    conn.commit()
    cur.close()
    conn.close()
    print("Database tables ready.")


# ── Write operations ──────────────────────────────────────────────────────────
def insert_calibration_rows(rows: list[dict]):
    """
    Insert calibration snapshot rows into calibration_history.
    Each row: {backend, qubit, snapshot_date, backend_ts, T1_us, T2_us, readout_error}
    Duplicate (backend, qubit, snapshot_date) rows are ignored (INSERT IGNORE).
    """
    if not rows:
        return 0
    sql = """
        INSERT IGNORE INTO calibration_history
            (backend, qubit, snapshot_date, backend_ts, T1_us, T2_us, readout_error)
        VALUES
            (%(backend)s, %(qubit)s, %(snapshot_date)s, %(backend_ts)s,
             %(T1_us)s, %(T2_us)s, %(readout_error)s)
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.executemany(sql, rows)
    inserted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return inserted


def insert_ranking_rows(rows: list[dict]):
    """
    Insert qubit viability rankings.
    Each row: {backend, qubit, snapshot_date, viability_score, viability_rank,
               label, T1_us, T2_us, readout_error}
    """
    if not rows:
        return 0
    sql = """
        INSERT IGNORE INTO qubit_rankings
            (backend, qubit, snapshot_date, viability_score, viability_rank,
             label, T1_us, T2_us, readout_error)
        VALUES
            (%(backend)s, %(qubit)s, %(snapshot_date)s, %(viability_score)s,
             %(viability_rank)s, %(label)s, %(T1_us)s, %(T2_us)s, %(readout_error)s)
    """
    conn = get_connection()
    cur  = conn.cursor()
    cur.executemany(sql, rows)
    inserted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return inserted


# ── Read operations ───────────────────────────────────────────────────────────
def get_history_for_backend(backend: str, n_days: int = 10) -> pd.DataFrame:
    """
    Load the last n_days of calibration history for a backend.
    Used by collector.py to compute historical features for today's prediction.
    """
    sql = """
        SELECT backend, qubit, snapshot_date, T1_us, T2_us, readout_error
        FROM calibration_history
        WHERE backend = %s
        ORDER BY qubit, snapshot_date
    """
    conn = get_connection()
    df   = pd.read_sql(sql, conn, params=(backend,))
    conn.close()
    return df


def get_latest_rankings(backend: str = None, top_n: int = None) -> pd.DataFrame:
    """
    Load the most recent qubit rankings.
    Used by dashboard.py to display the live ranking table.
    """
    where = "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM qubit_rankings)"
    params = []
    if backend:
        where += " AND backend = %s"
        params.append(backend)

    limit  = f"LIMIT {top_n}" if top_n else ""
    sql    = f"""
        SELECT backend, qubit, viability_rank, viability_score,
               label, T1_us, T2_us, readout_error, snapshot_date
        FROM qubit_rankings
        {where}
        ORDER BY backend, viability_rank
        {limit}
    """
    conn = get_connection()
    df   = pd.read_sql(sql, conn, params=params if params else None)
    conn.close()
    return df


def get_ranking_history(backend: str, qubit: int) -> pd.DataFrame:
    """Load ranking history for a specific qubit over all collected days."""
    sql = """
        SELECT snapshot_date, viability_score, viability_rank, label,
               T1_us, T2_us, readout_error
        FROM qubit_rankings
        WHERE backend = %s AND qubit = %s
        ORDER BY snapshot_date
    """
    conn = get_connection()
    df   = pd.read_sql(sql, conn, params=(backend, qubit))
    conn.close()
    return df


def get_all_dates() -> list:
    """Return all snapshot dates collected so far."""
    sql  = "SELECT DISTINCT snapshot_date FROM qubit_rankings ORDER BY snapshot_date"
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(sql)
    dates = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return dates


def get_summary_stats() -> dict:
    """Return high-level summary for dashboard header."""
    conn = get_connection()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT COUNT(DISTINCT snapshot_date) as days FROM qubit_rankings")
    days = cur.fetchone()['days']
    cur.execute("SELECT COUNT(*) as total FROM qubit_rankings WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM qubit_rankings)")
    total = cur.fetchone()['total']
    cur.execute("SELECT snapshot_date FROM qubit_rankings ORDER BY snapshot_date DESC LIMIT 1")
    row = cur.fetchone()
    latest = row['snapshot_date'] if row else None
    cur.close()
    conn.close()
    return {'days_collected': days, 'qubits_latest': total, 'latest_date': latest}
