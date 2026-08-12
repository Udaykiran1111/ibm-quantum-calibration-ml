"""
database.py — QubitTelemetry MySQL layer
Supports:
    Model A = historical XGBoost
    Model B = 7-day Random Forest
    Model C = 30-day Random Forest

Design guarantees:
    * one calibration row per backend/qubit/date
    * one ranking row per model/backend/qubit/date
    * historical feature queries can explicitly exclude the current date
    * dashboard comparisons can use a common date across all three models
"""

import os
import time
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_NAMES = ("model_a", "model_b", "model_c")


def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE"),
        connection_timeout=60,
        autocommit=False,
    )


CREATE_CALIBRATION = """
CREATE TABLE IF NOT EXISTS calibration_history (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    backend         VARCHAR(50) NOT NULL,
    qubit           INT NOT NULL,
    snapshot_date   DATE NOT NULL,
    backend_ts      DATETIME NOT NULL,
    T1_us           FLOAT,
    T2_us           FLOAT,
    readout_error   FLOAT,
    collected_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_backend_qubit_date (backend, qubit, snapshot_date)
);
"""

CREATE_RANKINGS = """
CREATE TABLE IF NOT EXISTS qubit_rankings (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    model_name      VARCHAR(20) NOT NULL DEFAULT 'model_a',
    backend         VARCHAR(50) NOT NULL,
    qubit           INT NOT NULL,
    snapshot_date   DATE NOT NULL,
    viability_score FLOAT NOT NULL,
    viability_rank  INT NOT NULL,
    label           TINYINT NOT NULL,
    T1_us           FLOAT,
    T2_us           FLOAT,
    readout_error   FLOAT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_model_backend_qubit_date
        (model_name, backend, qubit, snapshot_date)
);
"""


def _index_rows(cur, table_name):
    cur.execute(
        """
        SELECT INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """,
        (table_name,),
    )
    rows = cur.fetchall()
    indexes = {}
    for name, non_unique, seq, column in rows:
        indexes.setdefault(name, {"non_unique": non_unique, "columns": []})
        indexes[name]["columns"].append(column)
    return indexes


def _ensure_rankings_schema(cur):
    # Ensure model_name exists for an older deployment.
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'qubit_rankings'
          AND COLUMN_NAME = 'model_name'
        """
    )
    has_model_name = cur.fetchone()[0] > 0

    if not has_model_name:
        cur.execute(
            """
            ALTER TABLE qubit_rankings
            ADD COLUMN model_name VARCHAR(20) NOT NULL DEFAULT 'model_a'
            AFTER id
            """
        )

    indexes = _index_rows(cur, "qubit_rankings")

    # Old deployments may have used uq_backend_qubit_date.
    old_index = indexes.get("uq_backend_qubit_date")
    if old_index and old_index["columns"] == ["backend", "qubit", "snapshot_date"]:
        cur.execute(
            "ALTER TABLE qubit_rankings DROP INDEX uq_backend_qubit_date"
        )

    indexes = _index_rows(cur, "qubit_rankings")
    desired = indexes.get("uq_model_backend_qubit_date")

    if not desired or desired["columns"] != [
        "model_name", "backend", "qubit", "snapshot_date"
    ]:
        if desired:
            cur.execute(
                "ALTER TABLE qubit_rankings DROP INDEX uq_model_backend_qubit_date"
            )
        cur.execute(
            """
            ALTER TABLE qubit_rankings
            ADD UNIQUE KEY uq_model_backend_qubit_date
                (model_name, backend, qubit, snapshot_date)
            """
        )


def setup_database():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(CREATE_CALIBRATION)
        cur.execute(CREATE_RANKINGS)
        _ensure_rankings_schema(cur)
        conn.commit()
        print("[DB] Schema ready.")
    finally:
        cur.close()
        conn.close()


def insert_calibration_rows(rows):
    if not rows:
        return 0

    sql = """
        INSERT IGNORE INTO calibration_history
            (backend, qubit, snapshot_date, backend_ts,
             T1_us, T2_us, readout_error)
        VALUES
            (%(backend)s, %(qubit)s, %(snapshot_date)s, %(backend_ts)s,
             %(T1_us)s, %(T2_us)s, %(readout_error)s)
    """

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.executemany(sql, rows)
        inserted = cur.rowcount
        conn.commit()
        return inserted
    finally:
        cur.close()
        conn.close()


def insert_ranking_rows(rows, model_name="model_a"):
    if not rows:
        return 0

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name: {model_name}")

    payload = []
    for row in rows:
        item = dict(row)
        item["model_name"] = model_name
        payload.append(item)

    sql = """
        INSERT IGNORE INTO qubit_rankings
            (model_name, backend, qubit, snapshot_date,
             viability_score, viability_rank, label,
             T1_us, T2_us, readout_error)
        VALUES
            (%(model_name)s, %(backend)s, %(qubit)s, %(snapshot_date)s,
             %(viability_score)s, %(viability_rank)s, %(label)s,
             %(T1_us)s, %(T2_us)s, %(readout_error)s)
    """

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.executemany(sql, payload)
        inserted = cur.rowcount
        conn.commit()
        return inserted
    finally:
        cur.close()
        conn.close()


def get_history_for_backend(backend, before_date=None):
    """
    Return calibration history strictly before before_date when supplied.
    This is the safety boundary for live feature generation.
    """
    sql = """
        SELECT backend, qubit, snapshot_date,
               T1_us, T2_us, readout_error
        FROM calibration_history
        WHERE backend = %s
    """
    params = [backend]

    if before_date is not None:
        sql += " AND snapshot_date < %s"
        params.append(before_date)

    sql += " ORDER BY qubit, snapshot_date"

    for attempt in range(3):
        conn = None
        try:
            conn = get_connection()
            df = pd.read_sql(sql, conn, params=params)
            return df
        except mysql.connector.Error as exc:
            if attempt == 2:
                raise
            print(f"[DB] History read failed; retry {attempt + 1}/3: {exc}")
            time.sleep(2)
        finally:
            if conn is not None:
                conn.close()

    return pd.DataFrame()


def get_latest_calibration_date():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT MAX(snapshot_date) FROM calibration_history")
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


def get_latest_rankings(backend=None, model_name="model_a", top_n=None):
    if model_name == "all":
        where = ["1=1"]
        params = []
        date_sub = "(SELECT MAX(snapshot_date) FROM qubit_rankings)"
    else:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model_name: {model_name}")
        where = ["model_name = %s"]
        params = [model_name]
        date_sub = """
            (SELECT MAX(snapshot_date)
             FROM qubit_rankings
             WHERE model_name = %s)
        """
        params.append(model_name)

    if backend:
        where.append("backend = %s")
        params.append(backend)

    sql = f"""
        SELECT model_name, backend, qubit, viability_rank,
               viability_score, label, T1_us, T2_us,
               readout_error, snapshot_date
        FROM qubit_rankings
        WHERE snapshot_date = {date_sub}
          AND {" AND ".join(where)}
        ORDER BY model_name, backend, viability_rank
    """

    if top_n is not None:
        sql += f" LIMIT {int(top_n)}"

    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def get_latest_common_model_date():
    """
    Latest date for which all three models have at least one ranking row.
    """
    conn = get_connection()
    sql = """
        SELECT MAX(snapshot_date)
        FROM qubit_rankings
        WHERE model_name IN ('model_a','model_b','model_c')
        GROUP BY snapshot_date
        HAVING COUNT(DISTINCT model_name) = 3
        ORDER BY MAX(snapshot_date) DESC
        LIMIT 1
    """
    try:
        df = pd.read_sql(sql, conn)
        if df.empty:
            return None
        return df.iloc[0, 0]
    finally:
        conn.close()


def get_all_models_latest(backend=None, common_date=True):
    target_date = get_latest_common_model_date() if common_date else None

    if target_date is None and common_date:
        return pd.DataFrame()

    where = ["model_name IN ('model_a','model_b','model_c')"]
    params = []

    if target_date is not None:
        where.append("snapshot_date = %s")
        params.append(target_date)

    if backend:
        where.append("backend = %s")
        params.append(backend)

    sql = f"""
        SELECT model_name, backend, qubit, viability_rank,
               viability_score, label, T1_us, T2_us,
               readout_error, snapshot_date
        FROM qubit_rankings
        WHERE {" AND ".join(where)}
        ORDER BY backend, qubit, model_name
    """

    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def get_model_comparison_latest(backend=None):
    df = get_all_models_latest(backend=backend, common_date=True)
    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.pivot_table(
            index=["backend", "qubit"],
            columns="model_name",
            values="viability_score",
            aggfunc="mean",
        )
        .reset_index()
    )

    ref = df[df["model_name"] == "model_a"][
        ["backend", "qubit", "T1_us", "T2_us", "readout_error", "label"]
    ].copy()

    pivot = pivot.merge(ref, on=["backend", "qubit"], how="left")

    for col in MODEL_NAMES:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    pivot["score_std"] = pivot[list(MODEL_NAMES)].std(axis=1, skipna=True)
    pivot["score_range"] = (
        pivot[list(MODEL_NAMES)].max(axis=1)
        - pivot[list(MODEL_NAMES)].min(axis=1)
    )
    pivot["agreement"] = pd.cut(
        pivot["score_range"],
        bins=[-float("inf"), 0.05, 0.15, float("inf")],
        labels=["HIGH", "MODERATE", "LOW"],
    )

    return pivot


def get_ranking_history(backend, qubit, model_name="model_a"):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name: {model_name}")

    sql = """
        SELECT snapshot_date, model_name, viability_score,
               viability_rank, label, T1_us, T2_us, readout_error
        FROM qubit_rankings
        WHERE backend = %s AND qubit = %s AND model_name = %s
        ORDER BY snapshot_date
    """
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=(backend, int(qubit), model_name))
    finally:
        conn.close()


def get_all_models_history(backend, qubit):
    sql = """
        SELECT snapshot_date, model_name, viability_score,
               viability_rank, label
        FROM qubit_rankings
        WHERE backend = %s AND qubit = %s
        ORDER BY snapshot_date, model_name
    """
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=(backend, int(qubit)))
    finally:
        conn.close()


def get_all_dates(model_name="model_a"):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name: {model_name}")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT snapshot_date
            FROM qubit_rankings
            WHERE model_name = %s
            ORDER BY snapshot_date
            """,
            (model_name,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_summary_stats():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT COUNT(DISTINCT snapshot_date) AS days "
            "FROM calibration_history"
        )
        days = cur.fetchone()["days"]

        cur.execute(
            "SELECT MAX(snapshot_date) AS d FROM calibration_history"
        )
        latest = cur.fetchone()["d"]

        cur.execute(
            """
            SELECT COUNT(DISTINCT CONCAT(backend, '-', qubit)) AS n
            FROM calibration_history
            WHERE snapshot_date = (
                SELECT MAX(snapshot_date) FROM calibration_history
            )
            """
        )
        qubits = cur.fetchone()["n"]

        cur.execute(
            "SELECT DISTINCT model_name FROM qubit_rankings "
            "ORDER BY model_name"
        )
        models_live = [row["model_name"] for row in cur.fetchall()]

        return {
            "days_collected": days,
            "qubits_latest": qubits,
            "latest_date": latest,
            "models_live": models_live,
        }
    finally:
        cur.close()
        conn.close()


def get_model_data_coverage():
    conn = get_connection()
    sql = """
        SELECT
            model_name,
            MIN(snapshot_date) AS first_date,
            MAX(snapshot_date) AS last_date,
            COUNT(DISTINCT snapshot_date) AS days,
            COUNT(*) AS `rows`
        FROM qubit_rankings
        GROUP BY model_name
        ORDER BY model_name
    """
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    setup_database()
    print(get_summary_stats())