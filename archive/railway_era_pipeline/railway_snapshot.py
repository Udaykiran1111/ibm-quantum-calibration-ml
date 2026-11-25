import os, json
from pathlib import Path
import mysql.connector
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "railway"
OUT.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

def req(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value

def main():
    conn = mysql.connector.connect(
        host=req("MYSQL_HOST"), port=int(req("MYSQL_PORT")),
        user=req("MYSQL_USER"), password=req("MYSQL_PASSWORD"),
        database=req("MYSQL_DATABASE")
    )
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT MAX(snapshot_date) AS d FROM calibration_history")
    latest = cur.fetchone()["d"]
    if latest is None:
        raise RuntimeError("No calibration data found.")

    cur.execute("""
        SELECT backend, COUNT(*) AS rows_count,
               AVG(T1_us) AS avg_T1_us, AVG(T2_us) AS avg_T2_us,
               AVG(readout_error) AS avg_readout_error
        FROM calibration_history WHERE snapshot_date=%s
        GROUP BY backend ORDER BY backend
    """, (latest,))
    calibration = cur.fetchall()

    cur.execute("""
        SELECT model_name, backend, COUNT(*) AS rows_count,
               AVG(viability_score) AS avg_score
        FROM qubit_rankings WHERE snapshot_date=%s
        GROUP BY model_name, backend ORDER BY model_name, backend
    """, (latest,))
    rankings = cur.fetchall()

    report = {
        "latest_snapshot_date": str(latest),
        "calibration": calibration,
        "rankings": rankings,
        "read_only": True
    }
    (OUT / "latest_railway_snapshot.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    conn.close()

if __name__ == "__main__":
    main()
