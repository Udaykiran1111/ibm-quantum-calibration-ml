# run this locally
from dotenv import load_dotenv
load_dotenv()
from database import get_connection
import pandas as pd

conn = get_connection()
df = pd.read_sql("""
    SELECT snapshot_date, backend, COUNT(*) as qubits
    FROM calibration_history
    GROUP BY snapshot_date, backend
    ORDER BY snapshot_date, backend
""", conn)
conn.close()
print(df.to_string())
print(f"\nTotal unique dates: {df['snapshot_date'].nunique()}")