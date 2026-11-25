# export_mysql_backup.py
from dotenv import load_dotenv
load_dotenv()
from database import get_connection
import pandas as pd
from datetime import datetime

conn = get_connection()

# Export calibration_history
print("Exporting calibration_history...")
cal = pd.read_sql("SELECT * FROM calibration_history ORDER BY backend, qubit, snapshot_date", conn)
cal.to_csv(f'backup_calibration_history_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
print(f"Saved {len(cal)} rows")

# Export qubit_rankings
print("Exporting qubit_rankings...")
rank = pd.read_sql("SELECT * FROM qubit_rankings ORDER BY backend, qubit, snapshot_date", conn)
rank.to_csv(f'backup_qubit_rankings_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)
print(f"Saved {rank['snapshot_date'].nunique()} days, {len(rank)} rows")

conn.close()

# Verify
print("\n=== BACKUP SUMMARY ===")
print(f"calibration_history: {len(cal)} rows")
print(f"qubit_rankings: {len(rank)} rows")
print(f"Unique dates in rankings: {sorted(rank['snapshot_date'].unique())}")
print(f"Backends: {sorted(rank['backend'].unique())}")
print("\nBackup complete. Files saved locally.")