# day7_final_report.py
from dotenv import load_dotenv
load_dotenv()
from database import get_connection
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

conn = get_connection()

# ── 1. How many days collected per backend ────────────────────────────────────
days_query = """
SELECT backend, 
       COUNT(DISTINCT snapshot_date) as days_collected,
       MIN(snapshot_date) as first_day,
       MAX(snapshot_date) as last_day,
       COUNT(*) as total_rows
FROM qubit_rankings
GROUP BY backend
"""
days_df = pd.read_sql(days_query, conn)
print("=== COLLECTION SUMMARY ===")
print(days_df.to_string(index=False))

# ── 2. Live prediction vs next-day actual (the key result) ───────────────────
live_query = """
SELECT 
    r1.backend,
    r1.qubit,
    r1.viability_score  as predicted_score,
    r1.snapshot_date    as prediction_date,
    r2.label            as actual_next_day_label,
    r2.snapshot_date    as actual_date,
    r2.T1_us, r2.T2_us, r2.readout_error
FROM qubit_rankings r1
JOIN qubit_rankings r2 
    ON  r1.backend       = r2.backend 
    AND r1.qubit         = r2.qubit
    AND r2.snapshot_date = DATE_ADD(r1.snapshot_date, INTERVAL 1 DAY)
ORDER BY r1.backend, r1.qubit, r1.snapshot_date
"""
live_df = pd.read_sql(live_query, conn)
print(f"\n=== LIVE PREDICTION PAIRS: {len(live_df)} ===")

if len(live_df) > 0 and live_df['actual_next_day_label'].nunique() > 1:
    print("\nPer-backend live AUC:")
    for b in live_df['backend'].unique():
        sub = live_df[live_df['backend']==b]
        if sub['actual_next_day_label'].nunique() < 2:
            print(f"  {b}: single class, skip")
            continue
        auc  = roc_auc_score(sub['actual_next_day_label'], sub['predicted_score'])
        bacc = balanced_accuracy_score(
            sub['actual_next_day_label'],
            (sub['predicted_score'] > 0.5).astype(int))
        mcc  = matthews_corrcoef(
            sub['actual_next_day_label'],
            (sub['predicted_score'] > 0.5).astype(int))
        print(f"  {b}: AUC={auc:.4f}  BalAcc={bacc:.4f}  MCC={mcc:.4f}  Pairs={len(sub)}")

    overall = roc_auc_score(live_df['actual_next_day_label'], live_df['predicted_score'])
    print(f"\n  OVERALL LIVE AUC: {overall:.4f}")

# ── 3. Consistency table ──────────────────────────────────────────────────────
consistency_query = """
SELECT backend, qubit,
       COUNT(DISTINCT snapshot_date)  as days_observed,
       SUM(label)                     as days_viable,
       ROUND(AVG(viability_score), 4) as mean_score,
       ROUND(STDDEV(viability_score), 4) as score_std,
       ROUND(AVG(T1_us), 1)           as mean_T1,
       ROUND(AVG(T2_us), 1)           as mean_T2,
       ROUND(AVG(readout_error), 4)   as mean_RE
FROM qubit_rankings
GROUP BY backend, qubit
ORDER BY backend, mean_score DESC
"""
cons_df = pd.read_sql(consistency_query, conn)
conn.close()

print("\n=== 7-DAY CONSISTENCY PER BACKEND ===")
for b in cons_df['backend'].unique():
    sub = cons_df[cons_df['backend']==b]
    total = len(sub)
    always = (sub['days_viable'] == sub['days_observed']).sum()
    never  = (sub['days_viable'] == 0).sum()
    mostly = (sub['days_viable'] >= sub['days_observed'] * 0.7).sum()
    print(f"\n{b} ({total} qubits tracked):")
    print(f"  Always viable (all days): {always} qubits ({100*always/total:.1f}%)")
    print(f"  Mostly viable (≥70% days): {mostly} qubits ({100*mostly/total:.1f}%)")
    print(f"  Never viable (0 days):    {never} qubits ({100*never/total:.1f}%)")
    print(f"  Top 5 most stable qubits:")
    top5 = sub.head(5)
    for _, r in top5.iterrows():
        print(f"    Q{int(r['qubit']):3d}: score={r['mean_score']:.3f} "
              f"viable={int(r['days_viable'])}/{int(r['days_observed'])} days "
              f"T1={r['mean_T1']:.0f}μs RE={r['mean_RE']:.4f}")

cons_df.to_csv('results/7day_consistency.csv', index=False)
live_df.to_csv('results/live_validation.csv', index=False)
print("\nSaved: results/7day_consistency.csv")
print("Saved: results/live_validation.csv")