from pathlib import Path
root=Path(__file__).resolve().parent
required=[
'run_level1.py','railway_snapshot.py','circuits/bell_state.py',
'simulation/simulate.py','analysis/analyze_counts.py',
'hardware/check_ibm.py','hardware/run_level2.py',
'tests/test_level1.py','requirements.txt','.env'
]
missing=[x for x in required if not (root/x).exists()]
if missing: raise SystemExit('Missing: '+', '.join(missing))
print('PROJECT STRUCTURE: PASS')
