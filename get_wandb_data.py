import os
import pandas as pd
import wandb
from datetime import datetime, timezone

ENTITY  = "asherlabovich-brown-university"
PROJECT = "deep-thinking"
OUT_DIR = "ccot"

def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return to_dict(dict(obj))
    else:
        return obj

api = wandb.Api()
print("Fetching runs...")
runs = api.runs(f"{ENTITY}/{PROJECT}", filters = {
    'config.problem.name': {'$in': ['sudoku', 'chess']}, 
    'config.sweep_name': 'ccot',
    'state': 'finished'
})

runs_list = list(runs)
print(f"Found {len(runs_list)} finished runs matching filters...")

os.makedirs(OUT_DIR, exist_ok=True)

processed = 0
for i, run in enumerate(runs_list):
    if i % 10 == 0:
        print(f"Processing run {i+1}/{len(runs_list)}...")
    
    print(f"Processing run {run.id}...")
    rows = []
    try:
        for row in run.scan_history(page_size=10000):
            rows.append(row)
    except Exception as e:
        print("Skipped", run.id, "scan_history failed:", e)
        continue

    if not rows:
        print("Skipped", run.id, "no history rows found")
        continue

    df = pd.DataFrame(rows)

    # Save history
    out_path = os.path.join(OUT_DIR, f"{run.id}.parquet")
    df.to_parquet(out_path, index=False)

    # Also save run config/summary for labeling (tiny, but helpful)
    meta = {
        "run_id": run.id,
        "name": run.name,
        "config": to_dict(run.config),
        "summary": to_dict(run.summary)
    }
    
    meta_path = os.path.join(OUT_DIR, f"{run.id}.meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    processed += 1
    print(f"Saved {run.id} -> {out_path} ({processed} processed so far)")
