import os
import pandas as pd
import wandb

ENTITY  = "asherlabovich-brown-university"
PROJECT = "deep-thinking"
OUT_DIR = "wandb_full_history_sweep3"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

os.makedirs(OUT_DIR, exist_ok=True)

for run in runs:
    if run.config.get("sweep_num") != 3:
        continue

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
        "config": run.config,
        "summary": dict(run.summary),
    }
    meta_path = os.path.join(OUT_DIR, f"{run.id}.meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print("Saved", run.id, "->", out_path)
