#!/usr/bin/env python3
"""
Clean up submitit logs and consolidate outputs to ablations/ with standard naming.

Run this after jobs complete to:
1. Move log files from submitit_logs/ to ablations/ with standard naming
2. Remove submitit internal files (.sh, .pkl)
"""

import glob
import os
import shutil

submitit_folder = "ablations/submitit_logs"

# Remove submitit internal files (but keep .pkl until jobs are done!)
for pattern in ["*.sh", ".submission_file_*.sh"]:
    for file in glob.glob(os.path.join(submitit_folder, pattern)):
        try:
            os.remove(file)
        except:
            pass

# Move actual log files from submitit_logs to ablations/ with standard naming
# submitit creates files like 265857_0_log.out (contains actual command output)
# Move them to ablations/265857_0.out and remove empty files created by SLURM -o
moved = 0
for log_file in glob.glob(os.path.join(submitit_folder, "*_log.out")):
    basename = os.path.basename(log_file)
    # Extract job_id and task_id: "265857_0_log.out" -> "265857", "0"
    parts = basename.replace("_log.out", "").split("_")
    if len(parts) >= 2:
        job_id, task_id = parts[0], parts[1]
        target = f"ablations/{job_id}_{task_id}.out"
        # Remove empty file created by SLURM -o if it exists
        if os.path.exists(target) and os.path.getsize(target) == 0:
            os.remove(target)
        # Move submitit's log file (has actual output) to standard location
        if os.path.exists(log_file):
            shutil.move(log_file, target)
            moved += 1

print(f"Moved {moved} log files to ablations/ with standard naming")
print("Removed submitit internal files (.sh)")
