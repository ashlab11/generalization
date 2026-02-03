#!/usr/bin/env python3
"""
Automatically submit SLURM array jobs for parameter sweeps.

Usage:
    python submit_sweep.py
"""

import itertools
import subprocess
import submitit

# ============================================================================
# CONFIGURATION - Edit this section
# ============================================================================

# All parameters (sweep vars override these)
FIXED_PARAMS = {
    "name": "init_ablation",
    "problem": "prefix_sums",
    "problem/model": "transformer",
    "problem.hyp.epochs": 100,
    "problem.hyp.optimizer": "adamw",
    "problem.hyp.weight_decay": 0.01,
    "problem.hyp.lr": 0.001,
    "problem.hyp.eps": 1e-8,  
    "problem.hyp.use_amp": True,  
    "problem.hyp.train_mode": "softmin", 
    "problem.hyp.rand_method": "basic",
    "problem.model.test_iterations.low": 1,
    "problem.model.test_iterations.high": 500,
    "problem.model.hidden_dim": 256,
    "problem.model.norm_type": "peri",
    "problem.model.attn_type": "conv",
    "problem.model.num_sinks": 1,
    "problem.model.injection_type": "concat",
    "problem.model.recall_inner": True,
    "problem.model.residual_method": "gru",
    "problem.model.init_method": "default",  
    "+problem.model.qk_normalization": True,
    "+sweep_name": "softmin_sweep",
}

# Variables to sweep (overrides FIXED_PARAMS)
SWEEP_VARS = {
    "EPS": {
        "values": [1e-4, 1e-8],
        "config_path": "problem.hyp.eps",
    },
    "USE_AMP": {
        "values": [True, False],
        "config_path": "problem.hyp.use_amp",
    }
}

SLURM_CONFIG = {
    "job_name": "sweep",
    "time": "4:00:00",
    "mem": "64G",
    "gres": "gpu:1",
    "partition": "gpu-he",
    "constraint": "nomig",
    "cpus_per_task": 4,
}

# ============================================================================
# SUBMISSION
# ============================================================================

def format_value(v):
    return str(v).lower() if isinstance(v, bool) else str(v)

def build_command(combo, var_names, var_config_paths):
    """Build python train_model.py command for a combination."""
    cmd = ["python", "train_model.py"]
    
    # Start with fixed params
    params = FIXED_PARAMS.copy()
    
    # Override with sweep values
    for var_name, config_path in zip(var_names, var_config_paths):
        idx = var_names.index(var_name)
        params[config_path] = combo[idx]
    
    # Add all params to command
    for key, value in params.items():
        cmd.append(f"{key}={format_value(value)}")
    
    # Generate run_id from sweep vars
    name_parts = [format_value(combo[var_names.index(v)]) for v in var_names]
    cmd.append(f"+run_id={'_'.join(name_parts)}")
    
    return cmd

def main():
    # Generate combinations
    var_names = list(SWEEP_VARS.keys())
    var_values = [SWEEP_VARS[v]["values"] for v in var_names]
    var_config_paths = [SWEEP_VARS[v]["config_path"] for v in var_names]
    combinations = list(itertools.product(*var_values))
    
    print(f"Submitting {len(combinations)} jobs...")
    for i, combo in enumerate(combinations):
        combo_dict = {var_names[j]: combo[j] for j in range(len(var_names))}
        print(f"  {i}: {combo_dict}")
    
    # Setup submitit executor
    # Submitit captures output internally in submitit_logs/*_log.out
    executor = submitit.AutoExecutor(folder="ablations/submitit_logs")
    executor.update_parameters(
        name=SLURM_CONFIG["job_name"],
        slurm_time=SLURM_CONFIG["time"],
        mem_gb=int(SLURM_CONFIG["mem"].rstrip("G")),
        gpus_per_node=int(SLURM_CONFIG["gres"].split(":")[1]),
        cpus_per_task=SLURM_CONFIG["cpus_per_task"],
        slurm_partition=SLURM_CONFIG["partition"],
        slurm_constraint=SLURM_CONFIG["constraint"],
        stderr_to_stdout=True,  # Combine stderr into stdout
    )
    
    # Submit jobs
    jobs = []
    for combo in combinations:
        cmd = build_command(combo, var_names, var_config_paths)
        job = executor.submit(subprocess.run, cmd, check=True)
        jobs.append(job)
    
    print(f"\nSubmitted {len(jobs)} jobs!")
    print(f"Job IDs: {[job.job_id for job in jobs]}")
    print("\nNote: Output files will appear in ablations/submitit_logs/*_log.out")
    print("      Run 'python ablations/cleanup_submitit_logs.py' after jobs complete to consolidate.")
    print("      (Don't delete .pkl files - submitit needs them to run jobs!)")

if __name__ == "__main__":
    main()
