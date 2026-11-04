These files run the norm/lr combination experiments, designed for a slurm cluster. 

[run_all.bash](run_all.bash) runs all experiments except for high-lr sudoku, over 10 seeds (1440 runs). This will take substantial time (>4 days) on 1 GPU. [run_sudoku_high_lr.bash](run_sudoku_high_lr.bash) runs the remaining high-lr sudoku runs (160). All scripts were written for a B200 slurm cluster (`gpu-he` partition) with CUDA.

[run_sudoku_jacobian.bash](run_sudoku_jacobian.bash) trains the 16 sudoku checkpoints (1 and 2 blocks) in `rebuttal_runs/`, and [measure_all.bash](measure_all.bash) runs [../analyses/measure_jacobian.py](../analyses/measure_jacobian.py) over them (Appendix B.3, Tables 4-5).

## Where results live

- wandb logs `val/val_acc` and `val/hard_acc` (accuracy at the last test iteration, 500) every `val_period` epochs. The appendix tables use the final values ([../analyses/get_full_results.py](../analyses/get_full_results.py)).
- Per-iteration accuracy (iterations 1-500) is only in the training log and in `stats.json` in the run directory. wandb truncates each long line of its log copy (`output.log`), so it cannot be used for per-iteration curves. Keep the slurm `.out` files if you want to run [../analyses/schoop_out.py](../analyses/schoop_out.py).

## Known gaps

- Prefix-sums tests on 512-bit inputs, but the public `easy-to-hard-data` tarball only goes up to 128 bits, and its downloader first pings a dead hit-counter URL. The 512-bit data must be supplied in `data/prefix_sums_data/len_512/`.
