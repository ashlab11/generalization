These files conduct analysis on existing experiments. Unlike files in the experiments folder, these do not begin training runs but rather output either latex tables or plots. Each file gets data for the figure/section below:

- [anistropy.py](anistropy.py) conducts the anisotropy experiment in Appendix B.2
- [measure_jacobian.py](measure_jacobian.py) measures full-loop Jacobian spectral radii and token cosine similarity for B.3 (Tables 4-5); run it over all Jacobian checkpoints with [../experiments/measure_all.bash](../experiments/measure_all.bash)
- [spectral_analysis.py](spectral_analysis.py) gets results for B.4
- [schoop_out.py](schoop_out.py) plots accuracy vs. test iteration for the best autonomous configs (the "schoop" plot of Bansal et al. 2022). Not in the final paper, but fun to run for checking overthinking. It reads per-iteration accuracy from the slurm training logs, since wandb truncates those lines
- [get_full_results.py](get_full_results.py) gets seed-averaged results across all runs (B.6, B.7)
