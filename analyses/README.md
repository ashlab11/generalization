These files conduct analysis on existing experiments. Unlike files in the experiments folder, these do not begin training runs but rather output either latex tables or plots. Each file gets data for the figure/section below:

- [schoop_out.py](schoop_out.py) gets Figure 3 (requires training logs)
- [anisotropy.py](anisotropy.py) conducts the anisotropy experiment in Appendix B.2
- [spectral_analysis.py](spectral_analysis.py) gets results for B.4
- [get_full_results.py](get_full_results.py) gets seed-averaged results across all runs (B.5)
