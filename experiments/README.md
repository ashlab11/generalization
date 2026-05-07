These files run the norm/lr combination experiments, specifically on a slurm cluster. 

[run_all.bash](run_all.bash) runs all experiments except for high-lr sudoku. This will take substantial time (>4 days) on 1 GPU. [run_sudoku_high_lr.bash](run_sudoku_high_lr.bash) runs the remaining high-lr sudoku runs.