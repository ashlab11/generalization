import math
import wandb
import pandas as pd

#Will want to update this to include both table and plots, so you just need to run run.bash and analyze.py for any given experiment.

#Gets data
ENTITY  = "asherlabovich-brown-university"
PROJECT = "deep-thinking"
OUT_DIR = "experiments/residual_path"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters = {'state': 'finished', 
                                                  'config.sweep_name': 'residual_path_ablation',
                                                  'config.problem.hyp.seed': {'$exists': True}})

results = []

for run in runs:
    history_df = run.history()
    #Only get full-lr parts (skip warmup period, epochs 0-9)
    history_df = history_df[(history_df['_step'] >= 10) & (history_df['_step'] <= 60)]
    
    hard_acc, h_norm_ratio = history_df['val/hard_acc'], history_df['diagnostics/h_norm_ratio_mean']
    max_hard = hard_acc.max()
    ratio_when_max_hard = h_norm_ratio[(hard_acc == max_hard)]
    max_ratio = ratio_when_max_hard.max()
    den = hard_acc.notna().sum()
    percent_100 = (hard_acc == 100).sum() / den #Number of epochs at full step
    
    config = run.config['problem']
    lr = config['hyp']['lr']
    norm_type = config['model']['norm_type']
    residual_method = config['model']['residual_method']
    seed = config['hyp']['seed']
    results.append({
        'residual_method': residual_method, 
        'norm_type': norm_type, 
        'lr': lr, 
        'best_hard_acc': max_hard, 
        'top_h_ratio': max_ratio,
        'percent_100': percent_100, 
        'seed': seed
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/results.csv")
    
print(results_df)
# Print LaTeX
def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)
