import numpy as np
from scipy.optimize import minimize_scalar

#RUNS ANISOTROPY ANALYSES

INTERVALS = [(-20, 20)]
#--- Membership Tests---
def in_internal(x, y):
    return abs(1 + x*y) <= 1

def in_external(x, y):
    return abs((1 + y)*x) <= 1

#Projection helper
def project_to_curve(x0, y0, f):
    best_p = None
    best_d2 = np.inf

    for a, b in INTERVALS:
        obj = lambda x: (x - x0)**2 + (f(x) - y0)**2
        res = minimize_scalar(obj, bounds=(a, b), method="bounded")
        x = res.x
        p = np.array([x, f(x)])
        d2 = res.fun

        if d2 < best_d2:
            best_d2 = d2
            best_p = p

    return best_p, np.sqrt(best_d2)

#Project onto internal
def project_int(x0, y0):
    if in_internal(x0, y0):
        p = np.array([x0, y0])
        return p, 0
    
    candidates = []
    #Axis candidates
    candidates.append(np.array([0, y0]))
    candidates.append(np.array([x0, 0]))
    
    #Curve candidate
    fun = lambda x: -2 / x
    p3, _ = project_to_curve(x0, y0, fun)
    candidates.append(p3)
    
    dists = [np.linalg.norm(p - np.array([x0, y0])) for p in candidates]
    i = np.argmin(dists)
    return candidates[i], dists[i]

def project_ext(x0, y0):
    if in_external(x0, y0):
        p = np.array([x0, y0])
        return p, 0
    
    candidates = []
    
    #Curve candidates
    p1, _ = project_to_curve(x0, y0, lambda x:  (1 / x) - 1)
    candidates.append(p1)

    p2, _ = project_to_curve(x0, y0, lambda x: (-1 / x) - 1)
    candidates.append(p2)
    
    dists = [np.linalg.norm(p - np.array([x0, y0])) for p in candidates]
    i = np.argmin(dists)
    return candidates[i], dists[i]

# Anisotropy calcs
def anisotropy_log_ratio(points, eps=1e-8):
    """
    rA(p) = |log((|x|+eps)/(|y|+eps))|
    """
    x = np.abs(points[:, 0])
    y = np.abs(points[:, 1])
    return np.abs(np.log((x + eps) / (y + eps)))

def anisotropy_balance(points, eps=1e-8):
    """
    returns B(p) = (min(|x|,|y|) + eps) / (max(|x|,|y|)+eps)
    balanced ~1, anisotropic ~0
    """
    x = np.abs(points[:, 0])
    y = np.abs(points[:, 1])
    return (np.minimum(x, y) + eps) / (np.maximum(x, y) + eps) 

#Run experiment
def run_experiment(n = 10000, sigma = 1, seed = 0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, sigma, size=(n, 2))
    
    int_points = np.empty((n, 2))
    ext_points = np.empty((n, 2))
    
    for i, (x0, y0) in enumerate(pts):
        p_int, d_int = project_int(x0, y0)
        p_ext, d_ext = project_ext(x0, y0)
        int_points[i] = p_int
        ext_points[i] = p_ext
    
    return int_points, ext_points


def _se(a):
    n = max(len(a), 1)
    return float(a.std(ddof=1) / np.sqrt(n))

def _mean_se(mean, se):
    return f"{mean:.4f} ({se:.4f})"


def _row_tex(sigma, int_log, ext_log, int_bal, ext_bal):
    def cells(a_int, a_ext):
        return [
            _mean_se(a_int.mean(), _se(a_int)),
            _mean_se(a_ext.mean(), _se(a_ext)),
            f"{np.median(a_int):.4f}",
            f"{np.median(a_ext):.4f}",
        ]

    parts = [f"{sigma:g}"] + cells(int_log, ext_log) + cells(int_bal, ext_bal)
    return " & ".join(parts) + r" \\"


if __name__ == "__main__":
    sigmas = [0.1, 0.5, 1, 2, 4]
    n_samples = 10000
    table_rows = []
    for sigma in sigmas:
        int_pts, ext_pts = run_experiment(n=n_samples, sigma=sigma)
        table_rows.append(
            (
                sigma,
                anisotropy_log_ratio(int_pts),
                anisotropy_log_ratio(ext_pts),
                anisotropy_balance(int_pts),
                anisotropy_balance(ext_pts),
            )
        )

    lines = [
        r"% \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        (
            rf"\caption{{Projected-point anisotropy; $n = {n_samples}$ draws per $\sigma$. Parens after means: SE ($\hat\sigma/\sqrt{{n}}$)}}"
        ),
        r"\label{tab:anisotropy-projection}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l cc cc cc cc@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Log-ratio} & \multicolumn{4}{c}{Balance} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r" & \multicolumn{2}{c}{mean} & \multicolumn{2}{c}{median} & \multicolumn{2}{c}{mean} & \multicolumn{2}{c}{median} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"$\sigma$ & int & ext & int & ext & int & ext & int & ext \\",
        r"\midrule",
    ]
    for sigma, il, el, ib, eb in table_rows:
        lines.append(_row_tex(sigma, il, el, ib, eb))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(lines))
