"""
measure_jacobian.py

For each checkpoint, produces:
  1. Convergence curve ||x_{t+1}-x_t||/||x_t|| over 500 loop iterations from
     the model's true initialization (interim_thought = initial_thought)
  2. rho(full loop-step Jacobian) at the settled state, via power iteration
     with Jacobian-vector products (never materializes the (dL)x(dL) matrix)
  3. Same rho with post_norm_func knocked out to Identity (post-norm runs
     only) -> isolates the outer-normalization factor
  4. rho(W_x), the paper's original proxy, for the comparison column
  5. Mean pairwise token cosine similarity of the initial state x0 and the
     settled state (Table 5: token clustering at init vs at the fixed point)

Usage (same overrides as the training run, plus the checkpoint path):

  python measure_jacobian.py \
      problem=sudoku problem/model=transformer \
      problem.model.hidden_dim=256 problem.model.norm_type=post \
      problem.model.injection_type=linear problem.model.recall_inner=false \
      problem.model.residual_method=add problem.model.num_blocks=1 \
      problem.model.max_iters=30 problem.model.attn_type=full \
      problem.model.num_sinks=0 problem.model.ccot=none \
      +problem.model.qk_normalization=true \
      problem.model.model_path=/path/to/run_dir/model_hardbest.pth

Do NOT pass compile=true. Runs on CPU or GPU; fp32 throughout.
Output: jacobian_measurements.json next to the checkpoint + a markdown row.
"""

import json
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import deepthinking as dt

log = logging.getLogger()

N_INPUTS = 5      # validation puzzles measured per checkpoint
N_ROLLOUT = 500   # loop iterations toward the fixed point
N_POWER = 100     # power-iteration steps
CONV_TOL = 1e-5   # relative step below which we call the rollout converged


# ---------------------------------------------------------------------------
# Model plumbing (matched to DTTransformer / AttentionBlock in this repo)
# ---------------------------------------------------------------------------

def unwrap(net):
    return net.module if isinstance(net, nn.DataParallel) else net


def embed_input(model, x):
    """Replicates DTTransformer.forward's input handling exactly."""
    if x.dim() >= 2 and x.size(1) == model.in_channels:
        x = x.permute(0, *range(2, x.dim()), 1)      # (B,C,*S) -> (B,*S,C)
    elif x.dim() == 2:
        x = x.unsqueeze(-1)                          # (B,L) -> (B,L,1)
    return model.init_norm(model.projection(x))


def make_step_fn(model, x0):
    """One full loop iteration as a function of the hidden state, recall
    input x0 held fixed. Mirrors the training loop body with noise/velocity
    off (asserted) and an empty ccot buffer."""
    assert model.noise_prob == 0.0 and model.velocity == 0, \
        "step fn assumes noise/velocity disabled (as in the Jacobian runs)"
    ccot = model.ccot_tokens.unsqueeze(0).repeat(x0.shape[0], 1, 1)

    def step(h):
        out = h
        for block in model.recur_blocks:
            out, _ = block(out, x0, ccot)
        return out

    return step


def knock_out_outer_norm(model):
    """Replace each block's post_norm_func (the paper's outer phi) with
    Identity. Returns undo info; [] means nothing to knock out (pre/peri)."""
    saved = []
    for block in model.recur_blocks:
        if not isinstance(block.post_norm_func, nn.Identity):
            saved.append((block, block.post_norm_func))
            block.post_norm_func = nn.Identity()
    return saved


def restore_outer_norm(saved):
    for block, norm in saved:
        block.post_norm_func = norm


def rho_Wx(model):
    """Spectral radius of the recall matrix W_x (the paper's proxy)."""
    block = model.recur_blocks[0]
    if not hasattr(block, "Wx"):
        return None                                   # injection_type != linear
    W = block.Wx.weight.detach().float().cpu()
    return torch.abs(torch.linalg.eigvals(W)).max().item()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def token_cos_sim(h):
    """Mean off-diagonal cosine similarity between token vectors of one
    hidden state (1, *S, D) -> scalar; 1.0 means fully clustered tokens."""
    t = F.normalize(h.reshape(-1, h.shape[-1]).float(), dim=-1)
    n = t.shape[0]
    return ((t @ t.T).sum().item() - n) / (n * (n - 1))


def rollout(step, x_init, n_iters):
    deltas = []
    x = x_init
    with torch.no_grad():
        for _ in range(n_iters):
            x_next = step(x)
            deltas.append((x_next - x).norm().item() / (x.norm().item() + 1e-12))
            x = x_next
    return x.detach(), deltas


def _jvp(step, x, v):
    """J @ v at x. Tries forward-mode AD; falls back to the double-backward
    trick if any op lacks forward-mode support."""
    try:
        from torch.func import jvp as func_jvp
        _, Jv = func_jvp(step, (x,), (v,))
        return Jv
    except (RuntimeError, NotImplementedError):
        _, Jv = torch.autograd.functional.jvp(
            step, (x,), (v,), create_graph=False, strict=False)
        return Jv


def spectral_radius(step, x_star, n_steps=N_POWER, seed=0):
    """Power iteration: repeatedly apply the Jacobian to a direction; the
    per-step growth factor converges to |lambda_max| = rho. Final estimates
    are averaged to smooth oscillation from complex eigenvalue pairs."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(x_star.shape, generator=g).to(x_star.device, x_star.dtype)
    v = v / v.norm()
    estimates = []
    for _ in range(n_steps):
        Jv = _jvp(step, x_star, v)
        growth = Jv.norm().item()
        estimates.append(growth)
        if growth < 1e-12:
            return 0.0, estimates
        v = (Jv / Jv.norm()).detach()
    return float(np.mean(estimates[-10:])), estimates


# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="train_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False     # honest fp32 eigenvalues
    torch.backends.cudnn.allow_tf32 = False

    cfg.problem.model.test_iterations = [cfg.problem.model.max_iters]
    ckpt_path = cfg.problem.model.model_path
    assert ckpt_path, "set problem.model.model_path=/path/to/model_hardbest.pth"
    log.info(f"Measuring checkpoint: {ckpt_path}")

    # builds the identical architecture and loads weights via model_path
    net, _, _ = dt.utils.load_model_from_checkpoint(
        cfg.problem.name, cfg.problem.model, device)
    model = unwrap(net).float().eval()

    proxy = rho_Wx(model)

    loaders = dt.utils.get_dataloaders(cfg.problem)
    inputs, _ = next(iter(loaders["val"]))
    inputs = inputs[:N_INPUTS].to(device).float()

    results = []
    for i in range(inputs.shape[0]):
        x0 = embed_input(model, inputs[i:i + 1]).detach()
        step = make_step_fn(model, x0)

        # (1) does the loop settle, starting from where the model starts?
        x_star, deltas = rollout(step, x0.clone(), N_ROLLOUT)
        converged = deltas[-1] < CONV_TOL

        # (2) rho of the full loop-step Jacobian at the settled state
        rho_full, traj = spectral_radius(step, x_star)

        # (3) rho with outer normalization removed (post-norm runs only)
        saved = knock_out_outer_norm(model)
        rho_nonorm = None
        if saved:
            rho_nonorm, _ = spectral_radius(step, x_star)
            restore_outer_norm(saved)

        results.append({
            "input_idx": i,
            "converged": bool(converged),
            "final_rel_step": deltas[-1],
            "rho_full": rho_full,
            "rho_full_no_outer_norm": rho_nonorm,
            "cos_sim_init": token_cos_sim(x0),
            "cos_sim_fixed_point": token_cos_sim(x_star),
            "power_iter_last10_std": float(np.std(traj[-10:])),
            "deltas_every_25": deltas[::25],
        })
        log.info(f"[input {i}] converged={converged} "
                 f"final_step={deltas[-1]:.2e} rho_full={rho_full:.4f} "
                 f"rho_no_norm={rho_nonorm} rho_Wx={proxy}")

    conv = [r for r in results if r["converged"]]

    def mean_over(key, rs):
        vals = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "checkpoint": ckpt_path,
        "rho_Wx_proxy": proxy,
        "n_converged": len(conv),
        "n_inputs": len(results),
        # rho is only meaningful at an actual fixed point -> converged inputs
        "rho_full_mean_converged": mean_over("rho_full", conv),
        "rho_full_no_norm_mean_converged": mean_over("rho_full_no_outer_norm", conv),
        "cos_sim_init_mean_converged": mean_over("cos_sim_init", conv),
        "cos_sim_fixed_point_mean_converged": mean_over("cos_sim_fixed_point", conv),
        "per_input": results,
    }
    out_path = os.path.join(os.path.dirname(ckpt_path),
                            "jacobian_measurements.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote {out_path}")

    name = os.path.basename(os.path.dirname(ckpt_path))
    rf = summary["rho_full_mean_converged"]
    rn = summary["rho_full_no_norm_mean_converged"]
    fmt = lambda z: "-" if z is None else f"{z:.3f}"
    log.info("| config | rho(Wx) | rho(M_full) | rho(M_full, no outer norm) | converged |")
    log.info(f"| {name} | {fmt(proxy)} | {fmt(rf)} | {fmt(rn)} "
             f"| {len(conv)}/{len(results)} |")


if __name__ == "__main__":
    main()