"""Loss landscape comparison.

This script analyzes neural network loss landscapes using three complementary approaches:
1. Hessian analysis: Computes analytical curvature metrics (trace, max eigenvalue) using pyhessian
2. 1D perturbation: Perturbs parameters along random directions to measure robustness. Allows multiple models for comparison.
3. 2D perturbation: Visualizes a 2D slice of the landscape by perturbing along two fixed directions

The script generates both static (matplotlib) and interactive (plotly) visualizations.
"""
import argparse
import copy
import csv
import os
import sys
from contextlib import nullcontext

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from pyhessian import hessian

import deepthinking as dt
from deepthinking.utils.testing import get_predicted

#HELPER FUNCTIONS
#Hessian functions
class MaxIterWrapper(torch.nn.Module):
    def __init__(self, net, max_iters):
        super().__init__()
        self.net = net
        self.max_iters = max_iters
    
    def forward(self, x):
        outputs = self.net(x, iters_to_do=self.max_iters)[:, -1]
        logits = outputs.view(outputs.size(0), outputs.size(1), -1)
        return logits
    
#Uses pyhessian to analyze curvature of loss landscape from an analytic perspective
def get_hessian_data(net, loader, device, max_iters, batch_limit=None):
    model = net.module if isinstance(net, torch.nn.DataParallel) else net
    model_norm = torch.nn.utils.get_total_norm(model.parameters())
    uses_natten = any(hasattr(m, 'attn_type') and getattr(m, 'attn_type') == 'local' for m in model.modules() if hasattr(m, 'attn_type'))
    if uses_natten:
        print("Warning: Model uses natten/local attention which doesn't support second-order gradients. Skipping Hessian computation.")
        return None
    
    criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    traces, max_eigs, count = 0, 0, 0
    max_net = MaxIterWrapper(net, max_iters)
    max_net.eval()
    max_net = max_net.float()
    for i, (inputs, targets) in enumerate(loader):
        if batch_limit is not None and i >= batch_limit:
            break
        inputs = inputs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long().view(targets.size(0), -1)
        hessian_comp = hessian(max_net, criterion, data=(inputs, targets), cuda=(device == 'cuda'))
        trace = hessian_comp.trace()
        max_eig = hessian_comp.eigenvalues()
        mean_trace = np.mean(trace)
        count+=1
        traces+=mean_trace
        max_eigs+=max_eig
    return traces / count, max_eigs / (count * model_norm)


#Perturbing functions
def get_perturbation_direction(params, device, seed = None):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    for p in params:
        w = p.norm()
        u = torch.randn(p.shape, generator=generator, device=device, dtype=p.dtype)
        yield w * (u / (u.norm() + 1e-12))
        
def perturb_in_place(params, rho, device, seed = None, direction=None):
    direction = get_perturbation_direction(params, device, seed) if direction is None else direction
    for p, u in zip(params, direction):
        p.add_(rho * u)

def restore_in_place(params, bases):
    for p, b in zip(params, bases):
        p.copy_(b)

#Gets loss from a perturbed network
def get_loss(net, loader, cfg, device, batch_limit = None):    
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    use_amp = bool(getattr(cfg.problem.hyp, "use_amp", False)) and device == "cuda"
    total_loss, total_count, total_correct, total_samples = 0.0, 0, 0, 0
    net.eval()
    for i, (inputs, targets) in enumerate(loader):
        if batch_limit is not None and i >= batch_limit:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long().view(targets.size(0), -1)
        with (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)):
            outputs = net(inputs, iters_to_do=cfg.problem.model.max_iters)[:, -1]
        logits = outputs.view(outputs.size(0), outputs.size(1), -1)
        losses = criterion(logits, targets)
        total_loss += losses.sum().item()
        total_count += int(losses.numel())
        total_correct += int(torch.amin(get_predicted(inputs, outputs, cfg.problem.name) == targets, dim=[1]).sum().item())
        total_samples += int(targets.size(0))
    return total_loss / total_count, 100.0 * total_correct / total_samples


#ACTUAL CODE
def run_model(net, loader, cfg, device, batches, n_seeds, rhos):
    for p in net.parameters():
        p.requires_grad = False
    bases = [p.clone() for p in net.parameters()]
    
    losses = torch.zeros((n_seeds, len(rhos)), device=device)
    accs = torch.zeros((n_seeds, len(rhos)), device=device)
    for i in range(n_seeds):
        for j, rho in enumerate(rhos):
            torch.manual_seed(i)
            if device == "cuda":
                torch.cuda.manual_seed_all(i)
            perturb_in_place(net.parameters(), 0.1 * rho, device)
            perturbed_loss, perturbed_acc = get_loss(net, loader, cfg, device, batch_limit = batches)
            losses[i, j] = perturbed_loss
            accs[i, j] = perturbed_acc
            restore_in_place(net.parameters(), bases)
    
    losses = torch.mean(losses, dim=0).detach().cpu().numpy()
    accs = torch.mean(accs, dim=0).detach().cpu().numpy()
    for p in net.parameters():
        p.requires_grad = True
    
    return losses, accs

def run_model_2d(net, loader, cfg, device, batches, n_seeds, rhos):
    #Creates necessary data for a 2d graph
    for p in net.parameters():
        p.requires_grad = False
    bases = [p.clone() for p in net.parameters()]
    direction1 = get_perturbation_direction(net.parameters(), device, seed=1)
    direction2 = get_perturbation_direction(net.parameters(), device, seed=2)
    
    losses = torch.zeros((n_seeds, len(rhos), len(rhos)), device=device)
    accs = torch.zeros((n_seeds, len(rhos), len(rhos)), device=device)
    for i in range(n_seeds):
        for j, rho1 in enumerate(rhos):
            for k, rho2 in enumerate(rhos):
                torch.manual_seed(i)
                if device == "cuda":
                    torch.cuda.manual_seed_all(i)
                perturb_in_place(net.parameters(), 0.1 * rho1, device, direction=direction1)
                perturb_in_place(net.parameters(), 0.1 * rho2, device, direction=direction2)
                perturbed_loss, perturbed_acc = get_loss(net, loader, cfg, device, batch_limit = batches)
                losses[i, j, k] = perturbed_loss
                accs[i, j, k] = perturbed_acc
                restore_in_place(net.parameters(), bases)
    
    losses = torch.mean(losses, dim=0).detach().cpu().numpy()
    accs = torch.mean(accs, dim=0).detach().cpu().numpy()
    for p in net.parameters():
        p.requires_grad = True
    
    return losses, accs


def main():
    p = argparse.ArgumentParser(description="Perturb prefix-sums checkpoints and compare robustness.")
    p.add_argument("model_folder", nargs="+", help="Path(s) to model folder(s)")
    p.add_argument("model_name", nargs="+", help="Model names for graph")
    p.add_argument("--type", type=str, default='1d') #1d, 2d, and hessian
    p.add_argument("--data", type=str, default="val")
    p.add_argument("--n-rhos", type=int, default=10)
    p.add_argument("--max-rho", type=int, default=1)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--batches", type=int, default=10)
    
    args = p.parse_args()
    folders = args.model_folder
    names = args.model_name
    assert len(folders) == len(names), "names and folders must be the same length"
    assert (args.type != '2d') or len(folders) == 1, "3d graph only supports a single model"
    assert args.type in ['1d', '2d', 'hessian'], "type must be 1d, 2d, or hessian"
    file_ending = "model_best.pth" if args.data == 'val' else "model_besthard.pth" #Change to val eventually
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rhos = [args.max_rho * i / (args.n_rhos - 1) for i in range(args.n_rhos)]
    
    #Necessary to get loader. Assumes that all models use the same data.
    cfg_first = OmegaConf.load(os.path.join(folders[0], ".hydra", "config.yaml"))
    original_cwd = os.getcwd()
    os.chdir(folders[0])
    loaders = dt.utils.get_dataloaders(cfg_first.problem)
    os.chdir(original_cwd)
    loader = loaders[args.data]
    
    if args.type == '1d':
        full_losses = np.empty((args.n_rhos, len(folders)))
        full_accs = np.empty((args.n_rhos, len(folders)))
        for i, folder in enumerate(folders): 
            cfg = OmegaConf.load(os.path.join(folder, ".hydra", "config.yaml"))
            cfg.problem.model.model_path = os.path.join(folder, file_ending)
            net, _, _ = dt.utils.load_model_from_checkpoint(cfg.problem.name, cfg.problem.model, device)
            losses, accs = run_model(net, loader, cfg, device, batches=args.batches, n_seeds = args.n_seeds, 
                                            rhos = rhos)
            full_losses[:, i] = losses
            full_accs[:, i] = accs
            del net
            if device == "cuda":
                torch.cuda.empty_cache()
        
        plt.figure("losses")
        for i, name in enumerate(names):
            plt.plot(rhos, full_losses[:, i], label=name)
        plt.xlabel("Rho")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.figure("accs")
        for i, name in enumerate(names):
            plt.plot(rhos, full_accs[:, i], label=name)
        plt.xlabel("Rho")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        
    elif args.type == '2d':
        cfg = OmegaConf.load(os.path.join(folders[0], ".hydra", "config.yaml"))
        cfg.problem.model.model_path = os.path.join(folders[0], file_ending)
        net, _, _ = dt.utils.load_model_from_checkpoint(cfg.problem.name, cfg.problem.model, device)
        losses, accs = run_model_2d(net, loader, cfg, device, batches=args.batches, n_seeds = args.n_seeds, 
                                    rhos=rhos)
        
        alpha = np.array(rhos)
        beta = np.array(rhos)
        Alpha, Beta = np.meshgrid(alpha, beta)
        
        fig_plotly = go.Figure(data=[go.Surface(z=losses, x=Alpha, y=Beta, colorscale='Viridis')])
        fig_plotly.update_layout(
            title='3D Loss Landscape',
            scene=dict(
                xaxis_title='Alpha',
                yaxis_title='Beta',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            )
        )
        fig_plotly.write_html("loss_landscape_3d.html")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Alpha, Beta, losses, cmap='viridis', alpha=0.9, edgecolor='none')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Loss')
        ax.view_init(elev=30, azim=45)
        plt.colorbar(surf, ax=ax, shrink=0.5)
        plt.title('3D Loss Landscape')
        plt.savefig("loss_landscape_3d.png")
        plt.show()  
    elif args.type == 'hessian':
        cfg = OmegaConf.load(os.path.join(folders[0], ".hydra", "config.yaml"))
        cfg.problem.model.model_path = os.path.join(folders[0], file_ending)
        net, _, _ = dt.utils.load_model_from_checkpoint(cfg.problem.name, cfg.problem.model, device)
        max_iters = cfg.problem.model.max_iters if args.data == 'val' else cfg.problem.model.test_iterations.high
        trace, eig = get_hessian_data(net, loader, device, max_iters, batch_limit=args.batches)
        if trace is not None:
            print("Average Hessian Trace:", trace)
            print("Average Max Eigenvalue:", eig)
        else:
            print("Hessian computation skipped (model uses natten/local attention)")
        

if __name__ == "__main__":
    main()