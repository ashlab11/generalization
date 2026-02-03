""" training.py
    Utilities for training models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import os
from dataclasses import dataclass
from random import randrange, gauss, random
import numpy as np
import math

import torch
from torch.nn import Softmin
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm

from deepthinking.utils.testing import get_predicted


def save_nan_debug(net, inputs, targets, epoch, batch_idx, h_stats=None, save_dir="."):
    """Save debugging info when NaN is detected."""
    debug_path = os.path.join(save_dir, "nan_debug.pt")
    grad_norms = {name: p.grad.norm().item() for name, p in net.named_parameters() 
                  if p.grad is not None}
    weight_stats = {name: {"mean": p.data.mean().item(), "std": p.data.std().item(), 
                           "max": p.data.abs().max().item()}
                    for name, p in net.named_parameters()}
    
    torch.save({
        'model_state': net.state_dict(),
        'inputs': inputs.cpu(),
        'targets': targets.cpu(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'h_stats': h_stats,
        'grad_norms': grad_norms,
        'weight_stats': weight_stats,
    }, debug_path)
    tqdm.write(f"NaN debug info saved to {debug_path}")


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"
    rand_method: "typing.Any" = "basic"  # 'basic', 'geiping', or int for min detach steps
    use_amp: bool = False  # Use automatic mixed precision (bfloat16)
    scaler: "typing.Any" = None  # GradScaler for mixed precision training
    softmin_beta: float = 1.0
    softmin_lambda: float = 0.1


def compute_first_n_iter_loss(all_outputs, targets, criterion, mask=None, n=5):
    """Compute average loss over first n iterations. Returns [n] tensor of per-iteration averages."""
    B, I, C, L = all_outputs.size()
    first_n_outputs = all_outputs[:, :n, :, :]  # [B, n, C, L]
    targets_exp = targets.unsqueeze(1).expand(-1, n, -1)  # [B, n, L]
    outputs_flat = first_n_outputs.reshape(B * n, C, L)
    targets_flat = targets_exp.reshape(B * n, L)
    losses = criterion(outputs_flat, targets_flat).reshape(B, n, L).float()
    if mask is not None:
        mask_exp = mask.unsqueeze(1).expand(-1, n, -1)
        losses = losses * mask_exp
        losses = losses[mask_exp > 0].reshape(B, n, -1) if losses.numel() > 0 else losses
    return losses.mean(dim=-1).mean(dim=0)  # [n]

def compute_loss_gradient_sensitivity(loss_all_outputs, beta, lam):
    """Compute gradient of total loss w.r.t. each per-iteration loss.
    Returns [I] tensor of per-iteration sensitivity (mean absolute gradient)."""
    B, I, L = loss_all_outputs.shape
    L_detached = loss_all_outputs.detach().clone().requires_grad_(True)
    
    # Recompute total loss using detached loss tensor
    zero_vec = torch.zeros_like(L_detached[:, :1, :])
    log_neg = -beta * L_detached
    log_sum_neg = torch.logsumexp(log_neg, dim=1)  # [B, L]
    softmin_loss = -log_sum_neg / beta + math.log(I) / beta
    
    relu_diff = F.relu(L_detached[:, 1:, :] - L_detached[:, :-1, :])
    relu_diff = torch.cat([zero_vec, relu_diff], dim=1)
    relu_diff_cumsum = torch.cumsum(relu_diff.flip(1), dim=1).flip(1)
    relu_suffix = torch.cat([relu_diff_cumsum[:, 1:, :], zero_vec], dim=1)
    weights = torch.softmax(log_neg, dim=1).detach()
    relu_loss = (weights * relu_suffix).sum(dim=1)
    
    total = ((1 - lam) * softmin_loss + lam * relu_loss).mean()
    
    # Compute gradients w.r.t. per-iteration losses
    g = torch.autograd.grad(total, L_detached, retain_graph=False)[0]  # [B, I, L]
    
    # Summarize per-iteration sensitivity (mean absolute gradient)
    g_iter = g.abs().mean(dim=(0, 2))  # [I] - average over batch and sequence
    return g_iter

def get_output_for_prog_loss(inputs, max_iters, net, rand_method = 'basic'):
    # get features from n iterations to use as input
    
    # Handle string integers from hydra config
    if rand_method == 'basic':
        n = randrange(0, max_iters)
        # do k iterations using intermediate features as input
        k = randrange(1, max_iters - n + 1)
    #Requiring 1 non-backprop every time
    elif rand_method == 'no_full_backprop':
        n = randrange(1, max_iters)
        k = randrange(1, max_iters - n + 1)
    elif rand_method == 'geiping':
        tau = gauss(np.log(max_iters / 2 - 1) - 1/8, 1/2)
        r = int(np.random.poisson(np.exp(tau), 1)) + 1
        n = min(max(r - 4, 0), 4)
        k = r - n
    elif rand_method == 'bimodal':
        if random() > 0.95:
            n = randrange(0, max_iters)
            # do k iterations using intermediate features as input
            k = randrange(1, max_iters - n + 1)
        else:
            n = randrange(0, max_iters * 20)
            k = 1
    elif rand_method == 'single':
        n = randrange(0, max_iters)
        # do k iterations using intermediate features as input
        k = 1
    elif rand_method == 'zipf':
        la = math.log1p(1)
        L  = math.log(max_iters * 20)
        while True:
            k = int(math.exp(random() * L))
            if k < 1: k = 1
            if random() < (1/k) * la / math.log1p(1/k):
                n = k
                break
        k = randrange(1, 15)
        
        

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k


def train(net, loaders, mode, train_setup, device, epoch=0):
    if mode == "progressive":
        train_loss, acc, bit_acc, first_five_ce_avg, grad_sensitivity = train_progressive(net, loaders, train_setup, device, epoch)
    elif mode == 'softmin':
        train_loss, acc, bit_acc, first_five_ce_avg, grad_sensitivity = train_softmin(net, loaders, train_setup, device, epoch)
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return train_loss, acc, bit_acc, first_five_ce_avg, grad_sensitivity

def train_softmin(net, loaders, train_setup, device, epoch=0, beta=None, lam=None):
    #Formula:
    #Ideal loss is the minimum loss x_* + lambda * sum(relu(x_k - x_*)) for k > *
    #If there is a minimum loss that can be reached, reach it. Afterwards, do not get worse.
    #Only two ways to reduce total loss: get to better loss eventually, or remain more stable 
    #after reaching ideal loss
    #This is a relaxation of that loss, to softmin on both ends
    
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    max_iters = train_setup.max_iters
    problem = train_setup.problem
    clip = train_setup.clip
    use_amp = train_setup.use_amp
    scaler = train_setup.scaler
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100, label_smoothing=0.01)
    if beta is None:
        beta = getattr(train_setup, "softmin_beta", 1.0)
    if lam is None:
        lam = getattr(train_setup, "softmin_lambda", 0.1)
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"softmin_beta must be > 0, got {beta}")
    
    train_loss = 0
    correct = 0
    total = 0
    bit_correct = 0
    bit_total = 0
    track_every_n = 10
    last_h_stats = None  # Keep track of last h_stats for debugging
    
    # Track CE loss for first 5 iterations
    first_five_iter_ce_losses = []
    # Track gradient sensitivity per iteration
    loss_grad_sensitivity = []
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        
        optimizer.zero_grad()

        # Conditionally apply autocast based on use_amp flag
        autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type='cuda', enabled=False)
        
        with autocast_context:
            #[B, Iters, C, L]
            all_outputs = net(inputs, iters_to_do=max_iters, return_all = True)
            outputs_max_iters = all_outputs[:, -1, :, :].squeeze(1)
            B, I, C, L = all_outputs.size()
            
            #NaN detection
            if torch.isnan(all_outputs).any():
                tqdm.write(f"NaN in outputs at batch {batch_idx}!")
                save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
                raise ValueError(f"NaN in outputs at epoch {epoch}, batch {batch_idx}")
            
            targets_exp = targets.unsqueeze(1).expand(-1, I, -1)
            loss_all_outputs = criterion(
                all_outputs.view(B * I, C, L),
                targets_exp.reshape(B * I, L),
            )
            loss_all_outputs = loss_all_outputs.view(B, I, L).float()
            
            # Compute average CE loss for first 5 iterations (for diagnostic)
            first_five_iter_means = compute_first_n_iter_loss(all_outputs, targets, criterion, n=15)
            first_five_iter_ce_losses.append(first_five_iter_means)
            
            zero_vec = torch.zeros_like(loss_all_outputs[:, :1, :])

            # Stable log-space computation with temperature beta.
            log_neg = -beta * loss_all_outputs
            log_sum_neg = torch.logsumexp(log_neg, dim=1)  # [B, L]
            softmin_loss = -log_sum_neg / beta + math.log(I) / beta #Term added for log mean rather than log sum
            
            relu_diff = F.relu(loss_all_outputs[:, 1:, :] - loss_all_outputs[:, :-1, :])
            relu_diff = torch.cat([zero_vec, relu_diff], dim=1) #[B, I, L]
            relu_diff_cumsum = torch.cumsum(relu_diff.flip(1), dim=1).flip(1) # sum_{r>=t} relu_diff[r]
            relu_suffix = torch.cat([relu_diff_cumsum[:, 1:, :], zero_vec], dim=1)
            
            weights = torch.softmax(log_neg, dim=1).detach()
            relu_loss = (weights * relu_suffix).sum(dim = 1)
            
            loss = ((1 - lam) * softmin_loss + lam * relu_loss).mean()
            
            # Compute gradient sensitivity diagnostic (on first batch only to save compute)
            if batch_idx == 0:
                g_iter = compute_loss_gradient_sensitivity(loss_all_outputs, beta, lam)
                loss_grad_sensitivity.append(g_iter)
            
            # NaN detection in loss (before backward)
            if torch.isnan(loss):
                tqdm.write(f"NaN in loss at batch {batch_idx}!")
                save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
                raise ValueError(f"NaN in loss at epoch {epoch}, batch {batch_idx}")
        
        # Use scaler if amp is enabled, otherwise regular backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Check for NaN in gradients
        grad_nan = any(torch.isnan(p.grad).any() for p in net.parameters() if p.grad is not None)
        if grad_nan:
            tqdm.write(f"NaN in gradients at batch {batch_idx}!")
            save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
            raise ValueError(f"NaN in gradients at epoch {epoch}, batch {batch_idx}")

        if clip is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        if problem in {"rule110", "cellular"}:
            mask = targets != -100
            eq = predicted == targets
            eq = eq | (~mask)
            correct += torch.amin(eq, dim=[-1]).sum().item()
            bit_correct += eq[mask].sum().item()
            bit_total += mask.sum().item()
        else:
            correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)
        if problem == "mazes":
            bit_correct += (predicted == targets)[mask].sum().item()
            bit_total += mask.sum().item()
        elif problem not in {"rule110", "cellular"}:
            bit_correct += (predicted == targets).sum().item()
            bit_total += targets.numel()

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total
    bit_acc = 100.0 * bit_correct / bit_total if bit_total > 0 else 0.0
    
    # Compute average CE loss over first 5 iterations
    first_five_avg = torch.stack(first_five_iter_ce_losses).mean().item() if first_five_iter_ce_losses else 0.0
    
    # Compute gradient sensitivity per iteration
    grad_sensitivity = torch.stack(loss_grad_sensitivity).mean(dim=0).cpu() if loss_grad_sensitivity else None

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc, bit_acc, first_five_avg, grad_sensitivity
    
    

def train_progressive(net, loaders, train_setup, device, epoch=0):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    rand_method = train_setup.rand_method
    use_amp = train_setup.use_amp
    scaler = train_setup.scaler
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    train_loss = 0
    correct = 0
    total = 0
    bit_correct = 0
    bit_total = 0
    track_every_n = 10
    last_h_stats = None  # Keep track of last h_stats for debugging
    
    # Track CE loss for first 5 iterations
    first_five_iter_ce_losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes":
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # Conditionally apply autocast based on use_amp flag
        autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type='cuda', enabled=False)
        
        with autocast_context:
            # Get outputs for first 5 iterations for diagnostic
            all_outputs = net(inputs, iters_to_do=max_iters, return_all=True)
            outputs_max_iters = all_outputs[:, -1, :, :]  # Last iteration output
            
            # Compute average CE loss for first 5 iterations (for diagnostic)
            first_five_iter_means = compute_first_n_iter_loss(all_outputs, targets, criterion, mask if problem == "mazes" else None, n=15)
            first_five_iter_ce_losses.append(first_five_iter_means)

            # NaN detection in outputs
            if torch.isnan(outputs_max_iters).any():
                tqdm.write(f"NaN in outputs at batch {batch_idx}!")
                save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
                raise ValueError(f"NaN in outputs at epoch {epoch}, batch {batch_idx}")
                
            if alpha != 1:
                outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                           outputs_max_iters.size(1), -1)
                loss_max_iters = criterion(outputs_max_iters, targets)
            else:
                loss_max_iters = torch.zeros_like(targets).float()

            if alpha != 0:
                outputs, k = get_output_for_prog_loss(inputs, max_iters, net, rand_method)
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                loss_progressive = criterion(outputs, targets)
            else:
                loss_progressive = torch.zeros_like(targets).float()

            if problem == "mazes":
                loss_max_iters = (loss_max_iters * mask)
                loss_max_iters = loss_max_iters[mask > 0]
                loss_progressive = (loss_progressive * mask)
                loss_progressive = loss_progressive[mask > 0]

            loss_max_iters_mean = loss_max_iters.mean()
            loss_progressive_mean = loss_progressive.mean()

            loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
            
            # NaN detection in loss (before backward)
            if torch.isnan(loss):
                tqdm.write(f"NaN in loss at batch {batch_idx}!")
                save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
                raise ValueError(f"NaN in loss at epoch {epoch}, batch {batch_idx}")
        
        # Use scaler if amp is enabled, otherwise regular backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Check for NaN in gradients
        grad_nan = any(torch.isnan(p.grad).any() for p in net.parameters() if p.grad is not None)
        if grad_nan:
            tqdm.write(f"NaN in gradients at batch {batch_idx}!")
            save_nan_debug(net, inputs, targets, epoch, batch_idx, last_h_stats)
            raise ValueError(f"NaN in gradients at epoch {epoch}, batch {batch_idx}")

        if clip is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        if problem in {"rule110", "cellular"}:
            mask = targets != -100
            eq = predicted == targets
            eq = eq | (~mask)
            correct += torch.amin(eq, dim=[-1]).sum().item()
            bit_correct += eq[mask].sum().item()
            bit_total += mask.sum().item()
        else:
            correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)
        if problem == "mazes":
            bit_correct += (predicted == targets)[mask].sum().item()
            bit_total += mask.sum().item()
        elif problem not in {"rule110", "cellular"}:
            bit_correct += (predicted == targets).sum().item()
            bit_total += targets.numel()

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total
    bit_acc = 100.0 * bit_correct / bit_total if bit_total > 0 else 0.0
    
    # Compute average CE loss over first 5 iterations
    first_five_avg = torch.stack(first_five_iter_ce_losses).mean().item() if first_five_iter_ce_losses else 0.0

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc, bit_acc, first_five_avg
