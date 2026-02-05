""" tools.py
    Utility functions that are common to all tasks

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import logging
import random
from datetime import datetime

import torch
from icecream import ic
from torch.optim import SGD, Adam, AdamW, Muon, Optimizer
from torch_optimizer import Shampoo
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import deepthinking.models as models
from deepthinking.models.init import apply_initialization
from .mazes_data import prepare_maze_loader
from .prefix_sums_data import prepare_prefix_loader
from .cellular_data import prepare_cellular_loader
from .chess_data import prepare_lichess_puzzle_loader
from .arc_data import prepare_arc_loader
from .warmup import ExponentialWarmup, LinearWarmup

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def get_dataloaders(problem_args):
    if problem_args.name == "prefix_sums":
        return prepare_prefix_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                     test_batch_size=problem_args.hyp.test_batch_size,
                                     train_data=problem_args.train_data,
                                     test_data=problem_args.test_data)
    elif problem_args.name in {"rule110", "cellular"}:
        extra = {k: v for k, v in dict(problem_args).items()
                 if k not in ["name", "hyp", "model", "train_data", "test_data"]}
        return prepare_cellular_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                       test_batch_size=problem_args.hyp.test_batch_size,
                                       train_data=problem_args.train_data,
                                       test_data=problem_args.test_data,
                                       **extra)
    elif problem_args.name == "mazes":
        return prepare_maze_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)
    elif problem_args.name == "chess":
        return prepare_lichess_puzzle_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                    test_batch_size=problem_args.hyp.test_batch_size,
                                    train_data=problem_args.train_data,
                                    test_data=problem_args.test_data)
    elif problem_args.name == "arc":
        extra = {k: v for k, v in dict(problem_args).items()
                 if k not in ["name", "hyp", "train_data", "test_data"]}
        return prepare_arc_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                  test_batch_size=problem_args.hyp.test_batch_size,
                                  train_data=problem_args.train_data,
                                  test_data=problem_args.test_data,
                                  **extra)
    else:
        raise ValueError(f"Invalid problem spec. {problem_args.name}")


def get_model(model, hidden_dim, max_iters, in_channels=3, **kwargs):
    model_lower = model.lower()
    model_name = model_lower if hasattr(models, model_lower) else model
    net = getattr(models, model_name)(hidden_dim=hidden_dim, in_channels=in_channels, max_iters=max_iters, **kwargs)
    return net


def get_optimizer(optim_args, model_args, net, state_dict):
    optimizer_name = optim_args.optimizer.lower()
    epochs = optim_args.epochs
    lr = optim_args.lr
    lr_decay = optim_args.lr_decay
    lr_schedule = optim_args.lr_schedule
    lr_factor = optim_args.lr_factor
    warmup_period = optim_args.warmup_period
    weight_decay = getattr(optim_args, 'weight_decay', 2e-4)
    eps = getattr(optim_args, 'eps', 1e-8)  # Default AdamW eps

    if optim_args.lr_throttle:
        # Reducing the lr here for the recurrent layers helps with stability,
        # To date (July 21, 2021), we may only need this for maze models.
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = model_args.max_iters
        all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1
        all_params = [{"params": base_params}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=weight_decay, eps=eps)
    elif optimizer_name == 'shampoo':        
        # For Shampoo, exclude bias and norm parameters from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in net.named_parameters():
            is_bias = name.endswith('.bias') or 'bias' in name.lower()
            is_norm = any(norm_name in name.lower() for norm_name in ['norm', 'ln', 'bn', 'gn', 'rmsnorm', 'layernorm'])
            
            if is_bias or is_norm:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        all_params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        optimizer = Shampoo(all_params, lr=lr)
    elif optimizer_name == "adamw":
        # For transformers, exclude bias and norm parameters from weight decay
        actual_net = net.module if isinstance(net, torch.nn.DataParallel) else net
        is_transformer = "Transformer" in type(actual_net).__name__
        
        if is_transformer:
            decay_params = []
            no_decay_params = []
            for name, param in net.named_parameters():
                # Check if parameter is bias or belongs to a normalization layer
                is_bias = name.endswith('.bias') or 'bias' in name.lower()
                is_norm = any(norm_name in name.lower() for norm_name in ['norm', 'ln', 'bn', 'gn', 'rmsnorm', 'layernorm'])
                
                if is_bias or is_norm:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            all_params = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0}
            ]
            optimizer = AdamW(all_params, lr=lr, eps=eps)
        else:
            optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay, eps=eps)
    elif optimizer_name == "muon":        
        class MultipleOptimizer(Optimizer):
            def __init__(self, *optimizers):
                param_groups = []
                for opt in optimizers:
                    param_groups.extend(opt.param_groups)
                super().__init__(param_groups, {})
                self.optimizers = optimizers

            def zero_grad(self):
                for op in self.optimizers:
                    op.zero_grad()

            def step(self, closure=None):
                for op in self.optimizers:
                    op.step()
            
            def state_dict(self):
                return {f'optimizer_{i}': op.state_dict() for i, op in enumerate(self.optimizers)}
            
            def load_state_dict(self, state_dict):
                for i, op in enumerate(self.optimizers):
                    op.load_state_dict(state_dict[f'optimizer_{i}'])
                    
        def use_muon(n, p):
            #Given a name, return whether muon should optimize the parameter
            return (n not in ['head', 'projection']) and (p.dim() > 1)
        
        muon_params = [p for n, p in net.named_parameters() if use_muon(n, p)]
        adam_params = [p for n, p in net.named_parameters() if not use_muon(n, p)]
        
        muon = Muon(muon_params, lr = lr, weight_decay = weight_decay)
        adam = AdamW(adam_params, lr = lr, weight_decay = weight_decay)
        optimizer = MultipleOptimizer(muon, adam)
        
    else:
        raise ValueError(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented.")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=0)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=warmup_period)

    if lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_schedule,
                                   gamma=lr_factor, last_epoch=-1)
    elif lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler


def load_model_from_checkpoint(problem, model_args, device):
    model = model_args.model
    model_path = model_args.model_path
    hidden_dim = getattr(model_args, 'hidden_dim', None) or getattr(model_args, 'width', None)
    if hidden_dim is None:
        raise ValueError("Must provide either 'hidden_dim' or 'width' in model config")
    max_iters = model_args.max_iters
    epoch = 0
    optimizer = None

    # Respect explicit config if provided; otherwise fall back to problem defaults.
    in_channels = getattr(model_args, "in_channels", None)
    if in_channels is None:
        if problem == "chess":
            in_channels = 12
        elif problem == "prefix_sums":
            in_channels = 1
        elif problem in {"rule110", "cellular"}:
            in_channels = 1
        else:
            in_channels = 3

    extra_args = {k: v for k, v in dict(model_args).items() 
                  if k not in ['model', 'model_path', 'width', 'hidden_dim', 'max_iters',
                               'test_iterations', 'in_channels', 'init_method']}
    net = get_model(model, hidden_dim, in_channels=in_channels, max_iters=max_iters, **extra_args)
    if model_path is None:
        apply_initialization(net, getattr(model_args, "init_method", "default"))
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]

    return net, epoch, optimizer


def now():
    return datetime.now().strftime("%Y%m%d %H:%M:%S")
