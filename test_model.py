""" test_model.py
    Test models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import logging
import os
import sys
from collections import OrderedDict

import json

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import deepthinking as dt

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if cfg.problem.hyp.save_period is None:
        cfg.problem.hyp.save_period = cfg.problem.hyp.epochs
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    # Initialize wandb with model info
    run_name = getattr(cfg, "run_id", None) or None
    wandb.init(
        project='deep-thinking',
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    if not getattr(cfg, "run_id", None):
        cfg.run_id = wandb.run.name
        wandb.config.update({"run_id": cfg.run_id}, allow_val_change=True)

    training_args = OmegaConf.load(os.path.join(cfg.problem.model.model_path, ".hydra/config.yaml"))
    cfg_keys_to_load = [("hyp", "alpha"),
                        ("hyp", "epochs"),
                        ("hyp", "lr"),
                        ("hyp", "lr_factor"),
                        ("model", "max_iters"),
                        ("model", "model"),
                        ("hyp", "optimizer"),
                        ("hyp", "train_mode"),
                        ("model", "hidden_dim")]
    for k1, k2 in cfg_keys_to_load:
        cfg["problem"][k1][k2] = training_args["problem"][k1][k2]
    cfg.problem.train_data = cfg.problem.train_data

    log.info(OmegaConf.to_yaml(cfg))

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    cfg.problem.model.model_path = os.path.join(cfg.problem.model.model_path, "model_best.pth")
    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    
    wandb.config.update({
        "total_params_M": round(pytorch_total_params / 1e6, 3),
        "model_architecture": cfg.problem.model.model,
    }, allow_val_change=True)
    
    log.info(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    ####################################################
    #        Test
    log.info("==> Starting testing...")
    if "feedforward" in cfg.problem.model.model:
        test_iterations = [cfg.problem.model.max_iters]
    else:
        test_iterations = list(range(cfg.problem.model.test_iterations["low"],
                                     cfg.problem.model.test_iterations["high"] + 1))

    use_amp = getattr(cfg.problem.hyp, 'use_amp', False)
    
    if cfg.quick_test:
        test_result = dt.test(net, [loaders["test"]], cfg.problem.hyp.test_mode, test_iterations,
                              cfg.problem.name, device, use_amp, return_bitwise=True)
        test_acc = test_result[0][0]
        test_bit_acc = test_result[2][0]
        val_acc, train_acc = None, None
        val_bit_acc, train_bit_acc = None, None
    else:
        test_result = dt.test(net,
                              [loaders["test"], loaders["val"], loaders["train"]],
                              cfg.problem.hyp.test_mode,
                              test_iterations,
                              cfg.problem.name, device, use_amp, return_bitwise=True)
        test_acc, val_acc, train_acc = test_result[0]
        test_bit_acc, val_bit_acc, train_bit_acc = test_result[2]

    log.info(f"{dt.utils.now()} Training accuracy: {train_acc}")
    log.info(f"{dt.utils.now()} Val accuracy: {val_acc}")
    log.info(f"{dt.utils.now()} Testing accuracy (hard data): {test_acc}")
    log.info(f"{dt.utils.now()} Training bitwise accuracy: {train_bit_acc}")
    log.info(f"{dt.utils.now()} Val bitwise accuracy: {val_bit_acc}")
    log.info(f"{dt.utils.now()} Testing bitwise accuracy (hard data): {test_bit_acc}")

    # Log to wandb
    last_iter = test_iterations[-1]
    wandb_dict = {}
    
    if not cfg.quick_test:
        if train_acc is not None:
            wandb_dict["test/train_acc"] = train_acc[last_iter]
        if val_acc is not None:
            wandb_dict["test/val_acc"] = val_acc[last_iter]
        if train_bit_acc is not None:
            wandb_dict["test/train_bit_acc"] = train_bit_acc[last_iter]
        if val_bit_acc is not None:
            wandb_dict["test/val_bit_acc"] = val_bit_acc[last_iter]
    
    if test_acc is not None:
        wandb_dict["test/hard_acc"] = test_acc[last_iter]
    if test_bit_acc is not None:
        wandb_dict["test/hard_bit_acc"] = test_bit_acc[last_iter]
    
    # Create accuracy vs iterations plot
    if not cfg.quick_test:
        plot = wandb.plot.line_series(
            xs=test_iterations,
            ys=[[test_acc[i] for i in test_iterations],
                [val_acc[i] for i in test_iterations] if val_acc else [],
                [train_acc[i] for i in test_iterations] if train_acc else []],
            keys=["test", "val", "train"],
            title="Accuracy vs Iterations",
            xname="iterations")
        wandb_dict['test/accuracy_plot'] = plot
    else:
        plot = wandb.plot.line_series(
            xs=test_iterations,
            ys=[[test_acc[i] for i in test_iterations]],
            keys=["test"],
            title="Accuracy vs Iterations",
            xname="iterations")
        wandb_dict['test/accuracy_plot'] = plot
    
    wandb.log(wandb_dict)

    model_name_str = f"{cfg.problem.model.model}_hidden_dim={cfg.problem.model.hidden_dim}"
    stats = OrderedDict([("epochs", cfg.problem.hyp.epochs),
                         ("lr", cfg.problem.hyp.lr),
                         ("lr_factor", cfg.problem.hyp.lr_factor),
                         ("max_iters", cfg.problem.model.max_iters),
                         ("model", model_name_str),
                         ("model_path", cfg.problem.model.model_path),
                         ("num_params", pytorch_total_params),
                         ("optimizer", cfg.problem.hyp.optimizer),
                         ("val_acc", val_acc),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_bit_acc", test_bit_acc),
                         ("test_data", cfg.problem.test_data),
                         ("test_iters", test_iterations),
                         ("test_mode", cfg.problem.hyp.test_mode),
                         ("train_data", cfg.problem.train_data),
                         ("train_acc", train_acc),
                         ("train_bit_acc", train_bit_acc),
                         ("train_batch_size", cfg.problem.hyp.train_batch_size),
                         ("train_mode", cfg.problem.hyp.train_mode),
                         ("alpha", cfg.problem.hyp.alpha),
                         ("val_bit_acc", val_bit_acc)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)
    ####################################################


if __name__ == "__main__":
    main()
