""" train_model.py
    Train, test, and save models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import time
import json
import logging
import os
import sys
from collections import OrderedDict
import wandb

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import deepthinking as dt
import deepthinking.utils.logging_utils as lg


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    # Set seed if provided via environment variable
    seed = int(os.environ.get("SEED", -1))
    if seed >= 0:
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"All seeds set to {seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    cfg.problem.model.test_iterations = list(range(cfg.problem.model.test_iterations["low"],
                                                   cfg.problem.model.test_iterations["high"] + 1))
    assert 0 <= cfg.problem.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."
    writer = SummaryWriter(log_dir=f"tensorboard-{cfg.problem.model.model}-{cfg.problem.hyp.alpha}")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    
    # Initialize wandb with model info
    wandb.init(
        project='deep-thinking', 
        name=cfg.run_id, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    wandb.config.update({
        "total_params_M": round(pytorch_total_params / 1e6, 3),
        "model_architecture": cfg.problem.model.model,
    }, allow_val_change=True)
    
    log.info(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")
    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer(cfg.problem.hyp,
                                                                       cfg.problem.model,
                                                                       net,
                                                                       optimizer_state_dict)
    rand_method = getattr(cfg.problem.hyp, 'rand_method', 'basic')
    use_amp = getattr(cfg.problem.hyp, 'use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    log.info(f"Using automatic mixed precision: {use_amp}")
    train_setup = dt.TrainingSetup(optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler,
                                   clip=cfg.problem.hyp.clip,
                                   alpha=cfg.problem.hyp.alpha,
                                   max_iters=cfg.problem.model.max_iters,
                                   problem=cfg.problem.name,
                                   rand_method=rand_method,
                                   use_amp=use_amp,
                                   scaler=scaler)
    ####################################################

    ####################################################
    #        Train
    log.info(f"==> Starting training for {max(cfg.problem.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    best_so_far = False

    prev_state = None  # Keep previous epoch state for debugging
    for epoch in range(start_epoch, cfg.problem.hyp.epochs):
        # Save state before training (so we can replay if NaN happens)
        prev_state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
        
        try:
            start = time.time()
            loss, acc, bit_acc = dt.train(net, loaders, cfg.problem.hyp.train_mode, train_setup, device, epoch)
            train_time = time.time() - start
        except ValueError as e:
            # NaN detected! Save the pre-crash checkpoint
            log.error(f"Training crashed: {e}")
            if prev_state is not None:
                torch.save(prev_state, "pre_crash_checkpoint.pt")
                log.info("Saved pre-crash checkpoint to pre_crash_checkpoint.pt")
            raise
            
        val_result = dt.test(net, [loaders["val"]], cfg.problem.hyp.test_mode, [cfg.problem.model.max_iters],
                          cfg.problem.name, device, use_amp, return_bitwise=True)
        val_acc = val_result[0][0][cfg.problem.model.max_iters]
        val_bit_acc = val_result[2][0][cfg.problem.model.max_iters]
        diag_stats = val_result[1]
        if val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = val_acc

        log.info(f"Training loss at epoch {epoch}: {loss}")
        log.info(f"Training accuracy at epoch {epoch}: {acc}")
        log.info(f"Training bitwise accuracy at epoch {epoch}: {bit_acc}")
        log.info(f"Val accuracy at epoch {epoch}: {val_acc}")
        log.info(f"Val bitwise accuracy at epoch {epoch}: {val_bit_acc}")

        # Prepare wandb log dict
        wandb_dict = {
            "train/loss": loss,
            "train/acc": acc,
            "train/bit_acc": bit_acc,
            "train/time": train_time, 
            "train/val_acc": val_acc,
            "train/val_bit_acc": val_bit_acc
        }
        
        #Diagnostics, only for transformer
        try:
            # Handle DataParallel wrapping
            model = net.module if isinstance(net, torch.nn.DataParallel) else net
            
            #QKV
            qkv = model.recur_blocks_inject[0].attn.qkv.weight
            Q, K, _ = qkv.chunk(3, dim=0)
            spectral_product = torch.linalg.matrix_norm(Q, 2) * torch.linalg.matrix_norm(K, 2)
            wandb_dict['diagnostics/spectral_product'] = spectral_product.item()
            
            O = model.recur_blocks_inject[0].attn.out_proj.weight
            wandb_dict['diagnostics/out_spectral'] = torch.linalg.matrix_norm(O, 2).item()
        except Exception as e:
            logging.getLogger().warning(f"Diagnostics calculation failed: {e}")
            pass
        
        try:    
            #Inject
            if cfg.problem.model.injection_type in ['concat', 'linear']:
                inject_block = model.recur_blocks_inject[0]
                Wx, Wh = inject_block.Wx, inject_block.Wh
                Wh_weight = Wh.weight
                Wx_weight = Wx.lin.weight if hasattr(Wx, 'lin') else Wx.weight
                Wh_2 = torch.linalg.matrix_norm(Wh_weight, 2).item()
                Wx_2 = torch.linalg.matrix_norm(Wx_weight, 2).item()
                wandb_dict['diagnostics/Wx_spectral'] = Wx_2
                wandb_dict['diagnostics/inject_spectral_ratio'] = Wx_2 / Wh_2
                Wx_svs = torch.linalg.svdvals(Wx_weight)
                wandb_dict['diagnostics/inject_condition'] = (Wx_svs[0] / Wx_svs[10]).item() if Wx_svs[10] > 0 else float('inf')
                wandb_dict['diagnostics/Wx_radius'] = torch.abs(torch.linalg.eigvals(Wx_weight)).max().item() 
          
        except Exception as e:
            logging.getLogger().warning(f"Diagnostics calculation failed: {e}")
            pass
        
        try:
            #MLP
            mlp = model.recur_blocks_inject[0].mlp
            W1 = torch.linalg.matrix_norm(mlp[0].weight, 2).item()
            W2 = torch.linalg.matrix_norm(mlp[2].weight, 2).item()
            wandb_dict['diagnostics/mlp_gain'] = W1 * W2
            
            # Add diagnostic stats from evaluation
            if diag_stats and len(diag_stats.get('h_norm', [])) > 0:
                wandb_dict.update({
                    "diagnostics/h_norm_mean": np.mean(diag_stats['h_norm']),
                    "diagnostics/h_norm_max": np.max(diag_stats['h_norm']),
                    "diagnostics/h_correlation": np.max(diag_stats.get('h_correlation', [1.0])),
                })
            if len(diag_stats.get('attn_entropy', [])) > 0:
                wandb_dict["diagnostics/attn_entropy_mean"] = np.mean(diag_stats['attn_entropy'])
            if len(diag_stats.get('attn_max', [])) > 0:
                wandb_dict["diagnostics/max_logit"] = np.max(diag_stats['attn_max'])
            if len(diag_stats.get('h_norm_ratio', [])) > 0:
                wandb_dict["diagnostics/h_norm_ratio_mean"] = np.mean(diag_stats['h_norm_ratio'])
            if len(diag_stats.get('convergence_cosine', [])) > 0:
                wandb_dict["diagnostics/convergence_cosine"] = np.mean(diag_stats['convergence_cosine'])
            if len(diag_stats.get('first_convergence_iter', [])) > 0:
                wandb_dict["diagnostics/first_convergence_iter"] = np.mean(diag_stats['first_convergence_iter'])
        except Exception as e:
            logging.getLogger().warning(f"Diagnostics calculation failed: {e}")
            pass
        
        wandb.log(wandb_dict, step=epoch)

        # if the loss is nan, then stop the training (kept for backward compat)
        if np.isnan(float(loss)):
            if prev_state is not None:
                torch.save(prev_state, "pre_crash_checkpoint.pt")
                log.info("Saved pre-crash checkpoint to pre_crash_checkpoint.pt")
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)
        writer.add_scalar("Accuracy/val_acc", val_acc, epoch)
        writer.add_scalar("Accuracy/bit_acc", bit_acc, epoch)
        writer.add_scalar("Accuracy/val_bit_acc", val_bit_acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}",
                              optimizer.param_groups[i]["lr"],
                              epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.problem.hyp.val_period == 0 or epoch + 1 == cfg.problem.hyp.epochs:
            start = time.time()
            test_result = dt.test(net,
                                  [loaders["test"],
                                   loaders["val"],
                                   loaders["train"]],
                                  cfg.problem.hyp.test_mode,
                                  cfg.problem.model.test_iterations,
                                  cfg.problem.name,
                                  device,
                                  use_amp,
                                  return_bitwise=True)
            test_acc, val_acc, train_acc = test_result[0]
            test_bit_acc, val_bit_acc, train_bit_acc = test_result[2]
            log.info(f"Training accuracy: {train_acc}")
            log.info(f"Val accuracy: {val_acc}")
            log.info(f"Test accuracy (hard data): {test_acc}")
            log.info(f"Training bitwise accuracy: {train_bit_acc}")
            log.info(f"Val bitwise accuracy: {val_bit_acc}")
            log.info(f"Test bitwise accuracy (hard data): {test_bit_acc}")
            
            #Calculating the first time the accuracy gets within 1% of the best accuracy
            first_iter_converge = {'val/train_first': None, 
                                   'val/val_first': None, 
                                   'val/hard_first': None}
            for acc, iter_key in zip([train_acc, val_acc, test_acc], first_iter_converge.keys()):
                max_acc = max(acc.values())
                threshold = 0.99 * max_acc
                for i, v in acc.items():
                    if v >= threshold:
                        first_iter_converge[iter_key] = i
                        break            
            
            tb_last = cfg.problem.model.test_iterations[-1]
            
            wandb.log({
                "val/train_acc": train_acc[tb_last],
                "val/train_acc_penalty": max(train_acc.values()) - train_acc[tb_last],
                "val/val_acc": val_acc[tb_last],
                "val/val_acc_penalty": max(val_acc.values()) - val_acc[tb_last],
                "val/hard_acc": test_acc[tb_last],
                "val/hard_acc_penalty": max(test_acc.values()) - test_acc[tb_last],
                "val/train_bit_acc": train_bit_acc[tb_last],
                "val/val_bit_acc": val_bit_acc[tb_last],
                "val/hard_bit_acc": test_bit_acc[tb_last],
                "val/time": time.time() - start,
                **first_iter_converge 
            }, step=epoch)
            
            lg.write_to_tb([train_acc[tb_last], val_acc[tb_last], test_acc[tb_last]],
                           ["train_acc", "val_acc", "test_acc"],
                           epoch,
                           writer)
            lg.write_to_tb([train_bit_acc[tb_last], val_bit_acc[tb_last], test_bit_acc[tb_last]],
                           ["train_bit_acc", "val_bit_acc", "test_bit_acc"],
                           epoch,
                           writer)
        # check to see if we should save
        save_now = (epoch + 1) % cfg.problem.hyp.save_period == 0 or \
                   (epoch + 1) == cfg.problem.hyp.epochs or best_so_far
        if save_now:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
            out_str = f"model_{'best' if best_so_far else ''}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)
    writer.flush()
    writer.close()

    # save some accuracy stats (can be used without testing to discern which models trained)
    stats = OrderedDict([("max_iters", cfg.problem.model.max_iters),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_data", cfg.problem.test_data),
                         ("test_iters", list(cfg.problem.model.test_iterations)),
                         ("test_mode", cfg.problem.hyp.test_mode),
                         ("train_data", cfg.problem.train_data),
                         ("train_acc", train_acc),
                         ("val_acc", val_acc),
                         ("train_bit_acc", train_bit_acc),
                         ("val_bit_acc", val_bit_acc),
                         ("test_bit_acc", test_bit_acc)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)
    ####################################################


if __name__ == "__main__":
    # Only generate random run_id if not provided via command line
    if not any("run_id=" in arg for arg in sys.argv):
        run_id = dt.utils.generate_run_id()
        sys.argv.append(f"+run_id={run_id}")
    main()
