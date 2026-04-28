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
from collections import OrderedDict
import wandb

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.profiler import profile, ProfilerActivity

import deepthinking as dt


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    # Set seed if provided via config or environment variable
    seed = getattr(cfg.problem.hyp, 'seed', int(os.environ.get("SEED", -1)))
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
    full_only_hard = bool(getattr(cfg.problem.hyp, "full_only_hard", False))
    profile_mode = bool(getattr(cfg, "profile", False))
    compile_mode = bool(getattr(cfg, "compile", False))
    if profile_mode:
        log.info("Profile mode enabled.")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device)        
    if compile_mode:
        FULLGRAPH = False
        DYNAMIC = True
        MODE = "default"
        try:
            torch._logging.set_logs(recompiles=True)
            log.info("Enabled TorchDynamo recompile logging (recompiles=True).")
        except Exception:
            log.warning("Could not enable torch._logging recompile logs; set TORCH_LOGS=recompiles manually if needed.")
        model = net.module if isinstance(net, torch.nn.DataParallel) else net
        compiled_blocks = 0
        if hasattr(model, "recur_blocks"):
            for i, block in enumerate(model.recur_blocks):
                model.recur_blocks[i] = torch.compile(block, fullgraph=FULLGRAPH, dynamic=DYNAMIC, mode=MODE)
                compiled_blocks += 1
        log.info(f"torch.compile enabled for {compiled_blocks} recurrent attention blocks (fullgraph={FULLGRAPH}, dynamic={DYNAMIC}, mode={MODE}).")
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    
    # Initialize wandb with model info
    run_id = getattr(cfg, "run_id", None)
    wandb.init(
        project='deep-thinking',
        name=run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(
            x_graphql_timeout_seconds=120,
            x_file_stream_timeout_seconds=120,
        ),
    )
    if not getattr(cfg, "run_id", None):
        cfg.run_id = wandb.run.name
        wandb.config.update({"run_id": cfg.run_id}, allow_val_change=True)
    
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
    if device == "cuda" and not use_amp:
        # Ensure true FP32 for fp32 runs (avoid TF32 on Ampere+).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log.info("TF32 disabled for fp32 run.")
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
                                   scaler=scaler,
                                   softmin_beta=getattr(cfg.problem.hyp, "softmin_beta", 1.0),
                                   softmin_lambda=getattr(cfg.problem.hyp, "softmin_lambda", 0.1))
    ####################################################

    ####################################################
    #        Train
    if profile_mode:
        final_epoch_exclusive = min(cfg.problem.hyp.epochs, start_epoch + 2)
        profiled_epoch = start_epoch + 1 if final_epoch_exclusive - start_epoch >= 2 else start_epoch
        log.info(f"Profile mode: running {max(final_epoch_exclusive - start_epoch, 0)} epoch(s), profiling epoch {profiled_epoch}.")
    else:
        final_epoch_exclusive = cfg.problem.hyp.epochs
        profiled_epoch = None
    log.info(f"==> Starting training for {max(final_epoch_exclusive - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    highest_val_milestone_acc_so_far = -1
    highest_hard_milestone_acc_so_far = -1
    best_so_far = False

    prev_state = None  # Keep previous epoch state for debugging
    for epoch in range(start_epoch, final_epoch_exclusive):
        # Save state before training (so we can replay if NaN happens)
        prev_state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
        
        try:
            start = time.time()
            if profile_mode and epoch == profiled_epoch:
                activities = [ProfilerActivity.CPU]
                if device == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, record_shapes=False, profile_memory=True, with_stack=False) as prof:
                    loss, acc, bit_acc, first_five_ce_avg, grad_sensitivity = dt.train(
                        net, loaders, cfg.problem.hyp.train_mode, train_setup, device, epoch
                    )
                log.info("Top 10 ops by self CPU time:")
                log.info("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                log.info("Top 20 ops by self CPU memory:")
                log.info("\n" + prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
                if device == "cuda":
                    log.info("Top 10 ops by self CUDA time:")
                    log.info("\n" + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                    log.info("Top 20 ops by self CUDA memory:")
                    log.info("\n" + prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
                    log.info("Top 20 ops by CUDA memory (total):")
                    log.info("\n" + prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
            else:
                loss, acc, bit_acc, first_five_ce_avg, grad_sensitivity = dt.train(
                    net, loaders, cfg.problem.hyp.train_mode, train_setup, device, epoch
                )
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
        if len(diag_stats.get('h_norm_ratio', [])) > 0:
            log.info(f"Val h_norm_ratio (last/first hidden norm) at epoch {epoch}: {np.mean(diag_stats['h_norm_ratio']):.6f}")
        log.info(f"Average CE loss over first 5 iterations at epoch {epoch}: {first_five_ce_avg:.6f}")
        
        # Log LSTM weight norm diagnostic
        try:
            model = net.module if isinstance(net, torch.nn.DataParallel) else net
            lstm_weight_norm = model.recur_blocks[0].lstm.weight_hh.norm().item()
            log.info(f"LSTM recurrent weight (weight_hh) L2 norm at epoch {epoch}: {lstm_weight_norm:.6f}")
        except (AttributeError, IndexError) as e:
            log.info(f"Error: {e}")
            lstm_weight_norm = None
        
        # Log gradient sensitivity per iteration
        if grad_sensitivity is not None:
            grad_sens_str = ", ".join([f"iter{i}: {v:.6f}" for i, v in enumerate(grad_sensitivity)])
            log.info(f"Loss gradient sensitivity per iteration at epoch {epoch}: {grad_sens_str}")
            # Log fraction of mass per iteration
            total_mass = grad_sensitivity.sum().item()
            if total_mass > 0:
                fraction_mass = grad_sensitivity / total_mass
                fraction_str = ", ".join([f"iter{i}: {v:.3f}" for i, v in enumerate(fraction_mass)])
                log.info(f"Fraction of gradient mass per iteration: {fraction_str}")

        # Prepare wandb log dict
        wandb_dict = {
            "train/loss": loss,
            "train/acc": acc,
            "train/bit_acc": bit_acc,
            "train/time": train_time, 
            "train/val_acc": val_acc,
            "train/val_bit_acc": val_bit_acc,
            "train/first_five_iter_ce_avg": first_five_ce_avg
        }
        
        # Add LSTM weight norm to wandb
        if lstm_weight_norm is not None:
            wandb_dict["diagnostics/lstm_weight_hh_norm"] = lstm_weight_norm
        
        #Diagnostics, only for transformer
        try:
            # Handle DataParallel wrapping
            model = net.module if isinstance(net, torch.nn.DataParallel) else net
            
            #QKV
            qkv = model.recur_blocks[0].attn.qkv.weight
            Q, K, _ = qkv.chunk(3, dim=0)
            spectral_product = torch.linalg.matrix_norm(Q, 2) * torch.linalg.matrix_norm(K, 2)
            wandb_dict['diagnostics/spectral_product'] = spectral_product.item()
            
            O = model.recur_blocks[0].attn.out_proj.weight
            wandb_dict['diagnostics/out_spectral'] = torch.linalg.matrix_norm(O, 2).item()
        except Exception as e:
            logging.getLogger().warning(f"Diagnostics calculation failed: {e}")
            pass
        
        try:    
            #Inject
            if cfg.problem.model.injection_type in ['concat', 'linear']:
                inject_block = model.recur_blocks[0]
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
            mlp = model.recur_blocks[0].mlp
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
            raise ValueError(f"Loss is nan, exiting...")

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.problem.hyp.val_period == 0 or epoch + 1 == final_epoch_exclusive:
            start = time.time()
            if full_only_hard:
                hard_result = dt.test(net,
                                      [loaders["test"]],
                                      cfg.problem.hyp.test_mode,
                                      cfg.problem.model.test_iterations,
                                      cfg.problem.name,
                                      device,
                                      use_amp,
                                      return_bitwise=True)
                easy_result = dt.test(net,
                                      [loaders["val"], loaders["train"]],
                                      cfg.problem.hyp.test_mode,
                                      [cfg.problem.model.max_iters],
                                      cfg.problem.name,
                                      device,
                                      use_amp,
                                      return_bitwise=True)
                test_acc = hard_result[0][0]
                test_bit_acc = hard_result[2][0]
                val_acc, train_acc = easy_result[0]
                val_bit_acc, train_bit_acc = easy_result[2]
            else:
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
            train_val_iter = cfg.problem.model.max_iters if full_only_hard else tb_last
            final_val_acc = val_acc[train_val_iter]
            final_hard_acc = test_acc[tb_last]

            # Save milestone checkpoints at first epoch that reaches a new best.
            if final_val_acc > highest_val_milestone_acc_so_far:
                highest_val_milestone_acc_so_far = final_val_acc
                state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
                log.info(f"Saving model to: model_valbest.pth (val acc={final_val_acc:.4f})")
                torch.save(state, "model_valbest.pth")
            if final_hard_acc > highest_hard_milestone_acc_so_far:
                highest_hard_milestone_acc_so_far = final_hard_acc
                state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
                log.info(f"Saving model to: model_hardbest.pth (hard acc={final_hard_acc:.4f})")
                torch.save(state, "model_hardbest.pth")
            
            wandb.log({
                "val/train_acc": train_acc[train_val_iter],
                "val/train_acc_penalty": max(train_acc.values()) - train_acc[train_val_iter],
                "val/val_acc": final_val_acc,
                "val/val_acc_penalty": max(val_acc.values()) - val_acc[train_val_iter],
                "val/hard_acc": final_hard_acc,
                "val/hard_acc_penalty": max(test_acc.values()) - test_acc[tb_last],
                "val/train_bit_acc": train_bit_acc[train_val_iter],
                "val/val_bit_acc": val_bit_acc[train_val_iter],
                "val/hard_bit_acc": test_bit_acc[tb_last],
                "val/time": time.time() - start,
                **first_iter_converge 
            }, step=epoch)
            
        # check to see if we should save
        save_now = (epoch + 1) % cfg.problem.hyp.save_period == 0 or \
                   (epoch + 1) == final_epoch_exclusive or best_so_far
        if save_now:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
            out_str = f"model_{'best' if best_so_far else ''}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)

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
    main()
