""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import einops
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def _disable_diag(net, diag_state):
    diag_state["enabled"] = False
    for m in net.modules():
        if hasattr(m, "_compute_attn_entropy"):
            m._compute_attn_entropy = False
        if hasattr(m, "_compute_attn_stats"):
            m._compute_attn_stats = False
        if hasattr(m, "_compute_h_norm_ratio"):
            m._compute_h_norm_ratio = False
        if hasattr(m, "_compute_convergence"):
            m._compute_convergence = False


def test(net, loaders, mode, iters, problem, device, use_amp=False, return_bitwise=False):
    accs = []
    bit_accs = []
    diag_stats = defaultdict(list)
    hooks = []
    diag_state = {"enabled": True, "remaining_batches": 1, "track_h_norm_only": False, "h_norm_ratio_values": []}
    model = net.module if isinstance(net, torch.nn.DataParallel) else net
    is_compiled_runtime = bool(getattr(model, "compile", False)) or any(hasattr(m, "_orig_mod") for m in model.modules())
    if is_compiled_runtime:
        _disable_diag(net, diag_state)
        diag_state["track_h_norm_only"] = True
        model._compute_h_norm_ratio = True
    
    # Setup diagnostic hooks
    for m in net.modules():
        if not diag_state["enabled"]:
            break
        if hasattr(m, '__class__'):
            if 'AttentionBlock' in str(type(m)):
                def block_hook(module, input, output):
                    if not diag_state["enabled"]:
                        return
                    with torch.no_grad():
                        h = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
                        if h.dim() == 3:
                            token_norm = h.norm(dim=-1)
                            diag_stats['h_norm'].append(token_norm.mean().item())
                        else:
                            diag_stats['h_norm'].append(h.reshape(h.size(0), -1).norm(dim=-1).mean().item())
                        if h.dim() == 3:
                            h_norm = F.normalize(h, dim=-1)
                            sim = torch.einsum("bld,bmd->blm", h_norm, h_norm)
                            L = sim.size(-1)
                            off_diag = ~torch.eye(L, device=sim.device, dtype=torch.bool)
                            mean_corr = sim[:, off_diag].mean().item()
                            diag_stats['h_correlation'].append(mean_corr)
                hooks.append(m.register_forward_hook(block_hook))
            elif 'DTTransformer' in str(type(m)):
                m._compute_h_norm_ratio = True
                m._compute_convergence = True
                def norm_ratio_hook(module, input, output):
                    if not diag_state["enabled"]:
                        return
                    if hasattr(module, "_last_h_norm_ratio"):
                        diag_stats["h_norm_ratio"].append(module._last_h_norm_ratio)
                    if hasattr(module, "_convergence_cosine"):
                        diag_stats['convergence_cosine'].append(module._convergence_cosine)
                    if hasattr(module, "_first_convergence_iter"):
                        diag_stats['first_convergence_iter'].append(module._first_convergence_iter)
                hooks.append(m.register_forward_hook(norm_ratio_hook))
            elif 'MHA' in str(type(m)):
                m._compute_attn_stats = True
                def attn_hook(module, input, output):
                    if not diag_state["enabled"]:
                        return
                    if hasattr(module, '_last_attn_entropy'):
                        diag_stats['attn_entropy'].append(module._last_attn_entropy)
                    if hasattr(module, '_last_attn_max'):
                        diag_stats['attn_max'].append(module._last_attn_max)
                hooks.append(m.register_forward_hook(attn_hook))
    
    for loader in loaders:
        if mode == "default":
            if return_bitwise:
                accuracy, bit_accuracy = test_default(net, loader, iters, problem, device, use_amp, diag_state, return_bitwise=True)
                bit_accs.append(bit_accuracy)
            else:
                accuracy = test_default(net, loader, iters, problem, device, use_amp, diag_state)
        elif mode == "max_conf":
            if return_bitwise:
                accuracy, bit_accuracy = test_max_conf(net, loader, iters, problem, device, use_amp, diag_state, return_bitwise=True)
                bit_accs.append(bit_accuracy)
            else:
                accuracy = test_max_conf(net, loader, iters, problem, device, use_amp, diag_state)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)

    if len(diag_state["h_norm_ratio_values"]) > 0:
        diag_stats["h_norm_ratio"].extend(diag_state["h_norm_ratio_values"])
    
    # Cleanup hooks
    for h in hooks: h.remove()
    for m in net.modules():
        if hasattr(m, '_compute_attn_entropy'):
            m._compute_attn_entropy = False
        if hasattr(m, '_compute_attn_stats'):
            m._compute_attn_stats = False
        if hasattr(m, '_compute_h_norm_ratio'):
            m._compute_h_norm_ratio = False
        if hasattr(m, '_compute_convergence'):
            m._compute_convergence = False
    
    if return_bitwise:
        return accs, diag_stats, bit_accs
    return accs, diag_stats


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted


def test_default(net, testloader, iters, problem, device, use_amp=False, diag_state=None, return_bitwise=False):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters, device=device)
    bit_corrects = torch.zeros(max_iters, device=device)
    total = 0
    bit_total = torch.zeros((), device=device)

    autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.cuda.amp.autocast(enabled=False)

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            targets = targets.view(targets.size(0), -1)
            if return_bitwise and problem == "mazes":
                mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            with autocast_context:
                all_outputs = net(inputs, iters_to_do=max_iters)

            if diag_state is not None and diag_state.get("track_h_norm_only", False):
                model = net.module if isinstance(net, torch.nn.DataParallel) else net
                if hasattr(model, "_last_h_norm_ratio"):
                    diag_state["h_norm_ratio_values"].append(float(model._last_h_norm_ratio))

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                if problem in {"rule110", "cellular"}:
                    mask = targets != -100
                    eq = predicted == targets
                    eq = eq | (~mask)
                    corrects[i] += torch.amin(eq, dim=[1]).sum()
                    if return_bitwise:
                        bit_corrects[i] += eq[mask].sum()
                else:
                    corrects[i] += torch.amin(predicted == targets, dim=[1]).sum()
                    if return_bitwise:
                        if problem == "mazes":
                            #& mask instead of [mask] stops syncs
                            bit_corrects[i] += ((predicted == targets) & mask).sum()
                        else:
                            bit_corrects[i] += (predicted == targets).sum()

            total += targets.size(0)
            if return_bitwise:
                if problem == "mazes":
                    bit_total += mask.sum()
                elif problem in {"rule110", "cellular"}:
                    bit_total += (targets != -100).sum()
                else:
                    bit_total += targets.numel()
            if diag_state is not None and (diag_state["enabled"] or diag_state.get("track_h_norm_only", False)):
                diag_state["remaining_batches"] -= 1
                if diag_state["remaining_batches"] <= 0:
                    if diag_state["enabled"]:
                        _disable_diag(net, diag_state)
                    if diag_state.get("track_h_norm_only", False):
                        diag_state["track_h_norm_only"] = False
                        model = net.module if isinstance(net, torch.nn.DataParallel) else net
                        model._compute_h_norm_ratio = False

    accuracy = (100.0 * corrects / total).cpu()
    if return_bitwise:
        bit_total_value = bit_total.item()
        bit_accuracy = (100.0 * bit_corrects / bit_total).cpu() if bit_total_value > 0 else bit_corrects.cpu()
    ret_acc = {}
    ret_bit_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
        if return_bitwise:
            ret_bit_acc[ite] = bit_accuracy[ite-1].item()
    if return_bitwise:
        return ret_acc, ret_bit_acc
    return ret_acc


def test_max_conf(net, testloader, iters, problem, device, use_amp=False, diag_state=None, return_bitwise=False):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    bit_corrects = torch.zeros(max_iters).to(device)
    total = 0
    bit_total = torch.zeros((), device=device)
    softmax = torch.nn.functional.softmax

    autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else torch.cuda.amp.autocast(enabled=False)

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)
            if return_bitwise and problem == "mazes":
                mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            with autocast_context:
                all_outputs = net(inputs, iters_to_do=max_iters)

            if diag_state is not None and diag_state.get("track_h_norm_only", False):
                model = net.module if isinstance(net, torch.nn.DataParallel) else net
                if hasattr(model, "_last_h_norm_ratio"):
                    diag_state["h_norm_ratio_values"].append(float(model._last_h_norm_ratio))

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            if return_bitwise:
                bit_corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                if problem in {"rule110", "cellular"}:
                    rule_mask = targets != -100
                    eq = predicted == targets
                    eq_masked = eq | (~rule_mask)
                    corrects_array[i] = torch.amin(eq_masked, dim=[1])
                    if return_bitwise:
                        bit_corrects_array[i] = (eq & rule_mask).sum(dim=1)
                else:
                    corrects_array[i] = torch.amin(predicted == targets, dim=[1])
                    if return_bitwise:
                        if problem == "mazes":
                            bit_corrects_array[i] = (predicted == targets)[mask].sum(dim=1)
                        else:
                            bit_corrects_array[i] = (predicted == targets).sum(dim=1)

            best_idx = torch.cummax(confidence_array, dim=0)[1]
            correct_this_iter = corrects_array[best_idx, torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)
            if return_bitwise:
                bit_correct_this_iter = bit_corrects_array[best_idx, torch.arange(bit_corrects_array.size(1))]
                bit_corrects += bit_correct_this_iter.sum(dim=1)
            if diag_state is not None and (diag_state["enabled"] or diag_state.get("track_h_norm_only", False)):
                diag_state["remaining_batches"] -= 1
                if diag_state["remaining_batches"] <= 0:
                    if diag_state["enabled"]:
                        _disable_diag(net, diag_state)
                    if diag_state.get("track_h_norm_only", False):
                        diag_state["track_h_norm_only"] = False
                        model = net.module if isinstance(net, torch.nn.DataParallel) else net
                        model._compute_h_norm_ratio = False
            if return_bitwise:
                if problem == "mazes":
                    bit_total += mask.sum()
                elif problem in {"rule110", "cellular"}:
                    bit_total += (targets != -100).sum()
                else:
                    bit_total += targets.numel()

    accuracy = 100 * corrects.long().cpu() / total
    if return_bitwise:
        bit_total_value = bit_total.item()
        bit_accuracy = 100.0 * bit_corrects.float().cpu() / bit_total_value if bit_total_value > 0 else bit_corrects.float().cpu()
    ret_acc = {}
    ret_bit_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
        if return_bitwise:
            ret_bit_acc[ite] = bit_accuracy[ite-1].item()
    if return_bitwise:
        return ret_acc, ret_bit_acc
    return ret_acc
