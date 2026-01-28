""" cellular_data.py
    1D cellular automaton dataloaders (Wolfram rules).
"""

import torch
from torch.utils import data

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def _rule_step(state, rule_num):
    left = torch.roll(state, shifts=1, dims=1)
    center = state
    right = torch.roll(state, shifts=-1, dims=1)
    pattern = (left << 2) | (center << 1) | right
    rule = torch.tensor(int(rule_num), device=state.device, dtype=torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(rule, pattern), 1).to(state.dtype)


def _evolve_rule(state, steps, max_steps, rule_num):
    final = state.clone()
    if max_steps == 0:
        return final
    current = state
    for step in range(1, max_steps + 1):
        current = _rule_step(current, rule_num)
        mask = steps == step
        if mask.any():
            final[mask] = current[mask]
    return final


class CellularDataset(data.Dataset):
    def __init__(self, num_samples, t_min, t_max, state_bits=24, rule_num=110, seed=42):
        generator = torch.Generator().manual_seed(seed)
        self.t_values = torch.randint(t_min, t_max + 1, (num_samples,), generator=generator, dtype=torch.int64)
        self.x0 = torch.randint(0, 2, (num_samples, state_bits), generator=generator, dtype=torch.int64)
        max_steps = int(t_max)
        self.xT = _evolve_rule(self.x0, self.t_values, max_steps, rule_num)
        t_scalar = self.t_values.unsqueeze(1).float()
        inputs_bits = torch.cat([t_scalar, self.x0.float() - 0.5], dim=1)
        ignore_t = torch.full_like(t_scalar, -100)
        targets_bits = torch.cat([ignore_t, self.xT], dim=1).long()
        self.inputs = inputs_bits.unsqueeze(1)
        self.targets = targets_bits


    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


def prepare_cellular_loader(train_batch_size, test_batch_size, train_data, test_data,
                            train_split=0.8, shuffle=True, train_samples=10000, test_samples=10000,
                            state_bits=24, test_t_min=None, exclude_t=None, rule_num=110):
    train_t_min = 0
    train_t_max = int(train_data)
    if test_t_min is None:
        test_t_min = train_t_max + 1
    test_t_max = int(test_data)

    trainset_full = CellularDataset(num_samples=train_samples,
                                    t_min=train_t_min,
                                    t_max=train_t_max,
                                    state_bits=state_bits,
                                    rule_num=rule_num,
                                    seed=42)
    testset = CellularDataset(num_samples=test_samples,
                              t_min=int(test_t_min),
                              t_max=test_t_max,
                              state_bits=state_bits,
                              rule_num=rule_num,
                              seed=123)

    if exclude_t:
        exclude = torch.as_tensor(exclude_t, device=trainset_full.t_values.device)
        keep = ~torch.isin(trainset_full.t_values, exclude)
        keep_idx = keep.nonzero(as_tuple=False).view(-1).tolist()
        trainset_full = torch.utils.data.Subset(trainset_full, keep_idx)
    train_split = int(train_split * len(trainset_full))
    trainset, valset = torch.utils.data.random_split(
        trainset_full,
        [train_split, int(len(trainset_full) - train_split)],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    valloader = data.DataLoader(valset, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


Rule110Dataset = CellularDataset
prepare_rule110_loader = prepare_cellular_loader
