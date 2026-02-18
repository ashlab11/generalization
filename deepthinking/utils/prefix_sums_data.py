""" prefix_sums_data.py
    Prefix sum related dataloaders

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import os
import torch
from torch.utils import data
from easy_to_hard_data import PrefixSumDataset

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_prefix_loader(train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, exact = True, shuffle=True):

    if exact:
        dataset = PrefixSumDataset("../../../data", num_bits=train_data)
    else:
        num_bit_list = range(16, train_data + 1) #prefix-sums starts at 16-bit
        datasets = []
        for bits in num_bit_list:
            datasets.append(PrefixSumDataset("../../../data", num_bits=bits)) 
        dataset = data.ConcatDataset(datasets) 
            
    testset = PrefixSumDataset("../../../data", num_bits=test_data)

    train_split = int(train_split * len(dataset))

    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [train_split,
                                                      int(len(dataset) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    num_workers = min(6, max(1, os.cpu_count() or 1))
    loader_settings = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    trainloader = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True, **loader_settings)
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    valloader = data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
