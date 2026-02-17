""" mazes_data.py
    Maze related dataloaders

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import os
import shutil
import tarfile
import urllib.request as ur
import torch
from torch.utils import data
from easy_to_hard_data import MazeDataset

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611

MAZE_BASE_URL = "https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2"
MAZE_ROOT = "../../../data"


def _maze_files_exist(root, folder_name):
    inputs_path = os.path.join(root, folder_name, "inputs.npy")
    solutions_path = os.path.join(root, folder_name, "solutions.npy")
    return os.path.exists(inputs_path) and os.path.exists(solutions_path)


def _ensure_maze_split(root, train, size):
    folder_name = f"maze_data_{'train' if train else 'test'}_{size}"
    if _maze_files_exist(root, folder_name):
        return

    os.makedirs(root, exist_ok=True)
    archive_name = f"{folder_name}.tar.gz"
    archive_path = os.path.join(root, archive_name)
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
        url = f"{MAZE_BASE_URL}/{archive_name}"
        print(f"Downloading {url}")
        with ur.urlopen(url) as response, open(archive_path, "wb") as f:
            shutil.copyfileobj(response, f)

    with tarfile.open(archive_path) as archive:
        archive.extractall(root)

    if not _maze_files_exist(root, folder_name):
        raise RuntimeError(f"Maze data missing after extract: {folder_name}")


def prepare_maze_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    _ensure_maze_split(MAZE_ROOT, train=True, size=train_data)
    _ensure_maze_split(MAZE_ROOT, train=False, size=test_data)
    train_data = MazeDataset(MAZE_ROOT, train=True, size=train_data, download=False)
    testset = MazeDataset(MAZE_ROOT, train=False, size=test_data, download=False)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    num_workers = min(16, max(1, os.cpu_count() or 1))
    loader_settings = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    trainloader = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True, **loader_settings)
    valloader = data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
