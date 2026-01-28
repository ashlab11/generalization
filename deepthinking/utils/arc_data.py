""" arc_data.py
    ARC-AGI-1 dataset loader with optional precomputed augmentations.

    Mirrors the easy_to_hard_data dataset style:
    - download once
    - cache processed tensors
    - __getitem__ returns (input, target)
"""

import errno
import json
import os
import os.path
import tarfile
import urllib.request as ur
from typing import Optional, Callable, List
import random
import hashlib

import torch
from torch.utils import data

ARC_AGI_TARBALL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.tar.gz"
ARC_AGI_FOLDER = "ARC-AGI-master"
ARC_AGI_DATA_DIR = "data"
MAX_ARC_SIZE = 30


def extract_tar(path, folder):
    tar = tarfile.open(path)
    tar.extractall(folder)
    tar.close()


def download_url(url, folder):
    filename = url.rpartition("/")[2]
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("Using existing file", filename)
        return path
    print("Downloading", url)
    makedirs(folder)

    data = ur.urlopen(url)
    size = int(data.info()["Content-Length"])
    chunk_size = 1024 * 1024
    num_iter = int(size / chunk_size) + 2

    downloaded_size = 0
    try:
        with open(path, "wb") as f:
            for _ in range(num_iter):
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                downloaded_size += len(chunk)
                f.write(chunk)
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError("Stopped downloading due to interruption.") from exc

    return path


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def _pad_grid(grid: torch.Tensor, pad_size: int, pad_value: int = 0) -> torch.Tensor:
    h, w = grid.shape
    padded = torch.full((pad_size, pad_size), pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def _load_json_tasks(tasks_dir: str) -> List[str]:
    files = [f for f in os.listdir(tasks_dir) if f.endswith(".json")]
    files.sort()
    return [os.path.join(tasks_dir, f) for f in files]


def _color_permutations(count: int, seed: int = 0) -> List[List[int]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    perms = [list(range(10))]
    while len(perms) < count:
        perm = list(range(10))
        rng.shuffle(perm)
        if perm not in perms:
            perms.append(perm)
    return perms


def _apply_color_perm(grid: torch.Tensor, perm: List[int]) -> torch.Tensor:
    perm_tensor = torch.tensor(perm, dtype=torch.long)
    return perm_tensor[grid.long()].to(grid.dtype)

def _hash_pair(inp: torch.Tensor, tgt: torch.Tensor) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(inp.contiguous().cpu().numpy().tobytes())
    h.update(tgt.contiguous().cpu().numpy().tobytes())
    return h.digest()


class ArcAgiDataset(data.Dataset):
    """ARC-AGI-1 dataset with optional precomputed augmentations."""

    def __init__(
        self,
        root: str,
        split: str = "training",
        include_test_pairs: bool = False,
        pad_size: int = MAX_ARC_SIZE,
        pad_value: int = 0,
        augment_rot90: bool = True,
        color_perm_count: int = 10,
        precompute: bool = True,
        max_tasks: Optional[int] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 0,
    ):
        self.root = root
        self.split = split
        self.include_test_pairs = include_test_pairs
        self.pad_size = pad_size
        self.pad_value = pad_value
        self.augment_rot90 = augment_rot90
        self.color_perm_count = color_perm_count
        self.precompute = precompute
        self.max_tasks = max_tasks
        self.transform = transform
        self.seed = seed

        self.base_folder = "arc_agi"
        self.raw_folder = os.path.join(self.base_folder, "raw")
        self.processed_folder = os.path.join(self.base_folder, "processed")

        if download:
            self.download()

        if precompute:
            cache_name = self._cache_name()
            cache_path = os.path.join(self.root, self.processed_folder, cache_name)
            if os.path.exists(cache_path):
                cached = torch.load(cache_path)
                self.inputs = cached["inputs"]
                self.targets = cached["targets"]
            else:
                self.inputs, self.targets = self._build_cache()
                makedirs(os.path.join(self.root, self.processed_folder))
                torch.save({"inputs": self.inputs, "targets": self.targets}, cache_path)
        else:
            self.inputs, self.targets = self._build_cache()

    def _cache_name(self) -> str:
        rot_tag = "rot90" if self.augment_rot90 else "norot"
        perm_tag = f"perm{self.color_perm_count}"
        test_tag = "withtest" if self.include_test_pairs else "notest"
        max_tag = f"max{self.max_tasks}" if self.max_tasks is not None else "all"
        return f"{self.split}_{rot_tag}_{perm_tag}_{test_tag}_{max_tag}_pad{self.pad_size}.pt"

    def _raw_data_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder, ARC_AGI_FOLDER, ARC_AGI_DATA_DIR, self.split)

    def _build_cache(self):
        tasks_dir = self._raw_data_dir()
        task_files = _load_json_tasks(tasks_dir)
        if self.max_tasks is not None:
            task_files = task_files[: self.max_tasks]

        perms = _color_permutations(self.color_perm_count, seed=self.seed)
        if not perms:
            perms = [list(range(10))]

        inputs = []
        targets = []

        for task_path in task_files:
            with open(task_path, "r", encoding="utf-8") as f:
                task = json.load(f)
            pairs = list(task.get("train", []))
            if self.include_test_pairs:
                pairs.extend(task.get("test", []))

            for pair in pairs:
                in_grid = torch.tensor(pair["input"], dtype=torch.uint8)
                out_grid = torch.tensor(pair["output"], dtype=torch.uint8)

                rotations = [0, 1, 2, 3] if self.augment_rot90 else [0]
                for k in rotations:
                    in_rot = torch.rot90(in_grid, k=k, dims=(0, 1)) if k else in_grid
                    out_rot = torch.rot90(out_grid, k=k, dims=(0, 1)) if k else out_grid
                    for perm in perms:
                        in_aug = _apply_color_perm(in_rot, perm)
                        out_aug = _apply_color_perm(out_rot, perm)

                        in_pad = _pad_grid(in_aug, self.pad_size, self.pad_value)
                        out_pad = _pad_grid(out_aug, self.pad_size, self.pad_value)

                        if self.transform is not None:
                            stacked = torch.stack([in_pad, out_pad], dim=0)
                            stacked = self.transform(stacked)
                            in_pad = stacked[0].to(torch.uint8)
                            out_pad = stacked[1].to(torch.uint8)

                        inputs.append(in_pad)
                        targets.append(out_pad)

        if len(inputs) == 0:
            raise RuntimeError("ARC-AGI dataset produced zero samples.")

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)
        return inputs, targets

    def __getitem__(self, index):
        return self.inputs[index].long(), self.targets[index].long()

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        data_dir = self._raw_data_dir()
        return os.path.exists(data_dir)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        raw_root = os.path.join(self.root, self.raw_folder)
        makedirs(raw_root)
        path = download_url(ARC_AGI_TARBALL, raw_root)
        extract_tar(path, raw_root)
        os.unlink(path)


def prepare_arc_loader(
    train_batch_size,
    test_batch_size,
    train_data=None,
    test_data=None,
    shuffle=True,
    dedupe_test=True,
    **kwargs,
):
    trainset = ArcAgiDataset(
        "../../../data",
        split="training",
        max_tasks=train_data,
        **kwargs,
    )
    testset = ArcAgiDataset(
        "../../../data",
        split="evaluation",
        max_tasks=test_data,
        **kwargs,
    )

    if dedupe_test:
        train_hashes = {_hash_pair(trainset.inputs[i], trainset.targets[i]) for i in range(len(trainset))}
        keep_indices = []
        for i in range(len(testset)):
            if _hash_pair(testset.inputs[i], testset.targets[i]) not in train_hashes:
                keep_indices.append(i)
        if len(keep_indices) != len(testset):
            testset = data.Subset(testset, keep_indices)

    trainloader = data.DataLoader(
        trainset,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    valloader = data.DataLoader(
        testset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    testloader = data.DataLoader(
        testset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loaders = {"train": trainloader, "test": testloader, "val": valloader}
    return loaders
