""" sudoku_data.py
    Sudoku (easy to extreme) dataset loader using the sapientinc/sudoku-extreme CSVs.
"""

import csv
import errno
import os
import urllib.request as ur
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch.utils import data

SUDOKU_EXTREME_TRAIN_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv"
SUDOKU_EXTREME_TEST_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv"


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def download_url(url, folder):
    filename = url.rpartition("/")[2]
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("Using existing file", filename)
        return path
    print("Downloading", url)
    makedirs(folder)

    data = ur.urlopen(url)
    size = int(data.info().get("Content-Length", 0))
    chunk_size = 1024 * 1024
    num_iter = int(size / chunk_size) + 2 if size else None

    downloaded_size = 0
    try:
        with open(path, "wb") as f:
            if num_iter is None:
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break
                    downloaded_size += len(chunk)
                    f.write(chunk)
            else:
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


def _parse_grid(grid_str: str) -> torch.Tensor:
    if len(grid_str) != 81:
        raise ValueError(f"Expected 81-char grid, got {len(grid_str)}")
    vals = []
    for ch in grid_str:
        if ch in {".", "0"}:
            vals.append(0)
        else:
            vals.append(int(ch))
    return torch.tensor(vals, dtype=torch.long).view(9, 9)


def _estimate_rating_quantiles(
    csv_path: str,
    quantiles=(0.2, 0.4, 0.6, 0.8),
    sample_size: int = 200000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    sample = []
    seen = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rating = float(row.get("rating", row.get("Rating")))
            except (TypeError, ValueError):
                continue
            seen += 1
            if len(sample) < sample_size:
                sample.append(rating)
            else:
                j = rng.integers(0, seen)
                if j < sample_size:
                    sample[j] = rating
    if not sample:
        raise RuntimeError("Could not estimate rating quantiles from dataset.")
    return tuple(np.quantile(np.array(sample), quantiles).tolist())


def _difficulty_bounds_from_quantiles(qs: Tuple[float, float, float, float]) -> Dict[str, Tuple[float, float]]:
    q1, q2, q3, q4 = qs
    return {
        "easy": (-float("inf"), q1),
        "medium": (q1, q2),
        "hard": (q2, q3),
        "expert": (q3, q4),
        "extreme": (q4, float("inf")),
    }


class SudokuExtremeDataset(data.Dataset):
    """Sudoku dataset with optional difficulty filtering based on rating quantiles."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        difficulty: Optional[str] = None,
        difficulty_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        download: bool = True,
        cache: bool = True,
        max_samples: Optional[int] = None,
        one_hot: bool = True,
        quantile_sample_size: int = 200000,
        seed: int = 0,
    ):
        self.root = root
        self.split = split
        self.difficulty = difficulty
        self.cache = cache
        self.max_samples = max_samples
        self.one_hot = one_hot
        self.quantile_sample_size = quantile_sample_size
        self.seed = seed

        self.base_folder = "sudoku_extreme"
        self.raw_folder = os.path.join(self.base_folder, "raw")
        self.processed_folder = os.path.join(self.base_folder, "processed")

        if download:
            self.download()

        self.csv_path = self._split_path()
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Sudoku CSV not found at {self.csv_path}. "
                "Provide the file or enable download."
            )

        self.difficulty_bounds = difficulty_bounds
        if self.difficulty is not None and self.difficulty_bounds is None:
            qs = _estimate_rating_quantiles(
                self.csv_path,
                sample_size=self.quantile_sample_size,
                seed=self.seed,
            )
            self.difficulty_bounds = _difficulty_bounds_from_quantiles(qs)

        if cache:
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

    def _split_path(self) -> str:
        raw_root = os.path.join(self.root, self.raw_folder)
        filename = "train.csv" if self.split == "train" else "test.csv"
        return os.path.join(raw_root, filename)

    def _cache_name(self) -> str:
        diff_tag = self.difficulty or "all"
        sample_tag = "all" if self.max_samples is None else f"n{self.max_samples}"
        oh_tag = "oh" if self.one_hot else "int"
        return f"{self.split}_{diff_tag}_{sample_tag}_{oh_tag}.pt"

    def _row_in_difficulty(self, rating: float) -> bool:
        if self.difficulty is None:
            return True
        if not self.difficulty_bounds or self.difficulty not in self.difficulty_bounds:
            raise ValueError(f"Unknown difficulty '{self.difficulty}'.")
        low, high = self.difficulty_bounds[self.difficulty]
        return low <= rating < high

    def _build_cache(self):
        inputs = []
        targets = []

        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("question", row.get("Question"))
                answer = row.get("answer", row.get("Answer"))
                if not question or not answer:
                    continue
                try:
                    rating = float(row.get("rating", row.get("Rating", 0)))
                except (TypeError, ValueError):
                    rating = 0.0

                if not self._row_in_difficulty(rating):
                    continue

                try:
                    inp = _parse_grid(question)
                    tgt = _parse_grid(answer)
                except ValueError:
                    continue

                inputs.append(inp)
                targets.append(tgt)

                if self.max_samples is not None and len(inputs) >= self.max_samples:
                    break

        if len(inputs) == 0:
            raise RuntimeError("Sudoku dataset produced zero samples.")

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)

        if self.one_hot:
            # Encode digits 0-9 into 10 channels.
            inputs = torch.nn.functional.one_hot(inputs.long(), num_classes=10).permute(0, 3, 1, 2).float()

        return inputs, targets

    def __getitem__(self, index):
        inp = self.inputs[index]
        if self.one_hot:
            inp = inp.float()
        return inp, self.targets[index].long()

    def __len__(self):
        return self.inputs.size(0)

    def download(self) -> None:
        raw_root = os.path.join(self.root, self.raw_folder)
        makedirs(raw_root)
        train_path = os.path.join(raw_root, "train.csv")
        test_path = os.path.join(raw_root, "test.csv")
        if not os.path.exists(train_path):
            download_url(SUDOKU_EXTREME_TRAIN_URL, raw_root)
        if not os.path.exists(test_path):
            download_url(SUDOKU_EXTREME_TEST_URL, raw_root)


def prepare_sudoku_loader(
    train_batch_size,
    test_batch_size,
    train_data=None,
    test_data=None,
    shuffle=True,
    train_max_difficulty: Optional[str] = "expert",
    test_difficulty: Optional[str] = "extreme",
    **kwargs,
):
    if train_max_difficulty not in {"easy", "medium", "hard", "expert", None}:
        raise ValueError("train_max_difficulty must be one of: easy, medium, hard, expert, None")

    trainset = SudokuExtremeDataset(
        "../../../data",
        split="train",
        difficulty=train_max_difficulty,
        max_samples=train_data,
        **kwargs,
    )
    train_split = int(0.8 * len(trainset))
    trainset, valset = torch.utils.data.random_split(
        trainset,
        [train_split, int(len(trainset) - train_split)],
        generator=torch.Generator().manual_seed(42),
    )

    testset = SudokuExtremeDataset(
        "../../../data",
        split="test",
        difficulty=test_difficulty,
        max_samples=test_data,
        **kwargs,
    )

    trainloader = data.DataLoader(
        trainset,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    valloader = data.DataLoader(
        valset,
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
