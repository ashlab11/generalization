"""Minimal Sudoku loader with optional difficulty bands and sample caps."""

import csv
import os
import urllib.request as ur

import numpy as np
import torch
from torch.utils import data

SUDOKU_TRAIN_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv"
SUDOKU_TEST_URL = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv"


def _loader_settings():
    num_workers = min(6, max(1, os.cpu_count() or 1))
    return {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}


def _download(url, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, url.rpartition("/")[2])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with ur.urlopen(url) as src, open(path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return path


def _parse_grid(text):
    if len(text) != 81:
        raise ValueError(f"Expected 81-char grid, got {len(text)}")
    vals = []
    for ch in text:
        vals.append(0 if ch in {".", "0"} else int(ch))
    return torch.tensor(vals, dtype=torch.long).view(9, 9)


def _difficulty_bounds(csv_path):
    ratings = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                ratings.append(float(row.get("rating", row.get("Rating"))))
            except (TypeError, ValueError):
                continue
    if not ratings:
        raise RuntimeError("Could not read Sudoku ratings.")

    q20, q40, q60, q80 = np.quantile(np.array(ratings), [0.2, 0.4, 0.6, 0.8]).tolist()
    return {
        "easy": (-float("inf"), q20),
        "medium": (q20, q40),
        "hard": (q40, q60),
        "expert": (q60, q80),
        "extreme": (q80, float("inf")),
    }


class SudokuExtremeDataset(data.Dataset):
    # Difficulty is a single band: easy/medium/hard/expert/extreme.
    def __init__(
        self,
        root,
        split="train",
        difficulty=None,
        difficulty_bounds=None,
        min_rating=None,
        max_rating=None,
        max_samples=None,
        one_hot=True,
        download=True,
        cumulative_difficulty=False,
    ):
        self.root = root
        self.split = split
        self.difficulty = difficulty
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.max_samples = max_samples
        self.one_hot = one_hot
        self.cumulative_difficulty = cumulative_difficulty

        raw_dir = os.path.join(root, "sudoku_extreme", "raw")
        if download:
            _download(SUDOKU_TRAIN_URL, raw_dir)
            _download(SUDOKU_TEST_URL, raw_dir)

        file_name = "train.csv" if split == "train" else "test.csv"
        self.csv_path = os.path.join(raw_dir, file_name)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Sudoku CSV not found: {self.csv_path}")

        self.bounds = difficulty_bounds if difficulty is not None else None
        self.inputs, self.targets = self._build_tensors()

    def _keep_row(self, rating):
        if self.min_rating is not None and rating < self.min_rating:
            return False
        if self.max_rating is not None and rating > self.max_rating:
            return False
        if self.difficulty is None:
            return True
        if self.difficulty not in self.bounds:
            raise ValueError(f"Unknown difficulty '{self.difficulty}'")
        low, high = self.bounds[self.difficulty]
        if self.cumulative_difficulty:
            # Keep everything up to the selected level.
            return rating < high
        return low <= rating < high

    def _build_tensors(self):
        inputs = []
        targets = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                question = row.get("question", row.get("Question"))
                answer = row.get("answer", row.get("Answer"))
                if not question or not answer:
                    continue
                try:
                    rating = int(row.get("rating", row.get("Rating", 0)))
                except (TypeError, ValueError):
                    continue
                if not self._keep_row(rating):
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

        if not inputs:
            raise RuntimeError("No Sudoku samples found with current filters.")

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0).long()
        if self.one_hot:
            inputs = torch.nn.functional.one_hot(inputs.long(), num_classes=10).permute(0, 3, 1, 2).float()
        return inputs, targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


def prepare_sudoku_loader(
    train_batch_size,
    test_batch_size,
    train_data,
    test_data,
    shuffle=True,
    max_train_samples=None,
    max_test_samples=None,
    **kwargs,
):
    def _normalize_level(value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"top20", "top_20", "top-20"}:
                return "extreme"
            if value.isdigit():
                return int(value)
        return value

    train_data = _normalize_level(train_data)
    test_data = _normalize_level(test_data)

    band_levels = {"easy", "medium", "hard", "expert", "extreme", None}
    using_band_levels = train_data in band_levels and test_data in band_levels
    using_int_levels = isinstance(train_data, int) and isinstance(test_data, int)
    if not using_band_levels and not using_int_levels:
        raise ValueError(
            "Use named levels (easy/medium/hard/expert/extreme/top20) for both train/test, "
            "or integer levels for both train/test."
        )

    # Ignore unrelated problem config keys (for example, model settings).
    dataset_kwargs = {}
    for key in ["one_hot", "download"]:
        if key in kwargs:
            dataset_kwargs[key] = kwargs[key]

    root = "../../../data"
    raw_dir = os.path.join(root, "sudoku_extreme", "raw")
    if dataset_kwargs.get("download", True):
        _download(SUDOKU_TRAIN_URL, raw_dir)
        _download(SUDOKU_TEST_URL, raw_dir)
    train_csv_path = os.path.join(raw_dir, "train.csv")

    train_kwargs = {}
    test_kwargs = {}
    if using_band_levels:
        # Always define split thresholds from train.csv for consistent train/test meaning.
        bounds = _difficulty_bounds(train_csv_path)
        train_kwargs = {"difficulty": train_data, "difficulty_bounds": bounds, "cumulative_difficulty": True}
        test_kwargs = {"difficulty": test_data, "difficulty_bounds": bounds}
    else:
        # Integer levels: train on one exact level, test on strictly higher levels.
        train_kwargs = {"min_rating": train_data, "max_rating": train_data}
        test_kwargs = {"min_rating": train_data + 1, "max_rating": test_data}

    trainset_full = SudokuExtremeDataset(
        root,
        split="train",
        max_samples=max_train_samples,
        **train_kwargs,
        **dataset_kwargs,
    )
    train_len = int(0.8 * len(trainset_full))
    trainset, valset = torch.utils.data.random_split(
        trainset_full,
        [train_len, int(len(trainset_full) - train_len)],
        generator=torch.Generator().manual_seed(42),
    )
    testset = SudokuExtremeDataset(
        root,
        split="test",
        max_samples=max_test_samples,
        **test_kwargs,
        **dataset_kwargs,
    )

    loader_settings = _loader_settings()
    trainloader = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True, **loader_settings)
    valloader = data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    return {"train": trainloader, "test": testloader, "val": valloader}
