"""Minimal Lichess puzzle loader with rating-based train/test filtering."""

import csv
import io
import os
import urllib.request as ur

import chess
import torch
from torch.utils import data

from .repo_paths import repo_data_dir

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_PUZZLE_FILENAME = "lichess_db_puzzle.csv.zst"
LICHESS_PUZZLE_CSV = "lichess_db_puzzle.csv"


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


def _open_csv_stream(path):
    if path.endswith(".csv"):
        return open(path, "r", encoding="utf-8", newline="")
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Install zstandard to read .zst chess files.") from exc
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", newline="")
    raise ValueError(f"Unsupported chess puzzle file extension: {path}")


def _square_to_rc(square):
    return 7 - chess.square_rank(square), chess.square_file(square)


def _normalize_rc(row, col):
    return 7 - row, 7 - col


def _board_to_planes(board):
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row, col = _square_to_rc(square)
        idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        planes[idx, row, col] = 1.0
    return planes


def _normalize_position(
    planes,
    move_from,
    move_to,
    to_move_is_black,
):
    # Canonicalize orientation so "side to move" is always white.
    if to_move_is_black:
        planes = torch.flip(planes, dims=[1, 2])
        planes = torch.cat([planes[6:12], planes[0:6]], dim=0)
        move_from = _normalize_rc(*move_from)
        move_to = _normalize_rc(*move_to)
    king_positions = (planes[5] > 0).nonzero(as_tuple=False)
    if king_positions.numel() > 0 and int(king_positions[0, 1]) < 4:
        planes = torch.flip(planes, dims=[2])
        move_from = (move_from[0], 7 - move_from[1])
        move_to = (move_to[0], 7 - move_to[1])
    return planes, move_from, move_to


class LichessPuzzleDataset(data.Dataset):
    def __init__(
        self,
        root,
        min_rating=None,
        max_rating=None,
        max_samples=None,
        normalize=True,
        download=True,
        puzzle_path=None,
    ):
        self.root = root
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.max_samples = max_samples
        self.normalize = normalize
        self.raw_folder = os.path.join(root, "lichess_puzzles", "raw")

        if download:
            _download(LICHESS_PUZZLE_URL, self.raw_folder)
        self.puzzle_path = puzzle_path or self._default_path()
        self.inputs, self.targets = self._build_tensors()

    def _default_path(self):
        csv_path = os.path.join(self.raw_folder, LICHESS_PUZZLE_CSV)
        if os.path.exists(csv_path):
            return csv_path
        return os.path.join(self.raw_folder, LICHESS_PUZZLE_FILENAME)

    def _keep_rating(self, rating):
        if self.min_rating is not None and rating < self.min_rating:
            return False
        if self.max_rating is not None and rating > self.max_rating:
            return False
        return True

    def _build_tensors(self):
        inputs = []
        targets = []
        with _open_csv_stream(self.puzzle_path) as f:
            for row in csv.DictReader(f):
                try:
                    rating = int(row.get("Rating", 0))
                except ValueError:
                    continue
                if not self._keep_rating(rating):
                    continue

                fen = row.get("FEN")
                moves = row.get("Moves", "").split()
                if not fen or len(moves) < 2:
                    continue
                try:
                    board = chess.Board(fen)
                    first_move = chess.Move.from_uci(moves[0])
                    if first_move not in board.legal_moves:
                        continue
                    board.push(first_move)
                    solution_move = chess.Move.from_uci(moves[1])
                    if solution_move not in board.legal_moves:
                        continue
                except Exception:
                    continue

                planes = _board_to_planes(board)
                move_from = _square_to_rc(solution_move.from_square)
                move_to = _square_to_rc(solution_move.to_square)
                if self.normalize:
                    planes, move_from, move_to = _normalize_position(
                        planes, move_from, move_to, to_move_is_black=board.turn == chess.BLACK
                    )

                target = torch.zeros((8, 8), dtype=torch.long)
                target[move_from[0], move_from[1]] = 1
                target[move_to[0], move_to[1]] = 1
                inputs.append(planes)
                targets.append(target)

                if self.max_samples is not None and len(inputs) >= self.max_samples:
                    break

        if not inputs:
            raise RuntimeError("No chess samples found with the current filters.")
        return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)

    def __getitem__(self, index):
        return self.inputs[index].float(), self.targets[index].long()

    def __len__(self):
        return self.inputs.size(0)


def prepare_lichess_puzzle_loader(
    train_batch_size,
    test_batch_size,
    train_data, #max train rating
    test_data, #max test rating
    shuffle=True,
    max_train_samples=None,
    max_test_samples=None,
    **kwargs,
):
    # Train on easier puzzles (<= X), test on harder puzzles (>= Y).

    trainset_full = LichessPuzzleDataset(
        repo_data_dir(),
        max_rating=train_data,
        max_samples=max_train_samples,
        **kwargs,
    )
    train_split = int(0.8 * len(trainset_full))
    trainset, valset = torch.utils.data.random_split(
        trainset_full,
        [train_split, int(len(trainset_full) - train_split)],
        generator=torch.Generator().manual_seed(42),
    )
    testset = LichessPuzzleDataset(
        repo_data_dir(),
        min_rating=train_data + 1,
        max_rating=test_data,
        max_samples=max_test_samples,
        **kwargs,
    )

    loader_settings = _loader_settings()
    trainloader = data.DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True, **loader_settings)
    valloader = data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, drop_last=False, **loader_settings)
    return {"train": trainloader, "test": testloader, "val": valloader}
