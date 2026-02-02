""" chess_puzzles_data.py
    Lichess puzzle dataset loader without easy_to_hard_data dependency.

    Downloads and parses the Lichess Open Puzzle Database CSV and converts
    puzzle positions into (12, 8, 8) tensors with binary targets marking
    the from/to squares of the first solution move.
"""

import csv
import errno
import io
import os
import urllib.request as ur
from typing import Optional, Tuple

import torch
from torch.utils import data
import chess

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_PUZZLE_FILENAME = "lichess_db_puzzle.csv.zst"
LICHESS_PUZZLE_CSV = "lichess_db_puzzle.csv"


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


def _open_puzzle_stream(path: str):
    if path.endswith(".csv"):
        return open(path, "r", encoding="utf-8", newline="")
    if path.endswith(".bz2"):
        import bz2
        return bz2.open(path, "rt", encoding="utf-8", newline="")
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading .zst requires the 'zstandard' package. "
                "Either install it or decompress the file to .csv."
            ) from exc
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding="utf-8", newline="")
    raise ValueError(f"Unsupported puzzle file extension: {path}")


def _square_to_rc(square: int) -> Tuple[int, int]:
    return 7 - chess.square_rank(square), chess.square_file(square)


def _normalize_rc(row: int, col: int) -> Tuple[int, int]:
    return 7 - row, 7 - col


def _board_to_planes(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row, col = _square_to_rc(square)
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            idx += 6
        planes[idx, row, col] = 1.0
    return planes


def _flip_horizontal(planes: torch.Tensor) -> torch.Tensor:
    return torch.flip(planes, dims=[2])


def _normalize_position(
    planes: torch.Tensor,
    move_from: Tuple[int, int],
    move_to: Tuple[int, int],
    to_move_is_black: bool,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    if to_move_is_black:
        planes = torch.flip(planes, dims=[1, 2])
        planes = torch.cat([planes[6:12], planes[0:6]], dim=0)
        move_from = _normalize_rc(*move_from)
        move_to = _normalize_rc(*move_to)

    king_positions = (planes[5] > 0).nonzero(as_tuple=False)
    if king_positions.numel() > 0:
        king_col = int(king_positions[0, 1])
        if king_col < 4:
            planes = _flip_horizontal(planes)
            move_from = (move_from[0], 7 - move_from[1])
            move_to = (move_to[0], 7 - move_to[1])

    return planes, move_from, move_to


class LichessPuzzleDataset(data.Dataset):
    """Lichess puzzle dataset with (12, 8, 8) inputs and binary from/to targets."""

    def __init__(
        self,
        root: str,
        idx_start: int = 0,
        idx_end: Optional[int] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        normalize: bool = True,
        download: bool = True,
        cache: bool = True,
        max_samples: Optional[int] = None,
        puzzle_path: Optional[str] = None,
    ):
        self.root = root
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.normalize = normalize
        self.cache = cache
        self.max_samples = max_samples

        self.base_folder = "lichess_puzzles"
        self.raw_folder = os.path.join(self.base_folder, "raw")
        self.processed_folder = os.path.join(self.base_folder, "processed")

        if download:
            self.download()

        self.puzzle_path = puzzle_path or self._default_puzzle_path()
        if not os.path.exists(self.puzzle_path):
            raise FileNotFoundError(
                f"Puzzle file not found at {self.puzzle_path}. "
                "Provide puzzle_path or enable download."
            )

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

    def _default_puzzle_path(self) -> str:
        raw_root = os.path.join(self.root, self.raw_folder)
        zst_path = os.path.join(raw_root, LICHESS_PUZZLE_FILENAME)
        csv_path = os.path.join(raw_root, LICHESS_PUZZLE_CSV)
        return csv_path if os.path.exists(csv_path) else zst_path

    def _cache_name(self) -> str:
        end_tag = "end" if self.idx_end is None else str(self.idx_end)
        min_tag = "min" if self.min_rating is None else f"min{self.min_rating}"
        max_tag = "max" if self.max_rating is None else f"max{self.max_rating}"
        norm_tag = "norm" if self.normalize else "raw"
        sample_tag = "all" if self.max_samples is None else f"n{self.max_samples}"
        return f"puzzles_{self.idx_start}_{end_tag}_{min_tag}_{max_tag}_{norm_tag}_{sample_tag}.pt"

    def _iter_rows(self):
        with _open_puzzle_stream(self.puzzle_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    def _build_cache(self):
        inputs = []
        targets = []

        seen = 0
        kept = 0
        for row in self._iter_rows():
            if self.idx_end is not None and kept >= (self.idx_end - self.idx_start):
                break
            try:
                rating = int(row.get("Rating", 0))
            except ValueError:
                continue
            if self.min_rating is not None and rating < self.min_rating:
                continue
            if self.max_rating is not None and rating > self.max_rating:
                continue

            if seen < self.idx_start:
                seen += 1
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
                    planes,
                    move_from,
                    move_to,
                    to_move_is_black=board.turn == chess.BLACK,
                )

            target = torch.zeros((8, 8), dtype=torch.long)
            target[move_from[0], move_from[1]] = 1
            target[move_to[0], move_to[1]] = 1

            inputs.append(planes)
            targets.append(target)
            kept += 1
            seen += 1

            if self.max_samples is not None and kept >= self.max_samples:
                break

        if len(inputs) == 0:
            raise RuntimeError("Lichess puzzle dataset produced zero samples.")

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)
        return inputs, targets

    def __getitem__(self, index):
        return self.inputs[index].float(), self.targets[index].long()

    def __len__(self):
        return self.inputs.size(0)

    def download(self) -> None:
        raw_root = os.path.join(self.root, self.raw_folder)
        makedirs(raw_root)
        path = os.path.join(raw_root, LICHESS_PUZZLE_FILENAME)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print("Files already downloaded and verified")
            return
        download_url(LICHESS_PUZZLE_URL, raw_root)


def prepare_lichess_puzzle_loader(
    train_batch_size,
    test_batch_size,
    train_data,
    test_data,
    shuffle=True,
    train_max_rating: Optional[int] = None,
    test_min_rating: Optional[int] = None,
    **kwargs,
):
    train_kwargs = dict(kwargs)
    if train_max_rating is not None:
        train_kwargs["max_rating"] = train_max_rating

    test_kwargs = dict(kwargs)
    if test_min_rating is not None:
        test_kwargs["min_rating"] = test_min_rating

    trainset = LichessPuzzleDataset(
        "../../../data",
        idx_start=0,
        idx_end=train_data,
        **train_kwargs,
    )
    train_split = int(0.8 * len(trainset))
    trainset, valset = torch.utils.data.random_split(
        trainset,
        [train_split, int(len(trainset) - train_split)],
        generator=torch.Generator().manual_seed(42),
    )

    testset = LichessPuzzleDataset(
        "../../../data",
        idx_start=train_data,
        idx_end=train_data + test_data,
        **test_kwargs,
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
