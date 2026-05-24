"""Encode TinyStories/OWT train+valid into uint16 token IDs (.npy)."""

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

NUM_CHUNKS = 200
SPECIAL_TOKENS = ["<|endoftext|>"]
SPLIT_TOKEN = b"<|endoftext|>"

JOBS = [
    # (output_name, data_filename, tokenizer_basename)
    ("tinystories_train", "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-train"),
    ("tinystories_valid", "TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train"),
    ("owt_train", "owt_train.txt", "owt_train"),
    ("owt_valid", "owt_valid.txt", "owt_train"),
]


_worker_tokenizer: Tokenizer | None = None


def _init_worker(vocab_path: str, merges_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=SPECIAL_TOKENS,
    )


def _encode_byte_range(args: tuple[str, int, int]) -> tuple[int, np.ndarray]:
    file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    assert _worker_tokenizer is not None
    ids = np.asarray(_worker_tokenizer.encode(text), dtype=np.uint16)
    return end - start, ids


def encode_file(data_path: Path, vocab_path: Path, merges_path: Path, out_path: Path) -> None:
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_CHUNKS, SPLIT_TOKEN)
    ranges = [(str(data_path), s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]
    n_total = len(ranges)
    total_size = data_path.stat().st_size
    print(f"  {n_total} chunks across {total_size:,} bytes", flush=True)

    chunks: list[np.ndarray] = []
    bytes_done = 0
    t0 = time.perf_counter()
    with mp.Pool(initializer=_init_worker, initargs=(str(vocab_path), str(merges_path))) as pool:
        for i, (n_bytes_this, ids) in enumerate(pool.imap(_encode_byte_range, ranges), 1):
            chunks.append(ids)
            bytes_done += n_bytes_this
            elapsed = time.perf_counter() - t0
            rate = bytes_done / elapsed if elapsed else 0.0
            remaining = (total_size - bytes_done) / rate if rate else 0.0
            print(
                f"    [{i:>3}/{n_total}] {bytes_done / total_size * 100:5.1f}% — "
                f"{rate / 1e6:5.2f} MB/s — ETA {remaining:6.0f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - t0

    ids_all = np.concatenate(chunks)
    np.save(out_path, ids_all)

    print(
        f"  -> {out_path.name}: {len(ids_all):,} tokens, "
        f"{total_size / len(ids_all):.2f} bytes/tok, "
        f"{elapsed:.1f}s elapsed, {total_size / elapsed / 1e6:.2f} MB/s",
        flush=True,
    )


def main() -> None:
    for name, data_name, tok_name in JOBS:
        data_path = DATA_DIR / data_name
        vocab_path = PROJECT_ROOT / f"{tok_name}-vocab.json"
        merges_path = PROJECT_ROOT / f"{tok_name}-merges.txt"
        out_path = DATA_DIR / f"{name}.npy"

        print(f"\n=== {name} ===")
        if out_path.exists():
            print(f"  skip: {out_path} already exists")
            continue
        encode_file(data_path, vocab_path, merges_path, out_path)


if __name__ == "__main__":
    main()
