import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

PILE_BYTES = 825 * 1024**3  # 825 GiB

SEED = 0
NUM_DOCS = 10
DOC_SEP = b"<|endoftext|>"
READ_CHUNK = 64 * 1024  # I/O buffer for forward scans; size is not load-bearing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CORPORA = {
    "tinystories": {
        "data": DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        "vocab": PROJECT_ROOT / "TinyStoriesV2-GPT4-train-vocab.json",
        "merges": PROJECT_ROOT / "TinyStoriesV2-GPT4-train-merges.txt",
    },
    "owt": {
        "data": DATA_DIR / "owt_train.txt",
        "vocab": PROJECT_ROOT / "owt_train-vocab.json",
        "merges": PROJECT_ROOT / "owt_train-merges.txt",
    },
}


def sample_docs(path: Path, num_docs: int, seed: int) -> list[str]:
    """Sample `num_docs` documents by seeking to random byte offsets and
    taking the next full <|endoftext|>-delimited document. Sampling is
    proportional to document length, which is fine for a quick sanity check."""
    rng = random.Random(seed)
    size = path.stat().st_size
    docs: list[str] = []
    with open(path, "rb") as f:
        while len(docs) < num_docs:
            f.seek(rng.randrange(size))
            buf = b""
            # Skip past the in-progress doc to the next boundary.
            while DOC_SEP not in buf:
                chunk = f.read(READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
            if DOC_SEP not in buf:
                continue  # landed past the last separator; resample
            buf = buf.split(DOC_SEP, 1)[1]
            # Read until the end of this doc.
            while DOC_SEP not in buf:
                chunk = f.read(READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
            if DOC_SEP not in buf:
                continue  # trailing partial doc with no closing separator; resample
            docs.append(buf.split(DOC_SEP, 1)[0].decode("utf-8", errors="replace"))
    return docs


def main():
    tokenizers = {
        name: Tokenizer.from_files(
            vocab_filepath=str(paths["vocab"]),
            merges_filepath=str(paths["merges"]),
            special_tokens=["<|endoftext|>"],
        )
        for name, paths in CORPORA.items()
    }
    samples = {name: sample_docs(paths["data"], NUM_DOCS, SEED) for name, paths in CORPORA.items()}

    for corpus_name, docs in samples.items():
        for tok_name, tokenizer in tokenizers.items():
            print(f"\n=== corpus={corpus_name} tokenizer={tok_name} ===")
            total_bytes = 0
            total_tokens = 0
            elapsed = 0.0
            for i, doc in enumerate(docs):
                n_bytes = len(doc.encode("utf-8"))
                t0 = time.perf_counter()
                ids = tokenizer.encode(doc)
                elapsed += time.perf_counter() - t0
                ratio = n_bytes / len(ids) if ids else 0.0
                print(f"  doc {i}: {n_bytes} bytes, {len(ids)} tokens, {ratio:.2f} bytes/tok")
                total_bytes += n_bytes
                total_tokens += len(ids)
            overall = total_bytes / total_tokens if total_tokens else 0.0
            throughput = total_bytes / elapsed if elapsed else 0.0
            pile_seconds = PILE_BYTES / throughput if throughput else float("inf")
            print(f"  TOTAL : {total_bytes} bytes, {total_tokens} tokens, {overall:.2f} bytes/tok")
            print(
                f"  THRU  : {elapsed:.2f}s elapsed, {throughput/1e6:.3f} MB/s — "
                f"Pile (825 GiB) ≈ {pile_seconds/3600:.1f} h ({pile_seconds/86400:.2f} days)"
            )


if __name__ == "__main__":
    main()
