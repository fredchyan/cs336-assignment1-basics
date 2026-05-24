import os
import regex as re
from collections import Counter, defaultdict
import argparse
import pathlib
from typing import BinaryIO
import multiprocessing
from functools import partial
import cProfile
import json
from cs336_basics.bpe_ops import merge_pair_in_pretoken, convert_string_to_tuple_of_utf8_bytes, get_pairs, PAT, bytes_to_unicode_str, special_tokens_pattern

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
NUM_CHUNKS_TO_PROCESS_AT_ONCE = 25


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(chunk: str, special_tokens: list[str]):
    pretoken_freq: Counter[tuple[bytes]] = Counter()
    # Read the text file
    # Split on special_tokens
    individual_docs = re.split(special_tokens_pattern(special_tokens), chunk)
    for doc in individual_docs:
        for m in re.finditer(PAT, doc):
            m_str = m.group(0)
            tuple_of_bytes = convert_string_to_tuple_of_utf8_bytes(m_str)
            pretoken_freq[tuple_of_bytes] += 1
    return pretoken_freq


def multiproc_pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 2000, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks = []
        pretoken_freq: Counter[tuple[bytes]] = Counter()
        last_end = boundaries[-1]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
            # Process NUM_CHUNKS_TO_PROCESS_AT_ONCE chunks at a time, this is to prevent memory usage from getting too large.
            if len(chunks) == NUM_CHUNKS_TO_PROCESS_AT_ONCE or end == last_end:
                with multiprocessing.Pool() as pool:
                    pretoken_freq_of_each_chunks = pool.map(partial(pretokenize, special_tokens=special_tokens), chunks)
                for curr_pretoken_freq in pretoken_freq_of_each_chunks:
                    pretoken_freq += curr_pretoken_freq
                chunks = []
    return pretoken_freq


def find_most_freq_pair(pair_freq: Counter[bytes]) -> tuple[bytes]:
    prev_v = 0
    pair_to_merge = None
    for k, v in pair_freq.items():
        if v > prev_v:
            prev_v = v
            pair_to_merge = k
        elif v == prev_v:
            # Take lexical largest
            if k > pair_to_merge:
                pair_to_merge = k
    return pair_to_merge


class Tracker:
    def __init__(self, pretoken_freq: Counter[tuple[bytes, ...]]):
        self.pretoken_freq: Counter[tuple[bytes, ...]] = pretoken_freq
        self.pair_freq: Counter[tuple[bytes, bytes]] = Counter()
        self.pair_to_pretokens: defaultdict[tuple[bytes], set[tuple[bytes]]] = defaultdict(set)
        for pretoken, p_count in pretoken_freq.items():
            for pair in get_pairs(pretoken):
                self.pair_freq[pair] += p_count
                self.pair_to_pretokens[pair].add(pretoken)

    def merge(self):
        pair_to_merge = find_most_freq_pair(self.pair_freq)
        new_pairs = []
        for pretoken in self.pair_to_pretokens[pair_to_merge]:
            if pretoken not in self.pretoken_freq:
                continue
            pretoken_count = self.pretoken_freq[pretoken]
            new_pretoken = merge_pair_in_pretoken(pretoken=pretoken, pair_to_merge=pair_to_merge)
            for old_pair in get_pairs(pretoken):
                self.pair_freq[old_pair] -= pretoken_count
            for new_pair in get_pairs(new_pretoken):
                self.pair_freq[new_pair] += pretoken_count
                self.pair_to_pretokens[new_pair].add(new_pretoken)
                new_pairs.append(new_pair)
            self.pretoken_freq[new_pretoken] += pretoken_count
            del self.pretoken_freq[pretoken]
        # Remove all pairs that are no longer in the pretokens,
        # for memory efficiency.
        self.pair_freq += Counter()
        return pair_to_merge


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab_list = [special.encode("utf-8") for special in special_tokens] + [bytes([i]) for i in range(256)]

    ## Run pre-tokenization on your chunk and store the counts for each pre-token
    pretoken_freq = multiproc_pretokenize(input_path=input_path, special_tokens=special_tokens)
    tracker = Tracker(pretoken_freq)
    merges = []
    while vocab_size > (len(vocab_list)):
        curr_merge = tracker.merge()
        merges.append(curr_merge)
        vocab_list.append(merges[-1][0] + merges[-1][1])
    vocab = {i: token for i, token in enumerate(vocab_list)}
    filename_without_extension, _ = os.path.splitext(os.path.basename(input_path))
    # Save vocab as JSON: {"unicode_encoded_token": token_id, ...}
    vocab_json = {bytes_to_unicode_str(token): idx for idx, token in vocab.items()}
    with open(f"{filename_without_extension}-vocab.json", "w") as f:
        json.dump(vocab_json, f)
    # Save merges as text: one merge per line, space-separated pair
    with open(f"{filename_without_extension}-merges.txt", "w") as f:
        for token1, token2 in merges:
            f.write(f"{bytes_to_unicode_str(token1)} {bytes_to_unicode_str(token2)}\n")
    return (vocab, merges)


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "dataset_type",
        help="Type of dataset to sample from.",
        choices=["tinystories_val", "tinystories_train", "owt_train"],
    )
    args = argp.parse_args()

    if args.dataset_type == "tinystories_val":
        train_bpe(
            input_path=DATA_PATH / "TinyStoriesV2-GPT4-valid.txt",
            vocab_size=1000,
            special_tokens=["<|endoftext|>"],
        )
    elif args.dataset_type == "tinystories_train":
        train_bpe(
            input_path=DATA_PATH / "TinyStoriesV2-GPT4-train.txt",
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
        )
    elif args.dataset_type == "owt_train":
        train_bpe(
            input_path=DATA_PATH / "owt_train.txt",
            vocab_size=32000,
            special_tokens=["<|endoftext|>"],
        )
    else:
        raise ValueError(f"Unknown dataset type in command line args: {args.dataset_type}")


if __name__ == "__main__":
    cProfile.run("main()")
