from collections import defaultdict
from collections.abc import Iterable, Iterator
import json
import regex as re
from cs336_basics.bpe_ops import (
    merge_pair_in_pretoken,
    convert_string_to_tuple_of_utf8_bytes,
    get_pairs,
    PAT,
    special_tokens_pattern,
    unicode_str_to_bytes,
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary,
        list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_utf8_bytes_to_idx_int = {utf8_bytes: idx_int for idx_int, utf8_bytes in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Classmethod that constructs and return a Tokenizer from a serialized vocabulary
        (JSON) and list of merges (text file) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath) as vocab_f:
            vocab_json = json.load(vocab_f)
        # vocab.json maps unicode-encoded token strings to int IDs.
        # Convert back to {int: bytes}.
        vocab = {idx: unicode_str_to_bytes(token_str) for token_str, idx in vocab_json.items()}
        # merges.txt has one merge per line, space-separated pair of unicode-encoded tokens.
        merges = []
        with open(merges_filepath) as merges_f:
            for line in merges_f:
                cleaned = line.rstrip()
                if cleaned and len(cleaned.split(" ")) == 2:
                    t1, t2 = cleaned.split(" ")
                    merges.append((unicode_str_to_bytes(t1), unicode_str_to_bytes(t2)))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _derive_word_to_pretokens(self, chunk: str) -> None:
        self._word_to_pretoken: dict[str, tuple[bytes]] = {}
        if self.special_tokens:
            individual_docs = re.split(special_tokens_pattern(self.special_tokens), chunk)
        else:
            individual_docs = [chunk]
        for doc in individual_docs:
            for m in re.finditer(PAT, doc):
                m_str = m.group(0)
                if m_str not in self._word_to_pretoken:
                    self._word_to_pretoken[m_str] = convert_string_to_tuple_of_utf8_bytes(m_str)

    def _derive_pair_to_words(self) -> None:
        self._pair_to_words: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
        for word, pretokens in self._word_to_pretoken.items():
            for pair in get_pairs(pretokens):
                self._pair_to_words[pair].add(word)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # Pretokenize
        self._derive_word_to_pretokens(chunk=text)
        self._derive_pair_to_words()
        # Apply the merges.
        # Iterate merges
        # If pair in pair_to_word, merge and update _word_to_pretoken and _pair_to_words.
        for merge in self.merges:
            if merge not in self._pair_to_words:
                continue
            new_pair_to_words = defaultdict(set)
            old_pair_to_words = defaultdict(set)
            for word in self._pair_to_words[merge]:
                existing_pretoken = self._word_to_pretoken[word]
                new_pretoken = merge_pair_in_pretoken(pretoken=existing_pretoken, pair_to_merge=merge)
                for old_pair in get_pairs(existing_pretoken):
                    old_pair_to_words[old_pair].add(word)
                new_pairs = get_pairs(new_pretoken)
                for new_pair in new_pairs:
                    new_pair_to_words[new_pair].add(word)
                self._word_to_pretoken[word] = new_pretoken
            for old_pair, words in old_pair_to_words.items():
                self._pair_to_words[old_pair] -= words
                if not self._pair_to_words[old_pair]:
                    del self._pair_to_words[old_pair]
            for new_pair, words in new_pair_to_words.items():
                self._pair_to_words[new_pair] |= words
        if self.special_tokens:
            individual_docs = re.split(f"({special_tokens_pattern(self.special_tokens)})", text)
        else:
            individual_docs = [text]
        encoded_indices = []
        for idx, doc in enumerate(individual_docs):
            if doc in self.special_tokens:
                encoded_indices.append(self.vocab_utf8_bytes_to_idx_int[doc.encode(encoding="utf-8")])
                continue
            for m in re.finditer(PAT, doc):
                m_str = m.group(0)
                pretoken = self._word_to_pretoken[m_str]
                for token in pretoken:
                    encoded_indices.append(self.vocab_utf8_bytes_to_idx_int[token])
        return encoded_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory.
        """
        for curr_str in iterable:
            yield from self.encode(curr_str)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
