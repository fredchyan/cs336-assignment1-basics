from functools import lru_cache

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def special_tokens_pattern(special_tokens: list[str]) -> str:
    # Sort by length desc so longer tokens (e.g. "<|eot|><|eot|>") match
    # before their prefixes — regex alternation is first-match, not longest-match.
    return "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Maps every byte (0-255) to a unique printable unicode character.
    This lets us represent arbitrary bytes as readable text in JSON/txt files.

    The mapping is one-to-one (every byte maps to exactly one character, and
    every character maps back to exactly one byte), so no information is lost.

    How it works:
    - 188 bytes are already printable ASCII/Latin (like '!', 'A', 'é'), so
      they map to themselves: byte 33 -> '!', byte 65 -> 'A', etc.
    - The remaining 68 bytes are non-printable (control chars, space, etc.).
      These get mapped to unicode characters starting at code point 256 (U+0100),
      which are unused by the first group. For example:
        byte 0 (null)  -> chr(256) = 'Ā'
        byte 32 (space) -> chr(288) = 'Ġ'
      The offset 256 ensures these never collide with the printable range.
    """
    # These 188 bytes have printable representations and map to themselves.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # The remaining 68 bytes are non-printable. Map them to chr(256 + n)
    # where n counts from 0..67, placing them in a unicode range that
    # doesn't overlap with any of the 188 printable bytes above.
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def bytes_to_unicode_str(token_bytes: bytes) -> str:
    """Convert raw bytes to a unicode string using the GPT-2 encoding.
    Each byte is individually mapped to a printable unicode character."""
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token_bytes)


def unicode_str_to_bytes(token_str: str) -> bytes:
    """Reverse of bytes_to_unicode_str — convert a GPT-2 unicode string back to raw bytes."""
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return bytes(byte_decoder[c] for c in token_str)


def merge_pair_in_pretoken(pretoken: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_pretoken = []
    just_merged = False
    for i in range(len(pretoken)):
        if just_merged:
            just_merged = False
            continue
        if i == len(pretoken) - 1:
            new_pretoken.append(pretoken[i])
            break
        pair = pretoken[i] + pretoken[i + 1]
        if pair == pair_to_merge[0] + pair_to_merge[1]:
            new_pretoken.append(pair)
            just_merged = True
        else:
            new_pretoken.append(pretoken[i])
    new_pretoken_tuple = tuple(new_pretoken)
    assert new_pretoken_tuple != pretoken
    return new_pretoken_tuple


def get_pairs(pretoken: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return [(pretoken[i], pretoken[i + 1]) for i in range(len(pretoken) - 1)]


def convert_string_to_tuple_of_utf8_bytes(inp_string: str) -> tuple[bytes]:
    # Iterating over a bytes object yields ints, so we wrap each in bytes([b])
    # to get a tuple of single-byte bytes objects, e.g. "hi" -> (b'h', b'i')
    return tuple(bytes([b]) for b in inp_string.encode("utf-8"))
