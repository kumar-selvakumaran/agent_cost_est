"""Canonical tokenizer (spec §5.1): tiktoken o200k_base."""
from __future__ import annotations

import functools

import numpy as np
import tiktoken


@functools.lru_cache(maxsize=1)
def canonical():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(texts: list[str], num_threads: int = 1) -> np.ndarray:
    enc = canonical()
    toks = enc.encode_ordinary_batch([t if isinstance(t, str) else "" for t in texts], num_threads=num_threads)
    return np.fromiter((len(t) for t in toks), dtype=np.int64, count=len(texts))
