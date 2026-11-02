"""Instruction-text statistics, keywords, TF-IDF + SVD (spec §6.1)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

PATH_RE = re.compile(r"[\w./-]+\.\w{1,5}")
URL_RE = re.compile(r"https?://\S+")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def text_stats(texts: pd.Series, keywords: list[str]) -> pd.DataFrame:
    t = texts.fillna("")
    out = pd.DataFrame(index=texts.index)
    out["ins_chars"] = np.log1p(t.str.len())
    out["ins_lines"] = np.log1p(t.str.count("\n") + 1)
    out["ins_code_blocks"] = t.str.count("```") // 2
    out["ins_paths"] = np.log1p(t.map(lambda s: len(PATH_RE.findall(s))))
    out["ins_urls"] = t.map(lambda s: len(URL_RE.findall(s)))
    out["ins_numbers"] = np.log1p(t.map(lambda s: len(NUM_RE.findall(s))))
    low = t.str.lower()
    for kw in keywords:
        out["kw_" + kw.replace(" ", "_")] = low.str.contains(kw, regex=False).astype(np.int8)
    return out


class TextSVD:
    """TF-IDF(1-2 grams, min_df, max_features, sublinear) -> TruncatedSVD; fit on training texts only."""

    def __init__(self, min_df=5, max_features=200_000, dim=128, seed=0):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                                   sublinear_tf=True, dtype=np.float32)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed % (2 ** 31))
        self.dim = dim

    def fit(self, texts: pd.Series):
        u = pd.Series(texts.unique())
        X = self.vec.fit_transform(u)
        self.svd.fit(X)
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        u, inv = np.unique(texts.values.astype(str), return_inverse=True)
        Z = self.svd.transform(self.vec.transform(u)).astype(np.float32)
        return Z[inv]
