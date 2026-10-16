"""Embedding backends behind one small interface.

- local:<model>  sentence-transformers on CPU (free). e5 models need the
                 "query: " / "passage: " prefixes, handled here.
- openai:<model> OpenAI embeddings API.
- hash:<dim>     deterministic bag-of-words hashing, for tests and CI only.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


class HashEmbedder:
    """Not semantic. Good enough to test plumbing deterministically."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash:{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._one(text))


class LocalEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"local:{model}"
        self.model = SentenceTransformer(model)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.e5 = "e5" in model.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200)

    def embed_query(self, text: str) -> np.ndarray:
        q = f"query: {text}" if self.e5 else text
        return self.model.encode([q], normalize_embeddings=True)[0]


class OpenAIEmbedder:
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str):
        from langchain_openai import OpenAIEmbeddings

        self.name = f"openai:{model}"
        self.client = OpenAIEmbeddings(model=model)
        self.dim = self.DIMS.get(model) or len(self.client.embed_query("dim probe"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            out.extend(self.client.embed_documents(texts[i:i + 256]))
        return np.asarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.client.embed_query(text), dtype=np.float32)


_cache: dict[str, Embedder] = {}


def get_embedder(spec: str) -> Embedder:
    if spec not in _cache:
        kind, _, arg = spec.partition(":")
        if kind == "hash":
            _cache[spec] = HashEmbedder(int(arg or 256))
        elif kind == "local":
            _cache[spec] = LocalEmbedder(arg)
        elif kind == "openai":
            _cache[spec] = OpenAIEmbedder(arg)
        else:
            raise ValueError(f"Unknown EMBED_MODEL '{spec}'. Use local:<model>, openai:<model> or hash:<dim>.")
    return _cache[spec]
