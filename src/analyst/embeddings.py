"""Text embeddings.

FastEmbedEmbedder runs a small ONNX model on the CPU: no service and no key.
HashingEmbedder needs no model at all; the tests use it, and it is good enough
to tell passages apart by the words they share with a query.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Protocol

from analyst.config import EMBEDDING_DIMENSION, embedding_model


class Embedder(Protocol):
    dimension: int

    def passages(self, texts: list[str]) -> list[list[float]]: ...

    def query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str | None = None):
        # imported here: loading it takes a second or two
        from fastembed import TextEmbedding

        self.model_name = model_name or embedding_model()
        self._model = TextEmbedding(self.model_name)
        self.dimension = len(self.query("dimension probe"))

    def passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbedder:
    """Bag of words hashed into a fixed number of buckets and normalised to length one."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = zlib.crc32(word.encode("utf-8"))
            vector[bucket % self.dimension] += 1.0 if bucket & 0x80000000 else -1.0
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]

    def passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def query(self, text: str) -> list[float]:
        return self._embed(text)
