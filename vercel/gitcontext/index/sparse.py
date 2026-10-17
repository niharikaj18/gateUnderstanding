"""BM25-style sparse vectors, computed client-side so the *same* code encodes documents and queries.

Qdrant stores them as a sparse vector with the IDF modifier, so the server contributes IDF from the live
corpus statistics while we provide the field-weighted, length-normalised term-frequency part:

    weight(term, doc) = tf' * (k1 + 1) / (tf' + k1 * (1 - b + b * len / avg_len))

where tf' counts a term 1.0x in the body, 0.5x as a split sub-token of an identifier (parse_user_id ->
parse, user, id) and 4x in the symbol/path - mirroring the column weights of the SQLite FTS5 backend.
Terms are hashed to 32-bit indices (collisions are negligible at repository scale).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from ..models import Chunk
from .lexical import STOP, QueryProfile, expand_identifiers, query_tokens

K1, B, AVG_LEN = 1.2, 0.75, 120.0
_TOK = re.compile(r"[a-z0-9_]+")


def term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "little")


def _words(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP and len(t) > 1]


def encode_chunk(c: Chunk) -> tuple[list[int], list[float]]:
    body = _words(c.text)
    tf: Counter = Counter()
    for t in body:
        tf[t] += 1.0
    for t in _words(expand_identifiers(c.text)):
        tf[t] += 0.5
    sym = f"{c.path} {c.symbol}"
    for t in _words(sym) + _words(expand_identifiers(sym)) + _words(re.sub(r"[/.]", " ", sym)):
        tf[t] += 4.0
    dl = max(len(body), 1)
    norm = K1 * (1 - B + B * dl / AVG_LEN)
    weights: dict[int, float] = {}
    for t, f in tf.items():
        i = term_id(t)
        weights[i] = max(weights.get(i, 0.0), f * (K1 + 1) / (f + norm))
    idx = sorted(weights)
    return idx, [round(weights[i], 4) for i in idx]


def encode_query(profile: QueryProfile) -> tuple[list[int], list[float]]:
    ids = sorted({term_id(t) for t in query_tokens(profile)})
    return ids, [1.0] * len(ids)


def encode_phrase_words(words: list[str]) -> tuple[list[int], list[float]]:
    ids = sorted({term_id(w) for w in words if w not in STOP and len(w) > 1})
    return ids, [1.0] * len(ids)
