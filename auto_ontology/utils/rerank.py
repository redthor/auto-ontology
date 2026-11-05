# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rerank candidate hits against a query with the configured ranking endpoint.

Mirrors :mod:`auto_ontology.utils.embedding`: the endpoint, model, and API key
are read from the environment here. Two request schemas are in use, picked by
the URL, and each built-in default uses a different one:

* ``…/rerank`` (inference-api, ``sk-`` keys) --
  ``{"query": str, "documents": [str], "top_n"}``, scored by
  ``results[].relevance_score``.
* ``…/reranking`` or ``…/ranking`` (NIM, ``nvapi-`` keys) --
  ``{"query": {"text"}, "passages": [{"text"}], "truncate"}``, scored by
  ``rankings[].logit``.

Any other URL is treated as a NIM base and gets ``/v1/ranking`` appended.

``RERANK_ENDPOINT=embed`` skips the ranking service: hits are scored by cosine
similarity of their embeddings to the query's, using the configured EMBED
endpoint. Weaker than a cross-encoder, but it needs no extra model and nothing
leaves the embedding host.
"""

from __future__ import annotations

from typing import Any

import math

import httpx

from auto_ontology.utils.embedding import embed_texts, get_embed_params
from auto_ontology.utils.model_config import resolve

# Hosted NeMo reranking endpoint — no local GPU required. Override RERANK_ENDPOINT
# to point at a self-hosted vLLM/NIM ranking server. Each field falls back to
# DEFAULT_MODELS_<field> when unset.
_RERANK_ENDPOINT = resolve("RERANK", "ENDPOINT")
_RERANK_MODEL = resolve("RERANK", "MODEL")
_RERANK_API_KEY = resolve("RERANK", "API_KEY")

_TIMEOUT_S = 30.0

# RERANK_ENDPOINT value that selects embedding-similarity scoring.
_EMBED_BACKEND = "embed"


def get_rerank_kwargs() -> dict[str, str]:
    """Keyword args for :func:`rerank_hits` (remote NeMo reranking endpoint)."""
    return {
        "rerank_invoke_url": _RERANK_ENDPOINT,
        "model_name": _RERANK_MODEL,
        "api_key": _RERANK_API_KEY,
    }


def _ranking_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith(("/reranking", "/ranking", "/rerank")):
        return url
    return f"{url}/v1/ranking"


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b)) / norm if norm else float("-inf")


def _embedding_scores(query: str, documents: list[str]) -> list[float]:
    """Cosine similarity of each document to *query*; ``-inf`` if not embeddable."""
    params = get_embed_params()
    (query_vec,) = embed_texts([query], params, "query")
    doc_vecs = embed_texts(documents, params, "passage")
    return [
        _cosine(query_vec, vec) if query_vec and vec else float("-inf")
        for vec in doc_vecs
    ]


def _scores(
    query: str, documents: list[str], *, url: str, model_name: str, api_key: str
) -> list[float]:
    """One score per document, higher is better; ``-inf`` for any not returned."""
    v1 = url.endswith("/rerank")
    if v1:
        payload: dict[str, Any] = {
            "model": model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
    else:
        payload = {
            "model": model_name,
            "query": {"text": query},
            "passages": [{"text": d} for d in documents],
            "truncate": "END",
        }
    headers = {"accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT_S)
    response.raise_for_status()
    data = response.json()

    items = (
        data.get("results" if v1 else "rankings") if isinstance(data, dict) else None
    )
    score_key = "relevance_score" if v1 else "logit"
    scores = [float("-inf")] * len(documents)
    for item in items if isinstance(items, list) else []:
        try:
            index, score = int(item["index"]), float(item[score_key])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(scores):
            scores[index] = score
    return scores


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    rerank_invoke_url: str,
    model_name: str,
    api_key: str = "",
    top_n: int | None = None,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """*hits* sorted by relevance to *query*, best first, each with ``_rerank_score``.

    A hit that already carries ``_rerank_score`` keeps it. Ties keep their
    input order. Raises on an endpoint error; the caller picks the fallback.
    """
    if not hits:
        return hits
    documents = [str(hit.get(text_key) or "") for hit in hits]
    if rerank_invoke_url.strip().lower() == _EMBED_BACKEND:
        scores = _embedding_scores(query, documents)
    else:
        scores = _scores(
            query,
            documents,
            url=_ranking_url(rerank_invoke_url),
            model_name=model_name,
            api_key=api_key,
        )
    ranked = sorted(
        ({"_rerank_score": score, **hit} for hit, score in zip(hits, scores)),
        key=lambda hit: hit["_rerank_score"],
        reverse=True,
    )
    return ranked[:top_n] if top_n is not None else ranked
