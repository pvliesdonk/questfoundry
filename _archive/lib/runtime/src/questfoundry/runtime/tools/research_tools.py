from __future__ import annotations

"""Research-oriented tools: web search and local lore index."""

import logging
import os
from collections.abc import Iterable
from typing import Any

import httpx
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearch(BaseTool):
    """SearxNG-backed web search with mock fallback.

    Configure with env vars:
    - SEARXNG_URL (required for real calls)
    - SEARXNG_API_TOKEN (optional Authorization header)
    """

    name: str = "web_search"
    description: str = "Search the web via SearxNG"

    def _run(self, query: str, max_results: int = 5) -> dict[str, Any]:  # type: ignore[override]
        base_url = os.getenv("SEARXNG_URL")
        if not base_url:
            return {
                "status": "mock",
                "provider": "searxng",
                "message": "SEARXNG_URL not set; returning mock result",
                "results": [],
            }

        headers = {}
        token = os.getenv("SEARXNG_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(
                    base_url.rstrip("/") + "/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "en",
                        "pageno": 1,
                        "engines": "google",
                        "safesearch": 1,
                        "results": max_results,
                    },
                    headers=headers,
                )
            resp.raise_for_status()
            data = resp.json()
            results = [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                    "engine": item.get("engine"),
                }
                for item in data.get("results", [])[:max_results]
            ]
            return {
                "status": "success",
                "provider": "searxng",
                "results": results,
            }
        except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
            logger.warning("web_search fallback to mock due to error: %s", exc)
            return {
                "status": "mock",
                "provider": "searxng",
                "message": f"search failed: {exc}",
                "results": [],
            }


class _KeywordScorer:
    @staticmethod
    def score(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        words = {w.lower() for w in query.split() if len(w) > 2}
        scores: list[float] = []
        for doc in docs:
            text = doc.lower()
            scores.append(sum(text.count(w) for w in words))
        return scores


class LoreIndex(BaseTool):
    """Local semantic (or keyword) search over provided documents.

    Priority order:
    1) Ollama embeddings when OLLAMA_MODEL is set (default nomic-embed-text:latest).
    2) TF-IDF cosine similarity (local, no network).
    3) Keyword scoring fallback.
    No external vector database; everything stays in-memory for now.
    """

    name: str = "lore_index"
    description: str = "Semantic or keyword search over provided lore documents"

    class _LoreIndexInput(BaseModel):
        query: str = Field(..., description="Search query")
        documents: list[str] | None = Field(None, description="Documents to search")
        k: int = Field(5, description="Number of results")

    args_schema = _LoreIndexInput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vectorizer: Any | None = None
        self._ollama_model = os.getenv("OLLAMA_MODEL", "nomic-embed-text:latest")
        self._ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if isinstance(input, dict):
            return self._run(**input)
        return self._run(query=str(input), **kwargs)

    def run(self, query: str, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._run(query=query, **kwargs)

    def _load_vectorizer(self) -> Any | None:
        if self._vectorizer is not None:
            return self._vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._vectorizer = TfidfVectorizer(stop_words="english")
        except Exception as exc:  # pragma: no cover - deterministic in tests via monkeypatch
            logger.warning(
                "LoreIndex falling back to keyword search (vectorizer unavailable: %s)", exc
            )
            self._vectorizer = None
        return self._vectorizer

    def _run(
        self,
        query: str,
        documents: Iterable[str] | None = None,
        k: int = 5,
    ) -> dict[str, Any]:  # type: ignore[override]
        docs = list(documents or [])
        if not docs:
            return {"status": "empty", "results": []}

        # 1) Ollama embeddings when available
        if os.getenv("OLLAMA_MODEL") or os.getenv("USE_OLLAMA_EMBEDDINGS"):
            embeddings = self._embed_with_ollama([query] + docs)
            if embeddings is not None and len(embeddings) == len(docs) + 1:
                query_emb = embeddings[0]
                doc_embs = np.asarray(embeddings[1:])
                sims = (
                    doc_embs
                    @ query_emb
                    / (np.linalg.norm(doc_embs, axis=1) * (np.linalg.norm(query_emb) + 1e-9))
                )
                ranked_idx = sims.argsort()[::-1][:k]
                results = [
                    {"text": docs[idx], "score": float(sims[idx])}
                    for idx in ranked_idx
                    if sims[idx] > 0
                ]
                return {"status": "semantic", "results": results, "model": self._ollama_model}

        # 2) TF-IDF cosine similarity
        vectorizer = self._load_vectorizer()
        if vectorizer is not None:
            doc_matrix = vectorizer.fit_transform(docs)
            query_vec = vectorizer.transform([query])
            doc_dense = np.asarray(
                doc_matrix.todense() if hasattr(doc_matrix, "todense") else doc_matrix
            )
            query_dense = np.asarray(
                query_vec.todense() if hasattr(query_vec, "todense") else query_vec
            )
            sims = (doc_dense @ query_dense.T).ravel()
            ranked_idx = sims.argsort()[::-1][:k]
            results = [
                {"text": docs[idx], "score": float(sims[idx])}
                for idx in ranked_idx
                if sims[idx] > 0
            ]
            return {"status": "semantic", "results": results, "model": "tfidf"}

        # 3) Keyword fallback
        scores = _KeywordScorer.score(query, docs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
        return {
            "status": "keyword",
            "results": [{"text": doc, "score": float(score)} for doc, score in ranked if score > 0],
        }

    def _embed_with_ollama(self, texts: list[str]) -> list[list[float]] | None:
        if not self._ollama_model:
            return None
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{self._ollama_host.rstrip('/')}/api/embeddings",
                    json={"model": self._ollama_model, "input": texts},
                )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings") or data.get("embedding")
            if embeddings is None:
                return None
            return embeddings
        except Exception as exc:  # pragma: no cover
            logger.warning("LoreIndex ollama embeddings unavailable: %s", exc)
            return None
