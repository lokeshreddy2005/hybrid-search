"""FastAPI service exposing the hybrid search engine.

Run with:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from api.schemas import HealthResponse, SearchResponse, SearchResultOut
from retrieval.engine import VALID_MODES, SearchEngine

STORE_DIR = os.environ.get("INDEX_STORE_DIR", "index/store")

engine_holder: dict[str, SearchEngine] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_holder["engine"] = SearchEngine.load(STORE_DIR)
    yield
    engine_holder.clear()


app = FastAPI(
    title="Hybrid Search & Retrieval Engine",
    description="BM25 (from scratch) + dense vector search (FAISS) + reciprocal rank fusion",
    version="1.0.0",
    lifespan=lifespan,
)


def get_engine() -> SearchEngine:
    engine = engine_holder.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not loaded yet")
    return engine


@app.get("/health", response_model=HealthResponse)
def health():
    engine = get_engine()
    return HealthResponse(
        status="ok",
        n_docs=engine.inverted_index.n_docs,
        embedding_model=engine.embedder.model_name,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("hybrid", description=f"One of {VALID_MODES}"),
    top_k: int = Query(10, ge=1, le=100),
):
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {VALID_MODES}")

    engine = get_engine()
    start = time.perf_counter()
    results = engine.search(q, mode=mode, top_k=top_k)
    took_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        query=q,
        mode=mode,
        top_k=top_k,
        took_ms=round(took_ms, 2),
        results=[SearchResultOut(**r.__dict__) for r in results],
    )
