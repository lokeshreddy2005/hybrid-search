"""API response correctness, using a tiny in-memory engine (no real
embedding model or on-disk index needed) monkeypatched in at startup."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from index.inverted_index import InvertedIndex
from retrieval.engine import SearchEngine
from retrieval.vector_index import VectorIndex

DOCS = [
    ("d1", "Cat", "domestic cat facts and behavior"),
    ("d2", "Dog", "domestic dog facts and behavior"),
    ("d3", "Car", "automobile engineering overview"),
]


class FakeEmbedder:
    """Deterministic stand-in for retrieval.embedder.Embedder — no model download."""

    model_name = "fake-embedder"
    dim = 4

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        vectors = []
        for text in texts:
            h = abs(hash(text))
            rng = np.random.default_rng(h % (2**32))
            v = rng.normal(size=self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
        return np.stack(vectors)


def build_fake_engine() -> SearchEngine:
    inv_index = InvertedIndex()
    inv_index.build(DOCS)

    embedder = FakeEmbedder()
    doc_ids = [d[0] for d in DOCS]
    doc_titles = [d[1] for d in DOCS]
    embeddings = embedder.encode([f"{t}. {txt}" for _, t, txt in DOCS])

    vector_index = VectorIndex(dim=embedder.dim)
    vector_index.build(embeddings, doc_ids=doc_ids, doc_titles=doc_titles)

    doc_texts = {d[0]: {"title": d[1], "text": d[2]} for d in DOCS}
    return SearchEngine(inv_index, vector_index, embedder, doc_texts)


@pytest.fixture()
def client(monkeypatch):
    from api import main as api_main

    monkeypatch.setattr(api_main.SearchEngine, "load", staticmethod(lambda *a, **k: build_fake_engine()))
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_docs"] == 3
    assert body["embedding_model"] == "fake-embedder"


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_search_valid_modes(client, mode):
    resp = client.get("/search", params={"q": "cat", "mode": mode, "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "cat"
    assert body["mode"] == mode
    assert body["top_k"] == 2
    assert len(body["results"]) <= 2
    assert isinstance(body["took_ms"], float)
    for i, r in enumerate(body["results"], start=1):
        assert r["rank"] == i
        assert set(r.keys()) == {"doc_id", "title", "snippet", "score", "rank"}


def test_bm25_search_finds_lexical_match(client):
    resp = client.get("/search", params={"q": "cat", "mode": "bm25", "top_k": 5})
    body = resp.json()
    doc_ids = [r["doc_id"] for r in body["results"]]
    assert "d1" in doc_ids  # only doc containing the literal term "cat"


def test_invalid_mode_returns_400(client):
    resp = client.get("/search", params={"q": "cat", "mode": "bogus"})
    assert resp.status_code == 400


def test_missing_query_returns_422(client):
    resp = client.get("/search", params={"mode": "hybrid"})
    assert resp.status_code == 422


def test_top_k_out_of_range_returns_422(client):
    resp = client.get("/search", params={"q": "cat", "top_k": 0})
    assert resp.status_code == 422
    resp = client.get("/search", params={"q": "cat", "top_k": 1000})
    assert resp.status_code == 422


def test_default_mode_is_hybrid(client):
    resp = client.get("/search", params={"q": "cat"})
    assert resp.json()["mode"] == "hybrid"
