from pydantic import BaseModel, Field


class SearchResultOut(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    took_ms: float
    results: list[SearchResultOut]


class HealthResponse(BaseModel):
    status: str
    n_docs: int
    embedding_model: str
