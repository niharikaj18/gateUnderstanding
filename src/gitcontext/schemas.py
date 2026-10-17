"""Request models shared by the server API and the serverless API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    rerank: bool = True
    source_types: list[str] | None = None
    path_prefix: str | None = None


class IndexRequest(BaseModel):
    rebuild: bool = False
    history: bool = True
    github: bool = True
