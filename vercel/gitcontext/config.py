from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GITCONTEXT_", env_file=".env", extra="ignore")

    # repository
    repo: str = ""
    data_dir: Path = Path(".gitcontext")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    # storage backend
    #   local  : Qdrant (vectors) + SQLite FTS5 (BM25, chunks, manifest) - best for a single server/laptop
    #   qdrant : everything in Qdrant (dense + sparse BM25-style vectors, chunk payloads, manifest, meta);
    #            stateless clients, so the query side can run in serverless functions (see vercel/)
    backend: str = "local"

    # models
    embedding_provider: str = "fastembed"  # fastembed | voyage | openai | hash
    embedding_dim: int | None = None  # voyage only (output_dimension)
    model_cache_dir: str | None = None  # where fastembed caches models (/tmp/... on serverless)
    voyage_api_key: str | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "cross-encoder"  # cross-encoder | lexical | none
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "anthropic"  # anthropic | openai | none
    llm_model: str = "claude-sonnet-5-5"
    openai_base_url: str | None = None
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    # vector store
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_in_memory: bool = False
    collection_name: str | None = None  # explicit collection; set the SAME value for the indexer and the serverless reader

    # chunking / ingestion
    max_chunk_chars: int = 2400
    min_chunk_chars: int = 80
    max_file_bytes: int = 400_000
    history_commits: int = 2000
    github_max_prs: int = 200
    github_max_issues: int = 200
    incident_labels: str = "incident,postmortem,post-mortem,outage,sev1,sev2,rca"

    # retrieval
    retrieve_k: int = 40
    rerank_k: int = 20
    rerank_blend: float = 0.2  # 1.0 = reranker decides alone; 0.0 = ignore it; in between = RRF-blend with fused rank
    rerank_max_chars: int = 1200  # cross-encoder cost grows with input length; 1200 chars ~ 300 tokens
    final_k: int = 8
    max_chunks_per_file: int = 3
    context_char_budget: int = 24000

    # deployment
    api_key: str | None = None  # if set, every endpoint except /health and the HMAC webhook needs `x-api-key`
    cors_origins: str = ""  # comma-separated origins allowed to call the API from a browser
    auto_index_on_start: bool = False  # index in the background on boot if the repo isn't indexed yet

    rate_limit_per_min: int = 20  # serverless: best-effort per-IP limit on /api/query (per warm instance)

    # freshness
    auto_sync_seconds: int = 0
    sync_on_query: bool = False
    webhook_secret: str | None = None

    @property
    def repo_slug(self) -> str:
        s = self.repo.rstrip("/").removesuffix(".git")
        s = re.sub(r"^(https?://|git@)", "", s)
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        return (s[-60:] or "repo").lower()

    @property
    def work_dir(self) -> Path:
        return Path(self.data_dir) / self.repo_slug

    @property
    def sqlite_path(self) -> Path:
        return self.work_dir / "index.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.work_dir / "qdrant"

    @property
    def collection(self) -> str:
        return self.collection_name or f"gitcontext_{self.repo_slug}"[:100]

    @property
    def github_slug(self) -> str | None:
        m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", self.repo)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        bare = re.fullmatch(r"([\w.-]+)/([\w.-]+)", self.repo)  # "owner/name" (and not a local directory)
        if bare and not Path(self.repo).expanduser().exists():
            return f"{bare.group(1)}/{bare.group(2)}"
        return None


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
