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

    # models
    embedding_provider: str = "fastembed"  # fastembed | openai | hash
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
    rerank_k: int = 30
    final_k: int = 8
    max_chunks_per_file: int = 3
    context_char_budget: int = 24000

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
        return f"gitcontext_{self.repo_slug}"[:100]

    @property
    def github_slug(self) -> str | None:
        m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", self.repo)
        return f"{m.group(1)}/{m.group(2)}" if m else None


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
