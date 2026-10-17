from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

NS = uuid.UUID("6f0a2d0e-9a43-4d6e-a6f1-0f1a2b3c4d5e")

SOURCE_TYPES = ("code", "doc", "commit", "pr", "issue", "incident")


@dataclass
class Chunk:
    """A retrievable unit of knowledge with provenance metadata."""

    source_type: str  # code | doc | commit | pr | issue | incident
    text: str
    path: str = ""
    symbol: str = ""  # e.g. "UserService.create_user" or heading path
    kind: str = ""  # function | class | method | module | section | block | commit | ...
    language: str = ""
    start_line: int = 0
    end_line: int = 0
    author: str = ""  # last author (code) / commit author / PR author
    date: str = ""  # ISO-8601: last modified (code) / commit / PR updated
    commit_sha: str = ""
    pr_number: int | None = None
    url: str = ""
    blob_sha: str = ""
    part: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- identity -------------------------------------------------------
    @property
    def key(self) -> str:
        """Stable, content-independent key used for evaluation gold labels."""
        if self.source_type == "commit":
            return f"commit:{self.commit_sha[:7]}"
        if self.source_type in ("pr", "issue") or (self.source_type == "incident" and not self.path):
            return f"{self.source_type}:{self.pr_number}"
        return f"{self.source_type}:{self.path}::{self.symbol}"

    @property
    def id(self) -> str:
        line = self.start_line if self.path else 0
        return str(uuid.uuid5(NS, f"{self.key}#{self.part}#{line}"))

    # ---- text views -------------------------------------------------------
    def header(self) -> str:
        if self.source_type == "commit":
            return f"Commit {self.commit_sha[:7]} by {self.author} on {self.date[:10]}"
        if self.source_type in ("pr", "issue") or (self.source_type == "incident" and not self.path):
            return f"{self.source_type.upper()} #{self.pr_number} by {self.author} ({self.date[:10]})"
        loc = f"{self.path}:{self.start_line}-{self.end_line}" if self.start_line else self.path
        sym = f" | {self.symbol} ({self.kind})" if self.symbol else ""
        return f"{loc}{sym}"

    def embed_text(self) -> str:
        """Text sent to the embedder / reranker: metadata header + body."""
        return f"{self.header()}\n{self.text}"

    def label(self) -> str:
        """Short human label for citations."""
        if self.source_type == "commit":
            return f"commit {self.commit_sha[:7]}"
        if self.source_type in ("pr", "issue") or (self.source_type == "incident" and not self.path):
            return f"{self.source_type} #{self.pr_number}"
        if self.start_line:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return self.path

    def payload(self) -> dict[str, Any]:
        """Filterable fields stored alongside the vector."""
        parts = self.path.split("/") if self.path else []
        prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
        return {
            "source_type": self.source_type,
            "path": self.path,
            "path_prefixes": prefixes,
            "language": self.language,
            "author": self.author,
            "date": self.date,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        d["key"] = self.key
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})
