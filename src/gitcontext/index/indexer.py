"""Indexing pipeline: ingest -> chunk -> enrich metadata -> embed -> store (vector + lexical).

Freshness model
---------------
Every indexed file is recorded in a manifest as (path, git blob SHA, chunk ids). A file is *stale* when
the blob SHA at HEAD differs from the manifest (or the file is new/deleted). `sync()` re-indexes exactly
those files, deletes orphaned chunks, appends new commits, and pulls PR/issue updates since the last sync.
Comparison is by content hash, so it is robust to rebases, force-pushes and reverts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import Settings
from ..ingest.chunker import chunk_file, should_index
from ..ingest.github import GitHubClient
from ..ingest.repo import FileMeta, GitRepo
from ..models import Chunk
from .embeddings import Embedder
from .lexical import LexicalStore
from .vectorstore import VectorStore

Progress = Callable[[str], None]


@dataclass
class SyncReport:
    head: str = ""
    pulled: bool = False
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    commits_added: int = 0
    github_items: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Indexer:
    def __init__(self, s: Settings, repo: GitRepo, embedder: Embedder, store: VectorStore, lex: LexicalStore):
        self.s, self.repo, self.embedder, self.store, self.lex = s, repo, embedder, store, lex
        slug = s.github_slug
        self.github = GitHubClient(slug, s.github_token) if slug else None

    # ---- helpers -----------------------------------------------------------------------------
    def _candidates(self) -> dict[str, tuple[str, int]]:
        tree = self.repo.ls_tree("HEAD")
        return {p: v for p, v in tree.items() if should_index(p, v[1], self.s.max_file_bytes)}

    def _write(self, chunks: list[Chunk]) -> None:
        """Embed + upsert into both indexes."""
        B = 64
        for i in range(0, len(chunks), B):
            batch = chunks[i : i + B]
            vecs = self.embedder.embed_documents([c.embed_text() for c in batch])
            self.store.upsert([c.id for c in batch], vecs, [c.payload() for c in batch])
            self.lex.upsert(batch)

    def _index_files(self, files: dict[str, tuple[str, int]], metas: dict[str, FileMeta], head: str,
                     progress: Progress | None, report: SyncReport) -> None:
        manifest = self.lex.manifest()
        paths = sorted(files)
        slug = self.s.github_slug
        for bstart in range(0, len(paths), 40):
            group = paths[bstart : bstart + 40]
            blobs = self.repo.read_blobs([files[p][0] for p in group])
            pending: list[Chunk] = []
            per_file: dict[str, list[str]] = {}
            for p in group:
                text = blobs.get(files[p][0])
                if text is None:
                    continue
                m = metas.get(p, FileMeta())
                chunks = chunk_file(p, text, self.s.max_chunk_chars, self.s.min_chunk_chars)
                for c in chunks:
                    c.blob_sha, c.author, c.date = files[p][0], m.author, m.date
                    c.commit_sha, c.pr_number = m.sha, m.pr_number
                    c.extra["top_authors"] = m.top_authors or []
                    c.extra["last_commit_subject"] = m.subject
                    if slug:
                        c.url = f"https://github.com/{slug}/blob/{head}/{p}#L{c.start_line}-L{c.end_line}"
                per_file[p] = [c.id for c in chunks]
                pending += chunks
                old_ids = manifest.get(p, ("", []))[1]
                orphan = [i for i in old_ids if i not in set(per_file[p])]
                if orphan:
                    self.store.delete(orphan)
                    self.lex.delete(orphan)
                    report.chunks_deleted += len(orphan)
            self._write(pending)
            report.chunks_written += len(pending)
            for p, ids in per_file.items():
                self.lex.set_file(p, files[p][0], ids, _now())
            if progress:
                progress(f"indexed {min(bstart + 40, len(paths))}/{len(paths)} files")

    def _index_commits(self, since: str | None, report: SyncReport, progress: Progress | None) -> None:
        commits = self.repo.commits(since_sha=since, limit=self.s.history_commits)
        self._write(commits)
        report.commits_added += len(commits)
        report.chunks_written += len(commits)
        if progress:
            progress(f"indexed {len(commits)} commits")

    def _index_github(self, report: SyncReport, progress: Progress | None) -> None:
        if not self.github:
            return
        since = self.lex.get_meta("github_synced_at") or None
        started = GitHubClient.now_iso()
        try:
            items = self.github.fetch(since, self.s.github_max_prs, self.s.github_max_issues,
                                      self.s.incident_labels.split(","), self.s.max_chunk_chars)
        except Exception as e:  # network/auth issues must not break code indexing
            report.notes.append(f"github skipped: {e}")
            return
        report.notes += list(dict.fromkeys(self.github.warnings))
        self._write(items)
        report.github_items += len(items)
        report.chunks_written += len(items)
        self.lex.set_meta("github_synced_at", started)
        if progress:
            progress(f"indexed {len(items)} PR/issue/incident chunks")

    def _finish(self, head: str, report: SyncReport) -> None:
        self.lex.set_meta("indexed_commit", head)
        self.lex.set_meta("indexed_at", _now())
        self.lex.set_meta("embedder", self.embedder.name)
        self.lex.set_meta("branch", self.repo.branch())

    # ---- public --------------------------------------------------------------------------------
    def index_all(self, rebuild: bool = False, history: bool = True, github: bool = True,
                  progress: Progress | None = None) -> SyncReport:
        t0, r = time.time(), SyncReport()
        if rebuild:
            self.store.reset()
            for st in ("code", "doc", "incident", "commit", "pr", "issue"):
                self.lex.delete_source(st)
            for p in list(self.lex.manifest()):
                self.lex.drop_file(p)
            self.lex.set_meta("github_synced_at", "")
        self.store.ensure(self.embedder.dim)
        prev = self.lex.get_meta("embedder")
        if prev and prev != self.embedder.name:
            raise RuntimeError(f"Index was built with {prev} but current embedder is {self.embedder.name}; use --rebuild")
        head = self.repo.head()
        r.head = head
        files = self._candidates()
        if progress:
            progress(f"{len(files)} indexable files at {head[:8]}")
        metas = self.repo.file_meta(self.s.history_commits * 2)
        self._index_files(files, metas, head, progress, r)
        r.files_added = len(files)
        if history:
            self._index_commits(None, r, progress)
        if github:
            self._index_github(r, progress)
        self._finish(head, r)
        r.seconds = round(time.time() - t0, 2)
        return r

    def stale_files(self) -> tuple[list[str], list[str], list[str]]:
        """(added, modified, deleted) relative to what is indexed."""
        current, manifest = self._candidates(), self.lex.manifest()
        added = [p for p in current if p not in manifest]
        modified = [p for p, (sha, _) in current.items() if p in manifest and manifest[p][0] != sha]
        deleted = [p for p in manifest if p not in current]
        return sorted(added), sorted(modified), sorted(deleted)

    def sync(self, github: bool = True, progress: Progress | None = None) -> SyncReport:
        """Incremental re-index of only what changed since the last run."""
        t0, r = time.time(), SyncReport()
        self.store.ensure(self.embedder.dim)
        r.pulled = self.repo.pull()
        head = r.head = self.repo.head()
        added, modified, deleted = self.stale_files()
        r.files_added, r.files_modified, r.files_deleted = len(added), len(modified), len(deleted)
        for p in deleted:
            _, ids = self.lex.manifest().get(p, ("", []))
            if ids:
                self.store.delete(ids)
                self.lex.delete(ids)
                r.chunks_deleted += len(ids)
            self.lex.drop_file(p)
        changed = added + modified
        if changed:
            current = self._candidates()
            metas = self.repo.file_meta_for(changed) if len(changed) <= 300 else self.repo.file_meta(self.s.history_commits * 2)
            self._index_files({p: current[p] for p in changed}, metas, head, progress, r)
        prev_head = self.lex.get_meta("indexed_commit")
        if prev_head != head:
            self._index_commits(prev_head or None, r, progress)
        if github:
            self._index_github(r, progress)
        self._finish(head, r)
        r.seconds = round(time.time() - t0, 2)
        return r

    def status(self) -> dict:
        head = self.repo.head()
        indexed = self.lex.get_meta("indexed_commit")
        added, modified, deleted = self.stale_files() if indexed else ([], [], [])
        stale = bool(added or modified or deleted)
        return {
            "repo": self.s.repo, "branch": self.lex.get_meta("branch"), "head": head, "indexed_commit": indexed,
            "indexed_at": self.lex.get_meta("indexed_at"), "embedder": self.lex.get_meta("embedder"),
            "indexed": bool(indexed), "stale": stale or (bool(indexed) and indexed != head),
            "stale_files": {"added": added[:20], "modified": modified[:20], "deleted": deleted[:20],
                            "total": len(added) + len(modified) + len(deleted)},
            "chunks": self.lex.counts_by_type(), "vectors": self.store.count() if indexed else 0,
        }
