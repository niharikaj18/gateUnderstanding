"""Thin wrapper around the git CLI. Content is always read from git objects (HEAD),
never the working tree, so indexing is reproducible and tied to a commit SHA."""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk

RS, US = "\x1e", "\x1f"
PR_RE = re.compile(r"\(#(\d+)\)|[Mm]erge pull request #(\d+)|\bPR[ #]*(\d+)\b")


class GitError(RuntimeError):
    pass


@dataclass
class FileMeta:
    sha: str = ""
    author: str = ""
    date: str = ""
    subject: str = ""
    pr_number: int | None = None
    top_authors: list[str] | None = None


def pr_from_message(msg: str) -> int | None:
    m = PR_RE.search(msg or "")
    return int(next(g for g in m.groups() if g)) if m else None


class GitRepo:
    def __init__(self, path: Path, remote_url: str | None = None):
        self.path = Path(path)
        self.remote_url = remote_url

    # ---- plumbing -----------------------------------------------------------
    def git(self, *args: str, check: bool = True, input: bytes | None = None) -> bytes:
        p = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, input=input)
        if check and p.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {p.stderr.decode(errors='replace')[:400]}")
        return p.stdout

    def text(self, *args: str, **kw) -> str:
        return self.git(*args, **kw).decode("utf-8", errors="replace")

    # ---- acquisition ----------------------------------------------------------
    @classmethod
    def open_or_clone(cls, repo: str, work_dir: Path, token: str | None = None) -> "GitRepo":
        local = Path(repo).expanduser()
        if local.exists() and (local / ".git").exists() or (local.exists() and (local / "HEAD").exists()):
            return cls(local.resolve())
        url = repo if "://" in repo or repo.startswith("git@") else f"https://github.com/{repo}"
        dest = work_dir / "clone"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            auth = url.replace("https://", f"https://x-access-token:{token}@") if token and url.startswith("https://") else url
            p = subprocess.run(["git", "clone", "--quiet", auth, str(dest)], capture_output=True)
            if p.returncode != 0:
                raise GitError(f"clone failed: {p.stderr.decode(errors='replace')[:400]}")
            subprocess.run(["git", "-C", str(dest), "remote", "set-url", "origin", url], capture_output=True)
        return cls(dest, remote_url=url)

    @property
    def is_clone(self) -> bool:
        return self.remote_url is not None

    def pull(self) -> bool:
        """Fast-forward a managed clone. Returns True if HEAD moved. No-op for user-owned local repos."""
        if not self.is_clone:
            return False
        before = self.head()
        self.git("fetch", "--quiet", "origin", check=False)
        self.git("merge", "--ff-only", "--quiet", "FETCH_HEAD", check=False)
        return self.head() != before

    # ---- tree -----------------------------------------------------------------------
    def head(self) -> str:
        return self.text("rev-parse", "HEAD").strip()

    def branch(self) -> str:
        return self.text("rev-parse", "--abbrev-ref", "HEAD", check=False).strip()

    def ls_tree(self, rev: str = "HEAD") -> dict[str, tuple[str, int]]:
        """path -> (blob_sha, size)"""
        out = self.text("ls-tree", "-r", "-l", "-z", rev)
        res = {}
        for ent in out.split("\0"):
            if not ent:
                continue
            meta, path = ent.split("\t", 1)
            _mode, typ, sha, size = meta.split()
            if typ == "blob":
                res[path] = (sha, int(size) if size.isdigit() else 0)
        return res

    def read_blobs(self, shas: list[str]) -> dict[str, str]:
        """Batch-read blobs via `git cat-file --batch`."""
        if not shas:
            return {}
        p = subprocess.run(["git", "-C", str(self.path), "cat-file", "--batch"], capture_output=True,
                           input=("\n".join(shas) + "\n").encode())
        data, i, out = p.stdout, 0, {}
        for sha in shas:
            nl = data.index(b"\n", i)
            header = data[i:nl].decode().split()
            if len(header) < 3:
                i = nl + 1
                continue
            size = int(header[2])
            body = data[nl + 1 : nl + 1 + size]
            i = nl + 1 + size + 1
            if b"\0" in body[:8000]:
                continue  # binary
            out[sha] = body.decode("utf-8", errors="replace")
        return out

    def diff_names(self, old: str, new: str = "HEAD") -> tuple[list[str], list[str]]:
        """(changed_or_added, deleted) paths between two commits."""
        out = self.text("diff", "--name-status", "--no-renames", "-z", old, new)
        parts = [x for x in out.split("\0") if x]
        changed, deleted = [], []
        for i in range(0, len(parts) - 1, 2):
            (deleted if parts[i].startswith("D") else changed).append(parts[i + 1])
        return changed, deleted

    def commit_exists(self, sha: str) -> bool:
        if not sha:
            return False
        p = subprocess.run(["git", "-C", str(self.path), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True)
        return p.returncode == 0

    # ---- history ----------------------------------------------------------------------
    def file_meta(self, max_commits: int = 5000) -> dict[str, FileMeta]:
        """Last-modifying commit and top authors per file, from a single pass over `git log`."""
        fmt = f"{RS}%H{US}%an{US}%aI{US}%s"
        out = self.text("log", f"-n{max_commits}", "--no-merges", "--name-only", f"--pretty=format:{fmt}", check=False)
        meta: dict[str, FileMeta] = {}
        authors: dict[str, Counter] = {}
        for rec in out.split(RS):
            if not rec.strip():
                continue
            head, *files = rec.split("\n")
            sha, author, date, subject = (head.split(US) + [""] * 4)[:4]
            for f in (x for x in files if x.strip()):
                authors.setdefault(f, Counter())[author] += 1
                if f not in meta:
                    meta[f] = FileMeta(sha, author, date, subject, pr_from_message(subject))
        for f, m in meta.items():
            m.top_authors = [a for a, _ in authors[f].most_common(3)]
        return meta

    def file_meta_for(self, paths: list[str]) -> dict[str, FileMeta]:
        res = {}
        for p in paths:
            out = self.text("log", "-n1", f"--pretty=format:%H{US}%an{US}%aI{US}%s", "--", p, check=False)
            if out.strip():
                sha, author, date, subject = (out.split(US) + [""] * 4)[:4]
                names = self.text("log", "-n200", "--pretty=format:%an", "--", p, check=False).split("\n")
                res[p] = FileMeta(sha, author, date, subject, pr_from_message(subject),
                                  [a for a, _ in Counter(n for n in names if n).most_common(3)])
        return res

    def commits(self, since_sha: str | None = None, limit: int = 2000) -> list[Chunk]:
        """Commit history as chunks (subject + body + touched files)."""
        fmt = f"{RS}%H{US}%an{US}%aI{US}%s{US}%b{US}"
        rng = f"{since_sha}..HEAD" if since_sha and self.commit_exists(since_sha) else "HEAD"
        out = self.text("log", rng, f"-n{limit}", "--no-merges", "--name-only", f"--pretty=format:{fmt}", check=False)
        chunks = []
        for rec in out.split(RS):
            if not rec.strip():
                continue
            parts = rec.split(US)
            if len(parts) < 6:
                continue
            sha, author, date, subject, body, files_blob = parts[:6]
            files = [f for f in files_blob.split("\n") if f.strip()]
            shown = files[:25] + ([f"... (+{len(files) - 25} more)"] if len(files) > 25 else [])
            text = f"{subject}\n\n{body.strip()}\n\nFiles changed:\n" + "\n".join(f"- {f}" for f in shown)
            pr = pr_from_message(subject + " " + body)
            url = ""
            if self.remote_url and "github.com" in self.remote_url:
                url = f"{self.remote_url.removesuffix('.git')}/commit/{sha}"
            chunks.append(Chunk(source_type="commit", text=text.strip(), kind="commit", author=author, date=date,
                                commit_sha=sha, pr_number=pr, url=url, extra={"files": files[:100], "subject": subject}))
        return chunks
