"""GitHub REST ingestion for PR discussions, issues and incident reports (issues labelled
postmortem/incident/outage/...). Incremental via `since`; tolerant of rate limits."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import httpx

from ..models import Chunk

API = "https://api.github.com"
INCIDENT_TITLE_RE = re.compile(r"post[-_ ]?mortem|incident|outage|\brca\b|\bsev[ -]?[12]\b", re.I)


class GitHubClient:
    def __init__(self, slug: str, token: str | None = None, transport: httpx.BaseTransport | None = None):
        self.slug = slug
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "gitcontext"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.warnings: list[str] = []
        self.http = httpx.Client(base_url=API, headers=headers, timeout=30, transport=transport)

    def _get(self, url: str, params: dict | None = None) -> httpx.Response | None:
        for attempt in range(3):
            r = self.http.get(url, params=params)
            if r.status_code in (403, 429) and r.headers.get("x-ratelimit-remaining") == "0":
                self.warnings.append("GitHub rate limit exhausted (set GITHUB_TOKEN); PR/issue data may be partial")
                return None  # degrade gracefully, caller keeps what it has
            if r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                self.warnings.append(f"GitHub API {r.status_code} for {url}")
                return None
            return r
        return None

    def _paged(self, url: str, params: dict, limit: int):
        n, page = 0, 1
        while n < limit:
            r = self._get(url, {**params, "per_page": 100, "page": page})
            if r is None:
                return
            items = r.json()
            if not items:
                return
            for it in items:
                yield it
                n += 1
                if n >= limit:
                    return
            if len(items) < 100:
                return
            page += 1

    # ---- thread rendering ------------------------------------------------------------
    @staticmethod
    def _user(o: dict) -> str:
        return (o.get("user") or {}).get("login", "unknown")

    def _comments(self, number: int, is_pr: bool) -> list[dict]:
        out = [dict(c, _kind="comment") for c in self._paged(f"/repos/{self.slug}/issues/{number}/comments", {}, 100)]
        if is_pr:
            out += [dict(c, _kind="review-comment") for c in self._paged(f"/repos/{self.slug}/pulls/{number}/comments", {}, 100)]
            out += [dict(c, _kind="review", body=c.get("body") or "") for c in
                    self._paged(f"/repos/{self.slug}/pulls/{number}/reviews", {}, 50) if c.get("body")]
        return sorted(out, key=lambda c: c.get("created_at") or c.get("submitted_at") or "")

    def _to_chunks(self, item: dict, st: str, max_chars: int, is_pr: bool) -> list[Chunk]:
        n = item["number"]
        labels = [lb["name"] for lb in item.get("labels", [])]
        title, body = item.get("title", ""), item.get("body") or ""
        head = f"{'Pull request' if is_pr else 'Issue'} #{n}: {title}\nState: {item.get('state')}  Labels: {', '.join(labels) or '-'}\n\n{body}".strip()
        blocks = [head]
        if item.get("comments", 0) or is_pr:
            for c in self._comments(n, is_pr):
                path = f" on {c['path']}" if c.get("path") else ""
                blocks.append(f"[{c['_kind']}{path}] @{self._user(c)}: {(c.get('body') or '').strip()}")
        # pack blocks into chunks <= max_chars (never split a comment unless it alone is too long)
        chunks, cur = [], ""
        for b in blocks:
            for piece in [b[i : i + max_chars] for i in range(0, len(b), max_chars)] or [""]:
                if cur and len(cur) + len(piece) + 2 > max_chars:
                    chunks.append(cur)
                    cur = ""
                cur += ("\n\n" if cur else "") + piece
        if cur:
            chunks.append(cur)
        return [Chunk(source_type=st, text=t, kind="pr" if is_pr else "issue", author=self._user(item),
                      date=item.get("updated_at") or item.get("created_at") or "", pr_number=n,
                      url=item.get("html_url", ""), part=i,
                      extra={"title": title, "labels": labels, "state": item.get("state")})
                for i, t in enumerate(chunks)]

    # ---- public API ------------------------------------------------------------------------
    def fetch(self, since: str | None, max_prs: int, max_issues: int, incident_labels: list[str],
              max_chars: int = 2400) -> list[Chunk]:
        labels_l = {x.strip().lower() for x in incident_labels}
        self.warnings.clear()
        out: list[Chunk] = []
        for pr in self._paged(f"/repos/{self.slug}/pulls",
                              {"state": "all", "sort": "updated", "direction": "desc"}, max_prs):
            if since and (pr.get("updated_at") or "") <= since:
                break
            out += self._to_chunks(pr, "pr", max_chars, True)
        params = {"state": "all", "sort": "updated", "direction": "desc"}
        if since:
            params["since"] = since
        for it in self._paged(f"/repos/{self.slug}/issues", params, max_issues):
            if "pull_request" in it:  # the issues endpoint also returns PRs
                continue
            names = {lb["name"].lower() for lb in it.get("labels", [])}
            is_incident = bool(names & labels_l) or bool(INCIDENT_TITLE_RE.search(it.get("title", "")))
            out += self._to_chunks(it, "incident" if is_incident else "issue", max_chars, False)
        return out

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
