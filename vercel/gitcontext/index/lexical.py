"""Lexical index: SQLite FTS5 (BM25) + chunk store + file manifest + metadata.

Why FTS5 instead of an in-memory BM25 lib: it is persistent, incremental (delete/insert by rowid),
supports phrase and NEAR queries (exact error messages) and column weighting (symbol/path boost).
Identifiers are indexed twice: whole (`parse_user_id`) and split (`parse user id`), so both
`parse_user_id` and "parse user" find the same function.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set("""a an the of to in on for and or is are was were be been how what why when where which who whom does do did
can could should would it its this that these those with from by as at into about there their we you i me my our not
no yes if then than so such via use used using get gets make makes work works code file files function class""".split())


def split_identifier(tok: str) -> list[str]:
    t = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
    t = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", t)
    return [p.lower() for p in re.split(r"[_\s]+", t) if p]


def is_identifier_like(tok: str) -> bool:
    return "_" in tok.strip("_") or bool(re.search(r"[a-z][A-Z]", tok)) or (tok.isupper() and len(tok) > 3)


def expand_identifiers(text: str) -> str:
    out = []
    for tok in _WORD.findall(text):
        if is_identifier_like(tok):
            parts = split_identifier(tok)
            if len(parts) > 1:
                out.extend(parts)
    return " ".join(out)


@dataclass
class QueryProfile:
    terms: list[str]
    identifiers: list[str]
    phrases: list[str]

    @property
    def exact_signal(self) -> bool:
        return bool(self.identifiers or self.phrases)


_ERR_RE = re.compile(r"(?:\b[\w.]*(?:Error|Exception|Warning)\b|error|failed|cannot|unable to|invalid|unexpected)[:\s-]+(.{12,})", re.I)


def profile_query(q: str) -> QueryProfile:
    phrases = [p.strip() for p in re.findall(r"\"([^\"]{4,})\"|'([^']{6,})'|`([^`]{4,})`", q) for p in p if p.strip()]
    quoted = {p for p in phrases}
    # a quoted single identifier is an identifier, not a phrase
    phrases = [p for p in phrases if len(_WORD.findall(p)) >= 2]
    m = _ERR_RE.search(q)
    if m:
        phrases.append(m.group(1).strip(" .?\"'`"))
    words = _WORD.findall(q)
    ids = [w for w in words if is_identifier_like(w)] + [w for w in quoted if _WORD.fullmatch(w)]
    ids += re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b", q)  # dotted paths
    terms = [w.lower() for w in words if w.lower() not in STOP and len(w) > 1]
    return QueryProfile(terms=list(dict.fromkeys(terms))[:24], identifiers=list(dict.fromkeys(ids)), phrases=phrases)


def _fts_quote(t: str) -> str:
    return '"' + t.replace('"', '""') + '"'


def query_tokens(profile: QueryProfile) -> list[str]:
    """Terms used for keyword retrieval: query words, identifiers, and identifier sub-tokens."""
    toks: list[str] = list(profile.terms)
    for ident in profile.identifiers:
        for part in re.split(r"[.]", ident):
            toks.append(part.lower())
            toks.extend(split_identifier(part))
    return [t for t in dict.fromkeys(toks) if t and t not in STOP][:40]


def build_match(profile: QueryProfile) -> str | None:
    return " OR ".join(_fts_quote(t) for t in query_tokens(profile)) or None


def build_phrase_match(profile: QueryProfile) -> str | None:
    qs = []
    for ph in profile.phrases:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", ph)][:12]
        if len(words) >= 2:
            qs.append(_fts_quote(" ".join(words)))
            qs.append("NEAR(" + " ".join(_fts_quote(w) for w in words) + ", 8)")
    return " OR ".join(qs) or None


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, key TEXT, source_type TEXT, path TEXT,
  symbol TEXT, author TEXT, date TEXT, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_key ON chunks(key);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  raw, split, symbols, tokenize="unicode61 tokenchars '_'");
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, blob_sha TEXT, chunk_ids TEXT, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


class LexicalStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # ---- writes ---------------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk]) -> None:
        with self.lock, self.db:
            self._delete_ids([c.id for c in chunks])
            for c in chunks:
                cur = self.db.execute(
                    "INSERT INTO chunks(id,key,source_type,path,symbol,author,date,data) VALUES(?,?,?,?,?,?,?,?)",
                    (c.id, c.key, c.source_type, c.path, c.symbol, c.author, c.date, json.dumps(c.to_dict())))
                sym_txt = f"{c.path} {c.symbol}"
                self.db.execute(
                    "INSERT INTO fts(rowid,raw,split,symbols) VALUES(?,?,?,?)",
                    (cur.lastrowid, c.text, expand_identifiers(c.text),
                     f"{sym_txt} {expand_identifiers(sym_txt)} {re.sub(r'[/.]', ' ', sym_txt)}"))

    def _delete_ids(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            ph = ",".join("?" * len(batch))
            rows = self.db.execute(f"SELECT rowid FROM chunks WHERE id IN ({ph})", batch).fetchall()
            if rows:
                rph = ",".join("?" * len(rows))
                self.db.execute(f"DELETE FROM fts WHERE rowid IN ({rph})", [r[0] for r in rows])
                self.db.execute(f"DELETE FROM chunks WHERE id IN ({ph})", batch)

    def delete(self, ids: list[str]) -> None:
        with self.lock, self.db:
            self._delete_ids(ids)

    def delete_source(self, source_type: str) -> list[str]:
        with self.lock, self.db:
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE source_type=?", (source_type,))]
            self._delete_ids(ids)
            return ids

    # ---- manifest / meta ----------------------------------------------------------------
    def manifest(self) -> dict[str, tuple[str, list[str]]]:
        with self.lock:
            return {r["path"]: (r["blob_sha"], json.loads(r["chunk_ids"]))
                    for r in self.db.execute("SELECT * FROM files")}

    def set_file(self, path: str, blob_sha: str, chunk_ids: list[str], when: str) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", (path, blob_sha, json.dumps(chunk_ids), when))

    def drop_file(self, path: str) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM files WHERE path=?", (path,))

    def get_meta(self, k: str, default: str = "") -> str:
        with self.lock:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
            return r["v"] if r else default

    def set_meta(self, k: str, v: str) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))

    # ---- reads ------------------------------------------------------------------------------
    def get_many(self, ids: list[str]) -> dict[str, Chunk]:
        out: dict[str, Chunk] = {}
        with self.lock:
            for i in range(0, len(ids), 500):
                batch = ids[i : i + 500]
                ph = ",".join("?" * len(batch))
                for r in self.db.execute(f"SELECT id,data FROM chunks WHERE id IN ({ph})", batch):
                    out[r["id"]] = Chunk.from_dict(json.loads(r["data"]))
        return out

    def count(self, source_type: str | None = None) -> int:
        with self.lock:
            if source_type:
                return self.db.execute("SELECT COUNT(*) FROM chunks WHERE source_type=?", (source_type,)).fetchone()[0]
            return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def counts_by_type(self) -> dict[str, int]:
        with self.lock:
            return {r[0]: r[1] for r in self.db.execute("SELECT source_type, COUNT(*) FROM chunks GROUP BY 1")}

    def iter_chunks(self, source_type: str | None = None):
        with self.lock:
            q, args = "SELECT data FROM chunks", ()
            if source_type:
                q, args = q + " WHERE source_type=?", (source_type,)
            rows = self.db.execute(q, args).fetchall()
        for r in rows:
            yield Chunk.from_dict(json.loads(r["data"]))

    # ---- search ---------------------------------------------------------------------------------
    def _filters(self, source_types, path_prefix):
        sql, args = "", []
        if source_types:
            sql += f" AND c.source_type IN ({','.join('?' * len(source_types))})"
            args += list(source_types)
        if path_prefix:
            sql += " AND (c.path = ? OR c.path LIKE ?)"
            args += [path_prefix.rstrip("/"), path_prefix.rstrip("/") + "/%"]
        return sql, args

    def _run(self, match: str | None, k: int, source_types, path_prefix) -> list[tuple[str, float]]:
        if not match:
            return []
        fsql, fargs = self._filters(source_types, path_prefix)
        q = ("SELECT c.id, bm25(fts, 1.0, 0.5, 4.0) AS s FROM fts JOIN chunks c ON c.rowid = fts.rowid "
             f"WHERE fts MATCH ?{fsql} ORDER BY s LIMIT ?")
        with self.lock:
            try:
                return [(r[0], -r[1]) for r in self.db.execute(q, [match, *fargs, k])]
            except sqlite3.OperationalError:
                return []  # malformed MATCH from odd user input: degrade to no lexical hits

    def bm25(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        return self._run(build_match(profile), k, source_types, path_prefix)

    def phrase(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        return self._run(build_phrase_match(profile), k, source_types, path_prefix)

    def symbol_exact(self, profile: QueryProfile, k: int, source_types=None, path_prefix=None):
        """Chunks whose symbol equals / ends with a queried identifier. A qualified name
        (`Option.get_help_record`) ranks its exact match before same-named methods of other classes."""
        names = list(dict.fromkeys(i.split(".")[-1] for i in profile.identifiers))
        if not names:
            return []
        qualified = list(dict.fromkeys(i for i in profile.identifiers if "." in i))
        fsql, fargs = self._filters(source_types, path_prefix)
        conds = " OR ".join(["c.symbol = ? OR c.symbol LIKE ?"] * len(names))
        args = [x for n in names for x in (n, f"%.{n}")]
        order, oargs = "", []
        if qualified:
            order = f" ORDER BY CASE WHEN c.symbol IN ({','.join('?' * len(qualified))}) THEN 0 ELSE 1 END"
            oargs = qualified
        with self.lock:
            rows = self.db.execute(
                f"SELECT c.id FROM chunks c WHERE ({conds}){fsql}{order} LIMIT ?", [*args, *fargs, *oargs, k]).fetchall()
        return [(r[0], 1.0) for r in rows]
