"""Structure-aware chunking.

Code is split on semantic boundaries (module preamble / function / class / method),
never on arbitrary token counts. Python uses the real AST; C-family languages use a
brace-depth scanner plus declaration patterns; Markdown is split by heading hierarchy.
Only oversize units are split further (by line windows) so every chunk stays embeddable.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..models import Chunk

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".php": "php", ".rb": "ruby", ".scala": "scala", ".swift": "swift",
}
DOC_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}
CONFIG_EXT = {".yml", ".yaml", ".toml", ".ini", ".cfg", ".json", ".sh", ".sql", ".tf", ".gradle", ".xml"}
CONFIG_NAMES = {"dockerfile", "makefile", "jenkinsfile"}
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv", "venv", ".tox", "target",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "site-packages", ".next", "coverage",
}
SKIP_SUFFIX = (".min.js", ".min.css", ".lock", ".map", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
               ".pdf", ".zip", ".gz", ".woff", ".woff2", ".ttf", ".pyc", ".so", ".class", ".jar", ".bin")
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "go.sum"}
INCIDENT_RE = re.compile(r"post[-_ ]?mortem|incident|\brca\b|outage|runbook", re.I)


# --------------------------------------------------------------------------- file classification
def should_index(path: str, size: int, max_bytes: int) -> bool:
    p = PurePosixPath(path)
    if size > max_bytes or size == 0:
        return False
    if any(part in SKIP_DIRS for part in p.parts[:-1]):
        return False
    if p.name in SKIP_NAMES or path.lower().endswith(SKIP_SUFFIX):
        return False
    ext = p.suffix.lower()
    return ext in LANG_BY_EXT or ext in DOC_EXT or ext in CONFIG_EXT or p.name.lower() in CONFIG_NAMES


def classify(path: str) -> tuple[str, str]:
    """Return (source_type, language)."""
    p = PurePosixPath(path)
    ext, low = p.suffix.lower(), path.lower()
    if ext in DOC_EXT:
        if INCIDENT_RE.search(low):
            return "incident", "markdown"
        return "doc", "markdown" if ext in (".md", ".markdown") else ext.lstrip(".")
    return "code", LANG_BY_EXT.get(ext, ext.lstrip(".") or p.name.lower())


# --------------------------------------------------------------------------- helpers
def _mk(path, source_type, lang, kind, symbol, lines, start, end, part=0, **extra) -> Chunk:
    """lines is a 0-indexed list; start/end are 0-indexed inclusive."""
    text = "\n".join(lines[start : end + 1]).rstrip()
    return Chunk(source_type=source_type, text=text, path=path, symbol=symbol, kind=kind, language=lang,
                 start_line=start + 1, end_line=end + 1, part=part, extra=extra)


def split_window(lines: list[str], start: int, end: int, max_chars: int, overlap: int = 3):
    """Split [start,end] into line windows <= max_chars. Yields (s, e)."""
    s = start
    while s <= end:
        size, e = 0, s
        while e <= end and (size + len(lines[e]) + 1 <= max_chars or e == s):
            size += len(lines[e]) + 1
            e += 1
        e -= 1
        yield s, e
        if e >= end:
            break
        s = max(e + 1 - overlap, s + 1)


def _emit_sized(out, path, st, lang, kind, symbol, lines, start, end, max_chars, **extra):
    seg = "\n".join(lines[start : end + 1])
    if len(seg) <= max_chars:
        out.append(_mk(path, st, lang, kind, symbol, lines, start, end, **extra))
        return
    for part, (s, e) in enumerate(split_window(lines, start, end, max_chars)):
        out.append(_mk(path, st, lang, kind, symbol, lines, s, e, part=part, **extra))


def chunk_lines(path, st, lang, lines, max_chars, kind="block") -> list[Chunk]:
    """Last-resort fallback for files with no recognisable structure (config, unparsable code)."""
    out: list[Chunk] = []
    if not lines:
        return out
    for s, e in split_window(lines, 0, len(lines) - 1, max_chars, overlap=5):
        out.append(_mk(path, st, lang, kind, f"{PurePosixPath(path).name}:{s + 1}-{e + 1}", lines, s, e))
    return out


# --------------------------------------------------------------------------- Python (AST)
def chunk_python(path: str, src: str, max_chars: int) -> list[Chunk]:
    st, lang = "code", "python"
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return chunk_lines(path, st, lang, lines, max_chars)
    out: list[Chunk] = []
    covered = [False] * len(lines)

    def start_of(node) -> int:
        s = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        while s > 0 and lines[s - 1].lstrip().startswith("#"):  # leading comments belong to the def
            s -= 1
        return s

    def emit(node, symbol, kind):
        s, e = start_of(node), node.end_lineno - 1
        for i in range(s, e + 1):
            covered[i] = True
        _emit_sized(out, path, st, lang, kind, symbol, lines, s, e, max_chars,
                    docstring=ast.get_docstring(node) or "")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            s, e = start_of(node), node.end_lineno - 1
            if len("\n".join(lines[s : e + 1])) <= max_chars:
                emit(node, node.name, "class")
            else:  # big class: header chunk (signature, docstring, method index) + one chunk per method
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                hdr_end = (start_of(methods[0]) - 1) if methods else e
                index = "\n".join(f"    # method: {m.name}  (line {m.lineno})" for m in methods)
                text = "\n".join(lines[s : max(hdr_end, s) + 1]).rstrip() + ("\n" + index if index else "")
                out.append(Chunk(source_type=st, text=text[: max_chars * 2], path=path, symbol=node.name,
                                 kind="class", language=lang, start_line=s + 1, end_line=max(hdr_end, s) + 1,
                                 extra={"docstring": ast.get_docstring(node) or ""}))
                for i in range(s, e + 1):
                    covered[i] = True
                for m in methods:
                    emit(m, f"{node.name}.{m.name}", "method")
    # module preamble: imports, constants, top-level statements that don't belong to a def/class
    pre = [i for i, c in enumerate(covered) if not c and lines[i].strip()]
    text_lines = [lines[i] for i in pre]
    if "\n".join(text_lines).strip():
        mod_doc = ast.get_docstring(tree) or ""
        sym = PurePosixPath(path).stem
        for part, (s, e) in enumerate(split_window(text_lines, 0, len(text_lines) - 1, max_chars, 0)):
            out.append(Chunk(source_type=st, text="\n".join(text_lines[s : e + 1]).strip(), path=path,
                             symbol=sym, kind="module", language=lang, start_line=pre[s] + 1,
                             end_line=pre[e] + 1, part=part, extra={"docstring": mod_doc}))
    return [c for c in out if c.text.strip()]


# --------------------------------------------------------------------------- brace languages
@dataclass
class Decl:
    line: int
    kind: str
    name: str


_ID = r"[A-Za-z_$][\w$]*"
_NOT_METHOD = {"if", "for", "while", "switch", "catch", "return", "else", "function", "new", "sizeof", "do", "try"}

PATTERNS: dict[str, list[tuple[str, str]]] = {
    "javascript": [
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+({_ID})", "function"),
        (rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+({_ID})", "class"),
        (rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|{_ID})\s*(?::[^=]+)?=>", "function"),
        (rf"^\s*(?:export\s+)?(?:interface|type|enum)\s+({_ID})", "type"),
        (rf"^\s+(?:(?:public|private|protected|static|async|override|readonly|get|set)\s+)*({_ID})\s*\([^;{{]*\)\s*(?::\s*[^{{;]+)?\s*\{{", "method"),
    ],
    "go": [
        (rf"^func\s+\([^)]*\)\s*({_ID})", "method"),
        (rf"^func\s+({_ID})", "function"),
        (rf"^type\s+({_ID})\s+(?:struct|interface)", "class"),
    ],
    "java": [
        (rf"^\s*(?:(?:public|private|protected|static|final|abstract|sealed|open|data|internal|partial)\s+)*(?:class|interface|enum|record|object)\s+({_ID})", "class"),
        (r"^\s*namespace\s+([\w.]+)", "namespace"),
        (rf"^\s+(?:(?:public|private|protected|static|final|abstract|override|virtual|async|suspend|synchronized|fun)\s+)+[\w<>\[\],.?\s]*?({_ID})\s*\([^;{{]*\)[^;{{]*\{{", "method"),
    ],
    "rust": [
        (rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+({_ID})", "function"),
        (rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+({_ID})", "class"),
        (rf"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?({_ID})", "class"),
    ],
    "cpp": [
        (rf"^\s*(?:class|struct)\s+({_ID})", "class"),
        (rf"^\s*namespace\s+({_ID})", "namespace"),
        (rf"^[\w:<>*&\s~]+?[\s*&]({_ID}(?:::{_ID})*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{{?\s*$", "function"),
    ],
}
PATTERNS["typescript"] = PATTERNS["javascript"]
for _l in ("kotlin", "scala", "csharp", "php", "swift"):
    PATTERNS[_l] = PATTERNS["java"]
PATTERNS["c"] = PATTERNS["cpp"]
_COMPILED = {k: [(re.compile(p), kind) for p, kind in v] for k, v in PATTERNS.items()}
CONTAINERS = {"class", "namespace"}


def brace_depths(lines: list[str]) -> tuple[list[int], list[int], list[bool]]:
    """Brace depth at the start/end of every line plus whether the line opens a block
    (ignoring braces in strings and comments)."""
    start, end, opens, depth, in_block = [], [], [], 0, False
    for ln in lines:
        start.append(depth)
        opened = False
        i, n, quote = 0, len(ln), None
        while i < n:
            ch, two = ln[i], ln[i : i + 2]
            if in_block:
                if two == "*/":
                    in_block, i = False, i + 1
            elif quote:
                if ch == "\\":
                    i += 1
                elif ch == quote:
                    quote = None
            elif two == "//":
                break
            elif two == "/*":
                in_block, i = True, i + 1
            elif ch in "\"'`":
                # a lone ' is often a lifetime/apostrophe: only treat as a string if it closes on this line
                if ch != "'" or ln.find("'", i + 1) != -1:
                    quote = ch
            elif ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth = max(depth - 1, 0)
            i += 1
        end.append(depth)
        opens.append(opened)
    return start, end, opens


def _block_end(i: int, d0: int, ds, de, opens, limit: int) -> int | None:
    """Index of the line closing the block opened at/after line i; None if there is no body."""
    opened = False
    for j in range(i, min(len(ds), limit)):
        if opens[j] or de[j] > d0 or ds[j] > d0:
            opened = True
        if opened and de[j] <= d0:
            return j
        if not opened and j - i >= 6:
            return None
    return len(ds) - 1 if opened else None


def chunk_braces(path: str, lang: str, src: str, max_chars: int) -> list[Chunk]:
    st = "code"
    lines = src.splitlines()
    pats = _COMPILED.get(lang)
    if not pats:
        return chunk_lines(path, st, lang, lines, max_chars)
    ds, de, opens = brace_depths(lines)
    covered = [False] * len(lines)
    out: list[Chunk] = []

    def find_decls(lo: int, hi: int, depth: int) -> list[Decl]:
        found = []
        for i in range(lo, hi + 1):
            if ds[i] != depth or covered[i]:
                continue
            for rx, kind in pats:
                m = rx.match(lines[i])
                if m and not (kind == "method" and m.group(1) in _NOT_METHOD):
                    if kind == "method" and depth == 0:
                        kind = "function"
                    found.append(Decl(i, kind, m.group(1)))
                    break
        return found

    def with_leading(i: int) -> int:
        while i > 0 and lines[i - 1].lstrip().startswith(("//", "*", "/*", "#[", "@", "///")):
            i -= 1
        return i

    def process(lo, hi, depth, prefix, level=0):
        for d in find_decls(lo, hi, depth):
            if covered[d.line]:
                continue
            end = _block_end(d.line, depth, ds, de, opens, hi + 1)
            if end is None:
                end = d.line
            s = with_leading(d.line)
            sym = f"{prefix}{d.name}"
            size = sum(len(lines[k]) + 1 for k in range(s, end + 1))
            if d.kind in CONTAINERS and (size > max_chars or d.kind == "namespace") and level < 3:
                inner = find_decls(d.line + 1, end, depth + 1)
                hdr_end = max((inner[0].line if inner else end) - 1, d.line)
                _emit_sized(out, path, st, lang, d.kind, sym, lines, s, hdr_end, max_chars)
                for k in range(s, hdr_end + 1):
                    covered[k] = True
                process(d.line + 1, end, depth + 1, f"{sym}." if d.kind != "namespace" else prefix, level + 1)
                for k in range(s, end + 1):
                    covered[k] = True
            else:
                for k in range(s, end + 1):
                    covered[k] = True
                kind = "method" if depth > 0 and d.kind == "function" else d.kind
                _emit_sized(out, path, st, lang, kind, sym, lines, s, end, max_chars)

    process(0, len(lines) - 1, 0, "")
    if not out:
        return chunk_lines(path, st, lang, lines, max_chars)
    pre = [i for i, c in enumerate(covered) if not c and lines[i].strip()]
    if pre:
        tl = [lines[i] for i in pre]
        for part, (s, e) in enumerate(split_window(tl, 0, len(tl) - 1, max_chars, 0)):
            out.append(Chunk(source_type=st, text="\n".join(tl[s : e + 1]), path=path,
                             symbol=PurePosixPath(path).stem, kind="module", language=lang,
                             start_line=pre[s] + 1, end_line=pre[e] + 1, part=part))
    return out


# --------------------------------------------------------------------------- indentation languages (Ruby)
def chunk_indent(path: str, lang: str, src: str, max_chars: int) -> list[Chunk]:
    lines = src.splitlines()
    rx = re.compile(r"^(\s*)(?:def|class|module)\s+([\w.:?!]+)")
    out: list[Chunk] = []
    i = 0
    while i < len(lines):
        m = rx.match(lines[i])
        if not m:
            i += 1
            continue
        ind = len(m.group(1))
        j = i + 1
        while j < len(lines) and (
            not lines[j].strip()
            or len(lines[j]) - len(lines[j].lstrip()) > ind
            or (lines[j].strip() == "end" and len(lines[j]) - len(lines[j].lstrip()) == ind)
        ):
            j += 1
        kind = "class" if lines[i].lstrip().startswith(("class", "module")) else "function"
        _emit_sized(out, path, "code", lang, kind, m.group(2), lines, i, j - 1, max_chars)
        i = j
    return out or chunk_lines(path, "code", lang, lines, max_chars)


# --------------------------------------------------------------------------- Markdown / RST / text
_MD_H = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_RST_U = re.compile(r"^([=\-~^\"`#*+])\1{2,}\s*$")
_RST_LEVELS = "=-~^\"`#*+"


def chunk_markdown(path: str, st: str, text: str, max_chars: int, min_chars: int) -> list[Chunk]:
    """Split by heading hierarchy; the heading path ("Install > Docker") becomes the symbol."""
    lines = text.splitlines()
    sections: list[tuple[list[str], int, int]] = []
    stack: list[tuple[int, str]] = []
    cur_start, cur_path, fence = 0, [PurePosixPath(path).stem], False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith(("```", "~~~")):
            fence = not fence
        level, title, skip = 0, "", 1
        if not fence:
            m = _MD_H.match(ln)
            if m:
                level, title = len(m.group(1)), m.group(2)
            elif i + 1 < len(lines) and ln.strip() and _RST_U.match(lines[i + 1]) \
                    and len(lines[i + 1].strip()) >= len(ln.strip()):
                level, title, skip = 1 + _RST_LEVELS.find(lines[i + 1][0]), ln.strip(), 2
        if level:
            if i > cur_start:
                sections.append((cur_path, cur_start, i - 1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, title)]
            cur_path, cur_start = [t for _, t in stack], i
            i += skip
            continue
        i += 1
    if cur_start < len(lines):
        sections.append((cur_path, cur_start, len(lines) - 1))

    merged: list[tuple[list[str], int, int]] = []  # fold tiny sections forward: no heading-only chunks
    for sec in sections:
        if merged and len("\n".join(lines[merged[-1][1] : merged[-1][2] + 1])) < min_chars:
            merged[-1] = (sec[0], merged[-1][1], sec[2])
        else:
            merged.append(sec)
    out: list[Chunk] = []
    for hp, s, e in merged:
        if "\n".join(lines[s : e + 1]).strip():
            _emit_sized(out, path, st, "markdown", "section", " > ".join(hp), lines, s, e, max_chars)
    return out


# --------------------------------------------------------------------------- dispatcher
def chunk_file(path: str, content: str, max_chars: int = 2400, min_chars: int = 80) -> list[Chunk]:
    st, lang = classify(path)
    if st in ("doc", "incident"):
        return chunk_markdown(path, st, content, max_chars, min_chars)
    if lang == "python":
        return chunk_python(path, content, max_chars)
    if lang in _COMPILED:
        return chunk_braces(path, lang, content, max_chars)
    if lang == "ruby":
        return chunk_indent(path, lang, content, max_chars)
    return chunk_lines(path, "code", lang, content.splitlines(), max_chars, kind="config")
