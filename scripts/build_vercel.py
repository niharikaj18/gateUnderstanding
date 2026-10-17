#!/usr/bin/env python3
"""Generate the self-contained Vercel project in vercel/.

Vercel deploys only what is inside the project's Root Directory (`vercel/`), so the query-side Python package
and the web UI are copied in. Run this after changing src/gitcontext or web/index.html; CI runs `--check`.

    python scripts/build_vercel.py          # regenerate vercel/gitcontext and vercel/public/index.html
    python scripts/build_vercel.py --check  # exit 1 if they are out of date
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "src" / "gitcontext", ROOT / "vercel"
EXCLUDE_DIRS = {"__pycache__", "ui", "evaluation"}  # the function never imports these (streamlit UI, eval harness)


def source_files() -> dict[str, bytes]:
    files = {}
    for p in sorted(SRC.rglob("*")):
        rel = p.relative_to(SRC)
        if p.is_file() and not (set(rel.parts[:-1]) & EXCLUDE_DIRS) and p.suffix != ".pyc":
            files[f"gitcontext/{rel.as_posix()}"] = p.read_bytes()
    files["public/index.html"] = (ROOT / "web" / "index.html").read_bytes()
    return files


def current_files() -> dict[str, bytes]:
    files = {}
    for sub in ("gitcontext", "public"):
        for p in sorted((OUT / sub).rglob("*")) if (OUT / sub).exists() else []:
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files[p.relative_to(OUT).as_posix()] = p.read_bytes()
    return files


def drift() -> list[str]:
    want, have = source_files(), current_files()
    return sorted([f"missing  {k}" for k in want.keys() - have.keys()]
                  + [f"stale    {k}" for k in want.keys() & have.keys() if want[k] != have[k]]
                  + [f"extra    {k}" for k in have.keys() - want.keys()])


def build() -> None:
    for sub in ("gitcontext", "public"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    for rel, data in source_files().items():
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


if __name__ == "__main__":
    if "--check" in sys.argv:
        d = drift()
        if d:
            print("vercel/ is out of date; run `python scripts/build_vercel.py`:\n  " + "\n  ".join(d))
            sys.exit(1)
        print("vercel/ is up to date")
    else:
        build()
        print(f"built {len(source_files())} files into {OUT}")
