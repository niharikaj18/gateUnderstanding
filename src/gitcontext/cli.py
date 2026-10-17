"""`gitcontext` command line interface."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import Settings
from .engine import Engine

app = typer.Typer(help="GitContext: ask questions about any Git repository.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


def _engine(repo: str | None, data_dir: str | None = None, **over) -> Engine:
    kw = {k: v for k, v in over.items() if v is not None}
    if repo:
        kw["repo"] = repo
    if data_dir:
        kw["data_dir"] = Path(data_dir)
    return Engine(Settings(**kw))


RepoOpt = typer.Option(None, "--repo", "-r", help="Local path, GitHub URL or owner/name (default: $GITCONTEXT_REPO)")


@app.command()
def index(repo: str = RepoOpt, rebuild: bool = typer.Option(False, help="Drop and rebuild the index"),
          no_history: bool = typer.Option(False), no_github: bool = typer.Option(False)):
    """Ingest code, docs, commits, PRs and incident reports."""
    eng = _engine(repo)
    r = eng.index(rebuild=rebuild, history=not no_history, github=not no_github,
                  progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print_json(json.dumps(r.to_dict()))


@app.command()
def sync(repo: str = RepoOpt, no_github: bool = typer.Option(False)):
    """Re-index only files/commits/PRs that changed since the last run."""
    r = _engine(repo).sync(github=not no_github)
    console.print_json(json.dumps(r.to_dict()))


@app.command()
def status(repo: str = RepoOpt):
    """Show index freshness."""
    st = _engine(repo).status()
    flag = "[red]STALE[/red]" if st["stale"] else "[green]fresh[/green]"
    console.print(f"{flag}  HEAD {st['head'][:8]}  indexed {st['indexed_commit'][:8] or '-'}  chunks {st['chunks']}")
    if st["stale"]:
        console.print_json(json.dumps(st["stale_files"]))


@app.command()
def search(query: str, repo: str = RepoOpt, k: int = 8, mode: str = "hybrid", no_rerank: bool = False):
    """Retrieval only; shows which ranker surfaced each hit."""
    res = _engine(repo).search(query, mode=mode, rerank=not no_rerank, top_k=k)
    t = Table("#", "type", "where", "symbol", "author", "date", "ranks", "score")
    for i, h in enumerate(res.hits, 1):
        c = h.chunk
        t.add_row(str(i), c.source_type, c.label(), c.symbol[:40], c.author, c.date[:10],
                  ",".join(f"{k}:{v}" for k, v in h.ranks.items()), f"{h.score:.2f}")
    console.print(t)
    console.print(f"[dim]{ {k: round(v) for k, v in res.timings_ms.items()} } ms[/dim]")


@app.command()
def ask(question: str, repo: str = RepoOpt, k: int = 8, mode: str = "hybrid", no_rerank: bool = False,
        source: list[str] = typer.Option(None, "--source", help="code|doc|commit|pr|issue|incident")):
    """Ask a question; answer comes with citations to files/commits/PRs."""
    a = _engine(repo).ask(question, top_k=k, mode=mode, rerank=not no_rerank, source_types=source or None)
    console.print(Markdown(a.answer))
    if a.citations:
        console.print("\n[bold]Sources[/bold]")
        for c in a.citations:
            console.print(f" [{c['n']}] {c['label']} — {c['author']} {c['date'][:10]} {c['url']}")
    for w in a.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    if a.index_status.get("behind_head"):
        console.print("[yellow]! index is behind HEAD — run `gitcontext sync`[/yellow]")
    console.print(f"[dim]{ {k: round(v) for k, v in a.timings_ms.items()} } ms · {a.model}[/dim]")


@app.command()
def watch(repo: str = RepoOpt, interval: int = typer.Option(60, help="seconds between polls")):
    """Poll the repo and incrementally re-index on change (stale-index auto-repair)."""
    import time

    eng = _engine(repo)
    console.print(f"watching every {interval}s (Ctrl-C to stop)")
    while True:
        st = eng.status()
        if st["stale"] or not st["indexed"]:
            console.print(f"[yellow]stale ({st['stale_files']['total']} files) — syncing[/yellow]")
            console.print_json(json.dumps(eng.sync().to_dict()))
        time.sleep(interval)


@app.command()
def serve(repo: str = RepoOpt, host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import os

    import uvicorn

    if repo:
        os.environ["GITCONTEXT_REPO"] = repo
    uvicorn.run("gitcontext.api:app_factory", factory=True, host=host, port=port)


@app.command()
def ui(api: str = "http://127.0.0.1:8000", port: int = 8501):
    """Launch the Streamlit web UI (talks to the API)."""
    script = Path(__file__).parent / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port), "--",
                    "--api", api])


@eval_app.command("generate")
def eval_generate(repo: str = RepoOpt, out: str = "eval_out/dataset.jsonl", n: int = 60, seed: int = 7):
    """Build a retrieval eval set from the indexed repo (docstrings, error messages, symbols, commits)."""
    from .evaluation.dataset import generate_dataset

    n_written = generate_dataset(_engine(repo), Path(out), n=n, seed=seed)
    console.print(f"wrote {n_written} questions to {out}")


@eval_app.command("run")
def eval_run(dataset: str = "eval_out/dataset.jsonl", repo: str = RepoOpt, out: str = "eval_out",
             k: int = 5, answers: bool = typer.Option(False, help="Also score generated answers (needs an LLM)")):
    """Run retrieval ablations (dense / bm25 / hybrid / hybrid+rerank) and optional answer metrics."""
    from .evaluation.runner import run_eval

    report = run_eval(_engine(repo), Path(dataset), Path(out), k=k, answers=answers)
    console.print(Markdown(report))


if __name__ == "__main__":
    app()
