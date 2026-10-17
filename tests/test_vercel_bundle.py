import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location("build_vercel", ROOT / "scripts" / "build_vercel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vercel_bundle_is_in_sync_with_source():
    """vercel/ is generated (Vercel can only deploy files inside its Root Directory). Run scripts/build_vercel.py."""
    assert _load_script().drift() == []


def test_bundle_is_importable_without_cli_or_ui_dependencies():
    """The function must not import typer/rich/streamlit/uvicorn/git-only modules."""
    code = (
        "import sys; sys.path.insert(0, 'vercel')\n"
        "import importlib\n"
        "for banned in ('typer', 'rich', 'streamlit', 'uvicorn', 'fastembed'):\n"
        "    sys.modules[banned] = None  # make any import of them fail loudly\n"
        "m = importlib.import_module('api.index')\n"
        "from fastapi.testclient import TestClient\n"
        "assert TestClient(m.app).get('/api/health').json()['mode'] == 'serverless'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-1500:]


def test_index_workflow_is_valid_and_opt_in():
    yaml = __import__("pytest").importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "index.yml").read_text())
    job = wf["jobs"]["index"]
    assert "GITCONTEXT_INDEXING" in job["if"]  # does nothing until the owner opts in
    assert job["env"]["GITCONTEXT_BACKEND"] == "qdrant"
    assert any("gitcontext sync" in (s.get("run") or "") for s in job["steps"])


def test_cli_sync_on_empty_index_does_a_full_first_index(repo, tmp_path, monkeypatch):
    """The workflow runs `gitcontext sync`; on a brand-new Qdrant it must index everything, not no-op."""
    import json

    from typer.testing import CliRunner

    from gitcontext.cli import app

    for k, v in {"GITCONTEXT_BACKEND": "qdrant", "GITCONTEXT_QDRANT_IN_MEMORY": "true", "GITCONTEXT_EMBEDDING_PROVIDER": "hash",
                 "GITCONTEXT_RERANKER": "none", "GITCONTEXT_LLM_PROVIDER": "none", "GITCONTEXT_REPO": str(repo),
                 "GITCONTEXT_DATA_DIR": str(tmp_path / "d")}.items():
        monkeypatch.setenv(k, v)
    res = CliRunner().invoke(app, ["sync", "--no-github"])
    assert res.exit_code == 0, res.output
    report = json.loads(res.output[res.output.index("{"):])
    assert report["files_added"] >= 4 and report["chunks_written"] > 5 and report["commits_added"] >= 1
