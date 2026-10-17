from conftest import git


def test_fresh_after_index(engine):
    st = engine.status()
    assert st["indexed"] and not st["stale"] and st["indexed_commit"] == st["head"]


def test_detects_and_resyncs_only_changed_files(engine, repo):
    p = repo / "billing/payments.py"
    p.write_text(p.read_text().replace("range(3)", "range(5)") + '\n\ndef new_helper_fn():\n    """Brand new."""\n    return 42\n')
    (repo / "web/app.js").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "Increase retries; drop invoice view (#2)")

    st = engine.status()
    assert st["stale"] and st["stale_files"]["modified"] == ["billing/payments.py"]
    assert st["stale_files"]["deleted"] == ["web/app.js"]

    r = engine.sync(github=False)
    assert (r.files_modified, r.files_deleted, r.files_added) == (1, 1, 0)
    assert r.commits_added == 1  # only the new commit, not the whole history
    assert not engine.status()["stale"]

    assert engine.search("new_helper_fn", top_k=3).hits[0].chunk.symbol == "new_helper_fn"
    assert not any(h.chunk.path == "web/app.js" for h in engine.search("renderInvoice", top_k=5).hits)
    # vector + lexical stores stay consistent
    assert engine.store.count() == engine.lex.count()


def test_noop_sync_writes_nothing(engine):
    r = engine.sync(github=False)
    assert r.chunks_written == 0 and r.files_modified == 0


def test_answer_flags_stale_cited_file(engine, repo):
    p = repo / "billing/payments.py"
    p.write_text(p.read_text() + "\n# touched\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "touch")
    a = engine.ask("how does charge_customer retry?")
    assert a.index_status["behind_head"]
    assert "billing/payments.py" in a.index_status["stale_cited_files"]
    assert any("changed since" in w for w in a.warnings)
