"""Streamlit chat UI. Talks to the FastAPI backend: `gitcontext serve` then `gitcontext ui`."""
import argparse
import sys

import httpx
import streamlit as st

p = argparse.ArgumentParser()
p.add_argument("--api", default="http://127.0.0.1:8000")
args, _ = p.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="GitContext", page_icon="🧭", layout="wide")
api = st.sidebar.text_input("API URL", args.api)
client = httpx.Client(base_url=api, timeout=180)

ICON = {"code": "🧩", "doc": "📄", "commit": "🔀", "pr": "🔃", "issue": "🐞", "incident": "🚨"}


def call(method, path, **kw):
    try:
        r = client.request(method, path, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"{e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"API unreachable: {e}")
    return None


# ---- sidebar: freshness + controls ----------------------------------------------------------
status = call("GET", "/status")
with st.sidebar:
    st.title("🧭 GitContext")
    if status:
        st.caption(f"{status['repo']} · {status.get('branch', '')}")
        if not status["indexed"]:
            st.warning("Not indexed yet.")
            if st.button("Index repository", type="primary"):
                call("POST", "/index", json={})
                st.info("Indexing started — refresh in a minute.")
        elif status["stale"]:
            n = status["stale_files"]["total"]
            st.warning(f"Index is stale: HEAD {status['head'][:7]} vs indexed {status['indexed_commit'][:7]} "
                       f"({n} changed files)")
            if st.button("Sync changed files", type="primary"):
                with st.spinner("Re-indexing changed files…"):
                    st.json(call("POST", "/sync") or {})
                st.rerun()
        else:
            st.success(f"Index fresh @ {status['indexed_commit'][:7]}")
        st.caption("Chunks: " + ", ".join(f"{k} {v}" for k, v in status["chunks"].items()))
        if status.get("job", {}).get("running"):
            st.info("Indexing job running…")
        m = status.get("models", {})
        st.caption(f"embedder: {m.get('embedder')}\n\nreranker: {m.get('reranker')}\n\nllm: {m.get('llm') or 'none (extractive)'}")
    st.divider()
    top_k = st.slider("Evidence chunks (top-k)", 3, 20, 8)
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    rerank = st.toggle("Cross-encoder rerank", True)
    types = st.multiselect("Sources", list(ICON), default=list(ICON))
    prefix = st.text_input("Limit to path prefix", "")

st.header("Ask your codebase")
st.caption("Answers are grounded in code, docs, commits, PR discussions and incident reports, with citations.")

if "chat" not in st.session_state:
    st.session_state.chat = []


def render(ans):
    st.markdown(ans["answer"])
    for w in ans.get("warnings", []):
        st.warning(w)
    if ans.get("index_status", {}).get("behind_head"):
        st.info("The index is behind the repository HEAD; results may miss recent changes.")
    cites = {c["n"]: c for c in ans.get("citations", [])}
    with st.expander(f"Evidence ({len(ans['evidence'])} chunks · {ans['timings_ms'].get('total_ms', 0):.0f} ms · {ans['model']})"):
        for i, e in enumerate(ans["evidence"], 1):
            cited = "✅ cited" if i in cites else ""
            title = f"[{i}] {ICON.get(e['source_type'], '')} {e['source_type']} · {e['path'] or e['key']}"
            if e["start_line"]:
                title += f":{e['start_line']}-{e['end_line']}"
            st.markdown(f"**{title}** {cited}")
            meta = [f"`{e['symbol']}`" if e["symbol"] else "", f"👤 {e['author']}" if e["author"] else "",
                    f"📅 {e['date'][:10]}" if e["date"] else "",
                    f"commit `{e['commit_sha'][:7]}`" if e["commit_sha"] and e["source_type"] != "commit" else "",
                    f"PR #{e['pr_number']}" if e["pr_number"] else "",
                    f"[link]({e['url']})" if e["url"] else "",
                    "ranks: " + ", ".join(f"{k}#{v}" for k, v in e["ranks"].items())]
            st.caption(" · ".join(m for m in meta if m))
            st.code(e["text"][:1500], language=e["language"] or None if e["source_type"] == "code" else "markdown")
    with st.expander("Latency breakdown"):
        st.json({k: round(v, 1) for k, v in ans["timings_ms"].items()})


for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        render(turn["a"])

if q := st.chat_input("e.g. Where is request retry handled, and why was it changed?"):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"), st.spinner("Retrieving and reasoning…"):
        a = call("POST", "/query", json={"question": q, "top_k": top_k, "mode": mode, "rerank": rerank,
                                         "source_types": types or None, "path_prefix": prefix or None})
        if a:
            render(a)
            st.session_state.chat.append({"q": q, "a": a})
