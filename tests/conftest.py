import os
import subprocess
from pathlib import Path

import pytest

from gitcontext.config import Settings
from gitcontext.engine import Engine

FILES = {
    "README.md": "# Demo\n\nDemo service for billing.\n\n## Retry policy\n\nRequests to the payment gateway are retried 3 times with exponential backoff.\n",
    "docs/postmortem-2024-01.md": "# Postmortem: duplicate charges\n\nRoot cause: the idempotency key was not forwarded by `charge_customer`.\n",
    "billing/payments.py": '''"""Payment helpers."""
import time


class GatewayError(Exception):
    pass


def charge_customer(customer_id, amount):
    """Charge a customer through the payment gateway, retrying on transient failures."""
    for attempt in range(3):
        try:
            return _post(customer_id, amount)
        except TimeoutError:
            time.sleep(2 ** attempt)
    raise GatewayError("payment gateway unreachable after 3 attempts")


def _post(customer_id, amount):
    return {"ok": True}


def refund_order(order_id):
    """Refund a previously captured order."""
    raise ValueError("order has not been captured yet")
''',
    "pkg/shapes.py": '''class Circle:
    def area(self):
        """Area of a circle."""
        return 3.14 * self.r ** 2


class Square:
    def area(self):
        """Area of a square."""
        return self.side ** 2
''',
    "web/app.js": "export function renderInvoice(inv) {\n  return `<div>${inv.id}</div>`;\n}\n",
}


def git(repo: Path, *args, env=None):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Alice", "-c", "user.email=a@x.io", *args],
                   check=True, capture_output=True, env=env)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "demo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    for p, c in FILES.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(c)
    git(r, "add", "-A")
    git(r, "commit", "-qm", "Initial billing service (#1)")
    return r


# Set GITCONTEXT_TEST_QDRANT_URL=http://localhost:6333 to also run everything against a real Qdrant server.
_SERVER = os.environ.get("GITCONTEXT_TEST_QDRANT_URL")


@pytest.fixture(params=["local", "qdrant"] + (["qdrant-server"] if _SERVER else []))
def engine(request, repo, tmp_path):
    """Every engine-level test runs against each storage backend (SQLite+Qdrant, all-in-Qdrant in-memory,
    and optionally all-in-Qdrant on a real server)."""
    server = request.param == "qdrant-server"
    s = Settings(repo=str(repo), data_dir=tmp_path / "data", embedding_provider="hash", reranker="lexical",
                 llm_provider="none", qdrant_in_memory=not server, qdrant_url=_SERVER if server else None,
                 github_token=None, backend="local" if request.param == "local" else "qdrant")
    e = Engine(s, llm=None)
    e.index()
    return e


@pytest.fixture(params=["local", "qdrant"])
def backend(request):
    """Settings overrides for a hand-built Engine (tests that need custom chunking)."""
    if request.param == "local":
        return {"backend": "local", "qdrant_in_memory": True}
    return {"backend": "qdrant", "qdrant_in_memory": True}
