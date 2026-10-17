"""Vercel entrypoint: an ASGI app serving /api/* (rewritten here by vercel.json)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -> `import gitcontext`

from gitcontext.serverless import create_serverless_app  # noqa: E402

app = create_serverless_app()
