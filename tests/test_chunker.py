from gitcontext.ingest.chunker import chunk_file, classify, should_index

PY = '''"""mod"""
import os

@decorator
def foo(a):
    """Foo."""
    return a

class Big:
    """Big."""
    def m1(self):
        return 1

    def m2(self):
        return 2
'''


def test_python_splits_by_symbol_not_tokens():
    cs = chunk_file("pkg/mod.py", PY, max_chars=80)
    by = {c.symbol: c for c in cs}
    assert by["foo"].kind == "function" and "@decorator" in by["foo"].text  # decorators stay attached
    assert by["foo"].extra["docstring"] == "Foo."
    assert by["Big"].kind == "class" and "Big.m1" in by and "Big.m2" in by  # big class -> per-method chunks
    assert by["mod"].kind == "module" and "import os" in by["mod"].text


def test_small_class_stays_whole():
    cs = chunk_file("m.py", PY, max_chars=5000)
    assert {c.symbol for c in cs} == {"foo", "Big", "m"}


def test_syntax_error_falls_back():
    cs = chunk_file("bad.py", "def (:\n  x", max_chars=100)
    assert cs and cs[0].kind == "block"


def test_js_methods_and_single_line_blocks():
    js = "class Svc {\n  constructor() { this.a = 1; }\n  run(x) {\n    if (x) { return 1; }\n    return 2;\n  }\n}\nfunction lone() { return 1; }\n"
    syms = {c.symbol for c in chunk_file("a.js", js, max_chars=60)}
    assert {"Svc.constructor", "Svc.run", "lone"} <= syms


def test_braces_in_strings_ignored():
    js = 'function a() {\n  const s = "}";\n  return s;\n}\nfunction b() { return 1; }\n'
    cs = {c.symbol: c for c in chunk_file("a.js", js, 2000)}
    assert cs["a"].end_line == 4 and "b" in cs


def test_go_and_rust():
    go = "package m\nfunc (s *S) Handle() error {\n\treturn nil\n}\nfunc main() {\n}\n"
    assert {"Handle", "main"} <= {c.symbol for c in chunk_file("a.go", go)}
    rs = "impl A {\n    pub fn new() -> Self { A }\n    fn get<'a>(&'a self) -> &'a i32 { &self.x }\n}\n"
    assert {"A.new", "A.get"} <= {c.symbol for c in chunk_file("a.rs", rs, max_chars=40)}


def test_markdown_heading_path_and_fences():
    md = "# T\nintro text long enough to keep as its own chunk ok.\n## Install\n```\n# not heading\n```\nrun\n## Use\nuse it a lot of the time because good.\n"
    syms = [c.symbol for c in chunk_file("README.md", md, 2000, 20)]
    assert syms == ["T", "T > Install", "T > Use"]


def test_classification_and_filters():
    assert classify("docs/postmortem-1.md")[0] == "incident"
    assert classify("README.md")[0] == "doc"
    assert classify("a/b.py") == ("code", "python")
    assert not should_index("node_modules/x/index.js", 10, 1000)
    assert not should_index("big.py", 10_000_000, 1000)
    assert not should_index("package-lock.json", 10, 1000)
