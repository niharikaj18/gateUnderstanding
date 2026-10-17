import { test, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { createRequire } from "node:module";

const handler = createRequire(import.meta.url)("../api/[...path].js");
let server, seen, port, mode;

before(async () => {
  server = http.createServer((req, res) => {
    let b = ""; req.on("data", (c) => (b += c));
    req.on("end", () => {
      seen.push({ method: req.method, url: req.url, key: req.headers["x-api-key"], body: b });
      if (mode === "401") { res.writeHead(401, { "content-type": "application/json" }); return res.end('{"detail":"nope"}'); }
      if (mode === "slow") return; // never answers
      res.writeHead(200, { "content-type": "application/json" }); res.end('{"ok":true}');
    });
  });
  await new Promise((r) => server.listen(0, r));
  port = server.address().port;
});
after(() => server.close());
beforeEach(() => {
  seen = []; mode = "ok";
  process.env.BACKEND_URL = `http://127.0.0.1:${port}/`; // trailing slash must be tolerated
  process.env.BACKEND_API_KEY = "secret-key"; delete process.env.ALLOW_ADMIN;
});

function call(method, url, body) {
  return new Promise((resolve) => {
    const out = { headers: {}, code: 200 };
    const res = {
      status(c) { out.code = c; return this; }, setHeader(k, v) { out.headers[k.toLowerCase()] = v; },
      json(o) { out.body = JSON.stringify(o); resolve(out); }, send(t) { out.body = t; resolve(out); },
    };
    handler({ method, url, body }, res);
  });
}

test("forwards query with server-side API key and body", async () => {
  const r = await call("POST", "/api/query", { question: "where is retry?" });
  assert.equal(r.code, 200);
  assert.equal(seen[0].url, "/query");
  assert.equal(seen[0].key, "secret-key");
  assert.deepEqual(JSON.parse(seen[0].body), { question: "where is retry?" });
  assert.equal(r.headers["cache-control"], "no-store");
});

test("status is GET-only and health works", async () => {
  assert.equal((await call("GET", "/api/status")).code, 200);
  assert.equal((await call("GET", "/api/health")).code, 200);
  const bad = await call("POST", "/api/status", {});
  assert.equal(bad.code, 405); assert.equal(bad.headers.allow, "GET");
});

test("unknown paths never reach the backend", async () => {
  for (const p of ["/api/docs", "/api/webhook/github", "/api/../etc", "/api/query/extra", "/api/"]) {
    assert.equal((await call("GET", p)).code, 404, p);
  }
  assert.equal(seen.length, 0);
});

test("admin endpoints are blocked unless ALLOW_ADMIN=true", async () => {
  assert.equal((await call("POST", "/api/index", {})).code, 403);
  assert.equal((await call("POST", "/api/sync", {})).code, 403);
  assert.equal(seen.length, 0);
  process.env.ALLOW_ADMIN = "true";
  assert.equal((await call("POST", "/api/sync", {})).code, 200);
  assert.equal(seen.length, 1);
});

test("missing BACKEND_URL is a clear 500", async () => {
  delete process.env.BACKEND_URL;
  const r = await call("GET", "/api/status");
  assert.equal(r.code, 500); assert.match(r.body, /BACKEND_URL/);
});

test("backend errors pass through; unreachable backend is 502", async () => {
  mode = "401";
  const r = await call("POST", "/api/query", { question: "hello" });
  assert.equal(r.code, 401); assert.match(r.body, /nope/);
  process.env.BACKEND_URL = "http://127.0.0.1:1"; // nothing listens here
  assert.equal((await call("GET", "/api/status")).code, 502);
});

test("oversized request bodies are rejected", async () => {
  const r = await call("POST", "/api/query", { question: "x".repeat(25000) });
  assert.equal(r.code, 413); assert.equal(seen.length, 0);
});

test("no key header is sent when BACKEND_API_KEY is unset", async () => {
  delete process.env.BACKEND_API_KEY;
  await call("GET", "/api/status");
  assert.equal(seen[0].key, undefined);
});
