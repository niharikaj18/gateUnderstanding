import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { JSDOM } from "jsdom";

const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const windows = [];
after(() => windows.forEach((w) => w.close())); // the page polls /status on an interval; stop it so node can exit
const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

const STATUS = { repo: "https://github.com/o/r", branch: "main", indexed: true, stale: false, head: "abc1234ffff",
  indexed_commit: "abc1234ffff", chunks: { code: 10 }, stale_files: { total: 0 }, job: { running: false } };
const ANSWER = {
  answer: "Retries live in `charge_customer` [1]. <img src=x onerror=\"window.__pwned=1\"> See [9] too.\n\n- first **bold** point [2]\n- second",
  citations: [{ n: 1 }], warnings: ["Some cited files changed since they were indexed: a.py"],
  index_status: { behind_head: true }, model: "fake", timings_ms: { total_ms: 12, rerank: 3 },
  evidence: [
    { source_type: "code", path: "billing/payments.py", start_line: 10, end_line: 20, symbol: "charge_customer", author: "Alice",
      date: "2024-05-01T00:00:00Z", commit_sha: "deadbeefcafe", pr_number: 12, key: "k1", url: "https://github.com/o/r/blob/x/billing/payments.py",
      ranks: { dense: 1, bm25: 2 }, text: "def charge_customer():\n    return '<script>window.__pwned=2</script>'" },
    { source_type: "commit", path: "", start_line: 0, symbol: "", author: "Bob", date: "2024-04-01", key: "k2",
      url: "javascript:window.__pwned=3", ranks: {}, text: "Fix retry" },
  ],
};

async function boot({ query = ANSWER, status = STATUS, failQuery = null } = {}) {
  const calls = [];
  const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, url: "https://app.example/",
    beforeParse(w) {
      w.matchMedia = () => ({ matches: false });
      w.Element.prototype.scrollIntoView = () => {};
      w.fetch = async (url, opts = {}) => {
        calls.push({ url, opts });
        const reply = (code, body) => ({ ok: code < 400, status: code, statusText: "x", json: async () => body });
        if (url.endsWith("/status")) return status instanceof Error ? Promise.reject(status) : reply(200, status);
        if (url.endsWith("/query")) return failQuery ? reply(failQuery, { detail: "boom" }) : reply(200, query);
        return reply(404, {});
      };
    } });
  windows.push(dom.window);
  await tick();
  return { dom, w: dom.window, d: dom.window.document, calls };
}
async function ask(ctx, q) {
  ctx.d.getElementById("q").value = q;
  ctx.d.getElementById("form").dispatchEvent(new ctx.w.Event("submit", { cancelable: true }));
  await tick(50);
}

test("shows fresh status and example questions", async () => {
  const { d } = await boot();
  assert.match(d.getElementById("status").textContent, /fresh @ abc1234/);
  assert.equal(d.querySelectorAll("#examples button").length, 3);
});

test("stale index shows warning pill and sync button", async () => {
  const { d } = await boot({ status: { ...STATUS, stale: true, stale_files: { total: 3 } } });
  assert.match(d.getElementById("status").textContent, /stale · 3 changed/);
  assert.equal(d.getElementById("sync").hidden, false);
});

test("serverless status: no sync button, wording when file count is unknown", async () => {
  const { d } = await boot({ status: { ...STATUS, stale: true, stale_files: { total: null }, can_sync: false } });
  assert.match(d.getElementById("status").textContent, /stale · new commits/);
  assert.equal(d.getElementById("sync").hidden, true);
});

test("unreachable backend is surfaced", async () => {
  const { d } = await boot({ status: new Error("down") });
  assert.match(d.getElementById("status").textContent, /unreachable/);
});

test("renders answer, citation chips, warnings and evidence", async () => {
  const ctx = await boot(); await ask(ctx, "where is retry?");
  const a = ctx.d.querySelector(".a:not(.loading)");
  assert.ok(a, "answer rendered");
  assert.equal(a.querySelectorAll(".cite").length, 2, "[1] and [2] are chips; [9] (out of range) is not");
  assert.ok(a.textContent.includes("[9]"));
  assert.equal(a.querySelectorAll("li").length, 2);
  assert.ok(a.querySelector("strong"));
  assert.equal(a.querySelectorAll(".warn-box").length, 2);
  assert.equal(a.querySelectorAll(".card").length, 2);
  assert.equal(a.querySelectorAll(".card.cited").length, 1);
  const q = JSON.parse(ctx.calls.find((c) => c.url.endsWith("/query")).opts.body);
  assert.equal(q.question, "where is retry?"); assert.equal(q.mode, "hybrid"); assert.equal(q.source_types, null);
  assert.equal(ctx.calls.find((c) => c.url.endsWith("/query")).url, "/api/query");
});

test("model/repo text cannot inject HTML or run script", async () => {
  const ctx = await boot(); await ask(ctx, "x?");
  assert.equal(ctx.d.querySelector(".a img"), null);
  assert.equal(ctx.d.querySelectorAll(".a script").length, 0);
  assert.equal(ctx.w.__pwned, undefined);
  const links = [...ctx.d.querySelectorAll(".a a")];
  assert.equal(links.length, 1, "javascript: URL is dropped, https URL is kept");
  assert.equal(links[0].rel, "noopener noreferrer");
});

test("clicking a citation chip opens its evidence card", async () => {
  const ctx = await boot(); await ask(ctx, "x?");
  const chip = ctx.d.querySelector(".cite[data-n='1']");
  chip.dispatchEvent(new ctx.w.MouseEvent("click", { bubbles: true }));
  assert.equal(ctx.d.querySelector(".ev").open, true);
  assert.equal(ctx.d.querySelectorAll(".card")[0].open, true);
});

test("API errors are shown, with a hint for 401", async () => {
  const ctx = await boot({ failQuery: 401 }); await ask(ctx, "x?");
  const e = ctx.d.querySelector(".a.err");
  assert.match(e.textContent, /401.*boom.*Deployment Protection.*BACKEND_API_KEY/);
});

test("settings persist and custom API base is used", async () => {
  const ctx = await boot();
  const base = ctx.d.getElementById("apibase");
  base.value = "https://api.example.com/"; base.dispatchEvent(new ctx.w.Event("change"));
  await ask(ctx, "x?");
  assert.ok(ctx.calls.some((c) => c.url === "https://api.example.com/query"));
});
