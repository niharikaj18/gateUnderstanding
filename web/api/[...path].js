// Vercel serverless proxy: browser -> /api/* -> GitContext backend.
//  * keeps BACKEND_URL and BACKEND_API_KEY server-side (never shipped to the browser)
//  * exposes only the read-only endpoints unless ALLOW_ADMIN=true (sync/index trigger heavy work)
//  * no CORS needed: the browser only talks to its own origin
// Env: BACKEND_URL (required), BACKEND_API_KEY (if the backend sets GITCONTEXT_API_KEY), ALLOW_ADMIN ("true" to enable /sync, /index)

const READ = { health: ["GET"], status: ["GET"], query: ["POST"], search: ["POST"] };
const ADMIN = { sync: ["POST"], index: ["POST"] };
const TIMEOUT_MS = 55_000;

module.exports = async function handler(req, res) {
  const backend = (process.env.BACKEND_URL || "").replace(/\/+$/, "");
  if (!backend) return res.status(500).json({ detail: "BACKEND_URL is not configured on the Vercel project" });

  const url = new URL(req.url, "http://local");
  const name = url.pathname.replace(/^\/api\/?/, "").replace(/\/+$/, "");
  const admin = process.env.ALLOW_ADMIN === "true";
  const methods = READ[name] || (admin ? ADMIN[name] : undefined);

  if (!methods) {
    const known = name in ADMIN;
    return res.status(known ? 403 : 404).json({
      detail: known ? "admin endpoints are disabled on this deployment (set ALLOW_ADMIN=true)" : "unknown endpoint",
    });
  }
  if (!methods.includes(req.method)) {
    res.setHeader("Allow", methods.join(", "));
    return res.status(405).json({ detail: "method not allowed" });
  }

  const headers = { accept: "application/json" };
  if (process.env.BACKEND_API_KEY) headers["x-api-key"] = process.env.BACKEND_API_KEY;
  let body;
  if (req.method === "POST") {
    headers["content-type"] = "application/json";
    body = typeof req.body === "string" ? req.body : JSON.stringify(req.body ?? {});
    if (body.length > 20_000) return res.status(413).json({ detail: "request too large" });
  }

  try {
    const upstream = await fetch(`${backend}/${name}`, {
      method: req.method, headers, body, signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader("content-type", upstream.headers.get("content-type") || "application/json");
    res.setHeader("cache-control", "no-store");
    return res.send(text);
  } catch (err) {
    const timeout = err && (err.name === "TimeoutError" || err.name === "AbortError");
    return res.status(timeout ? 504 : 502).json({
      detail: timeout ? "backend timed out" : `cannot reach backend: ${err && err.message}`,
    });
  }
};
