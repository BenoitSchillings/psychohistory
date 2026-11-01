// Reader notes for the Seldon report (Netlify Function + Netlify Blobs).
// GET    /api/notes            -> {notes: [...]}   every note, oldest first
// POST   /api/notes            {text, type, anchor, x, y, name, page} -> {note}
// DELETE /api/notes?id=ID      header x-admin-key: NOTES_ADMIN_KEY (site environment variable) -> {ok: true}
// Notes are stored and returned as plain data; the page renders them as text only. Per visitor (hashed IP, never stored
// raw) at most DAILY_LIMIT notes a day.
import { getStore } from "@netlify/blobs";

const TYPES = ["doubt", "missing", "comment"];
const MAX_TEXT = 1000;
const DAILY_LIMIT = 30;

const json = (o, status = 200) =>
  new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
const clamp01 = (v) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

async function hash(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("seldon-notes:" + s));
  return [...new Uint8Array(d)].slice(0, 12).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default async (req, context) => {
  // (draft and preview deploys keep their own notes, so tests never reach the public page)
  const prod = (context.deploy?.context ?? "production") === "production";
  const store = getStore({ name: prod ? "reader-notes" : "reader-notes-preview", consistency: "strong" });
  if (req.method === "GET") {
    const { blobs } = await store.list({ prefix: "n/" });
    const notes = (await Promise.all(blobs.map((b) => store.get(b.key, { type: "json" })))).filter(Boolean);
    notes.sort((a, b) => (a.ts < b.ts ? -1 : 1));
    return json({ notes });
  }
  if (req.method === "POST") {
    let b;
    try {
      b = await req.json();
    } catch {
      return json({ error: "bad request" }, 400);
    }
    const text = String(b.text ?? "").trim().slice(0, MAX_TEXT);
    if (!text) return json({ error: "empty note" }, 400);
    const day = new Date().toISOString().slice(0, 10);
    const rk = `r/${day}/${await hash(context.ip || "unknown")}`;
    const n = (await store.get(rk, { type: "json" }))?.n ?? 0;
    if (n >= DAILY_LIMIT) return json({ error: "daily limit reached" }, 429);
    await store.setJSON(rk, { n: n + 1 });
    const id = `${Date.now().toString(36)}-${crypto.randomUUID().slice(0, 8)}`;
    const note = {
      id,
      text,
      type: TYPES.includes(b.type) ? b.type : "comment",
      anchor: String(b.anchor ?? "").replace(/[^A-Za-z0-9_-]/g, "").slice(0, 120),
      x: clamp01(Number(b.x)),
      y: clamp01(Number(b.y)),
      name: String(b.name ?? "").trim().slice(0, 60),
      page: String(b.page ?? "index").replace(/[^a-z_]/g, "").slice(0, 40) || "index",
      ts: new Date().toISOString(),
    };
    await store.setJSON(`n/${id}`, note);
    return json({ note });
  }
  if (req.method === "DELETE") {
    const key = req.headers.get("x-admin-key");
    const admin = Netlify.env.get("NOTES_ADMIN_KEY");
    if (!admin || key !== admin) return json({ error: "forbidden" }, 403);
    const id = (new URL(req.url).searchParams.get("id") || "").replace(/[^a-z0-9-]/g, "");
    if (!id) return json({ error: "missing id" }, 400);
    await store.delete(`n/${id}`);
    return json({ ok: true });
  }
  return json({ error: "method not allowed" }, 405);
};

export const config = { path: "/api/notes" };
