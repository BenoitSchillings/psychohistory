// Page views for the Seldon report (Netlify Function + Netlify Blobs). No cookies; visitors are counted once a day per
// page by a salted hash of their address, never stored raw; obvious bots are skipped.
// POST /api/views {page}  -> {page, total}
// GET  /api/views          -> {pages: {page: {total, days: {YYYY-MM-DD: {views, visitors}}}}}
import { getStore } from "@netlify/blobs";

const PAGES = ["index", "methodology", "data_sources", "ethics", "tldr"];
const VIEWS_PER_VISITOR = 20;      // a day, per page: later calls are not counted (audit 2026-11-10: the count was unlimited)
const BOT = /bot|crawl|spider|slurp|preview|monitor|headless|lighthouse|curl|wget|python|httpclient/i;
const json = (o, status = 200) =>
  new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });

async function hash(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode("seldon-views:" + s));
  return [...new Uint8Array(d)].slice(0, 12).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default async (req, context) => {
  // (draft and preview deploys count separately, so tests never reach the public numbers)
  const prod = (context.deploy?.context ?? "production") === "production";
  const store = getStore({ name: prod ? "page-views" : "page-views-preview", consistency: "strong" });
  if (req.method === "GET") {
    const { blobs } = await store.list({ prefix: "p/" });
    const pages = {};
    for (const b of blobs) {
      const v = await store.get(b.key, { type: "json" });
      if (v) pages[b.key.slice(2)] = v;
    }
    return json({ pages });
  }
  if (req.method === "POST") {
    let page = "index";
    try {
      page = String((await req.json()).page ?? "index");
    } catch {
      /* default page */
    }
    if (!PAGES.includes(page)) return json({ error: "unknown page" }, 400);
    const key = `p/${page}`;
    const rec = (await store.get(key, { type: "json" })) ?? { total: 0, days: {} };
    if (BOT.test(req.headers.get("user-agent") || "")) return json({ page, total: rec.total });
    const day = new Date().toISOString().slice(0, 10);
    const d = (rec.days[day] ??= { views: 0, visitors: 0 });
    const uk = `u/${day}/${page}/${await hash((context.ip || "unknown") + day)}`;
    const seen = Number(await store.get(uk)) || 0;
    if (seen >= VIEWS_PER_VISITOR) return json({ page, total: rec.total });
    await store.set(uk, String(seen + 1));
    if (!seen) d.visitors += 1;
    d.views += 1;
    rec.total += 1;
    await store.setJSON(key, rec);
    return json({ page, total: rec.total });
  }
  return json({ error: "method not allowed" }, 405);
};

export const config = { path: "/api/views" };
