/* (bottom left: Netlify's own badge sits bottom right)
   Reader notes: post-its that any reader can stick on the page where they are not convinced, where something is
   missing, or to comment. Three layers (one per kind), each shown or hidden; every reader sees every note.
   Notes are anchored to the nearest element with an id (a section, a card, a chart) at a relative position inside it,
   so they stay in place across screen sizes. Stored by the site's /api/notes function; rendered as text only. */
(function () {
  "use strict";
  const API = "/api/notes";
  const PAGE = (location.pathname.replace(/^\/+|\.html$/g, "") || "index").replace(/[^a-z_]/g, "") || "index";
  const KINDS = {
    doubt: { label: "Not convinced", bg: "#ffe873", edge: "#d9b800" },
    missing: { label: "Something's missing", bg: "#ffb3c7", edge: "#e0577d" },
    comment: { label: "Comment", bg: "#b8dcff", edge: "#4a8fd6" },
  };
  const KEY = "seldon-notes-layers";
  let layers;
  try { layers = JSON.parse(localStorage.getItem(KEY)) || null; } catch (e) { layers = null; }
  layers = layers || { on: true, doubt: true, missing: true, comment: true };
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(layers)); } catch (e) { /* private window */ } };
  let notes = [], placing = false, openId = null;

  const css = `
  .sn-bar{position:fixed;left:16px;bottom:16px;z-index:2147483000;font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1a1a1a}
  .sn-btn{border:1px solid rgba(0,0,0,.25);background:#fff;color:#1a1a1a;border-radius:6px;padding:7px 11px;cursor:pointer;font:inherit;box-shadow:0 2px 8px rgba(0,0,0,.15)}
  .sn-btn:focus-visible{outline:2px solid #3446c9;outline-offset:2px}
  .sn-btn.sn-primary{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
  .sn-panel{position:absolute;left:0;bottom:44px;width:min(320px,calc(100vw - 32px));max-height:min(70vh,560px);overflow:auto;background:#fff;color:#1a1a1a;border:1px solid rgba(0,0,0,.2);border-radius:8px;box-shadow:0 8px 28px rgba(0,0,0,.2);padding:12px;display:grid;gap:10px}
  .sn-panel[hidden]{display:none}
  .sn-row{display:flex;align-items:center;gap:8px}
  .sn-sw{width:14px;height:14px;border-radius:3px;border:1px solid rgba(0,0,0,.3);flex:none}
  .sn-list{display:grid;gap:6px;margin:0;padding:0;list-style:none}
  .sn-list li{padding:6px 8px;border-radius:4px;cursor:pointer;font-size:12px}
  .sn-muted{color:#555;font-size:12px}
  .sn-layer{position:absolute;left:0;top:0;width:0;height:0;z-index:8000}
  .sn-note{position:absolute;transform:translate(-4px,-4px)}
  .sn-tab{width:22px;height:22px;border-radius:3px 3px 3px 9px;border:1px solid;box-shadow:0 2px 5px rgba(0,0,0,.3);cursor:pointer;padding:0;font:600 11px system-ui,sans-serif;color:#1a1a1a}
  .sn-tab:focus-visible{outline:2px solid #3446c9;outline-offset:2px}
  .sn-card{position:absolute;left:0;top:26px;width:min(260px,calc(100vw - 40px));padding:10px 12px;border-radius:3px 3px 3px 14px;border:1px solid;box-shadow:0 6px 18px rgba(0,0,0,.28);color:#1a1a1a;font:13px/1.45 system-ui,sans-serif;white-space:pre-wrap;word-wrap:break-word}
  .sn-card .sn-meta{display:block;margin-top:6px;font-size:11px;color:#333}
  .sn-form{position:absolute;z-index:9100;width:min(280px,calc(100vw - 32px));background:#fff;color:#1a1a1a;border:1px solid rgba(0,0,0,.25);border-radius:8px;box-shadow:0 8px 28px rgba(0,0,0,.25);padding:10px;display:grid;gap:8px;font:13px system-ui,sans-serif}
  .sn-form textarea,.sn-form input,.sn-form select{font:inherit;width:100%;box-sizing:border-box;padding:6px;border:1px solid #aaa;border-radius:4px;background:#fff;color:#1a1a1a}
  .sn-placing,.sn-placing *{cursor:crosshair !important}
  .sn-hint{position:fixed;left:50%;top:12px;transform:translateX(-50%);z-index:9200;background:#1a1a1a;color:#fff;padding:7px 12px;border-radius:6px;font:13px system-ui,sans-serif}
  @media print{.sn-bar,.sn-layer{display:none}}`;
  const st = document.createElement("style");
  st.textContent = css;
  document.head.appendChild(st);

  const el = (tag, attrs, kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") e.textContent = v;
      else if (k === "style") Object.assign(e.style, v);
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of kids || []) e.appendChild(c);
    return e;
  };

  // anchor: the nearest ancestor with an id (excluding our own widgets), and the click's relative position inside it
  function anchorFor(target, px, py) {
    let a = target;
    while (a && a !== document.body && !(a.id && !a.closest(".sn-bar,.sn-layer,.sn-form"))) a = a.parentElement;
    if (!a || a === document.body) a = document.documentElement;
    const r = a.getBoundingClientRect();
    return { anchor: a === document.documentElement ? "" : a.id,
             x: r.width ? (px - r.left) / r.width : 0, y: r.height ? (py - r.top) / r.height : 0 };
  }
  function positionOf(n) {
    const a = n.anchor ? document.getElementById(n.anchor) : document.documentElement;
    if (!a || (a.offsetParent === null && a !== document.documentElement)) return null;   // hidden section
    const r = a.getBoundingClientRect();
    return { left: r.left + window.scrollX + n.x * r.width, top: r.top + window.scrollY + n.y * r.height };
  }

  const layer = el("div", { class: "sn-layer", "aria-label": "Reader notes" });
  document.body.appendChild(layer);
  const count = el("span", { text: "" });
  const panel = el("div", { class: "sn-panel", hidden: "", role: "dialog", "aria-label": "Reader notes" });
  const toggle = el("button", { class: "sn-btn sn-primary", type: "button", "aria-expanded": "false",
    onclick: () => { const h = panel.hasAttribute("hidden"); h ? panel.removeAttribute("hidden") : panel.setAttribute("hidden", ""); toggle.setAttribute("aria-expanded", String(h)); } },
    [document.createTextNode("Reader notes "), count]);
  const bar = el("div", { class: "sn-bar" }, [panel, toggle]);
  document.body.appendChild(bar);

  function renderPanel() {
    panel.textContent = "";
    panel.appendChild(el("div", { style: { fontWeight: "600" }, text: "Post-its from readers" }));
    panel.appendChild(el("div", { class: "sn-muted", text: "Stick a note where you are not convinced, where something is missing, or to comment. Every reader sees every note." }));
    panel.appendChild(el("button", { class: "sn-btn sn-primary", type: "button", text: "+ Add a note", onclick: startPlacing }));
    const master = el("input", { type: "checkbox", id: "sn-all" });
    master.checked = layers.on;
    master.addEventListener("change", () => { layers.on = master.checked; save(); renderNotes(); });
    panel.appendChild(el("label", { class: "sn-row", for: "sn-all" }, [master, document.createTextNode("Show notes on the page")]));
    for (const [k, v] of Object.entries(KINDS)) {
      const cb = el("input", { type: "checkbox", id: "sn-l-" + k });
      cb.checked = layers[k];
      cb.addEventListener("change", () => { layers[k] = cb.checked; save(); renderNotes(); renderPanel(); });
      const n = notes.filter((x) => x.type === k && x.page === PAGE).length;
      panel.appendChild(el("label", { class: "sn-row", for: "sn-l-" + k },
        [cb, el("span", { class: "sn-sw", style: { background: v.bg, borderColor: v.edge } }), document.createTextNode(`${v.label} (${n})`)]));
    }
    const mine = notes.filter((x) => x.page === PAGE && layers[x.type]);
    const ul = el("ul", { class: "sn-list" });
    for (const n of mine.slice().reverse()) {
      ul.appendChild(el("li", { style: { background: KINDS[n.type].bg }, tabindex: "0", text: (n.text.length > 90 ? n.text.slice(0, 90) + "…" : n.text),
        onclick: () => jump(n), onkeydown: (e) => { if (e.key === "Enter") jump(n); } }));
    }
    if (mine.length) panel.appendChild(ul);
    else panel.appendChild(el("div", { class: "sn-muted", text: "No notes in the visible layers yet." }));
  }
  function jump(n) {
    const p = positionOf(n);
    if (!p) return;
    layers.on = true; layers[n.type] = true; save(); renderNotes();
    window.scrollTo({ top: Math.max(0, p.top - window.innerHeight / 3), behavior: "smooth" });
    openId = n.id; renderNotes();
  }

  function renderNotes() {
    layer.textContent = "";
    const here = notes.filter((n) => n.page === PAGE);
    count.textContent = here.length ? `(${here.length})` : "";
    if (!layers.on) return;
    const seen = {};
    for (const n of here) {
      if (!layers[n.type]) continue;
      const p = positionOf(n);
      if (!p) continue;
      const key = Math.round(p.left / 12) + ":" + Math.round(p.top / 12);           // stack notes dropped at one spot
      const k = (seen[key] = (seen[key] || 0) + 1) - 1;
      const kind = KINDS[n.type];
      const wrap = el("div", { class: "sn-note", style: { left: p.left + 7 * k + "px", top: p.top + 7 * k + "px", zIndex: String(8000 + k) } });
      const tab = el("button", { class: "sn-tab", type: "button", "aria-label": `${kind.label}: ${n.text.slice(0, 80)}`,
        style: { background: kind.bg, borderColor: kind.edge }, text: n.type === "doubt" ? "?" : n.type === "missing" ? "+" : "•",
        onclick: (e) => { e.stopPropagation(); openId = openId === n.id ? null : n.id; renderNotes(); } });
      wrap.appendChild(tab);
      if (openId === n.id) {
        const card = el("div", { class: "sn-card", style: { background: kind.bg, borderColor: kind.edge } }, [
          el("strong", { text: kind.label }), document.createTextNode("\n" + n.text),
          el("span", { class: "sn-meta", text: `${n.name ? n.name + " · " : ""}${new Date(n.ts).toLocaleDateString()}` })]);
        // keep the card on screen near the right edge
        if (p.left + 270 > window.scrollX + document.documentElement.clientWidth) card.style.left = "auto", card.style.right = "0";
        wrap.appendChild(card);
        wrap.style.zIndex = "8900";
      }
      layer.appendChild(wrap);
    }
  }

  let hint = null;
  function startPlacing() {
    if (placing) return;
    placing = true;
    panel.setAttribute("hidden", "");
    document.documentElement.classList.add("sn-placing");
    hint = el("div", { class: "sn-hint", text: "Click where the note belongs (Esc to cancel)" });
    document.body.appendChild(hint);
  }
  function stopPlacing() {
    placing = false;
    document.documentElement.classList.remove("sn-placing");
    if (hint) hint.remove(), hint = null;
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { stopPlacing(); closeForm(); } });
  document.addEventListener("click", (e) => {
    if (!placing) {
      if (openId && !e.target.closest(".sn-note")) { openId = null; renderNotes(); }
      return;
    }
    if (e.target.closest(".sn-bar,.sn-form")) return;
    e.preventDefault(); e.stopPropagation();
    stopPlacing();
    openForm(e.target, e.clientX, e.clientY);
  }, true);

  let form = null;
  function closeForm() { if (form) form.remove(), form = null; }
  function openForm(target, cx, cy) {
    closeForm();
    const a = anchorFor(target, cx, cy);
    const sel = el("select", { "aria-label": "Kind of note" }, Object.entries(KINDS).map(([k, v]) => el("option", { value: k, text: v.label })));
    const ta = el("textarea", { rows: "4", maxlength: "1000", placeholder: "What is unconvincing or missing here?", "aria-label": "Note" });
    const nm = el("input", { type: "text", maxlength: "60", placeholder: "Your name (optional)", "aria-label": "Name" });
    const msg = el("div", { class: "sn-muted", text: "Visible to every reader. Text only." });
    const ok = el("button", { class: "sn-btn sn-primary", type: "button", text: "Stick it", onclick: async () => {
      const text = ta.value.trim();
      if (!text) { ta.focus(); return; }
      ok.disabled = true; msg.textContent = "Saving…";
      try {
        const r = await fetch(API, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...a, text, type: sel.value, name: nm.value, page: PAGE }) });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.status);
        notes.push(j.note); layers.on = true; layers[j.note.type] = true; save();
        openId = j.note.id; closeForm(); renderNotes(); renderPanel();
      } catch (err) { ok.disabled = false; msg.textContent = "Could not save the note (" + err.message + ")."; }
    } });
    form = el("div", { class: "sn-form", role: "dialog", "aria-label": "New note" }, [sel, ta, nm,
      el("div", { class: "sn-row" }, [ok, el("button", { class: "sn-btn", type: "button", text: "Cancel", onclick: closeForm })]), msg]);
    const w = Math.min(280, document.documentElement.clientWidth - 32);
    form.style.left = Math.max(16, Math.min(cx + window.scrollX, window.scrollX + document.documentElement.clientWidth - w - 16)) + "px";
    form.style.top = cy + window.scrollY + 12 + "px";
    document.body.appendChild(form);
    ta.focus();
  }

  async function load() {
    try {
      const r = await fetch(API, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      notes = ((await r.json()).notes || []).filter((n) => KINDS[n.type]);
    } catch (e) { notes = []; }
    renderNotes(); renderPanel();
  }
  // charts render after load and change the layout: re-place notes when the page settles or resizes
  let t = null;
  const relayout = () => { clearTimeout(t); t = setTimeout(renderNotes, 150); };
  window.addEventListener("resize", relayout);
  if ("ResizeObserver" in window) new ResizeObserver(relayout).observe(document.body);
  load();
  setTimeout(renderNotes, 2000);
})();
