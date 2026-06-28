"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid?.nodeType ? kid : document.createTextNode(kid ?? ""));
  return n;
};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

const csv = (s) => (s || "").split(",").map((x) => x.trim()).filter(Boolean);

// --- tabs --------------------------------------------------------------------
document.querySelectorAll("#nav button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#nav button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "datasets") loadDatasets();
    if (b.dataset.tab === "gpu") loadGpu();
  })
);

// --- job widget (poll until terminal) ----------------------------------------
function renderJob(container, job) {
  container.classList.remove("empty");
  const pct = job.total ? Math.round((job.done / job.total) * 100) : (job.status === "done" ? 100 : 0);
  const fail = job.failures ? `, ${job.failures} failed` : "";
  let resultLine = "";
  if (job.status === "done" && job.result) {
    const r = job.result;
    if (r.open_url) resultLine = `<a class="result-link" href="${r.open_url}" target="_blank" rel="noopener">Open in Label Studio &rarr;</a>`;
    else resultLine = `<span class="muted">${escapeHtml(summarize(r))}</span>`;
  }
  container.innerHTML = "";
  container.append(
    el("div", { class: "head" },
      el("span", { class: "badge " + job.status }, job.status),
      el("span", { class: "muted" }, job.total ? `${job.done}/${job.total}${fail}` : (job.status === "running" ? "working…" : ""))
    ),
    el("div", { class: "bar" }, el("i", { style: `width:${pct}%` })),
  );
  if (job.error) container.append(el("div", { class: "err-text" }, job.error));
  if (resultLine) container.append(el("div", { html: resultLine }));
  if (job.log && job.log.length) container.append(el("div", { class: "log" }, job.log.join("\n")));
}

function summarize(r) {
  if ("uploaded" in r) return `Uploaded ${r.uploaded}/${r.total} images.`;
  if ("labeled" in r) return `Labeled ${r.labeled}/${r.total} frames.`;
  if ("verified" in r) return `Wrote ${r.verified} verified labels.`;
  if ("ready" in r) return r.ready ? `Serving ${(r.served || []).join(", ")} at ${r.endpoint}` : `Pod ${r.id} up; still loading. ${r.endpoint}`;
  return JSON.stringify(r);
}
const escapeHtml = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function pollJob(container, job, onDone) {
  renderJob(container, job);
  if (job.status === "done" || job.status === "error") { onDone && onDone(job); return; }
  setTimeout(async () => {
    try {
      const next = await api(`/api/jobs/${job.id}`);
      pollJob(container, next, onDone);
    } catch (e) {
      container.append(el("div", { class: "err-text" }, "poll failed: " + e.message));
    }
  }, 1000);
}

async function runStage(btn, container, request) {
  btn.disabled = true;
  container.innerHTML = "";
  container.classList.remove("empty");
  try {
    const job = await request();
    pollJob(container, job, () => { btn.disabled = false; loadDatasetsQuiet(); });
  } catch (e) {
    container.append(el("div", { class: "err-text" }, e.message));
    btn.disabled = false;
  }
}

// --- models ------------------------------------------------------------------
let MODELS = [];
async function loadModels() {
  MODELS = await api("/api/models");
  for (const id of ["pl-model", "gpu-model"]) {
    const sel = $("#" + id);
    sel.innerHTML = "";
    MODELS.forEach((m) => sel.append(el("option", { value: m.key }, m.key)));
  }
  // prefill default categories from the first model
  if (MODELS[0]) {
    $("#pl-categories").value = MODELS[0].categories.join(", ");
    $("#im-categories").value = MODELS[0].categories.join(", ");
  }
}
$("#pl-model")?.addEventListener("change", (e) => {
  const m = MODELS.find((x) => x.key === e.target.value);
  if (m) $("#pl-categories").value = m.categories.join(", ");
});

// --- datasets ----------------------------------------------------------------
function dsCard(o) {
  const m = o.manifest || {};
  const groups = m.groups || {};
  const card = el("div", { class: "ds-card" });
  card.append(el("h3", {}, m.dataset || "?"));
  const cats = (m.categories || []).join(", ");
  card.append(el("div", { class: "pill" }, cats ? `categories: ${cats}` : "no categories declared"));
  const table = el("table", {});
  table.append(el("tr", {},
    el("th", {}, "Group"), el("th", { class: "num" }, "Frames"),
    el("th", { class: "num" }, "Labels"), el("th", { class: "num" }, "Verified"),
    el("th", {}, "LS")
  ));
  const lsp = o.ls_projects || {};
  Object.entries(groups).forEach(([g, c]) => {
    const lsCell = lsp[g] != null ? el("span", {}, `project #${lsp[g]}`) : el("span", { class: "muted" }, "—");
    table.append(el("tr", {},
      el("td", {}, g),
      el("td", { class: "num" }, String(c.frames || 0)),
      el("td", { class: "num" }, String(c.labels || 0)),
      el("td", { class: "num" }, String(c.verified || 0)),
      el("td", {}, lsCell)
    ));
  });
  if (!Object.keys(groups).length) table.append(el("tr", {}, el("td", { colspan: "5", class: "muted" }, "no groups yet")));
  card.append(table);
  return card;
}

async function loadDatasets() {
  const host = $("#datasets");
  host.innerHTML = "<p class='muted'>loading…</p>";
  try {
    const data = await api("/api/datasets");
    host.innerHTML = "";
    if (!data.length) { host.innerHTML = "<p class='muted'>No datasets yet — upload some images to begin.</p>"; return; }
    data.forEach((o) => host.append(dsCard(o)));
  } catch (e) {
    host.innerHTML = `<p class='err-text'>${escapeHtml(e.message)}</p>`;
  }
}
const loadDatasetsQuiet = () => { if ($("#tab-datasets").classList.contains("active")) loadDatasets(); };
$("#refresh-datasets").addEventListener("click", loadDatasets);

// --- stage handlers ----------------------------------------------------------
$("#do-upload").addEventListener("click", () => {
  const files = $("#up-files").files;
  if (!files.length) { alert("choose image files first"); return; }
  const fd = new FormData();
  fd.append("dataset", $("#up-dataset").value.trim());
  fd.append("group", $("#up-group").value.trim() || "all");
  for (const f of files) fd.append("files", f);
  runStage($("#do-upload"), $("#job-upload"), () => api("/api/upload", { method: "POST", body: fd }));
});

$("#do-prelabel").addEventListener("click", () =>
  runStage($("#do-prelabel"), $("#job-prelabel"), () => postJSON("/api/jobs/prelabel", {
    dataset: $("#pl-dataset").value.trim(),
    group: $("#pl-group").value.trim() || "all",
    model: $("#pl-model").value,
    categories: csv($("#pl-categories").value),
    min_score: parseFloat($("#pl-minscore").value) || 0,
  }))
);

$("#do-import").addEventListener("click", () =>
  runStage($("#do-import"), $("#job-import"), () => postJSON("/api/jobs/import-ls", {
    dataset: $("#im-dataset").value.trim(),
    group: $("#im-group").value.trim() || "all",
    categories: csv($("#im-categories").value),
    ttl: parseInt($("#im-ttl").value) || 604800,
  }))
);

$("#do-verify").addEventListener("click", () =>
  runStage($("#do-verify"), $("#job-verify"), () => postJSON("/api/jobs/verify", {
    dataset: $("#ve-dataset").value.trim(),
    group: $("#ve-group").value.trim() || "all",
  }))
);

// --- GPU ---------------------------------------------------------------------
async function loadGpu() {
  const host = $("#gpu-status");
  const pods = $("#gpu-pods");
  host.textContent = "loading…";
  pods.innerHTML = "";
  try {
    const s = await api("/api/gpu/status");
    host.innerHTML = `balance <b>$${Number(s.balance).toFixed(2)}</b> &middot; spend $${s.spend_per_hr}/hr`;
    if (!s.pods.length) { pods.innerHTML = "<p class='muted'>no pods running</p>"; return; }
    const table = el("table", {});
    table.append(el("tr", {}, el("th", {}, "Pod"), el("th", {}, "Name"), el("th", {}, "$/hr"), el("th", {}, "Status"), el("th", {}, "")));
    s.pods.forEach((p) => {
      const stop = el("button", { class: "ghost" }, "Stop");
      stop.addEventListener("click", async () => {
        stop.disabled = true;
        try { await postJSON("/api/gpu/down", { pods: [p.id] }); loadGpu(); }
        catch (e) { alert(e.message); stop.disabled = false; }
      });
      table.append(el("tr", {},
        el("td", {}, p.id), el("td", {}, p.name || ""),
        el("td", {}, "$" + (p.cost_per_hr ?? "?")), el("td", {}, p.status || ""),
        el("td", {}, stop)));
    });
    pods.append(table);
  } catch (e) {
    host.innerHTML = `<span class='err-text'>${escapeHtml(e.message)}</span>`;
  }
}
$("#refresh-gpu").addEventListener("click", loadGpu);
$("#do-gpu-up").addEventListener("click", () =>
  runStage($("#do-gpu-up"), $("#job-gpu"), () => postJSON("/api/gpu/up", {
    model: $("#gpu-model").value,
    gpu: $("#gpu-type").value.trim() || "rtx4090",
    hours: parseFloat($("#gpu-hours").value) || 3,
  }))
);

// --- boot --------------------------------------------------------------------
document.querySelectorAll(".job").forEach((j) => j.classList.add("empty"));
loadModels().then(loadDatasets);
