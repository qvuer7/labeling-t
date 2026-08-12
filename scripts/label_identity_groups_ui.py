#!/usr/bin/env python3
"""Local web UI for the identity/group pass over track montage sheets.

Usage:
    python scripts/label_identity_groups_ui.py /home/andrii/Downloads/identity_labelling [--port 8790]

Expects <root>/<game>/<review_dir>/{groups.csv, ids_00.jpg ...}. Each sheet
holds 16 track rows of 150 px; sheeted=1 CSV rows map onto them in order, so
row i lives at sheet i//16, y = (i%16)*150. Rows with sheeted=0 are skipped
(kept untouched in the CSV on write).

Three things get labeled, only where needed:
  group       same person under several tids -> same token (blank = unique)
  group = x   junk: referee, bench, ghost, fragment box, unreadable mix
  team_truth  pre-filled with the model call; flip only where the kit disagrees

Hotkeys: j/k or arrows = move · space = select · shift+move = extend ·
g = group selected · 1-9 = assign palette group · u = ungroup · x = junk ·
t = flip team · p = pin · c = compare · z = zoom · r = reviewed ·
a = select all visible · esc = clear selection · ctrl+z = undo.

Every mutation rewrites the CSV atomically (a .bak is kept from the first
write). Review progress and pins live in a sidecar .review_state.json so the
CSV schema stays exactly as the scorer expects.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import string
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment guard
    sys.exit("Pillow is required: uv run --with pillow scripts/label_identity_groups_ui.py ...")

ROW_H = 150
ROWS_PER_SHEET = 16
# Sheets bake "tid N / Nf / a-bs" into the left 148 px of every row; the UI shows
# that in its own column, so list strips are trimmed and only the zoom view keeps it.
LABEL_W = 148
FPS_DEFAULT = 30.0
JUNK = "x"
# Group tokens: a, b, ... z, aa, ab, ... — 'x' is reserved for junk.
_TOKEN_LETTERS = [c for c in string.ascii_lowercase if c != JUNK]


def _token(n: int) -> str:
    """Nth group token in a, b, ..., aa, ab, ... order (skipping the junk token)."""
    base = len(_TOKEN_LETTERS)
    out = ""
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, base)
        out = _TOKEN_LETTERS[rem] + out
    return out


class Game:
    """One game's groups.csv plus its montage sheets."""

    def __init__(self, name: str, csv_path: Path, fps: float):
        self.name = name
        self.csv_path = csv_path
        self.dir = csv_path.parent
        self.fps = fps
        self.lock = threading.Lock()
        self.backed_up = False
        self._sheets: dict[int, Image.Image] = {}
        self._strips: dict[tuple[int, int, bool], bytes] = {}

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        missing = {"tid", "group", "team_model", "team_truth", "sheeted"} - set(self.fieldnames)
        if missing:
            raise SystemExit(f"{csv_path}: missing columns {sorted(missing)}")

        self.sheeted = [i for i, r in enumerate(self.rows) if (r.get("sheeted") or "").strip() == "1"]
        self.sheets = sorted(self.dir.glob("ids_*.jpg"))
        capacity = sum(self._open(s).height // ROW_H for s in range(len(self.sheets)))
        if capacity != len(self.sheeted):
            print(f"  ! {name}: {len(self.sheeted)} sheeted rows but sheets hold {capacity}", file=sys.stderr)

        self.state_path = self.dir / ".review_state.json"
        self.review: dict = {"reviewed": [], "pinned": []}
        if self.state_path.exists():
            try:
                self.review = json.loads(self.state_path.read_text())
            except (OSError, ValueError):
                pass

    def _open(self, sheet: int) -> Image.Image:
        if sheet not in self._sheets:
            self._sheets[sheet] = Image.open(self.sheets[sheet]).convert("RGB")
        return self._sheets[sheet]

    def strip(self, pos: int, full: bool = False) -> bytes:
        """JPEG bytes of the montage row for the pos-th sheeted track.

        full=True keeps the sheet's baked-in tid caption (used by the zoom view
        so the operator can eyeball that a strip really is the track claimed).
        """
        if not 0 <= pos < len(self.sheeted):
            raise IndexError(pos)
        sheet, srow = divmod(pos, ROWS_PER_SHEET)
        key = (sheet, srow, full)
        if key not in self._strips:
            im = self._open(sheet)
            box = (0 if full else LABEL_W, srow * ROW_H, im.width, min((srow + 1) * ROW_H, im.height))
            buf = io.BytesIO()
            im.crop(box).save(buf, "JPEG", quality=93)
            self._strips[key] = buf.getvalue()
        return self._strips[key]

    def windows(self) -> list[dict]:
        """Recording windows: spans of covered time, split where coverage gaps by a minute."""
        tracks = sorted((int(self.rows[i]["first_f"]), int(self.rows[i]["last_f"])) for i in self.sheeted)
        spans: list[list[int]] = []
        for first, last in tracks:
            if spans and first - spans[-1][1] <= self.fps * 60:
                spans[-1][1] = max(spans[-1][1], last)
            else:
                spans.append([first, last])
        return [
            {"k": k, "lo": lo, "hi": hi,
             "label": f"{int(lo / self.fps // 60)}–{-int(-hi / self.fps // 60)} min"}
            for k, (lo, hi) in enumerate(spans)
        ]

    def snapshot(self) -> dict:
        with self.lock:
            wins = self.windows()
            rows = []
            for pos, ri in enumerate(self.sheeted):
                r = self.rows[ri]
                first, last = int(r["first_f"]), int(r["last_f"])
                win = next((w["k"] for w in wins if w["lo"] <= first <= w["hi"]), 0)
                rows.append({
                    "ri": ri, "pos": pos, "tid": r["tid"],
                    "group": (r.get("group") or "").strip(),
                    "team_model": (r.get("team_model") or "").strip(),
                    "team_truth": (r.get("team_truth") or "").strip(),
                    "n_frames": r.get("n_frames", ""), "first_f": first, "last_f": last,
                    "t0": round(first / self.fps), "t1": round(last / self.fps),
                    "sheet": pos // ROWS_PER_SHEET, "srow": pos % ROWS_PER_SHEET, "win": win,
                })
            return {
                "name": self.name, "rows": rows, "windows": wins,
                "reviewed": self.review.get("reviewed", []),
                "pinned": self.review.get("pinned", []),
            }

    def apply(self, updates: list[dict]) -> None:
        """Write group / team_truth edits for the given CSV row indices."""
        with self.lock:
            for up in updates:
                row = self.rows[int(up["ri"])]
                if "group" in up:
                    row["group"] = str(up["group"]).strip()
                if "team_truth" in up:
                    row["team_truth"] = str(up["team_truth"]).strip()
            if not self.backed_up:
                bak = self.csv_path.with_suffix(".csv.bak")
                if not bak.exists():
                    shutil.copy2(self.csv_path, bak)
                self.backed_up = True
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.csv_path)

    def set_review(self, reviewed: list, pinned: list) -> None:
        with self.lock:
            self.review = {"reviewed": reviewed, "pinned": pinned}
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.review))
            os.replace(tmp, self.state_path)

    def next_token(self) -> str:
        used = {(r.get("group") or "").strip() for r in self.rows}
        n = 0
        while _token(n) in used:
            n += 1
        return _token(n)


def discover(root: Path, fps: float) -> list[Game]:
    csvs = sorted(root.glob("*/*/groups.csv")) or sorted(root.glob("*/groups.csv"))
    games = [Game(p.parts[-3] if len(p.parts) >= 3 else p.parent.name, p, fps) for p in csvs]
    if not games and (root / "groups.csv").exists():
        games = [Game(root.name, root / "groups.csv", fps)]
    if not games:
        raise SystemExit(f"no <game>/<dir>/groups.csv found under {root}")
    return games


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>identity groups</title>
<style>
  :root { --bg:#0e0f12; --panel:#16181d; --line:#272b33; --fg:#e6e8ec; --dim:#8b93a1;
          --sel:#4c8dff; --junk:#ff6b6b; --ok:#3ddc84; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-sans-serif,system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--panel); border-bottom:1px solid var(--line);
           padding:8px 12px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header .grow { flex:1; }
  select, button { background:#20242c; color:var(--fg); border:1px solid var(--line); border-radius:6px;
                   padding:5px 9px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--sel); }
  .stat { color:var(--dim); }
  .stat b { color:var(--fg); }
  main { display:flex; align-items:flex-start; }
  #list { flex:1; padding:10px 12px 60vh; min-width:0; }
  aside { width:300px; flex:none; position:sticky; top:47px; max-height:calc(100vh - 47px); overflow:auto;
          border-left:1px solid var(--line); padding:10px; background:#101216; }
  aside h3 { margin:12px 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim); }
  .card { display:flex; gap:10px; align-items:center; border:1px solid var(--line); border-radius:8px;
          margin-bottom:6px; padding:6px; background:#121419; cursor:pointer; }
  .card.cursor { border-color:#6b7280; }
  .card.sel { border-color:var(--sel); box-shadow:0 0 0 2px rgba(76,141,255,.35); background:#141a26; }
  .card.junk { opacity:.45; }
  .card.rev .tid::after { content:"\2713"; color:var(--ok); margin-left:5px; }
  .meta { width:118px; flex:none; font-variant-numeric:tabular-nums; }
  .tid { font-weight:600; }
  .meta .sub { color:var(--dim); font-size:11px; }
  .strip { flex:1; min-width:0; text-align:left; }
  .strip img { display:block; height:auto; border-radius:4px; width:calc(100% * var(--rowscale,1)); }
  .side { width:132px; flex:none; display:flex; flex-direction:column; gap:4px; align-items:flex-end; }
  .gtag { font-weight:700; padding:2px 8px; border-radius:5px; background:#20242c; }
  .gtag.none { color:var(--dim); font-weight:400; }
  .gtag.junk { background:var(--junk); color:#150000; }
  .team { font-size:11px; color:var(--dim); }
  .team b { color:var(--fg); }
  .team.changed b { color:#ffd166; }
  #pins:not(:empty) { border-bottom:1px solid var(--line); background:#0b0d10; padding:6px 0;
                      position:sticky; top:47px; z-index:15; max-height:46vh; overflow:auto; }
  #pins .card { margin:0 12px 6px; }
  .grp { display:flex; align-items:center; gap:6px; padding:3px 0; cursor:pointer; }
  .grp:hover { color:var(--sel); }
  .grp .sw { width:12px; height:12px; border-radius:3px; flex:none; }
  .grp .ids { color:var(--dim); font-size:11px; }
  .warn { color:#ffd166; }
  kbd { background:#20242c; border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:11px; }
  .note { color:var(--dim); font-size:11px; line-height:1.9; }
  #overlay { position:fixed; inset:0; background:rgba(6,7,9,.97); z-index:50; overflow:auto; padding:20px; display:none; }
  #overlay img { width:100%; max-width:1900px; display:block; margin:4px auto 16px; }
  #overlay .cap { max-width:1900px; margin:0 auto; color:var(--dim); }
  #toast { position:fixed; right:14px; bottom:14px; background:#20242c; border:1px solid var(--line);
           border-radius:8px; padding:8px 12px; opacity:0; transition:opacity .2s; z-index:60; pointer-events:none; }
</style>
<header>
  <select id="game"></select>
  <select id="win"></select>
  <select id="filter">
    <option value="all">all rows</option>
    <option value="todo">unreviewed</option>
    <option value="grouped">grouped</option>
    <option value="ungrouped">no group</option>
    <option value="junk">junk (x)</option>
    <option value="teamfix">team corrected</option>
  </select>
  <label class="stat">row size <input id="zoom" type="range" min="0.45" max="1" step="0.05" value="1"
         style="vertical-align:middle"></label>
  <span class="grow"></span>
  <span class="stat" id="stats"></span>
  <button id="markall">mark visible reviewed</button>
</header>
<div id="pins"></div>
<main>
  <div id="list"></div>
  <aside>
    <h3>groups <span id="gcount" class="stat"></span></h3>
    <div id="groups"></div>
    <h3>palette <span class="stat">(1-9)</span></h3>
    <div id="palette" class="note">use <kbd>g</kbd> once; recent groups land here</div>
    <h3>keys</h3>
    <div class="note">
      <kbd>j</kbd>/<kbd>k</kbd> move · <kbd>space</kbd> select · <kbd>shift</kbd>+move extend<br>
      <kbd>g</kbd> group selected · <kbd>1</kbd>-<kbd>9</kbd> palette group<br>
      <kbd>u</kbd> ungroup · <kbd>x</kbd> junk · <kbd>r</kbd> reviewed<br>
      <kbd>t</kbd> flip team (0 &harr; 1)<br>
      <kbd>p</kbd> pin · <kbd>c</kbd> compare selected · <kbd>z</kbd> zoom<br>
      <kbd>a</kbd> select visible · <kbd>esc</kbd> clear<br>
      <kbd>ctrl</kbd>+<kbd>z</kbd> undo last edit
    </div>
    <h3>reading the sheets</h3>
    <div class="note">
      6 crops sampled across the track's life. Each crop's border is the model's
      team call for that crop, so borders can disagree inside one row. Measured over
      all 303 rows: <b>team 0 draws a red border, team 1 blue</b>, grey = no team.
      The border does not track kit colour (team 0 wears blue in 1079292 and black in
      1080287) — judge the kit and set the number, ignoring the tint.<br><br>
      Label a mixed row by its dominant person; if it is unreadable, <kbd>x</kbd>.
    </div>
  </aside>
</main>
<div id="overlay"></div>
<div id="toast"></div>
<script>
const CFG = __CONFIG__;
let G = null, rows = [], byRi = {}, cursor = 0, sel = new Set(), pinned = [], reviewed = new Set(), recent = [];
const el = id => document.getElementById(id);
const norm = g => (g || '').trim();
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function hue(tok) { let h = 0; for (const c of tok) h = (h * 37 + c.charCodeAt(0)) % 360; return h; }
function color(tok) { return tok === 'x' ? '#ff6b6b' : `hsl(${hue(tok)} 72% 62%)`; }
function toast(msg) { const t = el('toast'); t.textContent = msg; t.style.opacity = 1;
  clearTimeout(t._h); t._h = setTimeout(() => t.style.opacity = 0, 1600); }

async function load(name) {
  G = await (await fetch('/api/state?game=' + encodeURIComponent(name))).json();
  rows = G.rows; byRi = {}; rows.forEach(r => byRi[r.ri] = r);
  reviewed = new Set(G.reviewed); pinned = (G.pinned || []).filter(ri => byRi[ri]);
  sel.clear(); cursor = 0; recent = [];
  el('win').innerHTML = '<option value="all">all windows</option>' +
    G.windows.map(w => `<option value="${w.k}">${w.label}</option>`).join('');
  render();
}

function visible() {
  const w = el('win').value, f = el('filter').value;
  return rows.filter(r => {
    if (w !== 'all' && String(r.win) !== w) return false;
    const g = norm(r.group);
    if (f === 'todo') return !reviewed.has(r.ri);
    if (f === 'grouped') return g && g !== 'x';
    if (f === 'ungrouped') return !g;
    if (f === 'junk') return g === 'x';
    if (f === 'teamfix') return r.team_truth !== r.team_model;
    return true;
  });
}

function groupMap() {
  const m = new Map();
  for (const r of rows) {
    const g = norm(r.group);
    if (!g || g === 'x') continue;
    if (!m.has(g)) m.set(g, []);
    m.get(g).push(r);
  }
  return m;
}

function card(r, isPin, isCursor) {
  const g = norm(r.group), cls = ['card'];
  if (isCursor) cls.push('cursor');
  if (sel.has(r.ri)) cls.push('sel');
  if (g === 'x') cls.push('junk');
  if (reviewed.has(r.ri)) cls.push('rev');
  const tag = g
    ? `<span class="gtag ${g === 'x' ? 'junk' : ''}" ${g !== 'x' ? `style="color:${color(g)}"` : ''}>${esc(g)}</span>`
    : '<span class="gtag none">&mdash;</span>';
  const changed = r.team_truth !== r.team_model;
  return `<div class="${cls.join(' ')}" data-ri="${r.ri}">
    <div class="meta"><div class="tid">tid ${esc(r.tid)}</div>
      <div class="sub">${esc(r.n_frames)}f &middot; ${r.t0}&ndash;${r.t1}s</div>
      <div class="sub">sheet ${r.sheet} / row ${r.srow}</div></div>
    <div class="strip"><img loading="lazy" src="/strip/${encodeURIComponent(G.name)}/${r.pos}.jpg"></div>
    <div class="side">${tag}
      <span class="team ${changed ? 'changed' : ''}">team <b>${esc(r.team_truth || '—')}</b>${changed ? ` (was ${esc(r.team_model)})` : ''}</span>
      ${isPin ? `<button data-unpin="${r.ri}">unpin</button>` : ''}</div></div>`;
}

function render() {
  const v = visible();
  cursor = v.length ? Math.max(0, Math.min(cursor, v.length - 1)) : 0;
  const curRi = v.length ? v[cursor].ri : -1;
  el('list').innerHTML = v.map(r => card(r, false, r.ri === curRi)).join('')
    || '<p class="stat">nothing matches this filter.</p>';
  el('pins').innerHTML = pinned.map(ri => card(byRi[ri], true, false)).join('');
  document.body.style.setProperty('--rowscale', el('zoom').value);

  const gm = groupMap();
  el('gcount').textContent = `(${gm.size})`;
  el('groups').innerHTML = [...gm.entries()].sort().map(([g, mem]) => {
    let flag = '';
    outer: for (let i = 0; i < mem.length; i++)
      for (let j = i + 1; j < mem.length; j++)
        if (mem[i].first_f <= mem[j].last_f && mem[j].first_f <= mem[i].last_f) {
          flag = ' <span class="warn" title="member time ranges overlap — one person cannot be two tracks at once">&#8987;</span>';
          break outer;
        }
    if (new Set(mem.map(m => m.win)).size > 1)
      flag += ' <span class="warn" title="group spans separate windows">&#10530;</span>';
    return `<div class="grp" data-g="${esc(g)}"><span class="sw" style="background:${color(g)}"></span>
      <span>${esc(g)}${flag}</span><span class="ids">${mem.map(m => esc(m.tid)).join(', ')}</span></div>`;
  }).join('') || '<div class="stat">none yet</div>';

  if (recent.length) el('palette').innerHTML = recent.map((t, i) =>
    `<div class="grp" data-g="${esc(t)}"><kbd>${i + 1}</kbd><span class="sw" style="background:${color(t)}"></span>${esc(t)}</div>`).join('');

  const junk = rows.filter(r => norm(r.group) === 'x').length;
  const fixed = rows.filter(r => r.team_truth !== r.team_model).length;
  el('stats').innerHTML = `<b>${reviewed.size}</b>/${rows.length} reviewed &middot;
    <b>${gm.size}</b> groups &middot; <b>${junk}</b> junk &middot; <b>${fixed}</b> team fixes &middot;
    <b>${sel.size}</b> selected &middot; showing <b>${v.length}</b>`;

  const c = el('list').children[cursor];
  if (c && c.scrollIntoView) c.scrollIntoView({ block: 'nearest' });
}

const post = (path, body) => fetch(path, { method: 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const save = updates => post('/api/save', { game: G.name, updates });
const saveReview = () => post('/api/review', { game: G.name, reviewed: [...reviewed], pinned });

// Every edit hits the CSV immediately, so keep a stack to walk back out of a stray keypress.
let history = [];
function remember_state(rs) {
  history.push(rs.map(r => ({ ri: r.ri, group: r.group, team_truth: r.team_truth })));
  if (history.length > 200) history.shift();
}
async function undo() {
  const batch = history.pop();
  if (!batch) { toast('nothing to undo'); return; }
  batch.forEach(o => { const r = byRi[o.ri]; r.group = o.group; r.team_truth = o.team_truth; });
  toast(`undo: ${batch.length} row${batch.length > 1 ? 's' : ''}`);
  render();
  await save(batch);
}

function targets() {
  if (sel.size) return [...sel].map(ri => byRi[ri]).filter(Boolean);
  const v = visible();
  return v.length ? [v[cursor]] : [];
}
const touch = rs => rs.forEach(r => reviewed.add(r.ri));
const remember = tok => { recent = [tok, ...recent.filter(t => t !== tok)].slice(0, 9); };

async function setGroup(tok) {
  const rs = targets();
  if (!rs.length) return;
  remember_state(rs);
  rs.forEach(r => r.group = tok);
  touch(rs);
  if (tok && tok !== 'x') remember(tok);
  sel.clear();                 // a group call is terminal for those rows; same as 'g'
  render();
  await save(rs.map(r => ({ ri: r.ri, group: tok })));
  saveReview();
}

async function groupSelected() {
  const rs = targets();
  if (!rs.length) return;
  const existing = [...new Set(rs.map(r => norm(r.group)).filter(g => g && g !== 'x'))];
  let tok, extra = [];
  remember_state([...rs, ...rows.filter(r => existing.slice(1).includes(norm(r.group)))]);
  if (existing.length === 1) {
    tok = existing[0];
  } else if (existing.length > 1) {           // merge: fold the other groups into the first
    tok = existing[0];
    extra = rows.filter(r => existing.slice(1).includes(norm(r.group)));
    extra.forEach(r => r.group = tok);
  } else {
    tok = (await (await fetch('/api/token?game=' + encodeURIComponent(G.name))).json()).token;
  }
  rs.forEach(r => r.group = tok);
  touch(rs); remember(tok);
  const tids = [...new Set([...rs, ...extra].map(r => r.tid))];
  toast(`group ${tok}: tid ${tids.join(', ')}`);
  sel.clear(); render();
  await save([...rs, ...extra].map(r => ({ ri: r.ri, group: tok })));
  saveReview();
}

async function setTeam(val) {
  const rs = targets();
  if (!rs.length) return;
  remember_state(rs);
  rs.forEach(r => r.team_truth = val === 'flip' ? (String(r.team_truth) === '1' ? '0' : '1') : String(val));
  touch(rs); render();
  await save(rs.map(r => ({ ri: r.ri, team_truth: r.team_truth })));
  saveReview();
}

function openOverlay(rs) {
  el('overlay').innerHTML = rs.map(r =>
    `<div class="cap">tid ${esc(r.tid)} &middot; ${esc(r.n_frames)}f &middot; ${r.t0}&ndash;${r.t1}s &middot;
      team ${esc(r.team_truth)} &middot; group ${esc(norm(r.group) || '—')}</div>
     <img src="/strip/${encodeURIComponent(G.name)}/${r.pos}.jpg?full=1">`).join('')
    + '<div class="cap">esc or click to close</div>';
  el('overlay').style.display = 'block';
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const overlayOpen = el('overlay').style.display === 'block';
  if (e.key === 'Escape') {
    if (overlayOpen) { el('overlay').style.display = 'none'; return; }
    sel.clear(); render(); return;
  }
  if (overlayOpen) return;
  const v = visible();
  const move = d => {
    const prev = cursor;
    cursor = Math.max(0, Math.min(v.length - 1, cursor + d));
    if (e.shiftKey) {
      const [a, b] = [Math.min(prev, cursor), Math.max(prev, cursor)];
      for (let i = a; i <= b; i++) sel.add(v[i].ri);
    }
    render();
  };
  switch (e.key) {
    case 'j': case 'ArrowDown': e.preventDefault(); move(1); break;
    case 'k': case 'ArrowUp': e.preventDefault(); move(-1); break;
    case ' ': {
      e.preventDefault();
      const r = v[cursor]; if (!r) break;
      sel.has(r.ri) ? sel.delete(r.ri) : sel.add(r.ri);
      render(); break;
    }
    case 'g': e.preventDefault(); groupSelected(); break;
    case 'u': setGroup(''); break;
    case 'x': setGroup('x'); break;
    case 't': setTeam('flip'); break;
    case 'r': { const rs = targets(); touch(rs); render(); saveReview(); break; }
    case 'p': {
      targets().forEach(r => { if (!pinned.includes(r.ri)) pinned.push(r.ri); });
      render(); saveReview(); break;
    }
    case 'c': { const rs = targets(); if (rs.length) openOverlay(rs); break; }
    case 'z': { const r = v[cursor]; if (r) openOverlay([r]); break; }
    case 'a': e.preventDefault(); v.forEach(r => sel.add(r.ri)); render(); break;
    default:
      if (/^[1-9]$/.test(e.key)) {                       // digits are palette-only: team is 't'
        const tok = recent[+e.key - 1];
        tok ? setGroup(tok) : toast('no palette group ' + e.key);
      }
  }
});

el('list').addEventListener('click', e => {
  const c = e.target.closest('.card');
  if (!c) return;
  const ri = +c.dataset.ri, v = visible(), i = v.findIndex(r => r.ri === ri);
  if (e.detail === 2 && e.target.tagName === 'IMG') { openOverlay([byRi[ri]]); return; }
  if (e.shiftKey) {
    const [a, b] = [Math.min(cursor, i), Math.max(cursor, i)];
    for (let k = a; k <= b; k++) sel.add(v[k].ri);
  } else {
    sel.has(ri) ? sel.delete(ri) : sel.add(ri);
  }
  cursor = i; render();
});
el('pins').addEventListener('click', e => {
  const b = e.target.closest('[data-unpin]');
  if (b) { pinned = pinned.filter(ri => ri !== +b.dataset.unpin); render(); saveReview(); return; }
  const c = e.target.closest('.card');
  if (c) { const ri = +c.dataset.ri; sel.has(ri) ? sel.delete(ri) : sel.add(ri); render(); }
});
function pickGroup(e) {
  const g = e.target.closest('[data-g]');
  if (!g) return;
  sel = new Set(rows.filter(r => norm(r.group) === g.dataset.g).map(r => r.ri));
  el('filter').value = 'all'; el('win').value = 'all'; render();
}
el('groups').addEventListener('click', pickGroup);
el('palette').addEventListener('click', pickGroup);
el('overlay').addEventListener('click', () => el('overlay').style.display = 'none');
el('markall').addEventListener('click', () => { visible().forEach(r => reviewed.add(r.ri)); render(); saveReview(); });
['win', 'filter'].forEach(id => el(id).addEventListener('change', () => { cursor = 0; el(id).blur(); render(); }));
el('zoom').addEventListener('input', render);
el('game').addEventListener('change', () => { el('game').blur(); load(el('game').value); });

el('game').innerHTML = CFG.games.map(g => `<option>${esc(g)}</option>`).join('');
load(CFG.games[0]);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    games: dict[str, Game] = {}

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Only the montage strips are immutable; the page and API must never be cached.
        self.send_header("Cache-Control", "max-age=3600" if ctype == "image/jpeg" else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            page = HTML.replace("__CONFIG__", json.dumps({"games": list(self.games)}))
            return self._send(200, page.encode(), "text/html; charset=utf-8")
        if u.path in ("/api/state", "/api/token"):
            game = self.games.get((q.get("game") or [""])[0])
            if not game:
                return self._send(404, b"no such game", "text/plain")
            return self._json(game.snapshot() if u.path == "/api/state" else {"token": game.next_token()})
        m = re.fullmatch(r"/strip/([^/]+)/(\d+)\.jpg", u.path)
        if m:
            game = self.games.get(m.group(1))
            if game:
                try:
                    return self._send(200, game.strip(int(m.group(2)), full=bool(q.get("full"))), "image/jpeg")
                except (IndexError, OSError):
                    return self._send(404, b"no such strip", "text/plain")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send(400, b"bad json", "text/plain")
        game = self.games.get(body.get("game", ""))
        if not game:
            return self._send(404, b"no such game", "text/plain")
        try:
            if u.path == "/api/save":
                game.apply(body.get("updates", []))
            elif u.path == "/api/review":
                game.set_review(body.get("reviewed", []), body.get("pinned", []))
            else:
                return self._send(404, b"not found", "text/plain")
        except (KeyError, IndexError, ValueError, OSError) as exc:
            return self._send(400, str(exc).encode(), "text/plain")
        self._json({"ok": True})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dir holding <game>/<review_dir>/groups.csv")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT, help="frames per second, for the second labels")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    games = discover(args.root.expanduser(), args.fps)
    Handler.games = {g.name: g for g in games}
    for g in games:
        wins = ", ".join(w["label"] for w in g.windows())
        print(f"{g.name}: {len(g.sheeted)} sheeted rows over {len(g.sheets)} sheets "
              f"({len(g.rows) - len(g.sheeted)} short tracks skipped) — windows {wins}")
    print(f"total {sum(len(g.sheeted) for g in games)} rows -> http://127.0.0.1:{args.port}")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; CSVs were saved after every edit.")


if __name__ == "__main__":
    main()
