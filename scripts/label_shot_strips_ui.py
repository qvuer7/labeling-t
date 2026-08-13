#!/usr/bin/env python3
"""Local web UI for the shot-strip review pass (tasks.csv + strips/*.jpg).

Usage:
    python scripts/label_shot_strips_ui.py /home/andrii/Downloads/review/tasks.csv [--port 8791]

Each CSV row is one shot-moment filmstrip (7 frames wide). The question: did
the shot actually go in? Fills `human_made`: 1 (make) / 0 (miss) / unsure /
not_a_shot, plus free-text `human_notes`. `cls` is the review bucket
(fp_candidate_make, fn_adjacent_miss, spotcheck_*, suspect_negative) and
`label_now` is the pipeline's current belief — both shown, never written.
If info_hard_positives/<id>.jpg exists next to the CSV it is shown as extra
context under the strip.

Hotkeys: 1 = make · 2 = miss · 3 = unsure · 4 = not a shot · n = notes ·
arrows = move (no save). Sidebar chips filter by bucket. Every save rewrites
the CSV atomically; re-run resumes at the first unlabeled row.
"""

import argparse
import csv
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MADE_COL = "human_made"
NOTES_COL = "human_notes"
INFO_DIR = "info_hard_positives"


class Store:
    """CSV-backed row store with atomic writes (same shape as the sibling UIs)."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        for col in (MADE_COL, NOTES_COL):
            if col not in self.fieldnames:
                self.fieldnames.append(col)

    def save(self, index: int, made: str, notes: str) -> None:
        with self.lock:
            self.rows[index][MADE_COL] = made
            self.rows[index][NOTES_COL] = notes
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.csv_path)

    def state(self) -> dict:
        info_dir = self.csv_path.parent / INFO_DIR
        clips_dir = self.csv_path.parent / "clips"
        with self.lock:
            return {"rows": [
                {"i": i, "id": r.get("id", ""), "cls": r.get("cls", ""),
                 "label_now": r.get("label_now", ""), "score": r.get("model_score", ""),
                 "strip": r.get("strip", ""),
                 "made": (r.get(MADE_COL) or "").strip(),
                 "notes": (r.get(NOTES_COL) or "").strip(),
                 "info": (info_dir / f"{r.get('id', '')}.jpg").exists(),
                 "clip": (clips_dir / f"{r.get('id', '')}.mp4").exists()}
                for i, r in enumerate(self.rows)
            ]}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Shot strip review</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #14141c; color: #eee;
         display: flex; height: 100vh; overflow: hidden; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center;
          padding: 14px; overflow-y: auto; }
  #meta { font: 15px/1.4 monospace; color: #9ad; margin-bottom: 6px; }
  #progress { color: #888; font-size: 13px; margin-bottom: 8px; }
  #strip { width: 98%; border-radius: 6px; border: 2px solid #333; }
  #info { width: 60%; margin-top: 10px; border-radius: 6px; border: 2px solid #665533;
          display: none; }
  #controls { margin-top: 12px; display: flex; gap: 10px; align-items: center;
              flex-wrap: wrap; justify-content: center; }
  button { font: 16px system-ui; padding: 10px 18px; border-radius: 6px;
           border: 1px solid #444; background: #2a2a36; color: #ddd; cursor: pointer; }
  button:hover { background: #3a3a4a; }
  button u { color: #789; text-decoration: none; margin-right: 4px; }
  .make { border-color: #4a4; color: #8e8; }
  .miss { border-color: #a44; color: #f88; }
  .unsure { border-color: #a84; color: #fc8; }
  .noshot { border-color: #578; color: #9cf; }
  #notes { width: 60%; margin-top: 10px; font: 14px monospace; padding: 8px;
           background: #1c1c26; color: #ddd; border: 1px solid #333; border-radius: 6px; }
  #hint { color: #666; font-size: 12px; margin-top: 8px; }
  #panel { width: 250px; border-left: 1px solid #2a2a36; padding: 14px; overflow-y: auto; }
  #panel h3 { margin: 10px 0 8px; font-size: 14px; color: #aaa; }
  .chip { display: flex; justify-content: space-between; padding: 6px 10px; margin: 4px 0;
          background: #22222e; border-radius: 6px; font: 13px monospace; cursor: default; }
  .chip .n { color: #789; }
  .chip.good { color: #8e8; } .chip.bad { color: #f88; }
  .chip.warn { color: #fc8; } .chip.cold { color: #9cf; }
  .bucket { cursor: pointer; }
  .bucket:hover { background: #2e2e3c; }
  .bucket.active { outline: 1px solid #9ad; }
</style></head><body>
<div id="main">
  <div id="meta"></div>
  <div id="progress"></div>
  <img id="strip">
  <video id="clip" style="width:52%;margin-top:10px;border-radius:6px;border:2px solid #357;display:none"
         controls muted loop playsinline></video>
  <img id="info">
  <div id="controls">
    <button onclick="nav(-1)">&#8592; prev</button>
    <button class="make" onclick="save('1')"><u>1</u> make</button>
    <button class="miss" onclick="save('0')"><u>2</u> miss</button>
    <button class="unsure" onclick="save('unsure')"><u>3</u> unsure</button>
    <button class="noshot" onclick="save('not_a_shot')"><u>4</u> not a shot</button>
    <button onclick="nav(1)">skip &#8594;</button>
  </div>
  <input id="notes" placeholder="notes (n to focus, Enter to save + next)">
  <div id="hint">hotkeys: <b>1</b> make &middot; <b>2</b> miss &middot; <b>3</b> unsure
    &middot; <b>4</b> not a shot &middot; <b>n</b> notes &middot; &#8592;/&#8594; move
    without saving &middot; click a bucket chip to filter</div>
</div>
<div id="panel">
  <h3>verdicts</h3><div id="verdicts"></div>
  <h3>buckets (click to filter)</h3><div id="buckets"></div>
</div>
<script>
let rows = [], view = [], cur = 0, bucket = '';
const $ = id => document.getElementById(id);

function rebuild() {
  view = bucket ? rows.filter(r => r.cls === bucket) : rows;
  if (!view.length) { bucket = ''; view = rows; }
  cur = Math.max(0, Math.min(view.length - 1, cur));
}
function render() {
  const r = view[cur];
  $('strip').src = `/img/${encodeURIComponent(r.strip)}`;
  const inf = $('info');
  inf.style.display = r.info ? 'block' : 'none';
  if (r.info) inf.src = `/img/${encodeURIComponent('info_hard_positives/' + r.id + '.jpg')}`;
  const clip = $('clip');
  clip.style.display = r.clip ? 'block' : 'none';
  if (r.clip) { clip.src = `/img/${encodeURIComponent('clips/' + r.id + '.mp4')}`; clip.play().catch(() => {}); }
  else { clip.pause(); clip.removeAttribute('src'); }
  $('meta').textContent =
    `${cur + 1}/${view.length}${bucket ? ' [' + bucket + ']' : ''} — ${r.id}` +
    ` — bucket ${r.cls} — label_now ${r.label_now}` +
    (r.score ? ` — score ${r.score}` : '') +
    (r.made ? ` — labeled: ${r.made}` : '');
  if (document.activeElement !== $('notes')) $('notes').value = r.notes || '';
  const done = rows.filter(x => x.made).length;
  $('progress').textContent = `${done}/${rows.length} labeled`;
  const n = v => rows.filter(x => x.made === v).length;
  $('verdicts').innerHTML =
    `<div class="chip good"><span>make (1)</span><span class="n">${n('1')}</span></div>` +
    `<div class="chip bad"><span>miss (0)</span><span class="n">${n('0')}</span></div>` +
    `<div class="chip warn"><span>unsure</span><span class="n">${n('unsure')}</span></div>` +
    `<div class="chip cold"><span>not a shot</span><span class="n">${n('not_a_shot')}</span></div>`;
  const buckets = [...new Set(rows.map(x => x.cls))];
  $('buckets').innerHTML = buckets.map(b => {
    const all = rows.filter(x => x.cls === b), d = all.filter(x => x.made).length;
    return `<div class="chip bucket${bucket === b ? ' active' : ''}" onclick="setBucket('${b}')">` +
           `<span>${b}</span><span class="n">${d}/${all.length}</span></div>`;
  }).join('');
}
function setBucket(b) { bucket = (bucket === b) ? '' : b; cur = 0; rebuild(); render(); }
function nav(d) { cur = Math.max(0, Math.min(view.length - 1, cur + d)); render(); }
async function save(v) {
  const r = view[cur];
  r.made = v;
  r.notes = $('notes').value.trim();
  await fetch('/save', { method: 'POST',
    body: JSON.stringify({ index: r.i, made: v, notes: r.notes }) });
  if (cur < view.length - 1) cur++;
  render();
}
document.addEventListener('keydown', e => {
  if (e.repeat) { e.preventDefault(); return; }  // held key must not machine-gun labels
  if (document.activeElement === $('notes')) {
    if (e.key === 'Enter') { const r = view[cur]; save(r.made || 'unsure'); $('notes').blur(); }
    if (e.key === 'Escape') $('notes').blur();
    return;
  }
  if (e.key === '1') { save('1'); e.preventDefault(); }
  else if (e.key === '2') { save('0'); e.preventDefault(); }
  else if (e.key === '3') { save('unsure'); e.preventDefault(); }
  else if (e.key === '4') { save('not_a_shot'); e.preventDefault(); }
  else if (e.key === 'n') { $('notes').focus(); e.preventDefault(); }
  else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { nav(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { nav(-1); e.preventDefault(); }
});
fetch('/state').then(r => r.json()).then(s => {
  rows = s.rows;
  rebuild();
  const first = view.findIndex(r => !r.made);
  cur = first === -1 ? 0 : first;
  render();
});
</script></body></html>"""


def make_handler(store: Store, base_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str = "text/html") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(PAGE.encode())
            elif path == "/state":
                self._send(json.dumps(store.state()).encode(), "application/json")
            elif path.startswith("/img/"):
                rel = unquote(path[len("/img/"):])
                target = (base_dir / rel).resolve()
                if not target.is_relative_to(base_dir.resolve()) or not target.exists():
                    self.send_error(404)
                    return
                ctype = "video/mp4" if target.suffix == ".mp4" else "image/jpeg"
                self._send(target.read_bytes(), ctype)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            store.save(int(data["index"]), str(data["made"]), str(data.get("notes", "")))
            self._send(b"{}", "application/json")

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="tasks.csv (strip paths resolved from its directory)")
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()

    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        print(f"no such file: {csv_path}", file=sys.stderr)
        return 1
    store = Store(csv_path)
    url = f"http://127.0.0.1:{a.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", a.port),
                                 make_handler(store, csv_path.parent))
    done = sum(1 for r in store.rows if (r.get(MADE_COL) or "").strip())
    print(f"{len(store.rows)} strips, {done} already labeled — {url} (Ctrl+C to stop; "
          "progress saves on every label)")
    threading.Timer(0.4, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — labels are saved in", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
