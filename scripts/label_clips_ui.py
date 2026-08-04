#!/usr/bin/env python3
"""Local web UI for labeling shot clips listed in a to_label.csv.

Usage:
    python scripts/label_clips_ui.py 1077305/to_label.csv [--port 8765]

Serves a single-page app: shows each clip (looping), buttons / hotkeys fill the
`label` column (make | miss | pass | not_a_shot | unclear) and optional `notes`.
Every label press rewrites the CSV atomically, so progress is never lost —
Ctrl+C and re-run resumes at the first unlabeled row.

If a clips_h264/ folder exists next to the CSV, videos are served from there
(browser-playable H.264) while CSV paths keep pointing at the original clips/.
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

LABELS = ["make", "miss", "pass", "not_a_shot", "unclear"]


class Store:
    """CSV-backed row store with atomic writes."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        for col in ("label", "notes"):
            if col not in self.fieldnames:
                self.fieldnames.append(col)
        for row in self.rows:
            row.setdefault("label", "")
            row.setdefault("notes", "")
        self.by_id = {row["id"]: row for row in self.rows}

    def set_label(self, row_id: str, label: str, notes: str | None) -> dict:
        with self.lock:
            row = self.by_id[row_id]
            row["label"] = label
            if notes is not None:
                row["notes"] = notes
            self._flush()
            return dict(row)

    def _flush(self) -> None:
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.csv_path)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Clip labeler</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.4 system-ui, sans-serif; background: #111; color: #ddd;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
  #bar { width: 100%; background: #222; padding: 8px 16px; box-sizing: border-box;
         display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  #progress { flex: 1; min-width: 120px; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }
  #progress > div { height: 100%; background: #4caf50; width: 0; transition: width .15s; }
  video { max-width: 96vw; max-height: 62vh; background: #000; margin-top: 10px; border-radius: 6px; }
  #meta { margin: 6px 0; color: #999; font-size: 13px; min-height: 18px; }
  #preds { color: #c90; }
  #buttons { display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; justify-content: center; }
  button.lab { font-size: 17px; padding: 12px 22px; border: 2px solid #444; border-radius: 8px;
               background: #1b1b1b; color: #eee; cursor: pointer; }
  button.lab:hover { border-color: #888; }
  button.lab.current { background: #16321a; box-shadow: 0 0 0 2px #4caf50; }
  button.lab .key { color: #777; font-size: 12px; margin-left: 6px; }
  #nav { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; justify-content: center; }
  #nav button, #bar button { padding: 6px 12px; background: #1b1b1b; color: #ccc;
        border: 1px solid #444; border-radius: 6px; cursor: pointer; }
  #notes { width: min(500px, 90vw); padding: 7px 10px; background: #1b1b1b; color: #eee;
           border: 1px solid #444; border-radius: 6px; }
  .make { border-color:#2e7d32 !important } .miss { border-color:#c62828 !important }
  .pass { border-color:#1565c0 !important } .not_a_shot { border-color:#6a1b9a !important }
  .unclear { border-color:#f9a825 !important }
  #done { color: #4caf50; font-size: 20px; margin: 20px; display: none; }
  kbd { background:#2a2a2a; border-radius:3px; padding:0 5px; font-size:12px; }
  #help { color:#666; font-size:12px; margin: 4px 0 16px; }
</style></head><body>
<div id="bar">
  <b id="counter"></b>
  <div id="progress"><div></div></div>
  <span id="labeled"></span>
  <button id="togglePreds">show predictions</button>
</div>
<video id="vid" autoplay loop muted playsinline></video>
<div id="meta"></div>
<div id="buttons"></div>
<div id="nav">
  <button id="prev">&larr; prev</button>
  <button id="nextUnlabeled">next unlabeled &darr;</button>
  <button id="next">next &rarr;</button>
  <input id="notes" placeholder="notes (Enter to save)">
</div>
<div id="help">hotkeys: <kbd>1</kbd>-<kbd>5</kbd> label &amp; advance · <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate ·
<kbd>u</kbd> next unlabeled · <kbd>space</kbd> pause/play · <kbd>,</kbd>/<kbd>.</kbd> step frame · <kbd>r</kbd> restart clip</div>
<div id="done">All clips labeled — CSV is saved. You can close this tab.</div>
<script>
const LABELS = %LABELS%;
let rows = [], idx = 0, showPreds = false;

const $ = id => document.getElementById(id);
const vid = $('vid');

function labeledCount() { return rows.filter(r => r.label).length; }

function render() {
  const r = rows[idx];
  $('counter').textContent = (idx + 1) + ' / ' + rows.length + '  —  ' + r.id;
  const n = labeledCount();
  $('labeled').textContent = n + ' labeled';
  $('progress').firstElementChild.style.width = (100 * n / rows.length) + '%';
  const src = '/video/' + encodeURIComponent(r.id);
  if (!vid.src.endsWith(src)) { vid.src = src; vid.play().catch(()=>{}); }
  let meta = 't=' + r.t_sec + 's · frame ' + r.peak_frame;
  if (showPreds) meta += ' · <span id="preds">geo: ' + r.geo_pred + ' · gru: ' + r.gru_pred +
      ' (' + r.gru_prob + ') · sb: ' + r.scoreboard + ' · ' + r.home + '-' + r.away + ' @ ' + r.timer + '</span>';
  $('meta').innerHTML = meta;
  document.querySelectorAll('button.lab').forEach(b => {
    b.classList.toggle('current', b.dataset.label === r.label);
  });
  $('notes').value = r.notes || '';
  $('done').style.display = rows.every(r => r.label) ? 'block' : 'none';
}

function go(i) { if (i >= 0 && i < rows.length) { idx = i; render(); } }
function nextUnlabeled(from) {
  for (let k = 0; k < rows.length; k++) {
    const i = (from + k) % rows.length;
    if (!rows[i].label) return go(i);
  }
  render();
}

async function setLabel(label) {
  const r = rows[idx];
  const resp = await fetch('/api/label', { method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ id: r.id, label, notes: $('notes').value }) });
  if (!resp.ok) { alert('save failed: ' + await resp.text()); return; }
  r.label = label; r.notes = $('notes').value;
  nextUnlabeled(idx + 1);
}

LABELS.forEach((lab, i) => {
  const b = document.createElement('button');
  b.className = 'lab ' + lab; b.dataset.label = lab;
  b.innerHTML = lab.replace(/_/g, ' ') + '<span class="key">' + (i + 1) + '</span>';
  b.onclick = () => setLabel(lab);
  $('buttons').appendChild(b);
});

$('prev').onclick = () => go(idx - 1);
$('next').onclick = () => go(idx + 1);
$('nextUnlabeled').onclick = () => nextUnlabeled(idx + 1);
$('togglePreds').onclick = () => { showPreds = !showPreds;
  $('togglePreds').textContent = showPreds ? 'hide predictions' : 'show predictions'; render(); };
$('notes').addEventListener('keydown', e => {
  e.stopPropagation();
  if (e.key === 'Enter') { e.preventDefault();
    const r = rows[idx];
    fetch('/api/label', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ id: r.id, label: r.label || '', notes: $('notes').value }) });
    r.notes = $('notes').value; $('notes').blur(); }
});

document.addEventListener('keydown', e => {
  if (e.target === $('notes')) return;
  const n = parseInt(e.key);
  if (n >= 1 && n <= LABELS.length) { setLabel(LABELS[n - 1]); return; }
  if (e.key === 'ArrowLeft') go(idx - 1);
  else if (e.key === 'ArrowRight') go(idx + 1);
  else if (e.key === 'u') nextUnlabeled(idx + 1);
  else if (e.key === ' ') { e.preventDefault(); vid.paused ? vid.play() : vid.pause(); }
  else if (e.key === ',') { vid.pause(); vid.currentTime = Math.max(0, vid.currentTime - 1/25); }
  else if (e.key === '.') { vid.pause(); vid.currentTime += 1/25; }
  else if (e.key === 'r') { vid.currentTime = 0; vid.play(); }
});

fetch('/api/state').then(r => r.json()).then(data => {
  rows = data.rows;
  nextUnlabeled(0);
});
</script></body></html>"""


def make_handler(store: Store, video_paths: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.replace("%LABELS%", json.dumps(LABELS)),
                           "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps({"rows": store.rows, "labels": LABELS}))
            elif path.startswith("/video/"):
                self._serve_video(unquote(path[len("/video/"):]))
            else:
                self._send(404, "not found", "text/plain")

        def _serve_video(self, row_id: str):
            file = video_paths.get(row_id)
            if not file or not file.exists():
                self._send(404, "no clip", "text/plain")
                return
            data = file.read_bytes()
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s or 0)
                end = int(end_s) if end_s else len(data) - 1
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                self._send(404, "not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                row_id, label = body["id"], body["label"]
                if label and label not in LABELS:
                    raise ValueError(f"bad label {label!r}")
                if row_id not in store.by_id:
                    raise KeyError(f"unknown id {row_id!r}")
                row = store.set_label(row_id, label, body.get("notes"))
                self._send(200, json.dumps(row))
            except Exception as e:  # surfaced in the UI via alert()
                self._send(400, str(e), "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="to_label.csv path")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    store = Store(csv_path)

    base = csv_path.parent
    h264 = base / "clips_h264"
    video_paths = {}
    missing = 0
    for row in store.rows:
        orig = base / row["clip"]
        cand = h264 / orig.name
        file = cand if cand.exists() else orig
        if not file.exists():
            missing += 1
        video_paths[row["id"]] = file
    if missing:
        print(f"WARNING: {missing} clips referenced in CSV are missing on disk")

    unlabeled = sum(1 for r in store.rows if not r["label"])
    print(f"{len(store.rows)} rows, {unlabeled} unlabeled — serving http://localhost:{args.port}")
    print("Labels are written to the CSV on every button press. Ctrl+C to stop.")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, video_paths))
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
