#!/usr/bin/env python3
"""Local web UI for reviewing ball crops (review.csv + crops/*.jpg).

Usage:
    python scripts/label_ball_crops_ui.py /home/andrii/Downloads/review/review.csv [--port 8789]

Each CSV row is one crop: does the ball actually exist in it? Fills the
`is_ball` column: yes (exists) / no (non exists) / unsure.

Hotkeys: 1 = exists (yes) · 2 = non exists (no) · 3 = unsure · arrows = move (no save).
Every save rewrites the CSV atomically; re-run resumes at the first unlabeled
row. Crop paths are resolved relative to the CSV directory.
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

LABEL_COL = "is_ball"


class Store:
    """CSV-backed row store with atomic writes (same shape as the sibling UIs)."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        if LABEL_COL not in self.fieldnames:
            self.fieldnames.append(LABEL_COL)

    def save(self, index: int, value: str) -> None:
        with self.lock:
            self.rows[index][LABEL_COL] = value
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.csv_path)

    def state(self) -> dict:
        with self.lock:
            return {"rows": [
                {"i": i, "crop": r["crop"], "split": r.get("split", ""),
                 "source": r.get("source", ""), "stem": r.get("stem", ""),
                 "conf": r.get("conf", ""),
                 "label": (r.get(LABEL_COL) or "").strip()}
                for i, r in enumerate(self.rows)
            ]}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Ball crop review</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #14141c; color: #eee;
         display: flex; height: 100vh; overflow: hidden; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center;
          padding: 16px; overflow-y: auto; }
  #meta { font: 15px/1.4 monospace; color: #9ad; margin-bottom: 8px; }
  #progress { color: #888; font-size: 13px; margin-bottom: 10px; }
  #crop { max-width: 90%; max-height: 62vh; min-width: 320px; border-radius: 6px;
          border: 2px solid #333; image-rendering: pixelated; }
  #controls { margin-top: 14px; display: flex; gap: 10px; align-items: center; }
  button { font: 16px system-ui; padding: 10px 18px; border-radius: 6px;
           border: 1px solid #444; background: #2a2a36; color: #ddd; cursor: pointer; }
  button:hover { background: #3a3a4a; }
  button u { color: #789; text-decoration: none; margin-right: 4px; }
  .yes { border-color: #4a4; color: #8e8; }
  .no { border-color: #a44; color: #f88; }
  .unsure { border-color: #a84; color: #fc8; }
  #hint { color: #666; font-size: 12px; margin-top: 10px; }
  #panel { width: 230px; border-left: 1px solid #2a2a36; padding: 14px; overflow-y: auto; }
  #panel h3 { margin: 4px 0 10px; font-size: 14px; color: #aaa; }
  .chip { display: flex; justify-content: space-between; padding: 6px 10px; margin: 4px 0;
          background: #22222e; border-radius: 6px; font: 14px monospace; }
  .chip .n { color: #789; }
  .chip.bad { color: #f88; }
  .chip.warn { color: #fc8; }
</style></head><body>
<div id="main">
  <div id="meta"></div>
  <div id="progress"></div>
  <img id="crop">
  <div id="controls">
    <button onclick="nav(-1)">&#8592; prev</button>
    <button class="yes" onclick="save('yes')"><u>1</u> exists</button>
    <button class="no" onclick="save('no')"><u>2</u> non exists</button>
    <button class="unsure" onclick="save('unsure')"><u>3</u> unsure</button>
    <button onclick="nav(1)">skip &#8594;</button>
  </div>
  <div id="hint">hotkeys: <b>1</b> exists &middot; <b>2</b> non exists &middot; <b>3</b> unsure
    &middot; &#8592;/&#8594; (or &#8593;/&#8595;) move without saving</div>
</div>
<div id="panel"><h3>counts</h3><div id="chips"></div></div>
<script>
let rows = [], cur = 0;
const $ = id => document.getElementById(id);

function render() {
  const r = rows[cur];
  $('crop').src = `/img/${encodeURIComponent(r.crop)}`;
  $('meta').textContent =
    `crop ${cur + 1}/${rows.length} — ${r.stem} (${r.source}/${r.split}) — conf ${r.conf}` +
    (r.label ? ` — labeled: ${r.label}` : '');
  const done = rows.filter(x => x.label).length;
  $('progress').textContent = `${done}/${rows.length} labeled`;
  const n = v => rows.filter(x => x.label === v).length;
  $('chips').innerHTML =
    `<div class="chip"><span>exists (yes)</span><span class="n">${n('yes')}</span></div>` +
    `<div class="chip bad"><span>non exists (no)</span><span class="n">${n('no')}</span></div>` +
    `<div class="chip warn"><span>unsure</span><span class="n">${n('unsure')}</span></div>`;
}
function nav(d) {
  cur = Math.max(0, Math.min(rows.length - 1, cur + d));
  render();
}
async function save(v) {
  const r = rows[cur];
  r.label = v;
  await fetch('/save', { method: 'POST',
    body: JSON.stringify({ index: r.i, value: v }) });
  if (cur < rows.length - 1) cur++;
  render();
}
document.addEventListener('keydown', e => {
  if (e.repeat) { e.preventDefault(); return; }  // held key must not machine-gun labels
  if (e.key === '1') { save('yes'); e.preventDefault(); }
  else if (e.key === '2') { save('no'); e.preventDefault(); }
  else if (e.key === '3') { save('unsure'); e.preventDefault(); }
  else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { nav(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { nav(-1); e.preventDefault(); }
});
fetch('/state').then(r => r.json()).then(s => {
  rows = s.rows;
  const first = rows.findIndex(r => !r.label);
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
                self._send(target.read_bytes(), "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            store.save(int(data["index"]), str(data["value"]))
            self._send(b"{}", "application/json")

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="review.csv (crop paths resolved from its directory)")
    ap.add_argument("--port", type=int, default=8789)
    a = ap.parse_args()

    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        print(f"no such file: {csv_path}", file=sys.stderr)
        return 1
    store = Store(csv_path)
    url = f"http://127.0.0.1:{a.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", a.port),
                                 make_handler(store, csv_path.parent))
    done = sum(1 for r in store.rows if (r.get(LABEL_COL) or "").strip())
    print(f"{len(store.rows)} crops, {done} already labeled — {url} (Ctrl+C to stop; "
          "progress saves on every label)")
    threading.Timer(0.4, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — labels are saved in", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
