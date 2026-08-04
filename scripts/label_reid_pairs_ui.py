#!/usr/bin/env python3
"""Local web UI for verifying ReID track pairs (pairs.csv + pair_*.jpg).

Usage:
    python scripts/label_reid_pairs_ui.py /home/andrii/Downloads/reid_pairs_1079311/pairs.csv [--port 8788]

Each CSV row is one pair image (OLD track strip on top, NEW below, headers
baked in): did the tracker's re-identification adopt the RIGHT player? Fill
`same_player`: yes / no / blank (can't tell).

Hotkeys: 1 = same (yes) · 2 = different (no) · 3 = blank · ↓/↑ = move (no save).
Every save rewrites the CSV atomically; re-run resumes at the first unlabeled
row. Images are resolved relative to the CSV directory.
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


class Store:
    """CSV-backed row store with atomic writes (same shape as the sibling UIs)."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        if "same_player" not in self.fieldnames:
            self.fieldnames.append("same_player")

    def save(self, index: int, value: str) -> None:
        with self.lock:
            self.rows[index]["same_player"] = value
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.csv_path)

    def state(self) -> dict:
        with self.lock:
            return {"rows": [
                {"i": i, "sheet": r["sheet"], "frame": r["frame"],
                 "old_tid": r["old_tid"], "new_tid": r["new_tid"],
                 "cid": r["cid"], "sim": r["sim"], "in_frozen": r["in_frozen"],
                 "label": (r.get("same_player") or "").strip()}
                for i, r in enumerate(self.rows)
            ]}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ReID pair verification</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #14141c; color: #eee;
         display: flex; height: 100vh; overflow: hidden; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center;
          padding: 16px; overflow-y: auto; }
  #meta { font: 15px/1.4 monospace; color: #9ad; margin-bottom: 8px; }
  #progress { color: #888; font-size: 13px; margin-bottom: 10px; }
  #pair { max-width: 1050px; width: 95%; border-radius: 6px; border: 2px solid #333; }
  #controls { margin-top: 14px; display: flex; gap: 10px; align-items: center; }
  button { font: 16px system-ui; padding: 10px 18px; border-radius: 6px;
           border: 1px solid #444; background: #2a2a36; color: #ddd; cursor: pointer; }
  button:hover { background: #3a3a4a; }
  button u { color: #789; text-decoration: none; margin-right: 4px; }
  .same { border-color: #4a4; color: #8e8; }
  .diff { border-color: #a44; color: #f88; }
  .blankb { border-color: #777; color: #bbb; }
  #hint { color: #666; font-size: 12px; margin-top: 10px; }
  #panel { width: 230px; border-left: 1px solid #2a2a36; padding: 14px; overflow-y: auto; }
  #panel h3 { margin: 4px 0 10px; font-size: 14px; color: #aaa; }
  .chip { display: flex; justify-content: space-between; padding: 6px 10px; margin: 4px 0;
          background: #22222e; border-radius: 6px; font: 14px monospace; }
  .chip .n { color: #789; }
  .chip.bad { color: #f88; }
</style></head><body>
<div id="main">
  <div id="meta"></div>
  <div id="progress"></div>
  <img id="pair">
  <div id="controls">
    <button onclick="nav(-1)">&#8593; prev</button>
    <button class="same" onclick="save('yes')"><u>1</u> same</button>
    <button class="diff" onclick="save('no')"><u>2</u> different</button>
    <button class="blankb" onclick="save('')"><u>3</u> blank</button>
    <button onclick="nav(1)">skip &#8595;</button>
  </div>
  <div id="hint">hotkeys: <b>1</b> same &middot; <b>2</b> different &middot; <b>3</b> blank
    &middot; &#8595;/&#8593; move (no save)</div>
</div>
<div id="panel"><h3>counts</h3><div id="chips"></div></div>
<script>
let rows = [], cur = 0;
const $ = id => document.getElementById(id);

function render() {
  const r = rows[cur];
  $('pair').src = `/img/${r.sheet}`;
  $('meta').textContent =
    `pair ${cur + 1}/${rows.length} — old tid${r.old_tid} (cid ${r.cid}) vs ` +
    `new tid${r.new_tid} @f${r.frame} — sim ${r.sim} — frozen: ${r.in_frozen}` +
    (r.label ? ` — labeled: ${r.label}` : '');
  const done = rows.filter(x => x.label).length;
  $('progress').textContent = `${done}/${rows.length} labeled`;
  const n = v => rows.filter(x => x.label === v).length;
  $('chips').innerHTML =
    `<div class="chip"><span>same (yes)</span><span class="n">${n('yes')}</span></div>` +
    `<div class="chip bad"><span>different (no)</span><span class="n">${n('no')}</span></div>` +
    `<div class="chip"><span>empty (blank/unseen)</span><span class="n">${n('')}</span></div>`;
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
  if (e.key === '1') { save('yes'); e.preventDefault(); }
  else if (e.key === '2') { save('no'); e.preventDefault(); }
  else if (e.key === '3') { save(''); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { nav(1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { nav(-1); e.preventDefault(); }
});
fetch('/state').then(r => r.json()).then(s => {
  rows = s.rows;
  const first = rows.findIndex(r => !r.label);
  cur = first === -1 ? 0 : first;
  render();
});
</script></body></html>"""


def make_handler(store: Store, img_dir: Path):
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
                name = unquote(path[len("/img/"):])
                target = (img_dir / name).resolve()
                if target.parent != img_dir.resolve() or not target.exists():
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
    ap.add_argument("csv", help="pairs.csv (pair images resolved from its directory)")
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args()

    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        print(f"no such file: {csv_path}", file=sys.stderr)
        return 1
    store = Store(csv_path)
    url = f"http://127.0.0.1:{a.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", a.port),
                                 make_handler(store, csv_path.parent))
    done = sum(1 for r in store.rows if (r.get("same_player") or "").strip())
    print(f"{len(store.rows)} pairs, {done} already labeled — {url} (Ctrl+C to stop; "
          "progress saves on every label)")
    threading.Timer(0.4, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — labels are saved in", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
