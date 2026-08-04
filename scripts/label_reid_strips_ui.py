#!/usr/bin/env python3
"""Local web UI for labeling ReID track-fragment strips (groups.csv + sheets).

Usage:
    python scripts/label_reid_strips_ui.py /home/andrii/Downloads/reid_group_1079311/groups.csv [--port 8767]

Each CSV row is one strip in a contact sheet (25 strips of 432x198 px per
sheet, CSV order = top-to-bottom strip order). The task: fill the `person`
column with a consistent identity label (team color + jersey: y23, p7;
referees: ref1), MIXED for mid-strip identity swaps, or blank for
can't-tell. The identity ROSTER with per-person fragment counts is shown and
clickable, so the same real person always gets byte-identical labels — that
consistency is what the downstream fragment-count analysis joins on.

Every save rewrites the CSV atomically; Ctrl+C and re-run resumes at the
first unlabeled row. Sheets are resolved relative to the CSV directory.

Hotkeys: Enter = save + next · ↓/↑ = next/prev (no save) · Esc = clear input.
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

STRIP_W, STRIP_H = 432, 198  # sheet geometry (measured; 25 strips/full sheet)


class Store:
    """CSV-backed row store with atomic writes (same shape as the sibling UIs)."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        if "person" not in self.fieldnames:
            self.fieldnames.append("person")
        # strip index WITHIN its sheet = row's rank among same-sheet rows (CSV order)
        seen: dict[str, int] = {}
        self.strip_idx: list[int] = []
        for r in self.rows:
            k = r["sheet"]
            self.strip_idx.append(seen.get(k, 0))
            seen[k] = seen.get(k, 0) + 1

    def save(self, index: int, person: str) -> None:
        with self.lock:
            self.rows[index]["person"] = person
            tmp = self.csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.csv_path)

    def state(self) -> dict:
        with self.lock:
            return {
                "rows": [
                    {"i": i, "tid": r["tid"], "seg": r["seg"],
                     "start_f": r["start_f"], "end_f": r["end_f"],
                     "n_frames": r["n_frames"], "sheet": r["sheet"],
                     "strip": self.strip_idx[i],
                     "person": (r.get("person") or "").strip()}
                    for i, r in enumerate(self.rows)
                ],
            }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ReID strip labeling</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #14141c; color: #eee;
         display: flex; height: 100vh; overflow: hidden; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center;
          padding: 16px; overflow-y: auto; }
  #meta { font: 15px/1.4 monospace; color: #9ad; margin-bottom: 8px; }
  #progress { color: #888; font-size: 13px; margin-bottom: 10px; }
  .strip { width: 864px; height: 396px; background-repeat: no-repeat;
           background-size: 864px auto; border-radius: 6px; border: 2px solid #333; }
  .strip.ctx { width: 648px; height: 297px; background-size: 648px auto;
               opacity: 0.35; border-color: transparent; }
  #controls { margin-top: 14px; display: flex; gap: 8px; align-items: center; }
  #person { font: 18px monospace; padding: 8px 12px; width: 180px; background: #222;
            color: #fff; border: 2px solid #555; border-radius: 6px; outline: none; }
  #person.known { border-color: #4a4; }
  #person.new { border-color: #ca4; }
  button { font: 14px system-ui; padding: 8px 14px; border-radius: 6px; border: 1px solid #444;
           background: #2a2a36; color: #ddd; cursor: pointer; }
  button:hover { background: #3a3a4a; }
  #mixed { border-color: #a44; color: #f88; }
  #hint { color: #666; font-size: 12px; margin-top: 10px; }
  #roster { width: 260px; border-left: 1px solid #2a2a36; padding: 14px; overflow-y: auto; }
  #roster h3 { margin: 4px 0 10px; font-size: 14px; color: #aaa; }
  .chip { display: flex; justify-content: space-between; padding: 6px 10px; margin: 4px 0;
          background: #22222e; border-radius: 6px; cursor: pointer; font: 14px monospace; }
  .chip:hover { background: #33334a; }
  .chip .n { color: #789; }
  .chip.special { color: #f88; }
</style></head><body>
<div id="main">
  <div id="meta"></div>
  <div id="progress"></div>
  <div id="prev" class="strip ctx"></div>
  <div id="strip" class="strip"></div>
  <div id="next" class="strip ctx"></div>
  <div id="controls">
    <button onclick="nav(-1)">&#8593; prev</button>
    <input id="person" autocomplete="off" spellcheck="false"
           placeholder="y23 / p7 / ref1 ...">
    <button onclick="save(document.getElementById('person').value)">save &#8594;</button>
    <button id="mixed" onclick="save('MIXED')">MIXED</button>
    <button onclick="save('')">blank &#8594;</button>
    <button onclick="nav(1)">skip &#8595;</button>
  </div>
  <div id="hint">Enter = save+next &nbsp;&middot;&nbsp; &#8595;/&#8593; = move (no save)
    &nbsp;&middot;&nbsp; green border = known name, orange = NEW name (check for typos)
    &nbsp;&middot;&nbsp; click roster to apply</div>
</div>
<div id="roster"><h3>identities</h3><div id="chips"></div></div>
<script>
let rows = [], cur = 0;
const STRIP_H = 198;
const $ = id => document.getElementById(id);

function counts() {
  const c = {};
  for (const r of rows) if (r.person && r.person !== 'MIXED')
    c[r.person] = (c[r.person] || 0) + 1;
  return c;
}
function setStrip(el, r, scale) {
  if (!r) { el.style.backgroundImage = 'none'; return; }
  el.style.backgroundImage = `url(/sheet/${r.sheet})`;
  el.style.backgroundPosition = `0px -${r.strip * STRIP_H * scale}px`;
}
function render() {
  const r = rows[cur];
  setStrip($('prev'), rows[cur - 1], 1.5);
  setStrip($('strip'), r, 2);
  setStrip($('next'), rows[cur + 1], 1.5);
  $('meta').textContent =
    `row ${cur + 1}/${rows.length} — tid${r.tid} seg${r.seg} f${r.start_f}-${r.end_f} ` +
    `(${r.n_frames}fr) — ${r.sheet} strip ${r.strip}` +
    (r.person ? ` — labeled: ${r.person}` : '');
  const done = rows.filter(x => x.person).length;
  $('progress').textContent = `${done}/${rows.length} labeled`;
  const inp = $('person');
  inp.value = r.person || '';
  inp.focus(); inp.select();
  markInput();
  const c = counts();
  const names = Object.keys(c).sort();
  $('chips').innerHTML =
    names.map(n => `<div class="chip" onclick="save('${n}')"><span>${n}</span>` +
                   `<span class="n">${c[n]}</span></div>`).join('') +
    `<div class="chip special" onclick="save('MIXED')"><span>MIXED</span>` +
    `<span class="n">${rows.filter(x => x.person === 'MIXED').length}</span></div>`;
}
function markInput() {
  const v = $('person').value.trim();
  const inp = $('person');
  inp.className = !v ? '' : (counts()[v] || v === 'MIXED') ? 'known' : 'new';
}
function nav(d) {
  cur = Math.max(0, Math.min(rows.length - 1, cur + d));
  render();
}
async function save(v) {
  const r = rows[cur];
  r.person = v.trim();
  await fetch('/save', { method: 'POST',
    body: JSON.stringify({ index: r.i, person: r.person }) });
  if (cur < rows.length - 1) cur++;
  render();
}
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') { save($('person').value); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { nav(1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { nav(-1); e.preventDefault(); }
  else if (e.key === 'Escape') { $('person').value = ''; markInput(); }
});
document.addEventListener('input', markInput);
fetch('/state').then(r => r.json()).then(s => {
  rows = s.rows;
  const first = rows.findIndex(r => !r.person);
  cur = first === -1 ? 0 : first;
  render();
});
</script></body></html>"""


def make_handler(store: Store, sheet_dir: Path):
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
            elif path.startswith("/sheet/"):
                name = unquote(path[len("/sheet/"):])
                target = (sheet_dir / name).resolve()
                if target.parent != sheet_dir.resolve() or not target.exists():
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
            store.save(int(data["index"]), str(data["person"]))
            self._send(b"{}", "application/json")

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="groups.csv (sheets resolved from its directory)")
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()

    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        print(f"no such file: {csv_path}", file=sys.stderr)
        return 1
    store = Store(csv_path)
    url = f"http://127.0.0.1:{a.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", a.port),
                                 make_handler(store, csv_path.parent))
    done = sum(1 for r in store.rows if (r.get("person") or "").strip())
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
