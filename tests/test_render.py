"""render: pixel-probe checks (no golden files), deterministic sampling, CLI.

Probes assert the right pixels CHANGED (box edge, mask interior, caption area)
rather than exact renderings — robust to font/PIL version drift.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from labeling_t.cli import main
from labeling_t.render import render_labels, render_set
from labeling_t.schema import BBox, Detection, ImageLabels
from labeling_t.storage import LocalStorage

BG = (40, 40, 40)


def _frame_bytes(w=100, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), BG).save(buf, format="PNG")
    return buf.getvalue()


def _rle(w=100, h=100, box=(60, 60, 90, 90)) -> dict:
    arr = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = 1
    rle = mask_utils.encode(np.asfortranarray(arr))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode()}  # on-disk shape: str counts


def _labels(dets, w=100, h=100, path="frame.png") -> ImageLabels:
    return ImageLabels(image_path=path, width=w, height=h, detections=dets)


def _px(png: bytes, x: int, y: int) -> tuple:
    return Image.open(io.BytesIO(png)).convert("RGB").getpixel((x, y))


def test_box_outline_and_caption_are_drawn():
    d = Detection(bbox=BBox(x1=20, y1=30, x2=60, y2=70), category="player", score=0.9)
    png = render_labels(_labels([d]), _frame_bytes())
    assert _px(png, 40, 30) != BG          # top edge of the box
    assert _px(png, 20, 50) != BG          # left edge
    assert _px(png, 40, 50) == BG          # interior untouched (no mask)
    assert _px(png, 22, 33) != BG          # caption band just under (x1, y1)


def test_mask_tints_interior_at_partial_alpha():
    d = Detection(bbox=BBox(x1=60, y1=60, x2=90, y2=90), category="rim",
                  mask=_rle(box=(60, 60, 90, 90)))
    png = render_labels(_labels([d]), _frame_bytes())
    inside, outside = _px(png, 75, 80), _px(png, 10, 90)
    assert outside == BG
    assert inside != BG                     # tinted...
    assert all(abs(c - b) < 200 for c, b in zip(inside, BG))  # ...but translucent, not solid


def test_text_second_line_changes_pixels_below_caption():
    base = Detection(bbox=BBox(x1=20, y1=20, x2=80, y2=80), category="scoreboard")
    with_text = base.model_copy(update={"text": "42:39"})
    png_a = render_labels(_labels([base]), _frame_bytes())
    png_b = render_labels(_labels([with_text]), _frame_bytes())
    assert png_a != png_b                   # the text line is visible somewhere


def test_stable_category_colors_across_calls():
    d1 = Detection(bbox=BBox(x1=10, y1=10, x2=50, y2=50), category="player")
    d2 = Detection(bbox=BBox(x1=10, y1=10, x2=50, y2=50), category="player")
    assert _px(render_labels(_labels([d1]), _frame_bytes()), 30, 10) == \
        _px(render_labels(_labels([d2]), _frame_bytes()), 30, 10)


def _seed_set(tmp_path, n=5) -> str:
    st = LocalStorage()
    frame = tmp_path / "frame.png"
    frame.write_bytes(_frame_bytes())
    prefix = str(tmp_path / "labels")
    for i in range(n):
        img = _labels([Detection(bbox=BBox(x1=10, y1=10, x2=40, y2=40), category="c")],
                      path=str(frame))
        st.write_text(f"{prefix}/f{i}.json", img.model_dump_json())
    return prefix


def test_render_set_sampling_is_deterministic(tmp_path):
    prefix = _seed_set(tmp_path, n=5)
    kw = dict(storage=LocalStorage(), out_dir=str(tmp_path / "out"), sample=2, seed=0)
    first = render_set(prefix, **kw)["stems"]
    second = render_set(prefix, **kw)["stems"]
    assert first == second and len(first) == 2  # same seed = same frames


def test_render_set_bad_file_is_a_failure_entry_not_a_crash(tmp_path):
    prefix = _seed_set(tmp_path, n=2)
    st = LocalStorage()
    ghost = _labels([], path=str(tmp_path / "missing.png"))
    st.write_text(f"{prefix}/ghost.json", ghost.model_dump_json())
    res = render_set(prefix, storage=st, out_dir=str(tmp_path / "out"))
    assert res["rendered"] == 2
    assert [f["stem"] for f in res["failures"]] == ["ghost"]
    assert "FileNotFoundError" in res["failures"][0]["error"]


def test_cli_render_local_labels_json_envelope(tmp_path, capsys):
    prefix = _seed_set(tmp_path, n=3)
    out_dir = str(tmp_path / "png")
    rc = main(["render", "--labels", prefix, "--out", out_dir, "--stems", "f0,f1", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["rendered"] == 2
    assert envelope["result"]["stems"] == ["f0", "f1"]
    assert sorted(p.name for p in Path(out_dir).glob("*.png")) == ["f0.png", "f1.png"]


def test_cli_render_empty_selection_fails(tmp_path, capsys):
    prefix = _seed_set(tmp_path, n=2)
    rc = main(["render", "--labels", prefix, "--out", str(tmp_path / "png"),
               "--stems", "nope", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and "nothing to render" in envelope["error"]["message"]


def test_keypoints_dots_and_skeleton_edges():
    from labeling_t.schema import Keypoint

    d = Detection(bbox=BBox(x1=10, y1=10, x2=90, y2=90), category="player",
                  keypoints=[Keypoint(x=30, y=60, name="left_shoulder"),
                             Keypoint(x=70, y=60, name="right_shoulder")])
    frame = _frame_bytes()
    png_plain = render_labels(_labels([d]), frame)
    assert _px(png_plain, 30, 60) != BG          # dot at the keypoint
    assert _px(png_plain, 50, 60) == BG          # no edge without a skeleton
    png_edges = render_labels(_labels([d]), frame,
                              skeleton=[["left_shoulder", "right_shoulder"],
                                        ["left_shoulder", "missing_point"]])
    assert _px(png_edges, 50, 60) != BG          # edge drawn between the dots
    # the missing_point edge is skipped silently, not an error


def test_render_set_passes_skeleton_through(tmp_path):
    from labeling_t.schema import Keypoint
    from labeling_t.storage import LocalStorage as LS

    st = LS()
    frame = tmp_path / "frame.png"
    frame.write_bytes(_frame_bytes())
    prefix = str(tmp_path / "labels")
    img = _labels([Detection(bbox=BBox(x1=10, y1=10, x2=90, y2=90), category="c",
                             keypoints=[Keypoint(x=20, y=50, name="a"),
                                        Keypoint(x=80, y=50, name="b")])],
                  path=str(frame))
    st.write_text(f"{prefix}/f0.json", img.model_dump_json())
    res = render_set(prefix, storage=st, out_dir=str(tmp_path / "out"),
                     skeleton=[["a", "b"]])
    assert res["rendered"] == 1
    png = (tmp_path / "out" / "f0.png").read_bytes()
    assert _px(png, 50, 50) != BG                # the a-b edge crosses the middle
