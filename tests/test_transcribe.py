"""Transcribe stage: category filter, per-detection resume, crop mechanics,
failure isolation — for both the local and cloud paths."""

import json
from pathlib import Path

import pytest
from PIL import Image

from labeling_t.schema import BBox, Detection, ImageLabels
from labeling_t.storage import LocalStorage
from labeling_t.transcribe import (
    FAILURES_NAME,
    _crop_png,
    clean_text,
    transcribe,
    transcribe_cloud,
)


class FakeOCRClient:
    """Records every crop it receives; returns canned replies in order (the
    last one repeats). Raises `boom` instead when set."""

    def __init__(self, replies=("87",), boom: Exception | None = None):
        self.replies = list(replies)
        self.calls: list[bytes] = []
        self.boom = boom

    def infer(self, image: bytes) -> str:
        if self.boom is not None:
            raise self.boom
        self.calls.append(image)
        i = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[i]


def _dataset(tmp_path, *, detections, stem="f0", size=(100, 100)):
    """A frame on disk + one ImageLabels JSON pointing at it. Returns the labels dir."""
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    frame = frames / f"{stem}.jpg"
    Image.new("RGB", size, (10, 20, 30)).save(frame)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(exist_ok=True)
    labels = ImageLabels(image_path=str(frame), width=size[0], height=size[1],
                         detections=detections)
    (labels_dir / f"{stem}.json").write_text(labels.model_dump_json())
    return labels_dir


def _det(cat, *, box=(10, 10, 40, 30), text=None, mask=None):
    return Detection(bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                     category=cat, text=text, mask=mask)


# --- clean_text ----------------------------------------------------------------

def test_clean_text_strips_decorations():
    assert clean_text("  87 \n") == "87"
    assert clean_text('"87"') == "87"
    assert clean_text("'87'") == "87"
    assert clean_text("```\n87\n```") == "87"
    assert clean_text("```text\n87:12\n```") == "87:12"
    assert clean_text("") == ""
    assert clean_text("  ") == ""


def test_clean_text_keeps_inner_content():
    # multi-word replies aren't a fence language tag — nothing gets eaten
    assert clean_text("HOME 87") == "HOME 87"


# --- _crop_png -----------------------------------------------------------------

def test_crop_png_pads_and_clamps_at_edges():
    im = Image.new("RGB", (100, 50))
    # box at the exact image edge: pad must clamp, not go negative/overflow
    png = _crop_png(im, BBox(x1=0, y1=0, x2=100, y2=50), pad=2)
    out = Image.open(__import__("io").BytesIO(png))
    assert out.size == (100, 50)
    # interior box: pad grows it by 2 on each side
    png = _crop_png(im, BBox(x1=10, y1=10, x2=20, y2=20), pad=2)
    assert Image.open(__import__("io").BytesIO(png)).size == (14, 14)


def test_crop_png_rejects_degenerate_box():
    im = Image.new("RGB", (100, 50))
    with pytest.raises(ValueError, match="degenerate"):
        _crop_png(im, BBox(x1=5, y1=5, x2=5, y2=5), pad=0)


# --- local transcribe ----------------------------------------------------------

def test_transcribe_fills_only_matching_categories(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("score_home"), _det("player")])
    client = FakeOCRClient(replies=("87",))
    n = transcribe(labels_dir, client, categories=["score_home"], max_concurrency=1)
    assert n == 1
    assert len(client.calls) == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    by_cat = {d.category: d for d in out.detections}
    assert by_cat["score_home"].text == "87"
    assert by_cat["player"].text is None
    # the untouched fields survive the rewrite
    assert by_cat["score_home"].bbox.x1 == 10


def test_transcribe_resume_is_per_detection(tmp_path):
    # one done ("" counts as attempted!), one todo -> exactly one API call
    labels_dir = _dataset(tmp_path, detections=[
        _det("timer", text=""), _det("timer", box=(50, 10, 80, 30)),
    ])
    client = FakeOCRClient(replies=("12:34",))
    n = transcribe(labels_dir, client, categories=["timer"], max_concurrency=1)
    assert n == 1 and len(client.calls) == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert [d.text for d in out.detections] == ["", "12:34"]

    # second run: everything attempted -> zero calls, zero files rewritten
    client2 = FakeOCRClient()
    assert transcribe(labels_dir, client2, categories=["timer"], max_concurrency=1) == 0
    assert client2.calls == []


def test_transcribe_failure_goes_to_jsonl_and_run_continues(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("timer")], stem="bad")
    # a second, healthy file
    frames = tmp_path / "frames"
    frame2 = frames / "good.jpg"
    Image.new("RGB", (100, 100)).save(frame2)
    good = ImageLabels(image_path=str(frame2), width=100, height=100,
                       detections=[_det("timer")])
    (labels_dir / "good.json").write_text(good.model_dump_json())
    # break the bad file's frame
    Path(json.loads((labels_dir / "bad.json").read_text())["image_path"]).unlink()

    client = FakeOCRClient(replies=("59",))
    n = transcribe(labels_dir, client, categories=["timer"], max_concurrency=1)
    assert n == 1  # the good one
    fails = (labels_dir / FAILURES_NAME).read_text().strip().splitlines()
    assert len(fails) == 1 and "bad.json" in json.loads(fails[0])["labels"]
    out = ImageLabels.model_validate_json((labels_dir / "good.json").read_text())
    assert out.detections[0].text == "59"


def test_transcribe_images_dir_override(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("timer")])
    # simulate from-ls labels: image_path is a bare name that doesn't resolve
    j = json.loads((labels_dir / "f0.json").read_text())
    j["image_path"] = "f0.jpg"
    (labels_dir / "f0.json").write_text(json.dumps(j))
    client = FakeOCRClient(replies=("31",))
    n = transcribe(labels_dir, client, categories=["timer"],
                   images_dir=tmp_path / "frames", max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert out.detections[0].text == "31"


# --- cloud transcribe ----------------------------------------------------------

def test_transcribe_cloud_in_place_preserves_masks(tmp_path):
    rle = {"size": [100, 100], "counts": "xyz"}
    labels_dir = _dataset(tmp_path, detections=[_det("score_home", mask=rle)])
    client = FakeOCRClient(replies=("103",))
    n = transcribe_cloud(str(labels_dir), client, storage=LocalStorage(),
                         categories=["score_home"], max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert out.detections[0].text == "103"
    assert out.detections[0].mask == rle  # enrichment didn't drop the mask


def test_transcribe_cloud_to_prefix_leaves_source_untouched(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("timer")])
    dest_dir = tmp_path / "labels-ocr"
    client = FakeOCRClient(replies=("07",))
    n = transcribe_cloud(str(labels_dir), client, storage=LocalStorage(),
                         categories=["timer"], to_prefix=str(dest_dir),
                         max_concurrency=1)
    assert n == 1
    src = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert src.detections[0].text is None
    out = ImageLabels.model_validate_json((dest_dir / "f0.json").read_text())
    assert out.detections[0].text == "07"

    # resume reads the enriched copy: nothing left to do, no calls
    client2 = FakeOCRClient()
    n = transcribe_cloud(str(labels_dir), client2, storage=LocalStorage(),
                         categories=["timer"], to_prefix=str(dest_dir),
                         max_concurrency=1)
    assert n == 0 and client2.calls == []


def test_transcribe_cloud_failure_flushed_once(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("timer")])
    client = FakeOCRClient(boom=RuntimeError("api down"))
    n = transcribe_cloud(str(labels_dir), client, storage=LocalStorage(),
                         categories=["timer"], max_concurrency=1)
    assert n == 0
    fails = (labels_dir / FAILURES_NAME).read_text().strip().splitlines()
    assert len(fails) == 1 and "api down" in json.loads(fails[0])["error"]
