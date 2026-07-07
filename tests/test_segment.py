"""Segment stage: box→mask enrichment, per-detection resume, failure isolation."""

import json
from pathlib import Path

from PIL import Image

from labeling_t.schema import BBox, Detection, ImageLabels
from labeling_t.storage import LocalStorage
from labeling_t.segment import FAILURES_NAME, segment_cloud

RLE = {"size": [100, 100], "counts": "fake"}


class FakeSegmenter:
    """Records segment() calls; returns one masked detection per box, in order."""

    def __init__(self, boom: Exception | None = None, short: bool = False):
        self.calls: list[tuple] = []
        self.boom = boom
        self.short = short  # return one fewer mask than boxes (contract breach)

    def segment(self, image_url, boxes, *, labels=None, scores=None):
        if self.boom is not None:
            raise self.boom
        self.calls.append((image_url, boxes, labels, scores))
        out = [{"bbox": b, "label": (labels or [None] * len(boxes))[i],
                "score": None, "mask": dict(RLE)} for i, b in enumerate(boxes)]
        return out[:-1] if self.short else out


def _dataset(tmp_path, *, detections, stem="f0", size=(100, 100)):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    frame = frames / f"{stem}.jpg"
    Image.new("RGB", size, (5, 5, 5)).save(frame)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(exist_ok=True)
    labels = ImageLabels(image_path=str(frame), width=size[0], height=size[1],
                         detections=detections)
    (labels_dir / f"{stem}.json").write_text(labels.model_dump_json())
    return labels_dir


def _det(cat, *, box=(10, 10, 40, 30), mask=None):
    return Detection(bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                     category=cat, mask=mask)


def test_segment_cloud_fills_masks_for_all_boxes(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("cat"), _det("dog", box=(50, 10, 90, 40))])
    client = FakeSegmenter()
    n = segment_cloud(str(labels_dir), client, storage=LocalStorage(), max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert all(d.mask == RLE for d in out.detections)
    # boxes and labels travelled to the segmenter in detection order
    (_, boxes, labels, _), = client.calls
    assert boxes == [[10, 10, 40, 30], [50, 10, 90, 40]]
    assert labels == ["cat", "dog"]


def test_segment_cloud_category_filter_and_resume(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[
        _det("cat"), _det("dog", box=(50, 10, 90, 40)), _det("cat", box=(0, 0, 5, 5), mask=dict(RLE)),
    ])
    client = FakeSegmenter()
    n = segment_cloud(str(labels_dir), client, storage=LocalStorage(),
                      categories=["cat"], max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    by_box = {d.bbox.x1: d for d in out.detections}
    assert by_box[10].mask == RLE        # unmasked cat -> segmented
    assert by_box[50].mask is None       # dog filtered out
    assert by_box[0].mask == RLE         # already-masked cat untouched
    # only the ONE unmasked cat box was prompted
    (_, boxes, _, _), = client.calls
    assert boxes == [[10, 10, 40, 30]]

    # second run: nothing left to do -> no calls
    client2 = FakeSegmenter()
    assert segment_cloud(str(labels_dir), client2, storage=LocalStorage(),
                         categories=["cat"], max_concurrency=1) == 0
    assert client2.calls == []


def test_segment_cloud_to_prefix_leaves_source_untouched(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("cat")])
    dest_dir = tmp_path / "labels-masked"
    n = segment_cloud(str(labels_dir), FakeSegmenter(), storage=LocalStorage(),
                      to_prefix=str(dest_dir), max_concurrency=1)
    assert n == 1
    src = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert src.detections[0].mask is None
    out = ImageLabels.model_validate_json((dest_dir / "f0.json").read_text())
    assert out.detections[0].mask == RLE


def test_segment_cloud_count_mismatch_is_a_failure_not_corruption(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("cat"), _det("dog", box=(50, 10, 90, 40))])
    n = segment_cloud(str(labels_dir), FakeSegmenter(short=True),
                      storage=LocalStorage(), max_concurrency=1)
    assert n == 0
    fails = (labels_dir / FAILURES_NAME).read_text().strip().splitlines()
    assert len(fails) == 1 and "2 boxes" in json.loads(fails[0])["error"]
    # the label file was NOT rewritten with misaligned masks
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert all(d.mask is None for d in out.detections)


def test_segment_cloud_failure_does_not_stop_run(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("cat")], stem="bad")
    frames = tmp_path / "frames"
    frame2 = frames / "good.jpg"
    Image.new("RGB", (100, 100)).save(frame2)
    good = ImageLabels(image_path=str(frame2), width=100, height=100,
                       detections=[_det("dog")])
    (labels_dir / "good.json").write_text(good.model_dump_json())
    (labels_dir / "bad.json").write_text("{not json")

    n = segment_cloud(str(labels_dir), FakeSegmenter(), storage=LocalStorage(),
                      max_concurrency=1)
    assert n == 1
    fails = (labels_dir / FAILURES_NAME).read_text().strip().splitlines()
    assert len(fails) == 1 and "bad.json" in json.loads(fails[0])["labels"]


def test_segment_cloud_stems_filter_restricts_the_run(tmp_path):
    _dataset(tmp_path, detections=[_det("cat")], stem="f0")
    labels_dir = _dataset(tmp_path, detections=[_det("cat")], stem="f1")
    client = FakeSegmenter()
    n = segment_cloud(str(labels_dir), client, storage=LocalStorage(),
                      stems={"f1"}, max_concurrency=1)
    assert n == 1 and len(client.calls) == 1
    # f0 untouched: no mask landed on it
    f0 = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert all(d.mask is None for d in f0.detections)
