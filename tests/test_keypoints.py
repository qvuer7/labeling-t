"""Keypoints stage: box→skeleton enrichment, per-detection resume, failure isolation.

Mirrors test_segment.py — the stages share a skeleton (pun intended) and their
contracts must stay symmetric.
"""

import json
from pathlib import Path

from PIL import Image

from labeling_t.keypoints import FAILURES_NAME, keypoints_cloud
from labeling_t.schema import BBox, Detection, ImageLabels, Keypoint
from labeling_t.storage import LocalStorage

KPS = [{"x": 20.0, "y": 20.0, "name": "nose", "score": 0.9},
       {"x": 15.0, "y": 25.0, "name": "left_eye", "score": 0.8}]


class FakePoser:
    """Records keypoints() calls; returns one skeleton per box, in order."""

    def __init__(self, boom: Exception | None = None, short: bool = False,
                 kps: list[dict] | None = None):
        self.calls: list[tuple] = []
        self.boom = boom
        self.short = short
        self.kps = KPS if kps is None else kps

    def keypoints(self, image_url, boxes, *, labels=None, scores=None):
        if self.boom is not None:
            raise self.boom
        self.calls.append((image_url, boxes, labels, scores))
        out = [{"bbox": b, "label": (labels or [None] * len(boxes))[i],
                "score": None, "keypoints": [dict(k) for k in self.kps]}
               for i, b in enumerate(boxes)]
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


def _det(cat, *, box=(10, 10, 40, 30), keypoints=None):
    return Detection(bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                     category=cat, keypoints=keypoints)


def test_keypoints_cloud_fills_skeletons_for_all_boxes(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("player"), _det("player", box=(50, 10, 90, 40))])
    client = FakePoser()
    n = keypoints_cloud(str(labels_dir), client, storage=LocalStorage(), max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert all(len(d.keypoints) == 2 for d in out.detections)
    assert out.detections[0].keypoints[0].name == "nose"  # dicts coerced to Keypoint
    (_, boxes, labels, _), = client.calls
    assert boxes == [[10, 10, 40, 30], [50, 10, 90, 40]]
    assert labels == ["player", "player"]


def test_keypoints_cloud_category_filter_and_per_detection_resume(tmp_path):
    already = [Keypoint(x=1, y=1, name="nose")]
    labels_dir = _dataset(tmp_path, detections=[
        _det("player"),                          # target
        _det("ball", box=(50, 50, 60, 60)),      # filtered out by category
        _det("player", box=(70, 10, 90, 40), keypoints=already),  # already posed
    ])
    client = FakePoser()
    n = keypoints_cloud(str(labels_dir), client, storage=LocalStorage(),
                        categories=["player"], max_concurrency=1)
    assert n == 1
    (_, boxes, _, _), = client.calls
    assert boxes == [[10, 10, 40, 30]]  # only the un-posed player box travelled
    # a second run finds nothing left to do and never calls the model
    client2 = FakePoser()
    n2 = keypoints_cloud(str(labels_dir), client2, storage=LocalStorage(),
                         categories=["player"], max_concurrency=1)
    assert n2 == 0 and client2.calls == []


def test_empty_skeleton_marks_attempted(tmp_path):
    # [] from the model = attempted, nothing found -> resume must NOT retry it
    labels_dir = _dataset(tmp_path, detections=[_det("player")])
    n = keypoints_cloud(str(labels_dir), FakePoser(kps=[]), storage=LocalStorage(),
                        max_concurrency=1)
    assert n == 1
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert out.detections[0].keypoints == []
    n2 = keypoints_cloud(str(labels_dir), FakePoser(), storage=LocalStorage(),
                         max_concurrency=1)
    assert n2 == 0


def test_out_of_frame_peak_is_clamped_not_fatal(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("player")])
    stray = [{"x": 100.4, "y": -0.2, "name": "nose", "score": 0.5}]
    n = keypoints_cloud(str(labels_dir), FakePoser(kps=stray), storage=LocalStorage(),
                        max_concurrency=1)
    assert n == 1  # not a failures.jsonl entry
    out = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    kp = out.detections[0].keypoints[0]
    assert (kp.x, kp.y) == (100.0, 0.0)


def test_to_prefix_leaves_source_untouched(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("player")])
    before = (labels_dir / "f0.json").read_bytes()
    n = keypoints_cloud(str(labels_dir), FakePoser(), storage=LocalStorage(),
                        to_prefix=str(tmp_path / "labels-pose"), max_concurrency=1)
    assert n == 1
    assert (labels_dir / "f0.json").read_bytes() == before
    copy = ImageLabels.model_validate_json((tmp_path / "labels-pose" / "f0.json").read_text())
    assert copy.detections[0].keypoints


def test_count_mismatch_is_a_failure_not_corruption(tmp_path):
    labels_dir = _dataset(tmp_path, detections=[_det("player"), _det("player", box=(50, 10, 90, 40))])
    before = (labels_dir / "f0.json").read_bytes()
    n = keypoints_cloud(str(labels_dir), FakePoser(short=True), storage=LocalStorage(),
                        max_concurrency=1)
    assert n == 0
    assert (labels_dir / "f0.json").read_bytes() == before  # source not half-written
    fails = (labels_dir / FAILURES_NAME).read_text().strip().splitlines()
    assert len(fails) == 1 and "ValueError" in json.loads(fails[0])["error"]


def test_stems_filter_restricts_the_run(tmp_path):
    _dataset(tmp_path, detections=[_det("player")], stem="f0")
    labels_dir = _dataset(tmp_path, detections=[_det("player")], stem="f1")
    client = FakePoser()
    n = keypoints_cloud(str(labels_dir), client, storage=LocalStorage(),
                        stems={"f1"}, max_concurrency=1)
    assert n == 1 and len(client.calls) == 1
    f0 = ImageLabels.model_validate_json((labels_dir / "f0.json").read_text())
    assert all(d.keypoints is None for d in f0.detections)
