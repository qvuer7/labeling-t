"""COCO export: emitted file must load back as valid COCO with correct boxes."""

import json
import os

import pytest
from PIL import Image

from labeling_t.adapters.coco import to_coco
from labeling_t.schema import BBox, Detection, ImageLabels


def _make_image(path, w, h):
    Image.new("RGB", (w, h), (0, 0, 0)).save(path)


def test_to_coco_roundtrips_box_and_category(tmp_path):
    imgp = str(tmp_path / "f1.jpg")
    _make_image(imgp, 640, 480)
    labels = [
        ImageLabels(
            image_path=imgp,
            width=640,
            height=480,
            detections=[
                Detection(bbox=BBox(x1=10, y1=20, x2=110, y2=220), category="player", score=0.9)
            ],
        )
    ]
    ann = str(tmp_path / "coco.json")
    n_imgs, n_anns = to_coco(labels, ann, classes=["player", "ball"])
    assert (n_imgs, n_anns) == (1, 1)

    coco = json.loads(open(ann).read())
    assert coco["images"][0]["width"] == 640
    assert coco["images"][0]["height"] == 480
    # COCO bbox is [x, y, w, h] absolute pixels
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    assert cat[coco["annotations"][0]["category_id"]] == "player"


def test_to_coco_handles_image_with_no_detections(tmp_path):
    imgp = str(tmp_path / "empty.jpg")
    _make_image(imgp, 100, 100)
    labels = [ImageLabels(image_path=imgp, width=100, height=100, detections=[])]
    ann = str(tmp_path / "coco.json")
    n_imgs, n_anns = to_coco(labels, ann, classes=["player"])
    assert n_imgs == 1 and n_anns == 0


def test_to_coco_rejects_category_outside_class_list(tmp_path):
    imgp = str(tmp_path / "f.jpg")
    _make_image(imgp, 100, 100)
    labels = [
        ImageLabels(
            image_path=imgp,
            width=100,
            height=100,
            detections=[Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="ufo")],
        )
    ]
    with pytest.raises(ValueError, match="not in class list"):
        to_coco(labels, str(tmp_path / "c.json"), classes=["player", "ball"])


def test_to_coco_derives_stable_sorted_classes(tmp_path):
    imgp = str(tmp_path / "f.jpg")
    _make_image(imgp, 100, 100)
    labels = [
        ImageLabels(
            image_path=imgp,
            width=100,
            height=100,
            detections=[
                Detection(bbox=BBox(x1=0, y1=0, x2=5, y2=5), category="zebra"),
                Detection(bbox=BBox(x1=5, y1=5, x2=9, y2=9), category="ant"),
            ],
        )
    ]
    ann = str(tmp_path / "c.json")
    to_coco(labels, ann)  # classes=None -> sorted unique
    coco = json.loads(open(ann).read())
    names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    assert names == ["ant", "zebra"]
