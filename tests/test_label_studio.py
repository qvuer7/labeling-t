"""Label Studio adapter: percent coords, config matches categories, round-trip."""

import math

from labeling_t.adapters.label_studio import (
    from_label_studio,
    generate_label_config,
    to_label_studio_tasks,
)
from labeling_t.schema import BBox, Detection, ImageLabels


def approx(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-4)


def _img():
    return ImageLabels(
        image_path="frame.jpg",
        width=1000,
        height=500,
        detections=[
            Detection(bbox=BBox(x1=100, y1=50, x2=600, y2=300), category="player", score=0.8)
        ],
    )


# --- labeling config ----------------------------------------------------------

def test_config_lists_every_category():
    cfg = generate_label_config(["player", "ball", "ref"])
    for c in ["player", "ball", "ref"]:
        assert f'value="{c}"' in cfg
    assert "<RectangleLabels" in cfg and 'toName="image"' in cfg


def test_config_escapes_special_chars():
    # Categories with XML-special chars must produce parseable XML whose label
    # values survive a round-trip (quoteattr may use single OR double quotes).
    import xml.etree.ElementTree as ET

    cats = ['a & b', 'x"y', "<weird>"]
    cfg = generate_label_config(cats)
    root = ET.fromstring(cfg)
    values = [el.get("value") for el in root.iter("Label")]
    assert values == cats


# --- to LS tasks --------------------------------------------------------------

def test_tasks_use_percent_coordinates():
    tasks = to_label_studio_tasks([_img()])
    v = tasks[0]["predictions"][0]["result"][0]["value"]
    # x=100/1000=10%, y=50/500=10%, w=500/1000=50%, h=250/500=50%
    assert approx(v["x"], 10.0) and approx(v["y"], 10.0)
    assert approx(v["width"], 50.0) and approx(v["height"], 50.0)
    assert v["rectanglelabels"] == ["player"]


def test_tasks_carry_dims_and_score():
    item = to_label_studio_tasks([_img()])[0]
    pred = item["predictions"][0]
    assert pred["result"][0]["original_width"] == 1000
    assert pred["result"][0]["original_height"] == 500
    assert approx(pred["score"], 0.8)
    assert item["data"]["image"] == "frame.jpg"


def test_tasks_use_presigned_url_when_presign_given():
    img = ImageLabels(
        image_path="s3://ml-cv-data/datasets/d/frames/g/g_000_00001.jpg",
        width=100, height=100,
        detections=[Detection(bbox=BBox(x1=10, y1=20, x2=60, y2=80), category="player")],
    )
    tasks = to_label_studio_tasks([img], presign=lambda uri: f"https://signed/{uri.split('/')[-1]}?sig=abc")
    assert tasks[0]["data"]["image"] == "https://signed/g_000_00001.jpg?sig=abc"


def test_image_with_no_detections_yields_empty_result():
    img = ImageLabels(image_path="e.jpg", width=10, height=10, detections=[])
    tasks = to_label_studio_tasks([img])
    pred = tasks[0]["predictions"][0]
    assert pred["result"] == []
    assert "score" not in pred  # no scores to average


# --- from LS export -----------------------------------------------------------

def test_from_label_studio_parses_back_to_pixels():
    export = [
        {
            "data": {"image": "frame.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "original_width": 1000,
                            "original_height": 500,
                            "value": {
                                "x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0,
                                "rectanglelabels": ["player"],
                            },
                        }
                    ]
                }
            ],
        }
    ]
    out = from_label_studio(export)
    assert len(out) == 1
    d = out[0].detections[0]
    assert approx(d.bbox.x1, 100) and approx(d.bbox.y1, 50)
    assert approx(d.bbox.x2, 600) and approx(d.bbox.y2, 300)
    assert d.category == "player"
    assert d.score is None  # verified labels carry no model score


def test_full_roundtrip_schema_to_ls_and_back():
    original = _img()
    tasks = to_label_studio_tasks([original])
    # read back from the predictions we just wrote (source=predictions)
    back = from_label_studio(tasks, result_source="predictions")
    assert len(back) == 1
    ob, bb = original.detections[0].bbox, back[0].detections[0].bbox
    assert approx(ob.x1, bb.x1) and approx(ob.y1, bb.y1)
    assert approx(ob.x2, bb.x2) and approx(ob.y2, bb.y2)
    assert back[0].detections[0].category == "player"
