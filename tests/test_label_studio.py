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


def test_polygon_config_uses_polygonlabels():
    cfg = generate_label_config(["player", "ball"], control="polygon")
    assert "<PolygonLabels" in cfg and "<RectangleLabels" not in cfg
    assert '<Label value="player"/>' in cfg
    # default stays rectangle (boxes)
    assert "<RectangleLabels" in generate_label_config(["player"])


def test_polygon_tasks_emit_polygon_regions_from_masks():
    import pytest
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    mu = pytest.importorskip("pycocotools.mask", reason="needs pycocotools")
    m = np.zeros((500, 1000), np.uint8); m[50:300, 100:600] = 1
    enc = mu.encode(np.asfortranarray(m))
    rle = {"size": [500, 1000], "counts": enc["counts"].decode("ascii")}
    img = ImageLabels(image_path="f.jpg", width=1000, height=500, detections=[
        Detection(bbox=BBox(x1=100, y1=50, x2=600, y2=300), category="player", mask=rle),
        Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="ball"),  # no mask -> skipped
    ])
    tasks = to_label_studio_tasks([img], control="polygon")
    results = tasks[0]["predictions"][0]["result"]
    assert len(results) == 1 and results[0]["type"] == "polygonlabels"
    assert results[0]["value"]["polygonlabels"] == ["player"]
    for x, y in results[0]["value"]["points"]:        # percent, in-bounds
        assert 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0


def test_from_ls_reads_polygon_into_box_and_mask():
    import pytest
    pytest.importorskip("cv2")
    pytest.importorskip("pycocotools.mask", reason="needs pycocotools")
    task = {"data": {"image": "f.jpg"}, "annotations": [{"result": [
        {"type": "polygonlabels", "original_width": 1000, "original_height": 500,
         "value": {"points": [[10, 10], [60, 10], [60, 60], [10, 60]],
                   "polygonlabels": ["player"]}}]}]}
    out = from_label_studio([task])
    assert len(out) == 1 and len(out[0].detections) == 1
    d = out[0].detections[0]
    assert d.category == "player" and d.mask is not None and d.mask["size"] == [500, 1000]
    # box from polygon extent: x 10..60% of 1000 -> 100..600, y 10..60% of 500 -> 50..300
    assert abs(d.bbox.x1 - 100) < 2 and abs(d.bbox.x2 - 600) < 2
    assert abs(d.bbox.y1 - 50) < 2 and abs(d.bbox.y2 - 300) < 2


def test_brush_config_and_tasks_emit_brush_regions_from_masks():
    import pytest
    np = pytest.importorskip("numpy")
    mu = pytest.importorskip("pycocotools.mask", reason="needs pycocotools")
    pytest.importorskip("label_studio_sdk.converter.brush", reason="needs label-studio-sdk")
    assert "<BrushLabels" in generate_label_config(["player"], control="brush")
    m = np.zeros((500, 1000), np.uint8); m[50:300, 100:600] = 1
    enc = mu.encode(np.asfortranarray(m))
    rle = {"size": [500, 1000], "counts": enc["counts"].decode("ascii")}
    img = ImageLabels(image_path="f.jpg", width=1000, height=500, detections=[
        Detection(bbox=BBox(x1=100, y1=50, x2=600, y2=300), category="player", mask=rle)])
    res = to_label_studio_tasks([img], control="brush")[0]["predictions"][0]["result"]
    assert len(res) == 1 and res[0]["type"] == "brushlabels"
    assert res[0]["value"]["format"] == "rle" and res[0]["value"]["brushlabels"] == ["player"]
    assert isinstance(res[0]["value"]["rle"], list) and len(res[0]["value"]["rle"]) > 0


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


def test_import_rejects_overlong_project_title():
    # hosted LS 400s on titles > 50 chars; we refuse client-side, before any I/O
    import pytest

    from labeling_t.adapters.label_studio import import_to_label_studio

    with pytest.raises(ValueError, match="50"):
        import_to_label_studio([], base_url="http://x", api_key="k",
                               project_title="x" * 51, categories=["player"])


def test_keypoint_config_and_tasks_emit_point_regions():
    from labeling_t.adapters.label_studio import generate_label_config, to_label_studio_tasks
    from labeling_t.schema import Keypoint

    cfg = generate_label_config(["home", "away", "timer"], control="keypoint")
    assert "<KeyPointLabels" in cfg and '<Label value="home"/>' in cfg

    img = ImageLabels(
        image_path="s3://b/frames/all/f1.jpg", width=1280, height=720,
        detections=[Detection(
            bbox=BBox(x1=70, y1=570, x2=405, y2=640), category="scoreboard",
            keypoints=[Keypoint(x=265, y=622.5, name="home"),
                       Keypoint(x=91.5, y=581.5, name="timer")],
        ), Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="scoreboard")],  # no kps -> skipped
    )
    (task,) = to_label_studio_tasks([img], control="keypoint",
                                    presign=lambda uri: "https://signed/" + uri.rsplit("/", 1)[-1])
    results = task["predictions"][0]["result"]
    assert [r["type"] for r in results] == ["keypointlabels", "keypointlabels"]
    home, timer = results
    # percent of 1280x720, one region per point, label = the POINT name
    assert home["value"]["keypointlabels"] == ["home"]
    assert abs(home["value"]["x"] - 265 / 1280 * 100) < 1e-9
    assert abs(home["value"]["y"] - 622.5 / 720 * 100) < 1e-9
    assert timer["value"]["keypointlabels"] == ["timer"]
    assert task["data"]["image"] == "https://signed/f1.jpg"


def test_keypoint_tasks_with_no_keypoints_have_empty_predictions():
    from labeling_t.adapters.label_studio import to_label_studio_tasks

    img = ImageLabels(image_path="f.jpg", width=100, height=100,
                      detections=[Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="c")])
    (task,) = to_label_studio_tasks([img], control="keypoint")
    assert task["predictions"][0]["result"] == []  # label-from-scratch project
