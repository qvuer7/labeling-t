"""End-to-end through the real CLI, model mocked, LS skipped.

Exercises the offline-capable spine: prelabel (fake remote model) -> per-frame
label JSON -> COCO export, plus from-ls and ls-config. The two pieces that need
live servers (the vLLM endpoint, the LS server) are the only things stubbed.
"""

import json
from pathlib import Path

from PIL import Image

from labeling_t.cli import main


class FakeChatClient:
    """Drop-in for ChatClient: carries a spec, returns canned (normalized) boxes."""

    def __init__(self, endpoint, spec, *, api_key=None, categories=None, **kw):
        self.spec = spec

    @classmethod
    def from_env(cls, spec, *, categories=None, **kw):
        return cls("http://fake", spec, categories=categories, **kw)

    def infer(self, image_path):
        # qwen3_vl spec uses coord_space="norm1000"; on a 100x100 image
        # [100,200,600,800] -> abs [10,20,60,80]
        return '[{"bbox_2d": [100, 200, 600, 800], "label": "person", "score": 0.95}]'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _images(tmp_path, n=3, w=100, h=100):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        Image.new("RGB", (w, h), (0, 0, 0)).save(d / f"f{i}.jpg")
    return str(d)


def test_prelabel_then_to_coco(tmp_path, monkeypatch, capsys):
    # qwen3_vl is the vllm backend, so client_for builds a ChatClient.
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    images = _images(tmp_path, n=3)
    labels_dir = str(tmp_path / "labels")

    rc = main([
        "prelabel", "--images", images, "--out", labels_dir,
        "--model", "qwen3_vl",
        "--categories", "person,ball", "--category-map", _write_map(tmp_path),
    ])
    assert rc == 0
    assert len(list(Path(labels_dir).glob("*.json"))) == 3

    coco_path = str(tmp_path / "out.coco.json")
    rc = main(["to-coco", "--labels", labels_dir, "--out", coco_path,
               "--classes", "player,ball"])
    assert rc == 0

    coco = json.loads(Path(coco_path).read_text())
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    # person -> player via the category map; abs box becomes COCO xywh
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert cats[coco["annotations"][0]["category_id"]] == "player"
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 50.0, 60.0]


def _write_map(tmp_path) -> str:
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"person": "player"}))
    return str(p)


def test_from_ls_writes_label_files(tmp_path, capsys):
    export = [
        {
            "data": {"image": "frame7.jpg"},
            "annotations": [
                {"result": [
                    {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
                     "value": {"x": 5.0, "y": 10.0, "width": 25.0, "height": 50.0,
                               "rectanglelabels": ["player"]}}
                ]}
            ],
        }
    ]
    exp_path = tmp_path / "export.json"
    exp_path.write_text(json.dumps(export))
    out_dir = tmp_path / "verified"

    rc = main(["from-ls", "--export", str(exp_path), "--out", str(out_dir)])
    assert rc == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1
    img = json.loads(written[0].read_text())
    assert img["detections"][0]["category"] == "player"
    assert img["detections"][0]["bbox"]["x1"] == 10.0  # 5% of 200


def test_ls_config_prints_xml(capsys):
    rc = main(["ls-config", "--categories", "player,ball"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '<RectangleLabels' in out and 'value="player"' in out
