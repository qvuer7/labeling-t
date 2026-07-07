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
    def from_env(cls, spec, *, endpoint=None, categories=None, **kw):
        return cls(endpoint or "http://fake", spec, categories=categories, **kw)

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


class FakeOCRChatClient:
    """Drop-in for ChatClient on the transcribe path: returns a fixed reading."""

    def __init__(self, endpoint, spec, *, api_key=None, categories=None, **kw):
        self.spec = spec

    @classmethod
    def from_env(cls, spec, *, endpoint=None, categories=None, **kw):
        return cls(endpoint or "http://fake", spec, categories=categories, **kw)

    def infer(self, image):
        assert isinstance(image, bytes)  # transcribe sends in-memory PNG crops
        return "42"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_prelabel_then_transcribe_then_to_coco(tmp_path, monkeypatch):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    images = _images(tmp_path, n=2)
    labels_dir = str(tmp_path / "labels")
    rc = main(["prelabel", "--images", images, "--out", labels_dir,
               "--model", "qwen3_vl",
               "--categories", "person", "--category-map", _write_map(tmp_path)])
    assert rc == 0

    # openai_ocr is the openai backend -> client_for builds a ChatClient
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeOCRChatClient)
    rc = main(["transcribe", "--labels", labels_dir, "--categories", "player",
               "--model", "openai_ocr"])
    assert rc == 0

    enriched = json.loads(next(iter(sorted(Path(labels_dir).glob("*.json")))).read_text())
    assert enriched["schema_version"] == "1"
    assert enriched["detections"][0]["text"] == "42"

    # the enriched labels still export
    coco_path = str(tmp_path / "out.coco.json")
    rc = main(["to-coco", "--labels", labels_dir, "--out", coco_path,
               "--classes", "player"])
    assert rc == 0
    assert len(json.loads(Path(coco_path).read_text())["images"]) == 2


def test_transcribe_rejects_transformers_backend(tmp_path, capsys):
    (tmp_path / "labels").mkdir()
    rc = main(["transcribe", "--labels", str(tmp_path / "labels"),
               "--categories", "timer", "--model", "sam2"])
    assert rc == 1
    assert "chat backend" in capsys.readouterr().err


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


# ---- --json envelope (agent output mode) ----------------------------------

def test_json_envelope_success_and_prose_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    images = _images(tmp_path, n=2)
    labels_dir = str(tmp_path / "labels")

    rc = main(["prelabel", "--images", images, "--out", labels_dir,
               "--model", "qwen3_vl", "--categories", "person",
               "--category-map", _write_map(tmp_path), "--json"])
    out, err = capsys.readouterr()
    assert rc == 0
    envelope = json.loads(out)  # stdout is EXACTLY one JSON envelope
    assert envelope["ok"] is True
    result = envelope["result"]
    assert result["labeled"] == 2 and result["total"] == 2 and result["out"] == labels_dir
    assert result["failures_file"].endswith("failures.jsonl")
    assert "labeled 2/2" in err  # prose demoted to stderr

    coco_path = str(tmp_path / "out.coco.json")
    rc = main(["to-coco", "--labels", labels_dir, "--out", coco_path,
               "--classes", "player", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    assert envelope["result"] == {"images": 2, "annotations": 2, "out": coco_path}


def test_json_envelope_error(tmp_path, capsys):
    (tmp_path / "labels").mkdir()
    rc = main(["transcribe", "--labels", str(tmp_path / "labels"),
               "--categories", "timer", "--model", "sam2", "--json"])
    out, err = capsys.readouterr()
    assert rc == 1
    envelope = json.loads(out)
    assert envelope["ok"] is False  # ok mirrors the exit code
    assert "chat backend" in envelope["error"]["message"]
    assert "chat backend" in err  # message still human-visible


def test_ls_config_json_wraps_xml(capsys):
    rc = main(["ls-config", "--categories", "player,ball", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    assert "<RectangleLabels" in envelope["result"]["xml"]


# ---- prelabel-cloud e2e vehicle (local base + LocalStorage + FakeChatClient) ---

def _seed_cloud_frames(tmp_path, n=4):
    base = str(tmp_path / "cloud")
    d = Path(base) / "datasets/d/frames/g"
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (100, 100), (0, 0, 0)).save(d / f"f{i}.jpg")
    return base


def test_prelabel_cloud_e2e_local_base(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    base = _seed_cloud_frames(tmp_path, n=3)
    rc = main(["prelabel-cloud", "--dataset", "d", "--group", "g", "--base", base,
               "--model", "qwen3_vl", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["labeled"] == 3
    assert "requested" not in envelope["result"]  # no subset asked for
    labels = Path(base) / "datasets/d/labels/g"
    assert len(list(labels.glob("*.json"))) == 3


def test_prelabel_cloud_stems_subset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    base = _seed_cloud_frames(tmp_path, n=4)
    rc = main(["prelabel-cloud", "--dataset", "d", "--group", "g", "--base", base,
               "--model", "qwen3_vl", "--stems", "f0,f2,ghost", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0
    # requested = size of the stem filter; matched = frames actually present
    assert envelope["result"]["requested"] == 3
    assert envelope["result"]["matched"] == 2
    assert envelope["result"]["labeled"] == 2
    stems = {p.stem for p in (Path(base) / "datasets/d/labels/g").glob("*.json")}
    assert stems == {"f0", "f2"}


def test_prelabel_cloud_stems_intersect_frames_from(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    base = _seed_cloud_frames(tmp_path, n=4)
    # a previous run's label set provides the --frames-from universe
    ref = Path(base) / "datasets/d/labels-ref/g"
    ref.mkdir(parents=True)
    for stem in ("f1", "f2"):
        ref.joinpath(f"{stem}.json").write_text('{"image_path":"x","width":1,"height":1,"detections":[]}')
    rc = main(["prelabel-cloud", "--dataset", "d", "--group", "g", "--base", base,
               "--model", "qwen3_vl", "--frames-from", "labels-ref", "--stems", "f2,f3",
               "--labels-name", "out", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    # intersection of {f1,f2} and {f2,f3} = {f2}
    assert rc == 0 and envelope["result"]["labeled"] == 1
    assert envelope["result"]["requested"] == 1 and envelope["result"]["matched"] == 1
    assert [p.stem for p in (Path(base) / "datasets/d/labels-out/g").glob("*.json")] == ["f2"]


def test_prelabel_cloud_empty_subset_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    base = _seed_cloud_frames(tmp_path, n=2)
    rc = main(["prelabel-cloud", "--dataset", "d", "--group", "g", "--base", base,
               "--model", "qwen3_vl", "--stems", "nope", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    assert "no frames match" in envelope["error"]["message"]


def test_prelabel_cloud_progress_on_stderr_envelope_alone_on_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("labeling_t.model_client.ChatClient", FakeChatClient)
    base = _seed_cloud_frames(tmp_path, n=3)
    rc = main(["prelabel-cloud", "--dataset", "d", "--group", "g", "--base", base,
               "--model", "qwen3_vl", "--json"])
    out, err = capsys.readouterr()
    assert rc == 0
    json.loads(out)  # stdout: EXACTLY one JSON envelope, nothing else
    assert len(out.strip().splitlines()) == 1
    events = [json.loads(ln) for ln in err.strip().splitlines()
              if ln.startswith("{") and '"event"' in ln]
    assert events and events[0]["stage"] == "prelabel-cloud"
    assert events[-1]["done"] == events[-1]["total"] == 3  # final item always reported
