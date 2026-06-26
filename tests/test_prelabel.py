"""Pre-label orchestration + parser: parsing, coords, filtering, resilience, resume.

coord_space / parse / model name now come from the ModelSpec the client carries;
category_map / min_score / strict_categories stay run-level args.
"""

import json

import pytest
from PIL import Image

from labeling_t.models import ModelSpec
from labeling_t.prelabel import parse_boxes, prelabel


# --- parse_boxes (T3b) --------------------------------------------------------

def test_parse_plain_json():
    out = parse_boxes('[{"bbox_2d":[1,2,3,4],"label":"player","score":0.9}]')
    assert out == [([1.0, 2.0, 3.0, 4.0], "player", 0.9)]


def test_parse_strips_markdown_fence():
    out = parse_boxes('```json\n[{"bbox":[0,0,1,1],"label":"ball"}]\n```')
    assert out == [([0.0, 0.0, 1.0, 1.0], "ball", None)]


def test_parse_rejects_non_list():
    with pytest.raises(ValueError, match="expected a JSON list"):
        parse_boxes('{"bbox_2d":[0,0,1,1],"label":"x"}')


def test_parse_rejects_missing_label():
    with pytest.raises(ValueError, match="missing box/label"):
        parse_boxes('[{"bbox_2d":[0,0,1,1]}]')


def test_parse_rejects_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_boxes("not json at all")


def test_parse_salvages_truncated_looping_output():
    # Real Qwen3-VL failure mode: it loops the same box and the JSON gets cut
    # off at the token cap. Salvage the valid objects + dedup the repeats.
    truncated = (
        '[{"bbox_2d": [657, 304, 728, 439], "label": "player"}, '
        '{"bbox_2d": [325, 231, 352, 437], "label": "player"}, '
        '{"bbox_2d": [325, 231, 352, 437], "label": "player"}, '
        '{"bbox_2d": [325, 231, 352, 437], "label": "play'  # truncated mid-string
    )
    out = parse_boxes(truncated)
    assert out == [
        ([657.0, 304.0, 728.0, 439.0], "player", None),
        ([325.0, 231.0, 352.0, 437.0], "player", None),  # the 3 repeats -> 1
    ]


def test_parse_dedups_repeats_in_clean_json():
    out = parse_boxes(
        '[{"bbox_2d":[1,2,3,4],"label":"a"},{"bbox_2d":[1,2,3,4],"label":"a"}]'
    )
    assert len(out) == 1


# --- orchestration (T3a) ------------------------------------------------------

def _spec(coord_space="abs"):
    return ModelSpec(
        key="t", name="test-model", env_prefix="T",
        prompt="{categories}", coord_space=coord_space,
    )


class FakeClient:
    """Transport double: carries a spec (coord_space/parse/name) + canned text."""

    def __init__(self, responses=None, default="[]", spec=None):
        self.responses = responses or {}
        self.default = default
        self.spec = spec or _spec("abs")
        self.calls: list[str] = []

    def infer(self, image_path):
        self.calls.append(str(image_path))
        return self.responses.get(str(image_path), self.default)


def _imgs(tmp_path, n, w=100, h=100):
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (w, h), (0, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def test_norm_coords_become_abs_pixels(tmp_path):
    [p] = _imgs(tmp_path, 1, w=200, h=100)
    client = FakeClient({p: '[{"bbox_2d":[0.0,0.0,0.5,0.5],"label":"car"}]'}, spec=_spec("norm"))
    out = prelabel([p], client, tmp_path / "out")
    box = out[0].detections[0].bbox
    assert (box.x1, box.y1, box.x2, box.y2) == (0.0, 0.0, 100.0, 50.0)


def test_out_of_bounds_box_is_clamped_not_dropped(tmp_path):
    [p] = _imgs(tmp_path, 1, w=100, h=100)
    client = FakeClient({p: '[{"bbox_2d":[-10,-10,150,150],"label":"car"}]'})  # abs spec
    out = prelabel([p], client, tmp_path / "out")
    box = out[0].detections[0].bbox
    assert (box.x1, box.y1, box.x2, box.y2) == (0.0, 0.0, 100.0, 100.0)


def test_min_score_filters_low_confidence(tmp_path):
    [p] = _imgs(tmp_path, 1)
    client = FakeClient({p: '[{"bbox_2d":[0,0,10,10],"label":"a","score":0.3},'
                            '{"bbox_2d":[0,0,10,10],"label":"b","score":0.9}]'})
    out = prelabel([p], client, tmp_path / "out", min_score=0.5)
    cats = [d.category for d in out[0].detections]
    assert cats == ["b"]


def test_category_map_drops_unmapped(tmp_path):
    [p] = _imgs(tmp_path, 1)
    client = FakeClient({p: '[{"bbox_2d":[0,0,5,5],"label":"person"},'
                            '{"bbox_2d":[0,0,5,5],"label":"noise"}]'})
    out = prelabel([p], client, tmp_path / "out", category_map={"person": "player"})
    cats = [d.category for d in out[0].detections]
    assert cats == ["player"]


def test_strict_categories_routes_to_failure_manifest(tmp_path):
    [p] = _imgs(tmp_path, 1)
    client = FakeClient({p: '[{"bbox_2d":[0,0,5,5],"label":"ufo"}]'})
    out = prelabel(
        [p], client, tmp_path / "out",
        category_map={"person": "player"}, strict_categories=True,
    )
    assert out == []
    manifest = (tmp_path / "out" / "failures.jsonl").read_text()
    assert "unmapped category" in manifest and p in manifest


def test_bad_frame_goes_to_manifest_and_run_continues(tmp_path):
    p_good, p_bad = _imgs(tmp_path, 2)
    client = FakeClient(
        {p_good: '[{"bbox_2d":[0,0,5,5],"label":"car"}]', p_bad: "garbage"}
    )
    out = prelabel([p_good, p_bad], client, tmp_path / "out")
    assert len(out) == 1 and out[0].image_path == p_good
    manifest = (tmp_path / "out" / "failures.jsonl").read_text()
    assert p_bad in manifest


def test_resume_skips_already_written_frames(tmp_path):
    [p] = _imgs(tmp_path, 1)
    client = FakeClient({p: '[{"bbox_2d":[0,0,5,5],"label":"car"}]'})
    prelabel([p], client, tmp_path / "out")
    assert client.calls == [p]
    client.calls.clear()
    out = prelabel([p], client, tmp_path / "out")
    assert client.calls == []
    assert out[0].detections[0].category == "car"


def test_source_is_the_model_name(tmp_path):
    [p] = _imgs(tmp_path, 1)
    client = FakeClient({p: '[{"bbox_2d":[0,0,5,5],"label":"car"}]'})
    out = prelabel([p], client, tmp_path / "out")
    assert out[0].detections[0].source == "test-model"
