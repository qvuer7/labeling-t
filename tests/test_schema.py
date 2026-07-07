"""Schema validation: the contract must reject bad data loudly."""

import pytest
from pydantic import ValidationError

from labeling_t.schema import BBox, Detection, ImageLabels


def test_detection_accepts_optional_mask_rle():
    rle = {"size": [720, 1280], "counts": "abc123"}
    d = Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="player", mask=rle)
    assert d.mask == rle
    assert Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="ball").mask is None
    img = ImageLabels(image_path="f.jpg", width=20, height=20, detections=[d])
    assert img.detections[0].mask["size"] == [720, 1280]


def test_detection_accepts_optional_text():
    d = Detection(bbox=BBox(x1=0, y1=0, x2=10, y2=10), category="score_home", text="87")
    assert d.text == "87"
    # None = never transcribed; "" = attempted, nothing legible — both valid.
    assert Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="ball").text is None
    blank = Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="timer", text="")
    assert blank.text == ""
    # round-trips through JSON
    again = Detection.model_validate_json(d.model_dump_json())
    assert again.text == "87"


def test_schema_version_defaults_and_tolerates_absence():
    img = ImageLabels(image_path="f.jpg", width=10, height=10)
    assert img.schema_version == "2"  # current contract: keypoints exist
    assert '"schema_version":"2"' in img.model_dump_json()
    # pre-versioning on-disk JSON (no schema_version, no text) must still load;
    # the pydantic default only says what CURRENT code writes — on-disk truth
    # for provenance comes from the raw JSON (labelset.stats counts "absent")
    legacy = (
        '{"image_path": "f.jpg", "width": 10, "height": 10, "detections": '
        '[{"bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}, "category": "player"}]}'
    )
    loaded = ImageLabels.model_validate_json(legacy)
    assert loaded.detections[0].text is None
    assert loaded.detections[0].keypoints is None


def test_valid_detection_and_image():
    img = ImageLabels(
        image_path="frame_0001.jpg",
        width=1920,
        height=1080,
        detections=[
            Detection(
                bbox=BBox(x1=10, y1=20, x2=110, y2=220),
                category="player",
                score=0.91,
                source="locate-anything-3b",
            )
        ],
    )
    assert img.detections[0].bbox.width == 100
    assert img.detections[0].bbox.height == 200
    assert img.detections[0].score == 0.91


def test_bbox_rejects_inverted_corners():
    with pytest.raises(ValidationError):
        BBox(x1=100, y1=0, x2=10, y2=50)  # x2 < x1
    with pytest.raises(ValidationError):
        BBox(x1=0, y1=100, x2=50, y2=10)  # y2 < y1


def test_bbox_rejects_negative():
    with pytest.raises(ValidationError):
        BBox(x1=-1, y1=0, x2=10, y2=10)


def test_zero_size_box_is_allowed():
    # A degenerate point box is valid geometry; the model may emit one.
    b = BBox(x1=5, y1=5, x2=5, y2=5)
    assert b.width == 0 and b.height == 0


def test_score_must_be_in_unit_range():
    with pytest.raises(ValidationError):
        Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="ball", score=1.5)
    with pytest.raises(ValidationError):
        Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="ball", score=-0.1)


def test_score_optional_means_verified():
    d = Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="ball")
    assert d.score is None


def test_empty_category_rejected():
    with pytest.raises(ValidationError):
        Detection(bbox=BBox(x1=0, y1=0, x2=1, y2=1), category="")


def test_extra_fields_forbidden_keeps_schema_from_sprawling():
    # The field-discipline guard: unknown fields are a hard error, so the
    # contract can't quietly accumulate per-case junk.
    with pytest.raises(ValidationError):
        Detection(
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            category="player",
            track_id=7,  # not in v0 — must be rejected, not silently stored
        )


def test_image_dims_must_be_positive():
    with pytest.raises(ValidationError):
        ImageLabels(image_path="x.jpg", width=0, height=100)


def test_detection_outside_image_bounds_rejected():
    with pytest.raises(ValidationError):
        ImageLabels(
            image_path="x.jpg",
            width=100,
            height=100,
            detections=[Detection(bbox=BBox(x1=0, y1=0, x2=200, y2=50), category="car")],
        )


def test_detection_at_exact_image_edge_allowed():
    img = ImageLabels(
        image_path="x.jpg",
        width=100,
        height=100,
        detections=[Detection(bbox=BBox(x1=0, y1=0, x2=100, y2=100), category="car")],
    )
    assert img.detections[0].bbox.x2 == 100


# ---- keypoints (schema_version 2) ----------------------------------------------

def test_keypoints_ride_on_detection():
    from labeling_t.schema import Keypoint

    d = Detection(
        bbox=BBox(x1=10, y1=10, x2=60, y2=90), category="player",
        keypoints=[Keypoint(x=20, y=30, name="left_knee", visible=True, score=0.9),
                   Keypoint(x=25, y=15, name="nose")],
    )
    img = ImageLabels(image_path="f.jpg", width=100, height=100, detections=[d])
    again = ImageLabels.model_validate_json(img.model_dump_json())
    assert again.detections[0].keypoints[0].name == "left_knee"
    assert again.detections[0].keypoints[1].visible is None  # unknown, not False


def test_keypoints_resume_contract_none_vs_empty():
    d_untried = Detection(bbox=BBox(x1=0, y1=0, x2=5, y2=5), category="c")
    d_tried = Detection(bbox=BBox(x1=0, y1=0, x2=5, y2=5), category="c", keypoints=[])
    assert d_untried.keypoints is None and d_tried.keypoints == []


def test_keypoint_out_of_bounds_rejected():
    from labeling_t.schema import Keypoint

    d = Detection(bbox=BBox(x1=0, y1=0, x2=50, y2=50), category="c",
                  keypoints=[Keypoint(x=200, y=10, name="stray")])
    with pytest.raises(ValidationError, match="keypoint 'stray'"):
        ImageLabels(image_path="f.jpg", width=100, height=100, detections=[d])


def test_keypoint_rejects_unknown_fields_and_bad_values():
    from labeling_t.schema import Keypoint

    with pytest.raises(ValidationError):
        Keypoint(x=1, y=1, name="n", z=3)          # extra="forbid" holds
    with pytest.raises(ValidationError):
        Keypoint(x=1, y=1, name="")                # empty name
    with pytest.raises(ValidationError):
        Keypoint(x=1, y=1, name="n", score=1.5)    # score bound


def test_schema_version_2_written_old_files_still_load():
    img = ImageLabels(image_path="f.jpg", width=10, height=10)
    assert '"schema_version":"2"' in img.model_dump_json()
    # a v1 file (no keypoints anywhere) loads untouched
    v1 = '{"schema_version":"1","image_path":"f.jpg","width":10,"height":10,"detections":[]}'
    assert ImageLabels.model_validate_json(v1).schema_version == "1"
