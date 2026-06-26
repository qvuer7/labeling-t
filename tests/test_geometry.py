"""Geometry conversions — the bug-prone zone. Round-trips + edges must hold.

A coordinate bug here silently corrupts the training set (boxes land in the
wrong place after a verify-and-export cycle), so these tests carry weight.
"""

import math

import pytest

from labeling_t.geometry import (
    abs_to_coco_xywh,
    abs_to_normalized,
    abs_to_percent,
    normalized_to_abs,
    percent_to_abs,
)
from labeling_t.schema import BBox

W, H = 1920, 1080
EPS = 1e-6


def approx(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-4)


# --- normalized <-> abs --------------------------------------------------------

def test_normalized_to_abs_basic_unit_scale():
    b = normalized_to_abs(0.0, 0.0, 0.5, 0.5, W, H)
    assert approx(b.x1, 0) and approx(b.y1, 0)
    assert approx(b.x2, 960) and approx(b.y2, 540)


def test_normalized_to_abs_qwen_style_1000_scale():
    b = normalized_to_abs(0, 0, 500, 500, W, H, scale=1000.0)
    assert approx(b.x2, 960) and approx(b.y2, 540)


@pytest.mark.parametrize(
    "box",
    [
        BBox(x1=0, y1=0, x2=0, y2=0),            # corner / zero-size
        BBox(x1=0, y1=0, x2=W, y2=H),            # full frame
        BBox(x1=123.4, y1=987.6, x2=1000, y2=1001),  # arbitrary, non-square
        BBox(x1=W, y1=H, x2=W, y2=H),            # bottom-right point
    ],
)
def test_abs_norm_roundtrip_is_identity(box):
    nx1, ny1, nx2, ny2 = abs_to_normalized(box, W, H)
    back = normalized_to_abs(nx1, ny1, nx2, ny2, W, H)
    assert approx(back.x1, box.x1) and approx(back.y1, box.y1)
    assert approx(back.x2, box.x2) and approx(back.y2, box.y2)


# --- abs <-> Label Studio percent ---------------------------------------------

def test_abs_to_percent_known_values_non_square():
    # half-width, quarter-height box on a 1920x1080 frame
    box = BBox(x1=0, y1=0, x2=960, y2=270)
    p = abs_to_percent(box, W, H)
    assert approx(p["x"], 0.0) and approx(p["y"], 0.0)
    assert approx(p["width"], 50.0)
    assert approx(p["height"], 25.0)


def test_abs_to_percent_within_0_100():
    box = BBox(x1=0, y1=0, x2=W, y2=H)
    p = abs_to_percent(box, W, H)
    for v in p.values():
        assert 0.0 <= v <= 100.0


def test_percent_roundtrip_is_identity():
    box = BBox(x1=100, y1=200, x2=900, y2=700)
    p = abs_to_percent(box, W, H)
    back = percent_to_abs(p["x"], p["y"], p["width"], p["height"], W, H)
    assert approx(back.x1, box.x1) and approx(back.y1, box.y1)
    assert approx(back.x2, box.x2) and approx(back.y2, box.y2)


# --- abs -> COCO xywh ----------------------------------------------------------

def test_abs_to_coco_xywh():
    box = BBox(x1=10, y1=20, x2=110, y2=220)
    assert abs_to_coco_xywh(box) == [10, 20, 100, 200]


def test_abs_to_coco_xywh_zero_size():
    box = BBox(x1=5, y1=5, x2=5, y2=5)
    assert abs_to_coco_xywh(box) == [5, 5, 0, 0]


# --- invalid dimensions raise, not silently divide by zero --------------------

@pytest.mark.parametrize("fn", [
    lambda: normalized_to_abs(0, 0, 1, 1, 0, 100),
    lambda: abs_to_percent(BBox(x1=0, y1=0, x2=1, y2=1), 0, 100),
    lambda: percent_to_abs(0, 0, 10, 10, 100, 0),
    lambda: abs_to_normalized(BBox(x1=0, y1=0, x2=1, y2=1), 0, 100),
])
def test_zero_dims_raise(fn):
    with pytest.raises(ValueError):
        fn()


def test_normalized_to_abs_rejects_bad_scale():
    with pytest.raises(ValueError):
        normalized_to_abs(0, 0, 1, 1, W, H, scale=0)
