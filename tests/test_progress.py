"""The on_progress hook on the batch stages: fires once per item, ends at total,
and is fully optional (default None reproduces the prior silent behavior)."""

from pathlib import Path

from PIL import Image

from labeling_t.ingest import ingest_images
from labeling_t.models import ModelSpec
from labeling_t.prelabel import prelabel, prelabel_cloud
from labeling_t.storage import LocalStorage


def _spec(coord_space="abs"):
    return ModelSpec(key="t", name="test-model", env_prefix="T",
                     prompt="{categories}", coord_space=coord_space)


class FakeClient:
    def __init__(self, default='[{"bbox_2d":[0,0,5,5],"label":"car"}]'):
        self.default = default
        self.spec = _spec("abs")

    def infer(self, image_path):
        return self.default


def _imgs(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (60, 40), (0, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def _seed_src(src: Path, n: int) -> None:
    src.mkdir()
    for i in range(n):
        Image.new("RGB", (10, 10), (0, 0, 0)).save(src / f"f{i}.jpg")


def _assert_progress(calls, total):
    """done values are 1..total (in some order, one per item), total constant."""
    assert all(t == total for _, t in calls)
    assert sorted(d for d, _ in calls) == list(range(1, total + 1))


def test_ingest_on_progress_fires_per_image(tmp_path):
    src = tmp_path / "src"
    _seed_src(src, 4)
    calls = []
    uploaded, total = ingest_images(
        str(src), str(tmp_path / "dest"), storage=LocalStorage(),
        on_progress=lambda d, t: calls.append((d, t)),
    )
    assert (uploaded, total) == (4, 4)
    _assert_progress(calls, 4)


def test_prelabel_on_progress_fires_per_frame(tmp_path):
    paths = _imgs(tmp_path, 3)
    calls = []
    out = prelabel(paths, FakeClient(), tmp_path / "out",
                   on_progress=lambda d, t: calls.append((d, t)))
    assert len(out) == 3
    _assert_progress(calls, 3)


def test_prelabel_cloud_on_progress_fires_per_frame(tmp_path):
    uris = _imgs(tmp_path, 3)
    calls = []
    n = prelabel_cloud(uris, FakeClient(), str(tmp_path / "labels"),
                       storage=LocalStorage(), on_progress=lambda d, t: calls.append((d, t)))
    assert n == 3
    _assert_progress(calls, 3)


def test_default_none_is_silent_and_unchanged(tmp_path):
    # No callback => no error, same return contract as before the hook existed.
    paths = _imgs(tmp_path, 2)
    out = prelabel(paths, FakeClient(), tmp_path / "out")
    assert len(out) == 2


# ---- progress_reporter: throttle + event shape + --progress-file ---------------

import argparse
import json as _json

from labeling_t import output as _output
from labeling_t.output import progress_reporter


def _ns(**kw):
    kw.setdefault("json", True)
    kw.setdefault("progress_file", None)
    return argparse.Namespace(**kw)


def test_reporter_event_shape_and_stderr(capsys):
    report = progress_reporter(_ns(), "prelabel-cloud")
    report(1, 100)
    out, err = capsys.readouterr()
    assert out == ""  # stdout belongs to the envelope
    event = _json.loads(err)
    assert event == {"event": "progress", "stage": "prelabel-cloud",
                     "done": 1, "total": 100, "elapsed_s": event["elapsed_s"]}


def test_reporter_throttles_first_every25_and_final(capsys, monkeypatch):
    monkeypatch.setattr(_output.time, "monotonic", lambda: 1000.0)  # clock frozen
    report = progress_reporter(_ns(), "s")
    for done in range(1, 61):
        report(done, 60)
    lines = capsys.readouterr().err.strip().splitlines()
    assert [_json.loads(ln)["done"] for ln in lines] == [1, 26, 51, 60]


def test_reporter_time_based_emission_when_items_are_slow(capsys, monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(_output.time, "monotonic", lambda: t["now"])
    report = progress_reporter(_ns(), "s")
    report(1, 100)          # first: always
    t["now"] += 2.0
    report(2, 100)          # 2s since last, 1 item -> suppressed
    t["now"] += 4.0
    report(3, 100)          # 6s since last -> emitted
    lines = capsys.readouterr().err.strip().splitlines()
    events = [_json.loads(ln) for ln in lines]
    assert [e["done"] for e in events] == [1, 3]
    assert events[1]["elapsed_s"] == 6.0


def test_reporter_progress_file_holds_latest_event(tmp_path, capsys):
    path = tmp_path / "progress.json"
    report = progress_reporter(_ns(progress_file=str(path)), "ocr")
    report(1, 2)
    report(2, 2)
    capsys.readouterr()
    event = _json.loads(path.read_text())
    assert event["done"] == 2 and event["total"] == 2 and event["stage"] == "ocr"
