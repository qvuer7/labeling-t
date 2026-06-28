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
