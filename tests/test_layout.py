"""Dataset layout: the one definition of the storage folder structure."""

from labeling_t.layout import DatasetLayout


def test_dataset_grouped_paths():
    lo = DatasetLayout(dataset="ipbl-basketball", base="s3://ml-cv-data")
    assert lo.root == "s3://ml-cv-data/datasets/ipbl-basketball"
    assert lo.frames("cats_vs_wolves_02") == "s3://ml-cv-data/datasets/ipbl-basketball/frames/cats_vs_wolves_02"
    assert lo.labels("cats_vs_wolves_02") == "s3://ml-cv-data/datasets/ipbl-basketball/labels/cats_vs_wolves_02"
    # a named namespace keeps a second model's pre-labels apart from the default
    assert lo.labels("g", "locateanything") == "s3://ml-cv-data/datasets/ipbl-basketball/labels-locateanything/g"
    assert lo.labels("g", "") == lo.labels("g")  # empty name == default labels/
    assert lo.verified("g") == "s3://ml-cv-data/datasets/ipbl-basketball/verified/g"
    assert lo.export("v1") == "s3://ml-cv-data/datasets/ipbl-basketball/export/v1"


def test_no_game_gives_stage_prefix():
    lo = DatasetLayout(dataset="d", base="s3://b")
    assert lo.frames() == "s3://b/datasets/d/frames"


def test_from_env_uses_s3_bucket(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "ml-cv-data")
    assert DatasetLayout.from_env("d").base == "s3://ml-cv-data"


def test_from_env_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert DatasetLayout.from_env("d").base == "data"


def test_base_trailing_slash_stripped():
    assert DatasetLayout.from_env("d", base="s3://b/").root == "s3://b/datasets/d"
