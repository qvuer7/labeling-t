"""PR-0 seam spike: structured /infer flows through the widened seam into the
neutral schema, with NO GPU and NO change to the vLLM path."""

import json
import sys

import httpx
from fastapi.testclient import TestClient

from labeling_t.model_client import TransformersClient
from labeling_t.models import OWLV2
from labeling_t.prelabel import RawInference, _raw_inference, prelabel_cloud
from labeling_t.schema import ImageLabels
from labeling_t.server.app import create_app

SPEC = OWLV2  # backend="transformers", coord_space="abs"


def _infer_response(detections, w=200, h=100):
    return httpx.Response(200, json={"width": w, "height": h, "detections": detections})


def test_infer_raw_parses_structured_to_rawinference():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url).endswith("/infer")
        assert body["queries"] == ["player", "ball"]
        assert "image_url" in body and "params" in body
        return _infer_response([{"bbox": [1, 2, 3, 4], "label": "player", "score": 0.9}])

    client = TransformersClient(
        "http://srv:8000", SPEC, categories=["player", "ball"],
        transport=httpx.MockTransport(handler),
    )
    raw = client.infer_raw("https://s/frame.jpg?sig=x")
    assert isinstance(raw, RawInference)
    assert raw.boxes == [([1, 2, 3, 4], "player", 0.9)]
    assert (raw.width, raw.height) == (200, 100)


def test_raw_inference_prefers_structured_backend():
    client = TransformersClient(
        "http://s:8000", SPEC,
        transport=httpx.MockTransport(lambda r: _infer_response([], w=5, h=5)),
    )
    raw = _raw_inference(client, "https://s/f.jpg?x")
    assert raw.width == 5 and raw.boxes == []


class _FakeStorage:
    """presigned_url + list + write_text; image_size MUST NOT be called for a
    structured backend (the server returns dims) — calling it fails the test."""

    def __init__(self):
        self.written: dict[str, str] = {}

    def list(self, prefix):
        return []

    def presigned_url(self, uri, ttl=3600):
        return f"https://signed/{uri}?sig=1"

    def image_size(self, uri):
        raise AssertionError("image_size must not be called for a structured backend")

    def write_text(self, uri, text):
        self.written[uri] = text


def test_prelabel_cloud_structured_end_to_end():
    # the spike's core claim: structured /infer -> ImageLabels in storage, dims
    # from the server (no ranged read), boxes correct, source/category set.
    def handler(request: httpx.Request) -> httpx.Response:
        return _infer_response(
            [{"bbox": [10, 20, 110, 90], "label": "player", "score": 0.8}], w=200, h=100
        )

    client = TransformersClient(
        "http://s:8000", SPEC, categories=["player"],
        transport=httpx.MockTransport(handler),
    )
    storage = _FakeStorage()
    n = prelabel_cloud(
        ["s3://b/d/frames/all/g_000_1.jpg"], client, "s3://b/d/labels/all", storage=storage,
    )
    assert n == 1
    [text] = storage.written.values()
    labels = ImageLabels.model_validate_json(text)
    assert (labels.width, labels.height) == (200, 100)
    d = labels.detections[0]
    assert (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) == (10, 20, 110, 90)
    assert d.category == "player" and d.source == "owlv2"


def test_stub_server_contract():
    c = TestClient(create_app())
    health = c.get("/health").json()
    assert health["status"] == "ok" and health["ready"] is True
    r = c.post("/infer", json={"image_url": "http://x/y.jpg", "queries": ["player", "ball"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"width", "height", "detections"}
    assert len(body["detections"]) == 2
    assert body["detections"][0]["bbox"] == [10.0, 10.0, 110.0, 110.0]


def test_client_path_is_torch_free():
    # CQ3: the thin client must never pull torch (it ships in the same wheel but
    # only the server entrypoint imports it).
    import labeling_t  # noqa: F401
    import labeling_t.cli  # noqa: F401
    import labeling_t.model_client  # noqa: F401

    assert "torch" not in sys.modules
