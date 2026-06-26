"""Model transport: request shape, response extraction, retry policy."""

import base64
import json

import httpx
import pytest
from PIL import Image

from labeling_t.model_client import VLLMClient, _data_uri
from labeling_t.models import ModelSpec

SPEC = ModelSpec(
    key="t", name="locate-anything-3b", env_prefix="T", prompt="Detect: {categories}."
)


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


def _img(tmp_path, name="f.jpg"):
    p = tmp_path / name
    Image.new("RGB", (20, 10), (1, 2, 3)).save(p)
    return str(p)


def test_infer_returns_assistant_text(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return _chat_response('[{"bbox_2d":[0,0,1,1],"label":"player"}]')

    client = VLLMClient(
        "http://gpu:8000", SPEC,
        categories=["player"], transport=httpx.MockTransport(handler),
    )
    out = client.infer(_img(tmp_path))
    assert out == '[{"bbox_2d":[0,0,1,1],"label":"player"}]'
    assert captured["url"].endswith("/v1/chat/completions")
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["type"] == "text" and "player" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_data_uri_rejects_unknown_type(tmp_path):
    p = tmp_path / "f.tiff"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported image type"):
        _data_uri(str(p))


def test_retries_5xx_then_succeeds(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return _chat_response("[]")

    client = VLLMClient(
        "http://gpu:8000", SPEC, max_retries=2, transport=httpx.MockTransport(handler)
    )
    assert client.infer(_img(tmp_path)) == "[]"
    assert calls["n"] == 2


def test_does_not_retry_4xx(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = VLLMClient(
        "http://gpu:8000", SPEC, max_retries=3, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.infer(_img(tmp_path))
    assert calls["n"] == 1  # no retries on a 4xx


def test_api_key_sets_auth_header(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _chat_response("[]")

    client = VLLMClient(
        "http://gpu:8000", SPEC, api_key="secret", transport=httpx.MockTransport(handler)
    )
    client.infer(_img(tmp_path))
    assert seen["auth"] == "Bearer secret"
