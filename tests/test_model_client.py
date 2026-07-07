"""Model transport: request shape, response extraction, retry policy."""

import base64
import json

import httpx
import pytest
from PIL import Image

from labeling_t.model_client import ChatClient, TransformersClient, client_for, _data_uri
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

    client = ChatClient(
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

    client = ChatClient(
        "http://gpu:8000", SPEC, max_retries=2, transport=httpx.MockTransport(handler)
    )
    assert client.infer(_img(tmp_path)) == "[]"
    assert calls["n"] == 2


def test_infer_accepts_raw_bytes_as_png_data_uri():
    # transcribe.py sends in-memory crops as bytes -> base64 PNG data URI
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("87")

    client = ChatClient("http://gpu:8000", SPEC, transport=httpx.MockTransport(handler))
    png = b"\x89PNG\r\n\x1a\nfakepixels"
    assert client.infer(png) == "87"
    url = captured["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def test_retries_429_rate_limit_then_succeeds(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return _chat_response("[]")

    client = ChatClient(
        "http://gpu:8000", SPEC, max_retries=2, transport=httpx.MockTransport(handler)
    )
    assert client.infer(_img(tmp_path)) == "[]"
    assert calls["n"] == 2


def test_429_honors_retry_after_header(tmp_path, monkeypatch):
    calls = {"n": 0}
    slept = []
    monkeypatch.setattr("labeling_t.model_client.time.sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited", headers={"retry-after": "7"})
        return _chat_response("[]")

    client = ChatClient(
        "http://gpu:8000", SPEC, max_retries=2, transport=httpx.MockTransport(handler)
    )
    assert client.infer(_img(tmp_path)) == "[]"
    assert slept == [7.0]  # server-directed wait, not the sub-second default


def test_429_without_header_backs_off_long(tmp_path, monkeypatch):
    calls = {"n": 0}
    slept = []
    monkeypatch.setattr("labeling_t.model_client.time.sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return _chat_response("[]")

    client = ChatClient(
        "http://gpu:8000", SPEC, max_retries=2, transport=httpx.MockTransport(handler)
    )
    assert client.infer(_img(tmp_path)) == "[]"
    assert slept and slept[0] >= 10  # a rate-limit window, not a 0.5s blip


def test_image_detail_rides_in_image_url(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("87")

    low_spec = ModelSpec(key="t2", name="m", env_prefix="T", prompt="Read.",
                         image_detail="low")
    client = ChatClient("http://x", low_spec, transport=httpx.MockTransport(handler))
    client.infer(b"\x89PNGxx")
    img = captured["body"]["messages"][0]["content"][1]["image_url"]
    assert img["detail"] == "low"

    # default spec (no image_detail): the key is absent entirely
    client2 = ChatClient("http://x", SPEC, transport=httpx.MockTransport(handler))
    client2.infer(b"\x89PNGxx")
    img2 = captured["body"]["messages"][0]["content"][1]["image_url"]
    assert "detail" not in img2


def test_does_not_retry_4xx(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = ChatClient(
        "http://gpu:8000", SPEC, max_retries=3, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.infer(_img(tmp_path))
    assert calls["n"] == 1  # no retries on a 4xx


def test_infer_with_url_passes_url_not_base64():
    # presigned/http image -> sent as a URL so the GPU fetches it (no base64)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("[]")

    client = ChatClient("http://gpu:8000", SPEC, transport=httpx.MockTransport(handler))
    client.infer("https://spaces.example/presigned/frame_00001.jpg?sig=abc")
    img = captured["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert img == "https://spaces.example/presigned/frame_00001.jpg?sig=abc"


def test_image_url_helper(tmp_path):
    from labeling_t.model_client import _image_url
    assert _image_url("http://x/y.jpg") == "http://x/y.jpg"
    assert _image_url(_img(tmp_path)).startswith("data:image/jpeg;base64,")


def test_api_key_sets_auth_header(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _chat_response("[]")

    client = ChatClient(
        "http://gpu:8000", SPEC, api_key="secret", transport=httpx.MockTransport(handler)
    )
    client.infer(_img(tmp_path))
    assert seen["auth"] == "Bearer secret"


def test_extra_body_merged_into_payload(tmp_path):
    # vLLM's repetition_penalty (and any provider knob) rides on the spec, not the
    # client — so a hosted API spec can omit it and avoid a 400.
    spec = ModelSpec(
        key="t", name="m", env_prefix="T", prompt="Detect: {categories}.",
        extra_body={"repetition_penalty": 1.1},
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("[]")

    ChatClient("http://gpu:8000", spec, transport=httpx.MockTransport(handler)).infer(_img(tmp_path))
    assert captured["body"]["repetition_penalty"] == 1.1
    # the base spec (no extra_body) sends none — what a hosted vendor needs.
    captured.clear()
    ChatClient("http://gpu:8000", SPEC, transport=httpx.MockTransport(handler)).infer(_img(tmp_path))
    assert "repetition_penalty" not in captured["body"]


def test_chat_path_from_spec(tmp_path):
    # Gemini's OpenAI-compat layer lives at a different route than vLLM/OpenAI.
    spec = ModelSpec(
        key="t", name="m", env_prefix="T", prompt="Detect: {categories}.",
        chat_path="/chat/completions",
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return _chat_response("[]")

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    ChatClient(base, spec, transport=httpx.MockTransport(handler)).infer(_img(tmp_path))
    assert seen["url"] == base + "/chat/completions"


def test_from_env_falls_back_to_spec_default_endpoint(monkeypatch):
    # A SaaS provider bakes its URL into the spec; only the key need be in .env.
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = ModelSpec(
        key="t", name="gpt-4o", env_prefix="OPENAI", prompt="Detect: {categories}.",
        default_endpoint="https://api.openai.com/v1",
    )
    assert spec.endpoint_from_env() == "https://api.openai.com/v1"
    client = ChatClient.from_env(spec, transport=httpx.MockTransport(lambda r: _chat_response("[]")))
    assert str(client._http.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_client_for_routes_by_backend(monkeypatch):
    monkeypatch.setenv("T_ENDPOINT", "http://x:8000")
    chat = ModelSpec(key="t", name="m", env_prefix="T", prompt="p", backend="openai")
    structured = ModelSpec(key="t", name="m", env_prefix="T", prompt="p", backend="transformers")
    assert isinstance(client_for(chat), ChatClient)
    assert isinstance(client_for(structured), TransformersClient)


def test_client_for_explicit_endpoint_beats_everything(monkeypatch):
    # --endpoint on the CLI must win over env and recorded pods alike.
    monkeypatch.setenv("T_ENDPOINT", "http://from-env:8000")
    spec = ModelSpec(key="t", name="m", env_prefix="T", prompt="p", backend="transformers")
    client = client_for(spec, endpoint="http://explicit:9000/")
    assert str(client._http.base_url).rstrip("/") == "http://explicit:9000"


def test_from_env_prefers_recorded_pod_over_env(monkeypatch):
    from labeling_t import podstate

    monkeypatch.setenv("T_ENDPOINT", "http://from-env:8000")
    podstate.record_pod({"id": "p1", "model": "t", "env_prefix": "T",
                         "endpoint": "http://pod:8000", "gpu": None, "cost_per_hr": 0.4,
                         "created_at": "2026-07-07T00:00:00Z",
                         "terminate_after": "2099-01-01T00:00:00Z", "ready": True})
    client = TransformersClient.from_env(ModelSpec(
        key="t", name="m", env_prefix="T", prompt="p", backend="transformers"))
    assert str(client._http.base_url).rstrip("/") == "http://pod:8000"


def test_from_env_error_names_the_recovery_commands(monkeypatch):
    monkeypatch.delenv("T_ENDPOINT", raising=False)
    spec = ModelSpec(key="t", name="m", env_prefix="T", prompt="p", backend="transformers")
    with pytest.raises(ValueError) as exc:
        client_for(spec)
    # the error must tell an agent HOW to get an endpoint, not just that it's missing
    assert "labeling-t-runpod up" in str(exc.value) and "T_ENDPOINT" in str(exc.value)
