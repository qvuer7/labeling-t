"""Model transport — talk to a REMOTE vLLM endpoint over HTTP.

The model lives on a rented GPU box running vLLM; this machine has no GPU and no
torch. The client is bound to a ModelSpec (the model's identity: served name +
prompt) and only needs the endpoint, which comes from the environment. The
labeling framework therefore stays decoupled from model hosting.

    image bytes ──base64 data URI──► chat request ──httpx POST──► vLLM
                                       (model name + prompt from the ModelSpec)

Returns the raw assistant TEXT. Turning that into boxes is spec.parse (in
models.py / prelabel.parse_boxes), kept separate from transport.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from .models import ModelSpec
from .prelabel import RawInference

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _data_uri(image_path: str | Path) -> str:
    p = Path(image_path)
    media = _MEDIA_TYPES.get(p.suffix.lower())
    if media is None:
        raise ValueError(f"unsupported image type: {p.suffix} ({p})")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{media};base64,{b64}"


def _image_url(ref: str | Path) -> str:
    """An http(s) URL (e.g. presigned S3) passes through so the GPU fetches it
    directly; a local path is base64-inlined."""
    s = str(ref)
    if s.startswith(("http://", "https://")):
        return s
    return _data_uri(ref)


class VLLMClient:
    """httpx client for one model (a ModelSpec) at one endpoint."""

    def __init__(
        self,
        endpoint: str,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        categories: list[str] | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
        repetition_penalty: float = 1.1,
        transport: httpx.BaseTransport | None = None,
    ):
        self.spec = spec
        # Per-run category override; falls back to the spec's defaults.
        self.categories = list(categories) if categories is not None else list(spec.categories)
        self.max_retries = max_retries
        # Cap generation: a box JSON is a few hundred tokens. Without this, vLLM
        # generates to the full context and a single request takes minutes.
        self.max_tokens = max_tokens
        # Grounding VLMs greedily loop, repeating the same box to the token cap.
        # A repetition penalty (vLLM honors it on the OpenAI route) stops that.
        self.repetition_penalty = repetition_penalty
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=endpoint.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,  # injectable for tests (httpx.MockTransport)
        )

    @classmethod
    def from_env(cls, spec: ModelSpec, *, categories: list[str] | None = None, **kw) -> "VLLMClient":
        """Build a client for `spec`, reading {PREFIX}_ENDPOINT / _API_KEY."""
        endpoint = spec.endpoint_from_env()
        if not endpoint:
            raise ValueError(f"{spec.env_prefix}_ENDPOINT is not set (.env)")
        return cls(endpoint, spec, api_key=spec.api_key_from_env(), categories=categories, **kw)

    def build_payload(self, image_path: str | Path) -> dict:
        prompt = self.spec.prompt.format(categories=", ".join(self.categories))
        return {
            "model": self.spec.name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_url(image_path)}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    def infer(self, image_path: str | Path) -> str:
        """Raw assistant text for one image. Retries transient (5xx/network)
        errors; raises on 4xx and on the final attempt so prelabel can record
        the failure."""
        payload = self.build_payload(image_path)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", 500)
                if isinstance(exc, httpx.HTTPStatusError) and status < 500:
                    raise
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "VLLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TransformersClient:
    """httpx client for OUR transformers model-server (structured `/infer`).

    Unlike vLLM (text out, then spec.parse), this backend returns boxes already
    parsed and in absolute pixels, plus the image dims — so it implements
    `infer_raw` directly and the framework never parses model text. The wire
    payload is uniform across every detector: {image_url, queries, params}; the
    server's ModelAdapter maps it per model. Mirrors VLLMClient's retry policy.

        presigned URL ──POST /infer {image_url,queries,params}──► server
        RawInference(boxes abs-px, width, height) ◄── {width,height,detections}
    """

    def __init__(
        self,
        endpoint: str,
        spec: ModelSpec,
        *,
        api_key: str | None = None,
        categories: list[str] | None = None,
        params: dict | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        self.spec = spec
        self.categories = list(categories) if categories is not None else list(spec.categories)
        self.params = dict(params) if params else {}
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=endpoint.rstrip("/"), headers=headers, timeout=timeout, transport=transport,
        )

    @classmethod
    def from_env(cls, spec: ModelSpec, *, categories: list[str] | None = None, **kw) -> "TransformersClient":
        endpoint = spec.endpoint_from_env()
        if not endpoint:
            raise ValueError(f"{spec.env_prefix}_ENDPOINT is not set (.env)")
        return cls(endpoint, spec, api_key=spec.api_key_from_env(), categories=categories, **kw)

    def build_payload(self, image_path: str | Path) -> dict:
        # image is sent as a URL the server fetches (presigned S3 passthrough), or
        # base64 for a local file — same _image_url helper as the vLLM path.
        return {"image_url": _image_url(image_path), "queries": self.categories, "params": self.params}

    def infer_raw(self, image_path: str | Path) -> RawInference:
        """Structured detections for one image. Retries transient (5xx/network);
        raises on 4xx and on the final attempt so prelabel records the failure."""
        payload = self.build_payload(image_path)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/infer", json=payload)
                resp.raise_for_status()
                data = resp.json()
                boxes = [
                    (d["bbox"], d["label"], d.get("score"))
                    for d in data.get("detections", [])
                ]
                return RawInference(boxes=boxes, width=data.get("width"), height=data.get("height"))
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", 500)
                if isinstance(exc, httpx.HTTPStatusError) and status < 500:
                    raise
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TransformersClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
