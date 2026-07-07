"""Model transport — talk to a model over HTTP from this GPU-less machine.

Two transports, picked per ModelSpec by `client_for`:

  ChatClient        — any OpenAI-compatible chat endpoint: a vLLM box we rent, or
                      a hosted vendor API (OpenAI, Google Gemini via its compat
                      layer). Same wire shape for all; they differ only in the
                      endpoint + a couple of quirk params, which ride on the spec.
  TransformersClient — our own FastAPI model-server's structured /infer.

The client is bound to a ModelSpec (the model's identity: served name + prompt)
and only needs the endpoint + key, which come from the environment (or, for a
SaaS provider, are baked into the spec). The framework stays decoupled from where
— and by whom — the model is hosted.

    image bytes ──base64 data URI──► chat request ──httpx POST──► vLLM / OpenAI / Gemini
                                       (model name + prompt from the ModelSpec)

ChatClient returns the raw assistant TEXT. Turning that into boxes is spec.parse
(in models.py / prelabel.parse_boxes), kept separate from transport.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from .models import ModelSpec
from .podstate import resolve_endpoint
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


def _data_uri_bytes(data: bytes, media_type: str = "image/png") -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{b64}"


def _retry_after_seconds(exc: Exception, *, default: float) -> float:
    """Seconds to wait after a 429: the server's Retry-After if parseable
    (clamped to sane bounds), else the caller's default."""
    header = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
    try:
        return min(max(float(header), 1.0), 120.0)
    except (TypeError, ValueError):
        return default


def _image_url(ref: str | Path | bytes) -> str:
    """An http(s) URL (e.g. presigned S3) passes through so the GPU fetches it
    directly; a local path is base64-inlined; raw bytes (an in-memory crop from
    transcribe.py) are base64-inlined as PNG."""
    if isinstance(ref, bytes):
        return _data_uri_bytes(ref)
    s = str(ref)
    if s.startswith(("http://", "https://")):
        return s
    return _data_uri(ref)


class ChatClient:
    """httpx client for one model (a ModelSpec) at one OpenAI-compatible chat
    endpoint — a rented vLLM box or a hosted vendor API (OpenAI, Gemini)."""

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
        transport: httpx.BaseTransport | None = None,
    ):
        self.spec = spec
        # Per-run category override; falls back to the spec's defaults.
        self.categories = list(categories) if categories is not None else list(spec.categories)
        self.max_retries = max_retries
        # Cap generation: a box JSON is a few hundred tokens. Without this, a
        # looping model generates to the full context and one request takes minutes.
        self.max_tokens = max_tokens
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
    def from_env(cls, spec: ModelSpec, *, endpoint: str | None = None,
                 categories: list[str] | None = None, **kw) -> "ChatClient":
        """Build a client for `spec`. The endpoint resolves per podstate
        precedence: explicit `endpoint` arg > recorded pod > {PREFIX}_ENDPOINT
        env (deprecated) > the spec's baked-in default (SaaS providers).
        The API key still comes from {PREFIX}_API_KEY."""
        url = resolve_endpoint(spec, endpoint)
        if not url:
            raise ValueError(_no_endpoint_msg(spec))
        return cls(url, spec, api_key=spec.api_key_from_env(), categories=categories, **kw)

    def build_payload(self, image_path: str | Path | bytes) -> dict:
        prompt = self.spec.prompt.format(categories=", ".join(self.categories))
        image = {"url": _image_url(image_path)}
        if self.spec.image_detail:
            image["detail"] = self.spec.image_detail
        payload = {
            "model": self.spec.name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        # Per-model quirk knobs (vLLM repetition_penalty, a provider response_format).
        # Merged last so a spec can also override a base field if it ever needs to.
        payload.update(self.spec.extra_body)
        return payload

    def infer(self, image_path: str | Path | bytes) -> str:
        """Raw assistant text for one image (path, URL, or in-memory bytes).
        Retries transient (5xx/network/429-rate-limit) errors; raises on other
        4xx and on the final attempt so the caller can record the failure."""
        payload = self.build_payload(image_path)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(self.spec.chat_path, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", 500)
                # 429 is transient (rate limit) — back off like a 5xx instead of
                # failing the frame; hosted APIs hit it under normal fan-out.
                if isinstance(exc, httpx.HTTPStatusError) and status < 500 and status != 429:
                    raise
                if attempt < self.max_retries:
                    if status == 429:
                        # Rate limits are minute-window quotas: a sub-second sleep
                        # just burns the retry. Honor Retry-After when sent, else
                        # wait long enough for the window to move.
                        time.sleep(_retry_after_seconds(exc, default=15.0 * (attempt + 1)))
                    else:
                        time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TransformersClient:
    """httpx client for OUR transformers model-server (structured `/infer`).

    Unlike vLLM (text out, then spec.parse), this backend returns boxes already
    parsed and in absolute pixels, plus the image dims — so it implements
    `infer_raw` directly and the framework never parses model text. The wire
    payload is uniform across every detector: {image_url, queries, params}; the
    server's ModelAdapter maps it per model. Mirrors ChatClient's retry policy.

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
    def from_env(cls, spec: ModelSpec, *, endpoint: str | None = None,
                 categories: list[str] | None = None, **kw) -> "TransformersClient":
        url = resolve_endpoint(spec, endpoint)
        if not url:
            raise ValueError(_no_endpoint_msg(spec))
        return cls(url, spec, api_key=spec.api_key_from_env(), categories=categories, **kw)

    def build_payload(self, image_path: str | Path) -> dict:
        # image is sent as a URL the server fetches (presigned S3 passthrough), or
        # base64 for a local file — same _image_url helper as the vLLM path.
        return {"image_url": _image_url(image_path), "queries": self.categories, "params": self.params}

    def _post_infer(self, payload: dict) -> dict:
        """POST /infer with the shared retry policy: retries transient (5xx/network);
        raises on 4xx and on the final attempt so prelabel records the failure."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/infer", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", 500)
                if isinstance(exc, httpx.HTTPStatusError) and status < 500:
                    raise
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def infer_raw(self, image_path: str | Path) -> RawInference:
        """Structured detections for one image (detector backends)."""
        data = self._post_infer(self.build_payload(image_path))
        boxes = [(d["bbox"], d["label"], d.get("score")) for d in data.get("detections", [])]
        return RawInference(boxes=boxes, width=data.get("width"), height=data.get("height"))

    def _prompt_boxes(
        self,
        image_path: str | Path,
        boxes: list[list[float]],
        labels: list[str] | None,
        scores: list[float | None] | None,
    ) -> list[dict]:
        """Shared box-prompt call for the stage-2 models (SAM2, VitPose): the
        boxes ride in `params`, the wire shape is otherwise the detector's."""
        payload = {
            "image_url": _image_url(image_path),
            "queries": [],
            "params": {**self.params, "boxes": boxes,
                       "labels": labels or [], "scores": scores or []},
        }
        return self._post_infer(payload).get("detections", [])

    def segment(
        self,
        image_path: str | Path,
        boxes: list[list[float]],
        *,
        labels: list[str] | None = None,
        scores: list[float | None] | None = None,
    ) -> list[dict]:
        """Stage-2 segmentation: send box prompts (a detector's output) and get one
        masked detection per box back. Returns the raw detection dicts
        [{bbox, label, score, mask:RLE}] — masks ride as COCO RLE."""
        return self._prompt_boxes(image_path, boxes, labels, scores)

    def keypoints(
        self,
        image_path: str | Path,
        boxes: list[list[float]],
        *,
        labels: list[str] | None = None,
        scores: list[float | None] | None = None,
    ) -> list[dict]:
        """Stage-2 pose: send box prompts and get one keypointed detection per
        box back: [{bbox, label, score, keypoints:[{x,y,name,score}]}]."""
        return self._prompt_boxes(image_path, boxes, labels, scores)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TransformersClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _no_endpoint_msg(spec: ModelSpec) -> str:
    return (
        f"no endpoint for model {spec.key!r}: no running pod recorded "
        f"(`labeling-t-runpod up --model {spec.key}`, or check `labeling-t-runpod "
        f"status`), no --endpoint given, and {spec.env_prefix}_ENDPOINT is unset"
    )


def client_for(spec: ModelSpec, *, endpoint: str | None = None,
               categories: list[str] | None = None, **kw):
    """Build the right transport for a spec's backend. The endpoint resolves per
    podstate precedence (explicit arg > recorded pod > env > spec default).

    The transformers backend returns structured boxes from our model-server; every
    other backend (vllm / openai / gemini) speaks OpenAI chat, so ChatClient drives
    them all. Callers stay backend-agnostic — prelabel duck-types infer vs infer_raw.
    """
    if spec.backend == "transformers":
        return TransformersClient.from_env(spec, endpoint=endpoint, categories=categories, **kw)
    return ChatClient.from_env(spec, endpoint=endpoint, categories=categories, **kw)
