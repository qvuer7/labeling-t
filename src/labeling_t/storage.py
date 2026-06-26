"""Object storage — local files or S3-compatible (DO Spaces).

The cloud source/sink for frames and labels. Lets prelabel read images and write
labels by URI, so local-vs-cloud is a backend choice, not a code change.

Creds use the standard AWS_* env (same as StreamScout), so existing DO Spaces
keys work as-is. Config:
    S3_ENDPOINT_URL   e.g. https://fra1.digitaloceanspaces.com
    S3_REGION         e.g. fra1
    S3_BUCKET         e.g. ml-cv-data
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

URIs: `s3://bucket/key` (or a bare key against the default bucket) for S3,
plain paths for local.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


class Storage(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...
    def write_bytes(self, uri: str, data: bytes) -> None: ...
    def write_text(self, uri: str, text: str) -> None: ...
    def list(self, prefix: str) -> list[str]: ...
    def presigned_url(self, uri: str, ttl: int = 3600) -> str: ...
    def image_size(self, uri: str) -> tuple[int, int]: ...


class LocalStorage:
    """Plain filesystem. presigned_url returns the path (local LS/dev)."""

    def read_bytes(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

    def write_bytes(self, uri: str, data: bytes) -> None:
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def write_text(self, uri: str, text: str) -> None:
        self.write_bytes(uri, text.encode())

    def list(self, prefix: str) -> list[str]:
        p = Path(prefix)
        if p.is_dir():
            return sorted(str(f) for f in p.rglob("*") if f.is_file())
        # prefix semantics (like S3): files in the parent whose name starts with it
        return sorted(str(f) for f in p.parent.glob(p.name + "*") if f.is_file())

    def presigned_url(self, uri: str, ttl: int = 3600) -> str:
        return uri

    def image_size(self, uri: str) -> tuple[int, int]:
        from PIL import Image

        with Image.open(uri) as im:
            return im.size


class S3Storage:
    """S3-compatible object store (AWS S3 / DigitalOcean Spaces)."""

    def __init__(self, bucket: str, *, endpoint_url: str | None = None, region: str | None = None):
        import boto3

        self.default_bucket = bucket
        self._s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    @classmethod
    def from_env(cls) -> "S3Storage":
        bucket = os.environ.get("S3_BUCKET", "").strip()
        if not bucket:
            raise ValueError("S3_BUCKET not set")
        return cls(
            bucket,
            endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            region=os.environ.get("S3_REGION") or None,
        )

    def _split(self, uri: str) -> tuple[str, str]:
        if is_s3(uri):
            u = urlparse(uri)
            return u.netloc, u.path.lstrip("/")
        return self.default_bucket, uri.lstrip("/")

    def read_bytes(self, uri: str) -> bytes:
        b, k = self._split(uri)
        return self._s3.get_object(Bucket=b, Key=k)["Body"].read()

    def write_bytes(self, uri: str, data: bytes) -> None:
        b, k = self._split(uri)
        self._s3.put_object(Bucket=b, Key=k, Body=data)

    def write_text(self, uri: str, text: str) -> None:
        self.write_bytes(uri, text.encode())

    def list(self, prefix: str) -> list[str]:
        b, k = self._split(prefix)
        out: list[str] = []
        token = None
        while True:
            kw = {"Bucket": b, "Prefix": k}
            if token:
                kw["ContinuationToken"] = token
            r = self._s3.list_objects_v2(**kw)
            out += [f"s3://{b}/{o['Key']}" for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                break
            token = r.get("NextContinuationToken")
        return out

    def presigned_url(self, uri: str, ttl: int = 3600) -> str:
        b, k = self._split(uri)
        return self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=ttl
        )

    def image_size(self, uri: str) -> tuple[int, int]:
        """Image dims without a full download: ranged read of the header,
        falling back to the whole object if the header didn't fit."""
        from PIL import Image

        b, k = self._split(uri)
        try:
            data = self._s3.get_object(Bucket=b, Key=k, Range="bytes=0-65535")["Body"].read()
            return Image.open(io.BytesIO(data)).size
        except Exception:
            return Image.open(io.BytesIO(self.read_bytes(uri))).size


def open_storage(uri: str | None = None) -> Storage:
    """S3 backend for s3:// URIs or when S3_BUCKET is configured; else local."""
    if (uri and is_s3(uri)) or os.environ.get("S3_BUCKET"):
        return S3Storage.from_env()
    return LocalStorage()
