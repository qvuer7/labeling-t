#!/usr/bin/env python
"""Set a CORS policy on the Spaces/S3 bucket so browsers can load frames.

Label Studio's image annotator fetches frames with crossOrigin (it needs canvas
pixel access to draw boxes), so the object store must return CORS headers or the
browser blocks the load. Presigned signatures remain the real access control —
CORS only lets the browser *read* a response it already holds a valid URL for.

    set -a; . ../StreamScout/.env; set +a   # S3_* + AWS_* creds
    uv run python scripts/set_bucket_cors.py --origin '*'
    uv run python scripts/set_bucket_cors.py --origin https://165-245-251-248.nip.io
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="set bucket CORS for browser frame loading")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="default: $S3_BUCKET")
    p.add_argument("--origin", action="append", default=None,
                   help="allowed origin (repeatable); default '*'")
    p.add_argument("--methods", default="GET,HEAD", help="comma-separated, default GET,HEAD")
    a = p.parse_args(argv)
    if not a.bucket:
        print("no --bucket and S3_BUCKET unset", file=sys.stderr)
        return 1

    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
                      region_name=os.environ.get("S3_REGION") or None)
    rule = {
        "AllowedOrigins": a.origin or ["*"],
        "AllowedMethods": [m.strip() for m in a.methods.split(",") if m.strip()],
        "AllowedHeaders": ["*"],
        "MaxAgeSeconds": 3000,
    }
    s3.put_bucket_cors(Bucket=a.bucket, CORSConfiguration={"CORSRules": [rule]})
    print(f"CORS set on {a.bucket}:")
    for r in s3.get_bucket_cors(Bucket=a.bucket)["CORSRules"]:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
