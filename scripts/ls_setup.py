#!/usr/bin/env python
"""One-time Label Studio setup: enable legacy API tokens so the SDK can auth.

LS 1.23+ disables legacy (SDK-style) tokens org-wide by default and only the
JWT personal-access tokens work in the UI. The label-studio-sdk sends a legacy
`Authorization: Token ...` header, so we flip `legacy_api_tokens_enabled` on via
the authenticated settings API, then print the token to use.

    uv run python scripts/ls_setup.py
    # -> prints the API token; put it in your import-ls --api-key
"""

from __future__ import annotations

import os
import re
import sys

import requests

BASE = os.environ.get("LS_URL", "http://localhost:8080")
EMAIL = os.environ.get("LS_EMAIL", "admin@labeling-t.local")
PW = os.environ.get("LS_PASSWORD", "labeling-t-admin")


def main() -> int:
    s = requests.Session()
    page = s.get(f"{BASE}/user/login/", timeout=15)
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page.text)
    csrf = m.group(1) if m else s.cookies.get("csrftoken")
    login = s.post(
        f"{BASE}/user/login/",
        data={"csrfmiddlewaretoken": csrf, "email": EMAIL, "password": PW},
        headers={"Referer": f"{BASE}/user/login/"},
        timeout=15,
    )
    if login.status_code != 200:
        print(f"login failed: {login.status_code}", file=sys.stderr)
        return 1

    # Enable legacy tokens org-wide.
    r = s.post(
        f"{BASE}/api/jwt/settings",
        json={"api_tokens_enabled": True, "legacy_api_tokens_enabled": True},
        headers={"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE},
        timeout=15,
    )
    if r.status_code not in (200, 201):
        print(f"could not enable legacy tokens: {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 1

    token = s.get(f"{BASE}/api/current-user/token", timeout=15).json().get("token")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
