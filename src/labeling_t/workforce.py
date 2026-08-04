"""Human-workforce provisioning — rent annotators like runpod.py rents GPUs.

Backend: RentAHuman (rentahuman.ai) — a marketplace where agents hire humans.
Same shape as the other integrations: the framework owns the workflow (post a
labeling bounty, watch applications, message the hire), the marketplace is a
swappable adapter behind one client. Auth = RENTAHUMAN_API_KEY in .env.

    labeling-t-workforce search --skill "data annotation"
    labeling-t-workforce post --title ... --price 25 --hours 6 --description-file b.md
    labeling-t-workforce status --bounty <id>
    labeling-t-workforce applications --bounty <id>
    labeling-t-workforce message --conversation <id> --text "..."

GUARDRAIL — money never moves from here. Escrow funding, accepting an
application, and payment release are DELIBERATELY not wrapped: do those in the
RentAHuman web UI, or via the rentahuman MCP server (registered in .mcp.json)
where each action gets an explicit human-visible confirmation. This CLI covers
the free read/post/message surface only.

Typical loop (how project 20's trial was run): post bounty (status
pending_deposit) -> user funds escrow in the web UI (status open) -> watch
`applications` -> user accepts in UI -> `message` the hire a dedicated LS
annotator login (NEVER admin creds; credentials go via platform messaging, not
the public bounty text).
"""

from __future__ import annotations

import argparse
import os

import httpx

from .config import load_env
from .output import emit, fail, json_flag

API_BASE = "https://rentahuman.ai/api"


class WorkforceClient:
    """Thin httpx client for the RentAHuman REST API (the surface the MCP
    server wraps). Transport injectable for tests, like model_client."""

    def __init__(self, api_key: str | None = None, *, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        key = api_key or os.environ.get("RENTAHUMAN_API_KEY") or ""
        headers = {"Content-Type": "application/json"}
        if key:
            headers["X-API-Key"] = key
        self._http = httpx.Client(base_url=API_BASE, headers=headers,
                                  timeout=timeout, transport=transport)

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self._http.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self._http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def search_humans(self, *, skill: str | None = None, max_rate: float | None = None,
                      country: str | None = None, limit: int = 10) -> list[dict]:
        data = self._get("/humans", {"skill": skill, "maxRate": max_rate,
                                     "country": country, "limit": limit})
        return data.get("humans", [])

    def create_bounty(self, *, title: str, description: str, criteria: str,
                      price: float, hours: float, category: str = "research") -> dict:
        data = self._post("/bounties", {
            "title": title, "description": description,
            "completionCriteria": criteria, "evidenceTypes": ["text"],
            "price": price, "priceType": "fixed",
            "category": category, "estimatedHours": hours,
        })
        return data.get("bounty", data)

    def get_bounty(self, bounty_id: str) -> dict:
        return self._get(f"/bounties/{bounty_id}").get("bounty", {})

    def list_applications(self, bounty_id: str) -> list[dict]:
        return self._get(f"/bounties/{bounty_id}/applications").get("applications", [])

    def send_message(self, conversation_id: str, text: str,
                     sender_id: str | None = None) -> dict:
        # The API requires senderType/senderId (discovered live 2026-07-20; the
        # docs omit them). senderId = the account's user id — visible as
        # `agentId` on any bounty you created; RENTAHUMAN_SENDER_ID in .env.
        sid = sender_id or os.environ.get("RENTAHUMAN_SENDER_ID") or ""
        return self._post(f"/conversations/{conversation_id}/messages",
                          {"content": text, "messageType": "text",
                           "senderType": "agent", "senderId": sid})

    def close(self) -> None:
        self._http.close()


def _human_row(h: dict) -> dict:
    loc = h.get("location") or {}
    return {"id": h.get("id"), "name": h.get("name"), "headline": h.get("headline"),
            "rate": h.get("hourlyRate"), "currency": h.get("currency"),
            "country": loc.get("country"), "remote": loc.get("isRemoteAvailable"),
            "skills": (h.get("skills") or [])[:8]}


def cmd_search(a: argparse.Namespace) -> int:
    c = WorkforceClient()
    humans = [_human_row(h) for h in c.search_humans(
        skill=a.skill, max_rate=a.max_rate, country=a.country, limit=a.limit)]
    lines = [f"{h['name']} ({h['country']}, ${h['rate']}/hr) — {h['headline']}" for h in humans]
    return emit(a, {"humans": humans, "count": len(humans)}, "\n".join(lines) or "no matches")


def cmd_post(a: argparse.Namespace) -> int:
    description = open(a.description_file).read() if a.description_file else a.description
    if not description:
        return fail(a, "need --description or --description-file")
    c = WorkforceClient()
    b = c.create_bounty(title=a.title, description=description, criteria=a.criteria,
                        price=a.price, hours=a.hours, category=a.category)
    return emit(a, {"bounty_id": b.get("id"), "status": b.get("status"),
                    "price": b.get("price")},
                f"bounty {b.get('id')} created (status {b.get('status')}; "
                f"pending_deposit = fund escrow in the RentAHuman web UI to go live)")


def cmd_status(a: argparse.Namespace) -> int:
    b = WorkforceClient().get_bounty(a.bounty)
    if not b:
        return fail(a, f"bounty {a.bounty!r} not found")
    res = {"bounty_id": b.get("id"), "status": b.get("status"), "title": b.get("title"),
           "price": b.get("price"), "views": b.get("viewCount"),
           "applications": b.get("applicationCount")}
    return emit(a, res, f"{res['status']} | views {res['views']} | applications {res['applications']}")


def cmd_applications(a: argparse.Namespace) -> int:
    apps = WorkforceClient().list_applications(a.bounty)
    rows = [{"id": x.get("id"), "human": x.get("humanName") or (x.get("human") or {}).get("name"),
             "status": x.get("status"), "message": (x.get("message") or "")[:200]}
            for x in apps]
    lines = [f"{r['human']} [{r['status']}] {r['message']}" for r in rows]
    return emit(a, {"applications": rows, "count": len(rows)},
                "\n".join(lines) or "no applications yet")


def cmd_message(a: argparse.Namespace) -> int:
    WorkforceClient().send_message(a.conversation, a.text)
    return emit(a, {"sent": True, "conversation": a.conversation}, "sent")


def main(argv: list[str] | None = None) -> int:
    load_env()
    jf = [json_flag()]
    ap = argparse.ArgumentParser(prog="labeling-t-workforce",
                                 description="rent human annotators (RentAHuman backend)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="browse annotators (free, no side effects)", parents=jf)
    s.add_argument("--skill", default="data annotation")
    s.add_argument("--max-rate", type=float, default=None)
    s.add_argument("--country", default=None)
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_search)

    p = sub.add_parser("post", help="post a labeling bounty (goes live after the "
                       "user funds escrow in the web UI)", parents=jf)
    p.add_argument("--title", required=True)
    p.add_argument("--description", default=None)
    p.add_argument("--description-file", default=None)
    p.add_argument("--criteria", required=True, help="completion criteria shown to workers")
    p.add_argument("--price", type=float, required=True, help="fixed USD price")
    p.add_argument("--hours", type=float, required=True, help="estimated hours")
    p.add_argument("--category", default="research")
    p.set_defaults(func=cmd_post)

    st = sub.add_parser("status", help="bounty status / views / application count", parents=jf)
    st.add_argument("--bounty", required=True)
    st.set_defaults(func=cmd_status)

    al = sub.add_parser("applications", help="list a bounty's applications", parents=jf)
    al.add_argument("--bounty", required=True)
    al.set_defaults(func=cmd_applications)

    m = sub.add_parser("message", help="send a message in an existing conversation "
                       "(e.g. LS annotator credentials to the accepted hire)", parents=jf)
    m.add_argument("--conversation", required=True)
    m.add_argument("--text", required=True)
    m.set_defaults(func=cmd_message)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except httpx.HTTPStatusError as exc:
        return fail(a, f"RentAHuman API {exc.response.status_code}: {exc.response.text[:200]}")
    except httpx.TransportError as exc:
        return fail(a, f"network error talking to RentAHuman: {exc}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
