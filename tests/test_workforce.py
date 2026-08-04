"""Workforce (RentAHuman) client + CLI: wire shape, auth header, envelopes.
All transport is httpx.MockTransport — no network, no real API key."""

import json

import httpx

from labeling_t.workforce import WorkforceClient, main


def _client(handler):
    return WorkforceClient(api_key="rah_test_key", transport=httpx.MockTransport(handler))


def test_search_sends_auth_and_params_and_parses_humans():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-api-key"] == "rah_test_key"
        assert req.url.path == "/api/humans"
        assert req.url.params["skill"] == "data annotation"
        assert "maxRate" not in req.url.params  # None params dropped
        return httpx.Response(200, json={"success": True, "humans": [
            {"id": "h1", "name": "Ada", "hourlyRate": 5,
             "location": {"country": "NG", "isRemoteAvailable": True}}]})

    humans = _client(handler).search_humans(skill="data annotation")
    assert humans[0]["name"] == "Ada"


def test_create_bounty_posts_fixed_price_body():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.url.path == "/api/bounties"
        assert body["priceType"] == "fixed" and body["price"] == 25
        assert body["completionCriteria"] == "all frames labeled"
        return httpx.Response(200, json={"success": True, "bounty": {
            "id": "b1", "status": "pending_deposit", "price": 25}})

    b = _client(handler).create_bounty(title="t", description="d",
                                       criteria="all frames labeled", price=25, hours=6)
    assert (b["id"], b["status"]) == ("b1", "pending_deposit")


def test_applications_and_message_paths():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if "applications" in req.url.path:
            return httpx.Response(200, json={"applications": [
                {"id": "a1", "humanName": "Ada", "status": "pending", "message": "hi"}]})
        return httpx.Response(200, json={"success": True})

    c = _client(handler)
    assert c.list_applications("b1")[0]["humanName"] == "Ada"
    c.send_message("conv1", "credentials inside")
    assert calls == [("GET", "/api/bounties/b1/applications"),
                     ("POST", "/api/conversations/conv1/messages")]


def test_cli_status_envelope(monkeypatch, capsys):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bounty": {
            "id": "b1", "status": "open", "title": "t", "price": 25,
            "viewCount": 11, "applicationCount": 0}})

    monkeypatch.setattr("labeling_t.workforce.WorkforceClient",
                        lambda *a, **k: WorkforceClient(
                            api_key="k", transport=httpx.MockTransport(handler)))
    rc = main(["status", "--bounty", "b1", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["ok"] is True
    assert out["result"]["status"] == "open" and out["result"]["views"] == 11


def test_cli_api_error_becomes_fail_envelope(monkeypatch, capsys):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    monkeypatch.setattr("labeling_t.workforce.WorkforceClient",
                        lambda *a, **k: WorkforceClient(
                            api_key="k", transport=httpx.MockTransport(handler)))
    rc = main(["status", "--bounty", "b1", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["ok"] is False
    assert "401" in out["error"]["message"]
