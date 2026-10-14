"""API-key authentication: off by default, enforced when API_KEY is set; health/metrics/docs stay open."""
import api.main as main


def test_auth_enforced_when_key_set(client, monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "s3cret")
    assert client.get("/incidents").status_code == 401                                  # no key
    assert client.get("/incidents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/incidents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/score", json={"flows": [{"a": 1.0}]}).status_code == 401       # before validation
    for open_path in ("/health", "/metrics", "/openapi.json"):
        assert client.get(open_path).status_code == 200, open_path


def test_auth_off_by_default(client):
    assert main.API_KEY is None
    assert client.get("/incidents").status_code == 200
