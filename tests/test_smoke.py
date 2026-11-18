"""Local smoke tests (no network): store roundtrip, consent URL, validators."""

import base64
import os
import tempfile

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_SECRET", "x" * 32)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SECRET_PATH", "testpath123")
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    monkeypatch.delenv("RUNTIME_ID", raising=False)
    # import AFTER env is set so boot validation passes
    import importlib

    import app.main as main

    importlib.reload(main)
    return main


def test_secret_path_validation(monkeypatch):
    monkeypatch.setenv("SECRET_PATH", "short")
    with pytest.raises(SystemExit):
        import importlib

        import app.main as main

        importlib.reload(main)


def test_store_roundtrip(tmp_path):
    from app.store import TokenStore

    store = TokenStore("a" * 32, str(tmp_path))
    store.save_token('{"refresh_token": "secret1"}')
    assert store.load_token() == '{"refresh_token": "secret1"}'
    assert store.token_exists()
    store.save_state({"runtime_id": "r1"})
    assert store.load_state() == {"runtime_id": "r1"}
    # different secret cannot decrypt (rotation behavior)
    store2 = TokenStore("b" * 32, str(tmp_path))
    assert store2.load_token() is None
    store.delete_token()
    assert not store.token_exists()


def test_consent_url(env):
    url = env.AUTH.consent_url()
    assert "accounts.google.com/o/oauth2/auth" in url
    assert "client_id=764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8080" in url
    assert "colaboratory" in url
    assert "access_type=offline" in url and "prompt=consent" in url
    assert "token_usage=remote" in url


def test_exchange_carries_matching_redirect(env, monkeypatch):
    """The token request must echo the exact redirect_uri of the auth request."""
    captured = {}

    class FakeResp:
        status_code = 200
        content = b"{}"
        text = "{}"

        def json(self):
            return {"refresh_token": "rt", "access_token": "at", "scope": "s"}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data=None, headers=None):
            captured["url"] = url
            captured["data"] = data
            return FakeResp()

    monkeypatch.setattr("app.auth.httpx.AsyncClient", FakeClient)
    env.AUTH._creds = None
    import asyncio

    from app.auth import TOKEN_URI

    asyncio.run(env.AUTH.connect("4/0ATESTCODE"))
    assert captured["url"] == TOKEN_URI
    assert captured["data"]["redirect_uri"] == "http://localhost:8080"
    assert captured["data"]["code"] == "4/0ATESTCODE"
    assert captured["data"]["grant_type"] == "authorization_code"
    assert env.AUTH.connected()


def test_variant_mapping():
    from app.colabapi import variant_for_accelerator

    assert variant_for_accelerator("T4") == "VARIANT_GPU"
    assert variant_for_accelerator("NONE") == "VARIANT_CPU"
    assert variant_for_accelerator("TPU-V6E") == "VARIANT_TPU"


def test_registry_reattach():
    import asyncio
    from app.kernel import ExecRecord, ExecutionRegistry

    async def scenario():
        reg = ExecutionRegistry()
        rec = ExecRecord("default", "e1", "print(1)")
        reg.register(rec, asyncio.create_task(_noop()))
        assert reg.get("default", "e1") is rec
        assert reg.task_for("default", "e1") is not None
        assert reg.running_count() == 1
        assert reg.get("default", "other") is None
        assert reg.task_for("default", "other") is None

    asyncio.run(scenario())


async def _noop():
    return None


def test_file_guard():
    from app.kernel import MAX_FILE_BYTES

    assert MAX_FILE_BYTES == 48 * 1024 * 1024
    big = base64.b64encode(b"x" * (MAX_FILE_BYTES + 10)).decode()
    assert len(big) > MAX_FILE_BYTES // 2
