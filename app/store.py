"""Encrypted token + bookkeeping storage for the Colab MCP bridge.

Everything sensitive (the Google OAuth refresh token) lives in ``token.enc``
(Fernet-encrypted JSON); ``state.json`` holds only non-secret bookkeeping
(cached runtime id, subscription tier cache).

TOKEN_SECRET (a Railway ``${{secret(32)}}``) is an arbitrary random string, not
a valid Fernet key by itself — we derive a proper key with SHA-256 so any
random string works:

    fernet_key = urlsafe_b64encode(sha256(TOKEN_SECRET.encode()).digest())

Rotating TOKEN_SECRET makes existing token.enc unreadable (by design) — the
deployer just re-runs the /connect flow.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile

from cryptography.fernet import Fernet, InvalidToken

_TOKEN_FILE = "token.enc"
_STATE_FILE = "state.json"


class StoreError(Exception):
    """Raised when the store cannot be initialized or written."""


def resolve_data_dir() -> str:
    """Resolve the data directory: DATA_DIR env, else /data, else ./data.

    On Railway a volume is mounted at /data. Locally (dev/CI) /data usually
    does not exist, so fall back to a sibling ./data directory.
    """
    candidates = [
        os.environ.get("DATA_DIR", "").strip() or None,
        "/data",
        os.path.join(os.getcwd(), "data"),
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".write-probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return cand
        except OSError:
            continue
    # Last resort: a per-user temp dir (never silently lose tokens though).
    fallback = os.path.join(tempfile.gettempdir(), "colab-mcp-bridge-data")
    os.makedirs(fallback, exist_ok=True)
    print(
        f"WARNING: no writable DATA_DIR found, using {fallback}. Tokens will "
        "not survive container replacement there.",
        file=sys.stderr,
        flush=True,
    )
    return fallback


def _fernet_key(secret: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())


class TokenStore:
    """Fernet-encrypted token file + plain bookkeeping state, atomically written."""

    def __init__(self, token_secret: str, data_dir: str | None = None) -> None:
        if not token_secret or len(token_secret) < 16:
            raise StoreError(
                "FATAL: TOKEN_SECRET must be set to at least 16 random characters. "
                "On Railway it is generated automatically (${{secret(32)}}); when "
                "running manually, export TOKEN_SECRET=$(python -c 'import secrets;"
                " print(secrets.token_urlsafe(32))')."
            )
        self._fernet = Fernet(_fernet_key(token_secret))
        self.dir = data_dir or resolve_data_dir()
        self.token_path = os.path.join(self.dir, _TOKEN_FILE)
        self.state_path = os.path.join(self.dir, _STATE_FILE)

    # ------------------------------------------------------------------ #
    # token.enc — encrypted Google credentials
    # ------------------------------------------------------------------ #

    def _atomic_write(self, path: str, data: bytes) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:  # best-effort hardening (POSIX only; volume is single-user)
            os.chmod(path, 0o600)
        except OSError:
            pass

    def save_token(self, creds_json: str) -> None:
        blob = self._fernet.encrypt(creds_json.encode("utf-8"))
        self._atomic_write(self.token_path, blob)

    def load_token(self) -> str | None:
        try:
            with open(self.token_path, "rb") as fh:
                blob = fh.read()
        except FileNotFoundError:
            return None
        try:
            return self._fernet.decrypt(blob).decode("utf-8")
        except InvalidToken:
            print(
                "WARNING: token.enc exists but cannot be decrypted with the "
                "current TOKEN_SECRET — treating as not connected. Re-run the "
                "/connect flow.",
                file=sys.stderr,
                flush=True,
            )
            return None

    def token_exists(self) -> bool:
        return os.path.exists(self.token_path)

    def delete_token(self) -> None:
        try:
            os.remove(self.token_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ #
    # state.json — non-secret bookkeeping (cached runtime id, tier)
    # ------------------------------------------------------------------ #

    def load_state(self) -> dict:
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, state: dict) -> None:
        self._atomic_write(
            self.state_path, json.dumps(state, indent=1, sort_keys=True).encode("utf-8")
        )
