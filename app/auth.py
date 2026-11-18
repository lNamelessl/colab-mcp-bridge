"""Google OAuth for the Colab MCP bridge.

The bridge uses the *loopback copy-paste* consent flow: the /connect page
shows a consent URL bound to Google's public cloud-SDK OAuth client with
``redirect_uri=http://localhost:8080`` (Google's own client accepts any
localhost loopback port — verified live; the old sdk.cloud.google.com
ADC landing page registration was REVOKED by Google and returns
``redirect_uri_mismatch``). After approval the browser lands on a
"can't connect" page at localhost (nothing is listening there) and the
user copies the ``code=4/0A...`` parameter from the ADDRESS BAR into the
/connect form. The bridge exchanges the code server-side — the token
request MUST carry the exact same redirect_uri — and stores the refresh
token encrypted on the volume (``store.TokenStore``). google-auth then
refreshes access tokens server-side forever — the agent never sees
Google credentials.

Scope set (colab-cli PUBLIC_SCOPES minus drive.file/profile):
  openid, cloud-platform, userinfo.email, colaboratory

The cloud-SDK client id/secret are Google's own published values (shipped in
colab-cli's oauth_config.json) — they are public by construction. They can be
overridden via GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET /
GOOGLE_OAUTH_REDIRECT_URI (any localhost port works with Google's client;
a custom non-loopback redirect requires your own registered client).
"""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode

import httpx
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials

from .store import TokenStore

# Google's public cloud-SDK client (colab-cli oauth_config.json; public values).
DEFAULT_CLIENT_ID = (
    "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
)
DEFAULT_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
# Loopback redirect accepted by Google's cloud-SDK client (the former
# sdk.cloud.google.com ADC landing page now returns redirect_uri_mismatch).
DEFAULT_REDIRECT_URI = "http://localhost:8080"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
]

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"


class AuthError(Exception):
    """Anticipated auth failure (not connected, consent refused, refresh dead)."""


def _env(key: str, default: str) -> str:
    return os.environ.get(key, "").strip() or default


class AuthManager:
    """Consent-URL builder, code exchanger, and credential refresher."""

    def __init__(self, store: TokenStore) -> None:
        self._store = store
        self._client_id = _env("GOOGLE_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID)
        self._client_secret = _env("GOOGLE_OAUTH_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)
        self._redirect_uri = _env("GOOGLE_OAUTH_REDIRECT_URI", DEFAULT_REDIRECT_URI)
        self._creds: Credentials | None = None
        self._refresh_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Copy-paste consent flow
    # ------------------------------------------------------------------ #

    def consent_url(self) -> str:
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            # Colab/gcloud-specific: marks this as the remote copy-paste flow.
            "token_usage": "remote",
        }
        return f"{AUTH_URI}?{urlencode(params)}"

    async def connect(self, code: str) -> dict:
        """Exchange an authorization code for tokens and store them encrypted."""
        code = (code or "").strip()
        if not code:
            raise AuthError("Empty authorization code — paste the code Google displayed.")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                TOKEN_URI,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": self._redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("error_description") or resp.json().get("error") or ""
            except ValueError:
                detail = resp.text[:200]
            raise AuthError(
                f"Token exchange failed (HTTP {resp.status_code}): {detail or 'unknown error'}. "
                "Codes are single-use and expire in a few minutes — restart the /connect "
                "flow to get a fresh one."
            )
        payload = resp.json()
        if not payload.get("refresh_token"):
            raise AuthError(
                "Google did not return a refresh token (no offline access). Re-run the "
                "/connect flow — it always requests prompt=consent + access_type=offline."
            )
        creds = Credentials(
            token=payload.get("access_token"),
            refresh_token=payload.get("refresh_token"),
            token_uri=TOKEN_URI,
            client_id=self._client_id,
            client_secret=self._client_secret,
            scopes=SCOPES,
            expiry=None,
        )
        self._creds = creds
        self._store.save_token(creds.to_json())
        return {"connected": True, "scopes": payload.get("scope", "")}

    def disconnect(self) -> None:
        self._store.delete_token()
        self._creds = None

    # ------------------------------------------------------------------ #
    # Server-side credential loading + refresh
    # ------------------------------------------------------------------ #

    def connected(self) -> bool:
        return self._store.token_exists()

    def validate_stored(self) -> bool:
        """True when stored credentials parse and carry a refresh token.

        Deliberately NO network call (the /health probe must stay cheap and
        quiet toward Google).
        """
        creds = self.load_credentials()
        return bool(creds and creds.refresh_token)

    def load_credentials(self) -> Credentials | None:
        raw = self._store.load_token()
        if not raw:
            return None
        try:
            creds = Credentials.from_authorized_user_info(json.loads(raw))
        except Exception as exc:  # corrupt file — treat as not connected
            raise AuthError(f"Stored credentials are unreadable: {exc}. Re-run /connect.")
        self._creds = creds
        return creds

    async def fresh_credentials(self) -> Credentials:
        """Return credentials with a live access token, refreshing if needed."""
        if self._creds is None:
            self.load_credentials()
        if self._creds is None:
            raise AuthError(
                "Google account not connected. Open <base-url>/connect in a browser and "
                "complete the one-time consent flow."
            )
        creds = self._creds
        if creds.valid:
            return creds
        async with self._refresh_lock:
            if creds.valid:
                return creds
            if not creds.refresh_token:
                raise AuthError(
                    "Stored credentials have no refresh token. Re-run /connect."
                )
            try:
                await asyncio.to_thread(creds.refresh, GoogleAuthRequest())
            except RefreshError as exc:
                raise AuthError(
                    f"Google token refresh failed ({exc.__class__.__name__}). The refresh "
                    "token may have been revoked or the consent withdrawn — re-run "
                    "/connect to re-authorize."
                ) from exc
            self._store.save_token(creds.to_json())
            return creds

    def access_token_sync_hint(self) -> str | None:
        return self._creds.token if self._creds else None
