"""Colab REST v1beta client + runtime lifecycle (the colab-vscode pattern).

Endpoints (https://colaboratory.googleapis.com):
  POST   /v1beta/runtimes?requestId=<uuid4>[&runtimeId=<id>]   (LRO operation)
  GET    /v1beta/runtimes/{id}          (also refreshes the proxy token)
  GET    /v1beta/runtimes               (list)
  DELETE /v1beta/runtimes/{id}?allowMissing=true
  GET    /v1beta/runtimespecs           (per-tier eligibility in `eligible`)
  GET    /v1beta/subscription           (tier: FREE | PRO | PRO_PLUS)
  GET    /v1/operations/{id}:wait?timeout=NNs   (LRO best-effort wait)

Runtime.connectionInfo carries the Jupyter base url + the short-lived
X-Colab-Runtime-Proxy-Token (~1 h; expired token surfaces as HTTP 404).
RuntimeManager refreshes it via getRuntime ~5 min before expiry (the
colab-vscode ConnectionRefresher cadence) and treats getRuntime 404 as
runtime-gone (RECLAIMED) — the next tool call auto-recreates a runtime.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from .auth import AuthManager
from .store import TokenStore

API_ROOT = "https://colaboratory.googleapis.com"
USER_AGENT = "colab-mcp-bridge/1.0"

VARIANT_CPU = "VARIANT_CPU"
VARIANT_GPU = "VARIANT_GPU"
VARIANT_TPU = "VARIANT_TPU"

CPU_ACCELERATORS = {"NONE", "CPU", ""}
REQUEST_TIMEOUT = httpx.Timeout(60.0)
LRO_WAIT_SLICE = "60s"
LRO_TOTAL_BUDGET_S = 300.0
TOKEN_REFRESH_BUFFER = timedelta(minutes=5)


class ColabError(Exception):
    """Anticipated Colab REST failure, classified for retry/auto-recreate."""

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind  # auth | gone | capacity | rate | http
        self.status = status


def variant_for_accelerator(accelerator: str) -> str:
    acc = (accelerator or "").strip().upper()
    if acc.startswith("TPU"):
        return VARIANT_TPU
    if acc in CPU_ACCELERATORS:
        return VARIANT_CPU
    return VARIANT_GPU


def _parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def runtime_id_from_name(name: str | None) -> str | None:
    if not name:
        return None
    return name.split("/")[-1]


class ColabAPI:
    """Thin async client over the v1beta REST surface."""

    def __init__(self, auth: AuthManager) -> None:
        self._auth = auth
        self._http: httpx.AsyncClient | None = None

    async def client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=API_ROOT,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT, "X-Goog-Api-Client": USER_AGENT},
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict | None:
        client = await self.client()
        for attempt in (1, 2):
            creds = await self._auth.fresh_credentials()
            headers = {
                "Authorization": f"Bearer {creds.token}",
                # ADC user credentials need a quota project for
                # colaboratory.googleapis.com (quirk: 403 SERVICE_DISABLED
                # "requires a quota project" without it).
                "x-goog-user-project": self._auth.quota_project(),
            }
            resp = await client.request(
                method, path, params=params, json=json_body, headers=headers
            )
            if resp.status_code in (200, 201):
                return resp.json() if resp.content else {}
            if resp.status_code == 204:
                return None

            message, reason = _google_error(resp)
            # Self-heal: ADC credentials without the Colab API enabled on the
            # quota project 403 with reason SERVICE_DISABLED — enable once and
            # retry the original call.
            if (
                attempt == 1
                and resp.status_code == 403
                and (reason or "").upper() in ("SERVICE_DISABLED", "PERMISSION_DENIED")
            ):
                enable = await self._auth.enable_colab_api(force=True)
                if enable.get("enabled"):
                    continue
            if resp.status_code in (401, 403):
                raise ColabError(
                    "auth",
                    message or f"Google authorization rejected (HTTP {resp.status_code})",
                    status=resp.status_code,
                )
            if resp.status_code == 404:
                raise ColabError(
                    "gone",
                    message or "Resource not found (runtime expired/reclaimed or never existed)",
                    status=404,
                )
            if resp.status_code == 429:
                raise ColabError("rate", message or "Rate limited by Google", status=429)
            if resp.status_code == 400 and "PERMISSION" in (reason or "").upper():
                raise ColabError("auth", message or reason or "Permission denied", status=400)
            raise ColabError(
                "http",
                message or f"Colab API error (HTTP {resp.status_code})",
                status=resp.status_code,
            )

    # ------------------------------------------------------------------ #
    # v1beta surface
    # ------------------------------------------------------------------ #

    async def list_runtimes(self) -> list[dict]:
        data = await self.request("GET", "/v1beta/runtimes")
        return data.get("runtimes") or []

    async def get_runtime(self, runtime_id: str) -> dict:
        return await self.request("GET", f"/v1beta/runtimes/{runtime_id}")

    async def delete_runtime(self, runtime_id: str) -> None:
        await self.request(
            "DELETE",
            f"/v1beta/runtimes/{runtime_id}",
            params={"allowMissing": "true"},
        )

    async def subscription(self) -> dict:
        return await self.request("GET", "/v1beta/subscription") or {}

    async def runtime_specs(self) -> list[dict]:
        data = await self.request("GET", "/v1beta/runtimespecs")
        return data.get("runtimeSpecs") or []

    async def create_runtime(
        self,
        *,
        accelerator: str,
        shape: str = "STANDARD",
        variant: str | None = None,
        runtime_id: str | None = None,
    ) -> dict:
        params: dict = {"requestId": str(uuid.uuid4())}
        if runtime_id:
            params["runtimeId"] = runtime_id
        body = {
            "runtimeSpec": {
                "variant": variant or variant_for_accelerator(accelerator),
                "accelerator": (accelerator or "NONE").strip().upper(),
                "shape": shape,
            }
        }
        op = await self.request("POST", "/v1beta/runtimes", params=params, json_body=body)
        return await self._await_operation(op)

    async def _await_operation(self, op: dict) -> dict:
        deadline = asyncio.get_event_loop().time() + LRO_TOTAL_BUDGET_S
        while True:
            if op.get("done"):
                if op.get("error"):
                    raise ColabError(
                        "capacity",
                        f"Runtime creation failed: {op['error'].get('message', op['error'])}",
                    )
                response = op.get("response")
                if not response:
                    raise ColabError("http", "Runtime creation returned no runtime payload")
                return response
            if asyncio.get_event_loop().time() > deadline:
                raise ColabError(
                    "http",
                    "Timed out waiting for runtime creation (LRO still running; try again)",
                )
            op_name = (op.get("name") or "").strip("/")
            if not op_name:
                raise ColabError("http", "Runtime creation returned no operation name")
            op = await self.request(
                "GET", f"/v1/{op_name}:wait", params={"timeout": LRO_WAIT_SLICE}
            )


def _google_error(resp: httpx.Response) -> tuple[str | None, str | None]:
    try:
        err = resp.json().get("error", {})
    except ValueError:
        return None, None
    message = err.get("message")
    reason = None
    for detail in err.get("details") or []:
        if isinstance(detail, dict) and "errors" in detail:
            reason = detail["errors"][0].get("reason", reason)
        if isinstance(detail, dict) and detail.get("@type", "").endswith("ErrorInfo"):
            reason = detail.get("reason", reason)
    return message, reason


class RuntimeManager:
    """Caches the default runtime, refreshes its proxy token, handles RECLAIMED."""

    def __init__(self, api: ColabAPI, store: TokenStore, pinned_runtime_id: str = "") -> None:
        self._api = api
        self._store = store
        self._pinned = pinned_runtime_id
        self._pinned_accelerator = ""
        self._pinned_shape = "STANDARD"
        self._explicit: str | None = None
        self._runtime: dict | None = None
        self._lock = asyncio.Lock()
        self.reclaimed_since_boot = 0

    # ---- accessors ---- #

    def runtime_id(self) -> str | None:
        return runtime_id_from_name(self._runtime.get("name")) if self._runtime else None

    def connection_info(self) -> dict | None:
        return self._runtime.get("connectionInfo") if self._runtime else None

    def current(self) -> dict | None:
        return self._runtime

    def set_explicit(self, runtime_id: str) -> None:
        """Pin subsequent ensure() calls to an explicitly requested runtime."""
        self._explicit = runtime_id
        self.invalidate()

    def adopt(self, runtime: dict) -> None:
        """Make an existing runtime object the active one (create_runtime tool)."""
        self._set(runtime)

    def runtime_summary(self) -> dict | None:
        if not self._runtime:
            return None
        info = self.connection_info() or {}
        spec = self._runtime.get("runtimeSpec") or {}
        return {
            "id": self.runtime_id(),
            "accelerator": spec.get("accelerator"),
            "shape": spec.get("shape"),
            "variant": spec.get("variant"),
            "token_expires_at": info.get("expireTime"),
        }

    def _token_expiry(self) -> datetime | None:
        return _parse_expiry((self.connection_info() or {}).get("expireTime"))

    def _set(self, runtime: dict) -> None:
        self._runtime = runtime
        rid = runtime_id_from_name(runtime.get("name"))
        if rid:
            state = self._store.load_state()
            state["runtime_id"] = rid
            self._store.save_state(state)

    def invalidate(self) -> None:
        self._runtime = None

    # ---- lifecycle ---- #

    async def refresh_connection(self) -> dict | None:
        """GetRuntime on the cached runtime; None when it is gone (404)."""
        rid = self.runtime_id()
        if not rid:
            return None
        try:
            runtime = await self._api.get_runtime(rid)
        except ColabError as exc:
            if exc.kind == "gone":
                self.invalidate()
                return None
            raise
        if runtime.get("connectionInfo"):
            self._set(runtime)
            return runtime
        self.invalidate()
        return None

    async def ensure(self) -> dict:
        """Return a runtime with a live connectionInfo, creating when needed."""
        async with self._lock:
            expiry = self._token_expiry()
            if self._runtime and self.connection_info() and expiry:
                remaining = expiry - datetime.now(timezone.utc)
                if remaining > timedelta(seconds=30):
                    if remaining < TOKEN_REFRESH_BUFFER + timedelta(minutes=1):
                        await self.refresh_connection()
                    return self._runtime  # type: ignore[return-value]

            wanted_id = self._explicit or self._pinned or self._store.load_state().get("runtime_id")
            if wanted_id and not self._runtime:
                try:
                    runtime = await self._api.get_runtime(str(wanted_id))
                except ColabError as exc:
                    if exc.kind != "gone":
                        raise
                    runtime = None  # RECLAIMED or expired — recreate below
                if runtime and runtime.get("connectionInfo"):
                    self._set(runtime)
                    return runtime

            if not self._runtime and not self._explicit:
                try:
                    existing = await self._api.list_runtimes()
                except ColabError as exc:
                    if exc.kind in ("auth", "gone"):
                        raise
                    existing = []
                for rt in existing:
                    if rt.get("connectionInfo"):
                        self._set(rt)
                        return rt

            runtime = await self._create_default()
            self.reclaimed_since_boot += 1
            self._set(runtime)
            return runtime

    async def _create_default(self) -> dict:
        accelerator = (self._pinned_accelerator or "T4").strip().upper()
        create_id = self._pinned or self._explicit or None
        chain: list[str] = [accelerator]
        try:
            eligible = [
                (s.get("key") or {}).get("accelerator", "").upper()
                for s in await self._api.runtime_specs()
                if s.get("eligible") and (s.get("key") or {}).get("accelerator")
            ]
            chain += [acc for acc in eligible if acc and acc not in chain]
        except ColabError:
            pass
        chain += ["NONE"] if "NONE" not in chain else []
        chain = chain[:5]

        last_error: ColabError | None = None
        for acc in chain:
            try:
                return await self._api.create_runtime(
                    accelerator=acc,
                    shape=self._pinned_shape,
                    runtime_id=create_id,
                )
            except ColabError as exc:
                last_error = exc
                if exc.kind not in ("capacity", "rate"):
                    raise
        raise last_error or ColabError("http", "Runtime creation failed")

    def configure_defaults(self, accelerator: str, shape: str) -> None:
        self._pinned_accelerator = accelerator
        self._pinned_shape = shape

    # ---- background refresher (proxy-token maintenance, colab-vscode style) ---- #

    async def refresher_loop(self) -> None:
        """Refresh the cached runtime's proxy token ~5 min before expiry.

        This is connection maintenance exactly like colab-vscode's
        ConnectionRefresher — it never executes code or pings the kernel
        (no synthetic keep-alive; idle runtimes are reclaimed by design).
        """
        while True:
            await asyncio.sleep(60)
            try:
                expiry = self._token_expiry()
                if self._runtime and expiry:
                    remaining = expiry - datetime.now(timezone.utc)
                    if remaining < TOKEN_REFRESH_BUFFER:
                        await self.refresh_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — refresher must never die
                print(f"refresher: {exc.__class__.__name__}: {exc}", flush=True)
