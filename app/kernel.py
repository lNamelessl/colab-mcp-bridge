"""Jupyter client for Colab runtimes: WS kernel channels + Contents API.

Colab runtimes ARE Jupyter servers (docs verbatim: "This is essentially a
Jupyter server that runs on Google infrastructure. Clients should use the
connection_info provided to connect to the runtime, and leverage the standard
Jupyter REST API and/or client library"). Every HTTP request and WS upgrade
carries the ``X-Colab-Runtime-Proxy-Token`` header (colab-vscode's
ColabWebSocket pattern), plus ``X-Colab-Client-Agent``.

Execution model:
  - POST /api/sessions -> session + kernel; kernel messages over
    WS /api/kernels/{kernel_id}/channels (shell + iopub + stdin multiplexed)
  - execute_request on shell; outputs collected from iopub; completion when
    the shell execute_reply arrives (status idle fallback)
  - the 600 s stream cutoff Google's hosted MCP has does NOT exist here: the
    bridge keeps draining the socket and waits for kernel idle for as long as
    timeout_s allows; if a tool call's wait budget runs out, the execution
    keeps running in the bridge and a re-call with the SAME session_id +
    execution_id reattaches to it (partial output + running flag returned)
  - colab_request auth prompts (e.g. drive.mount consent) are DENIED
    automatically with a colab_reply error so executions can't hang forever;
    plain input_request prompts are answered with an empty string + note
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
import websockets

MAX_FILE_BYTES = 48 * 1024 * 1024  # <50 MB per Contents-API request
REQUEST_TIMEOUT = httpx.Timeout(60.0)


class JupyterError(Exception):
    """Anticipated Jupyter-API failure."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _proxy_headers(token: str) -> dict[str, str]:
    return {
        "X-Colab-Runtime-Proxy-Token": token,
        "X-Colab-Client-Agent": "colab-mcp-bridge",
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JupyterClient:
    """Per-runtime Jupyter API client (base url + proxy token come from the manager)."""

    def __init__(self, runtime_provider) -> None:  # () -> dict (runtime w/ connectionInfo)
        self._provider = runtime_provider
        self._http: httpx.AsyncClient | None = None

    def _conn(self) -> tuple[str, str]:
        runtime = self._provider()
        info = runtime.get("connectionInfo") or {}
        base = (info.get("url") or "").rstrip("/")
        token = info.get("token") or ""
        if not base or not token:
            raise JupyterError("Runtime has no connectionInfo — recreate the runtime")
        return base, token

    async def client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #

    async def list_sessions(self) -> list[dict]:
        base, token = self._conn()
        client = await self.client()
        resp = await client.get(
            f"{base}/api/sessions", headers=_proxy_headers(token)
        )
        if resp.status_code != 200:
            raise JupyterError(
                f"GET /api/sessions failed (HTTP {resp.status_code})", resp.status_code
            )
        return resp.json() or []

    async def ensure_session(self, session_name: str) -> dict:
        """Return the Jupyter session dict for a named bridge session."""
        sessions = await self.list_sessions()
        for s in sessions:
            if s.get("name") == session_name:
                return s
        base, token = self._conn()
        client = await self.client()
        payload = {
            "path": f"bridge-{session_name}.ipynb",
            "name": session_name,
            "type": "notebook",
            "kernel": {"name": "python3"},
        }
        resp = await client.post(
            f"{base}/api/sessions", json=payload, headers=_proxy_headers(token)
        )
        if resp.status_code in (400, 500):
            # Kernel spec name may differ on Colab images — retry with the
            # server's default kernel.
            payload.pop("kernel")
            resp = await client.post(
                f"{base}/api/sessions", json=payload, headers=_proxy_headers(token)
            )
        if resp.status_code not in (200, 201):
            raise JupyterError(
                f"POST /api/sessions failed (HTTP {resp.status_code}): {resp.text[:200]}",
                resp.status_code,
            )
        return resp.json()

    async def delete_session(self, session_name: str) -> bool:
        sessions = await self.list_sessions()
        base, token = self._conn()
        client = await self.client()
        deleted = False
        for s in sessions:
            if s.get("name") == session_name and s.get("id"):
                resp = await client.delete(
                    f"{base}/api/sessions/{s['id']}", headers=_proxy_headers(token)
                )
                deleted = deleted or resp.status_code in (200, 204)
        return deleted

    # ------------------------------------------------------------------ #
    # Contents API (file put/get, base64, <50 MB per request)
    # ------------------------------------------------------------------ #

    async def put_file(self, path: str, content_b64: str) -> dict:
        base, token = self._conn()
        decoded_size = len(content_b64) * 3 // 4
        if decoded_size > MAX_FILE_BYTES:
            raise JupyterError(
                f"content_b64 decodes to ~{decoded_size // (1024 * 1024)} MB; the "
                "Contents API accepts <50 MB per request. Split the payload or "
                "write the file via execute_code in chunks."
            )
        client = await self.client()
        # Jupyter Contents API does NOT create ancestor directories implicitly
        # (Colab runtime 500s with "[Errno 2] No such file or directory").
        clean = path.lstrip("/")
        parts = [p for p in clean.split("/") if p and p != "."]
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            await client.put(
                f"{base}/api/contents/{quote(d)}",
                json={"type": "directory"},
                headers=_proxy_headers(token),
            )
        url = f"{base}/api/contents/{quote(clean)}"
        resp = await client.put(
            url,
            json={"type": "file", "format": "base64", "content": content_b64},
            headers=_proxy_headers(token),
        )
        if resp.status_code not in (200, 201):
            raise JupyterError(
                f"PUT /api/contents/{path} failed (HTTP {resp.status_code}): "
                f"{resp.text[:200]}",
                resp.status_code,
            )
        return resp.json() or {"path": path}

    async def get_file(self, path: str) -> dict:
        base, token = self._conn()
        client = await self.client()
        url = f"{base}/api/contents/{quote(path.lstrip('/'))}"
        resp = await client.get(
            url, params={"content": "1"}, headers=_proxy_headers(token)
        )
        if resp.status_code == 404:
            raise JupyterError(f"File not found: {path}", 404)
        if resp.status_code != 200:
            raise JupyterError(
                f"GET /api/contents/{path} failed (HTTP {resp.status_code})",
                resp.status_code,
            )
        data = resp.json() or {}
        return {
            "path": data.get("path", path),
            "size": data.get("size"),
            "mimetype": data.get("mimetype"),
            "format": data.get("format"),
            "content": data.get("content"),
        }

    # ------------------------------------------------------------------ #
    # Kernel execution over WS channels
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        code: str,
        session: dict,
        *,
        wait_s: float,
        sink: dict | None = None,
        on_note=None,
    ) -> dict:
        """Run code and wait up to wait_s; raises asyncio.TimeoutError if the
        execution is still running when the wait budget is exhausted.

        ``sink`` (optional) is the dict outputs accumulate into live — the
        caller (execution record) can read partial output while the job runs.
        """
        kernel_id = (session.get("kernel") or {}).get("id")
        if not kernel_id:
            raise JupyterError("Session payload missing kernel id")
        base, token = self._conn()
        # ONE client session id for both the channels URL (?session_id=) and
        # every message header.session — jupyterlab semantics. Colab's proxy
        # closes sockets whose message session and URL session disagree
        # (verified live: abrupt ConnectionClosedError after the first output).
        client_session = uuid.uuid4().hex
        ws_url = (
            f"{base.replace('https://', 'wss://').replace('http://', 'ws://')}"
            f"/api/kernels/{kernel_id}/channels?session_id={quote(client_session)}"
        )
        msg_id = uuid.uuid4().hex
        started = asyncio.get_event_loop().time()

        result: dict = sink if sink is not None else {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": "",
            "error": None,
            "execution_count": None,
            "notes": [],
        }

        def note(text: str) -> None:
            result["notes"].append(text)
            if on_note:
                on_note(text)

        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "colab-mcp-bridge",
                "session": client_session,
                "date": _now_iso(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": False,
            },
            "buffers": [],
            "channel": "shell",
        }

        async def _deny_stdin(ws, msg: dict) -> None:
            msg_type = (msg.get("header") or {}).get("msg_type")
            reply = {
                "header": {
                    "msg_id": uuid.uuid4().hex,
                    "username": "colab-mcp-bridge",
                    "session": client_session,
                    "date": _now_iso(),
                    "msg_type": "input_reply",
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "buffers": [],
                "channel": "stdin",
            }
            if msg_type == "colab_request":
                reply["content"] = {
                    "value": {
                        "type": "colab_reply",
                        "colab_msg_id": (msg.get("metadata") or {}).get("colab_msg_id"),
                        "error": (
                            "colab-mcp-bridge: interactive Google auth flows are not "
                            "available over the bridge. Avoid code that triggers "
                            "Drive/service-account consent prompts (e.g. drive.mount)."
                        ),
                    }
                }
                note(
                    "[bridge] DENIED an interactive Google auth prompt (colab_request). "
                    "The execution may continue without Drive access."
                )
            else:  # plain input_request
                reply["content"] = {"value": ""}
                note("[bridge] stdin prompt auto-answered with an empty string.")
            await ws.send(json.dumps(reply))

        async with websockets.connect(
            ws_url,
            additional_headers=_proxy_headers(token),
            open_timeout=30,
            max_size=64 * 1024 * 1024,
        ) as ws:
            await ws.send(json.dumps(execute_request))
            reply_seen = False
            while True:
                remaining = wait_s - (asyncio.get_event_loop().time() - started)
                if remaining <= 0:
                    result["status"] = "timeout"
                    return result
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    result["status"] = "timeout"
                    return result
                except websockets.exceptions.ConnectionClosed as exc:
                    result["status"] = "error"
                    result["error"] = {
                        "ename": exc.__class__.__name__,
                        "evalue": str(exc),
                        "traceback": "",
                    }
                    result["notes"].append(
                        "[bridge] kernel socket closed before the execute_reply; "
                        "partial output kept. Re-run the call to retry."
                    )
                    return result
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                parent = (msg.get("parent_header") or {}).get("msg_id")
                header = msg.get("header") or {}
                msg_type = header.get("msg_type")
                channel = msg.get("channel")
                content = msg.get("content") or {}

                if msg_type in ("input_request", "colab_request") and channel == "stdin":
                    await _deny_stdin(ws, msg)
                    continue
                if parent != msg_id:
                    continue

                if msg_type == "stream":
                    text = content.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if content.get("name") == "stderr":
                        result["stderr"] += text
                    else:
                        result["stdout"] += text
                elif msg_type == "execute_result":
                    data = content.get("data") or {}
                    result["result"] = data.get("text/plain", result["result"])
                    result["execution_count"] = content.get("execution_count")
                elif msg_type == "display_data":
                    data = content.get("data") or {}
                    if "text/plain" in data and not result["result"]:
                        result["result"] = data["text/plain"]
                elif msg_type == "error":
                    tb = content.get("traceback") or []
                    if isinstance(tb, list):
                        tb = "\n".join(tb)
                    result["error"] = {
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": tb,
                    }
                    result["status"] = "error"
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    if reply_seen:
                        result["duration_s"] = round(
                            asyncio.get_event_loop().time() - started, 2
                        )
                        return result
                elif msg_type == "execute_reply":
                    reply_seen = True
                    result["execution_count"] = content.get("execution_count")
                    reply_status = content.get("status", "ok")
                    if reply_status == "error" and result["error"] is None:
                        payload = content.get("error") or {}
                        result["error"] = {
                            "ename": payload.get("ename", "Error"),
                            "evalue": payload.get("evalue", ""),
                            "traceback": "\n".join(payload.get("traceback") or []),
                        }
                        result["status"] = "error"
                    if reply_status == "aborted":
                        result["status"] = "error"
                        result["error"] = {"ename": "Aborted", "evalue": "", "traceback": ""}
                    result["duration_s"] = round(
                        asyncio.get_event_loop().time() - started, 2
                    )
                    # Keep draining briefly for trailing iopub (flush outputs).
                    try:
                        while True:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                            trailing = json.loads(raw)
                            t_parent = (trailing.get("parent_header") or {}).get("msg_id")
                            t_type = (trailing.get("header") or {}).get("msg_type")
                            t_content = trailing.get("content") or {}
                            if t_parent == msg_id and t_type == "stream":
                                text = t_content.get("text", "")
                                if isinstance(text, list):
                                    text = "".join(text)
                                if t_content.get("name") == "stderr":
                                    result["stderr"] += text
                                else:
                                    result["stdout"] += text
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
                    return result


class ExecRecord:
    """Tracks one execution so identical re-calls can reattach.

    ``result`` is created upfront and filled live by the kernel drain, so
    reattachment calls can read partial output while the job still runs.
    """

    def __init__(self, session_name: str, execution_id: str, code: str) -> None:
        self.session_name = session_name
        self.execution_id = execution_id
        self.code = code
        self.started_at = _now_iso()
        self.done = False
        self.result: dict = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": "",
            "error": None,
            "execution_count": None,
            "notes": [],
        }


class ExecutionRegistry:
    """In-memory execution registry (reattach handle = session_id + execution_id)."""

    MAX_RECORDS = 64

    def __init__(self) -> None:
        self._records: dict[str, ExecRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def key(session_name: str, execution_id: str) -> str:
        return f"{session_name}|{execution_id}"

    def get(self, session_name: str, execution_id: str) -> ExecRecord | None:
        return self._records.get(self.key(session_name, execution_id))

    def task_for(self, session_name: str, execution_id: str) -> asyncio.Task | None:
        return self._tasks.get(self.key(session_name, execution_id))

    def register(self, record: ExecRecord, task: asyncio.Task) -> None:
        self._records[self.key(record.session_name, record.execution_id)] = record
        self._tasks[self.key(record.session_name, record.execution_id)] = task
        # prune done records beyond the cap
        done_keys = [k for k, r in self._records.items() if r.done]
        overflow = len(self._records) - self.MAX_RECORDS
        for k in done_keys[:overflow]:
            self._records.pop(k, None)
            old_task = self._tasks.pop(k, None)
            if old_task:
                old_task.cancel()

    def running_count(self) -> int:
        return sum(1 for r in self._records.values() if not r.done)
