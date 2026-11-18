"""Colab MCP Bridge — streamable-HTTP MCP server exposing Google Colab GPU
runtimes to any AI agent (Claude Code, Cursor, ChatGPT connectors).

The bridge holds the user's Google OAuth refresh token server-side (encrypted
on the Railway volume) and implements the Colab toolset DIRECTLY over the
v1beta REST API + standard Jupyter API (the colab-vscode pattern) — it is
orchestration, never a hosted service on a runtime, and it never sends
synthetic keep-alives (idle runtimes are reclaimed by design).

Auth model:
  - the MCP endpoint lives at a secret URL path /<SECRET_PATH>/mcp
  - an optional MCP_TOKEN adds a Bearer check on top (delete the variable for
    clients that cannot send headers, e.g. ChatGPT connectors)
  - one-time user step: open /connect in a browser, complete the gcloud-style
    copy-paste consent, paste the code — the bridge stores the refresh token
    encrypted (TOKEN_SECRET-derived Fernet key) at /data/token.enc
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import html
import json
import os
import re
import sys
import uuid
from urllib.parse import parse_qs

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import AuthError, AuthManager
from .colabapi import ColabAPI, ColabError, RuntimeManager, runtime_id_from_name
from .kernel import ExecRecord, ExecutionRegistry, JupyterClient, JupyterError
from .store import TokenStore

# --------------------------------------------------------------------------- #
# Environment (read once at boot; fail fast with actionable messages)
# --------------------------------------------------------------------------- #

VERSION = "1.0.0"

SECRET_PATH = os.environ.get("SECRET_PATH", "").strip()
# Hyphen tolerated: a custom Railway secret() charset argument is treated as a
# literal character set and may yield hyphens (quirk 56c).
if not re.fullmatch(r"[a-zA-Z0-9-]{8,64}", SECRET_PATH):
    print(
        "FATAL: SECRET_PATH must be 8-64 alphanumeric characters. On Railway it is "
        "generated automatically (${{secret(16)}}); when running manually, "
        "export SECRET_PATH=<your-secret-path>.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

TOKEN_SECRET = os.environ.get("TOKEN_SECRET", "").strip()
MCP_TOKEN = os.environ.get("MCP_TOKEN", "").strip()
RUNTIME_ID_PIN = os.environ.get("RUNTIME_ID", "").strip()
DEFAULT_ACCELERATOR = os.environ.get("DEFAULT_ACCELERATOR", "T4").strip().upper()
DEFAULT_SHAPE = os.environ.get("DEFAULT_SHAPE", "STANDARD").strip().upper()
EXEC_TIMEOUT_DEFAULT = min(max(int(os.environ.get("EXEC_TIMEOUT_DEFAULT", "600")), 30), 3600)
EXEC_TIMEOUT_MAX = min(max(int(os.environ.get("EXEC_TIMEOUT_MAX", "3600")), 60), 7200)
PORT = int(os.environ.get("PORT", "8000"))

MCP_PATH = f"/{SECRET_PATH}/mcp"

STORE = TokenStore(TOKEN_SECRET)
AUTH = AuthManager(STORE)
API = ColabAPI(AUTH)
RUNTIMES = RuntimeManager(API, STORE, pinned_runtime_id=RUNTIME_ID_PIN)
RUNTIMES.configure_defaults(DEFAULT_ACCELERATOR, DEFAULT_SHAPE)
JUPYTER = JupyterClient(lambda: RUNTIMES.current() or {})
REGISTRY = ExecutionRegistry()
SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_REFRESHER_TASK: asyncio.Task | None = None

# Headless connect: GOOGLE_CREDENTIALS_JSON (a full authorized_user credential
# JSON, e.g. from `gcloud auth application-default login`) is stored at boot
# when no token.enc exists yet — keeps fresh-deploy verification / IaC deploys
# fully unattended.
_GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
if _GOOGLE_CREDENTIALS_JSON and not AUTH.connected():
    try:
        result = AUTH.connect_credentials_json(_GOOGLE_CREDENTIALS_JSON)
        print(
            f"boot: stored GOOGLE_CREDENTIALS_JSON (quota project: "
            f"{result.get('quota_project')})",
            flush=True,
        )
    except AuthError as exc:
        print(f"boot: GOOGLE_CREDENTIALS_JSON invalid: {exc}", file=sys.stderr, flush=True)


def _session_lock(session_name: str) -> asyncio.Lock:
    if session_name not in SESSION_LOCKS:
        SESSION_LOCKS[session_name] = asyncio.Lock()
    return SESSION_LOCKS[session_name]


async def _ensure_session(session_name: str) -> dict:
    await RUNTIMES.ensure()
    return await JUPYTER.ensure_session(session_name)


def _with_gone_retry(op):
    """Run op(); on runtime-gone (RECLAIMED / token 404) recreate + retry once."""

    async def run() -> dict:
        runtime = await RUNTIMES.ensure()
        try:
            return await op(runtime)
        except ColabError as exc:
            if exc.kind != "gone":
                raise
            RUNTIMES.invalidate()
        except JupyterError as exc:
            if exc.status != 404:
                raise
            # 404 from the runtime = expired proxy token OR reclaimed runtime.
            refreshed = await RUNTIMES.refresh_connection()
            if refreshed is None:
                RUNTIMES.invalidate()
        runtime = await RUNTIMES.ensure()  # recreates when gone
        return await op(runtime)

    return run()


def _auth_guidance(exc: Exception) -> str:
    if isinstance(exc, AuthError):
        return f"AUTH REQUIRED: {exc}"
    if isinstance(exc, ColabError) and exc.kind == "auth":
        return f"AUTH REJECTED: {exc} If this persists, re-authorize via /connect."
    return f"{exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------- #
# MCP server + tools
# --------------------------------------------------------------------------- #

server = MCPServer(
    "colab-mcp-bridge",
    title="Colab MCP Bridge",
    version=VERSION,
    instructions=(
        "Execute Python code on the user's own Google Colab GPU runtimes. "
        "colab_execute_code runs code in a named session (state persists per "
        "session; files persist across sessions on the runtime). Long jobs: if "
        "the tool returns status=running, re-call with the SAME session_id and "
        "execution_id to reattach and get more output. colab_file_put/get "
        "transfer files via the Jupyter Contents API (<50 MB per request). "
        "colab_list_runtimes / colab_create_runtime / colab_delete_runtime "
        "manage runtimes (creating falls back through eligible accelerators). "
        "colab_subscription shows the account tier. colab_status shows "
        "connection state. The runtime token and Google auth are handled "
        "server-side by the bridge."
    ),
)


def _connected_hint() -> str:
    return (
        "NOT CONNECTED: no Google account is linked to this bridge yet. Open "
        "<base-url>/connect in a browser and complete the one-time Google "
        "consent (copy-paste flow), then retry."
    )


@server.tool(
    name="colab_execute_code",
    description=(
        "Execute Python code on a Google Colab runtime and return stdout, stderr "
        "and the last expression's result. State persists per session_id; files "
        "persist across sessions. For long jobs the tool waits up to timeout_s "
        "(default 600); if the job is still running it returns status=running "
        "plus the execution_id — re-call with the SAME session_id and "
        "execution_id (and the same code or empty reattachment) to reattach and "
        "collect the rest. Executions on one session run serially. Avoid code "
        "that triggers Google consent prompts (e.g. drive.mount) — they are "
        "denied automatically."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
)
async def colab_execute_code(
    code: str,
    session_id: str = "default",
    execution_id: str = "",
    timeout_s: int = EXEC_TIMEOUT_DEFAULT,
    runtime_id: str = "",
) -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    session_name = re.sub(r"[^a-zA-Z0-9._-]", "-", (session_id or "default").strip())[:63] or "default"
    wait_s = float(min(max(int(timeout_s or EXEC_TIMEOUT_DEFAULT), 10), EXEC_TIMEOUT_MAX))
    if runtime_id.strip() and runtime_id.strip() != RUNTIMES.runtime_id():
        RUNTIMES.set_explicit(runtime_id.strip())

    try:
        record = REGISTRY.get(session_name, execution_id) if execution_id else None

        if record is not None:
            # Reattach: same session + execution id.
            if record.done:
                out = dict(record.result or {})
                out.update(
                    {
                        "session_id": session_name,
                        "execution_id": record.execution_id,
                        "reattached": True,
                        "status": out.get("status", "ok"),
                    }
                )
                return out
            budget = min(wait_s, EXEC_TIMEOUT_MAX)
            task = REGISTRY.task_for(session_name, record.execution_id)
            if task is None:
                out = dict(record.result or {})
                out.update(
                    {
                        "session_id": session_name,
                        "execution_id": record.execution_id,
                        "reattached": True,
                        "note": "execution task is gone (bridge restarted since it started)",
                    }
                )
                return out
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=budget)
            except (asyncio.TimeoutError, TimeoutError):
                return _running_snapshot(record, session_name)
            out = dict(record.result or {})
            out.update(
                {
                    "session_id": session_name,
                    "execution_id": record.execution_id,
                    "reattached": True,
                }
            )
            return out

        if not code or not code.strip():
            raise ToolError(
                "code is required (or pass the same session_id + execution_id of a "
                "running execution to reattach to it)."
            )
        execution_id = (execution_id or "").strip() or uuid.uuid4().hex[:12]
        if REGISTRY.get(session_name, execution_id) is not None:
            raise ToolError(
                f"execution_id '{execution_id}' already exists on session "
                f"'{session_name}' — re-calls with it reattach to that execution."
            )

        session = await _with_gone_retry(lambda _rt: _ensure_session(session_name))
        record = ExecRecord(session_name, execution_id, code)
        task = asyncio.create_task(_run_execution(record, session, session_name))
        REGISTRY.register(record, task)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=wait_s)
        except (asyncio.TimeoutError, TimeoutError):
            return _running_snapshot(record, session_name)
        out = dict(record.result or {})
        out.update({"session_id": session_name, "execution_id": execution_id})
        return out
    except (AuthError, ColabError, JupyterError) as exc:
        raise ToolError(_auth_guidance(exc)) from exc


def _running_snapshot(record: ExecRecord, session_name: str) -> dict:
    result = dict(record.result or {})
    return {
        "status": "running",
        "session_id": session_name,
        "execution_id": record.execution_id,
        "started_at": record.started_at,
        "partial_output": {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "result": result.get("result", ""),
        },
        "note": (
            "Still running (executions continue server-side past this call). "
            "Re-call colab_execute_code with the SAME session_id and execution_id "
            "to reattach and collect the complete result."
        ),
    }


async def _run_execution(record: ExecRecord, session: dict, session_name: str) -> None:
    async with _session_lock(session_name):
        try:
            # sink=record.result accumulates output LIVE so reattachment calls
            # can read partial output while the job still runs.
            await JUPYTER.execute(
                record.code,
                session,
                wait_s=float(EXEC_TIMEOUT_MAX),
                sink=record.result,
            )
        except (ColabError, JupyterError) as exc:
            record.result.update(
                {
                    "status": "error",
                    "error": {
                        "ename": exc.__class__.__name__,
                        "evalue": str(exc),
                        "traceback": "",
                    },
                    "notes": (record.result.get("notes") or []) + [_auth_guidance(exc)],
                }
            )
        except Exception as exc:  # noqa: BLE001 — surface anything else as tool error text
            record.result.update(
                {
                    "status": "error",
                    "error": {
                        "ename": exc.__class__.__name__,
                        "evalue": str(exc),
                        "traceback": "",
                    },
                }
            )
        finally:
            record.done = True


@server.tool(
    name="colab_file_put",
    description=(
        "Write a file on the Colab runtime via the Jupyter Contents API. "
        "content_b64 is base64-encoded file bytes (<50 MB per request). "
        "Directories are created implicitly. Files persist across sessions for "
        "the runtime's lifetime."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
)
async def colab_file_put(path: str, content_b64: str) -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    path = (path or "").strip().lstrip("/")
    if not path or ".." in path.split("/"):
        raise ToolError("path must be a relative runtime path without '..' segments.")
    try:
        written = await _with_gone_retry(lambda _rt: JUPYTER.put_file(path, content_b64))
    except (AuthError, ColabError, JupyterError) as exc:
        raise ToolError(_auth_guidance(exc)) from exc
    return {
        "path": written.get("path", path),
        "uploaded": True,
        "approx_size_bytes": len(content_b64) * 3 // 4,
    }


@server.tool(
    name="colab_file_get",
    description=(
        "Read a file from the Colab runtime via the Jupyter Contents API. "
        "Returns content_b64 (always) plus text (when the file is text and "
        "<200 KB), size and mimetype. <50 MB per request."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True),
)
async def colab_file_get(path: str) -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    path = (path or "").strip().lstrip("/")
    if not path or ".." in path.split("/"):
        raise ToolError("path must be a relative runtime path without '..' segments.")
    try:
        data = await _with_gone_retry(lambda _rt: JUPYTER.get_file(path))
    except JupyterError as exc:
        if exc.status == 404:
            raise ToolError(f"File not found on the runtime: {path}") from exc
        raise ToolError(_auth_guidance(exc)) from exc
    except (AuthError, ColabError) as exc:
        raise ToolError(_auth_guidance(exc)) from exc
    content = data.get("content") or ""
    out: dict = {
        "path": data.get("path", path),
        "size": data.get("size"),
        "mimetype": data.get("mimetype"),
        "format": data.get("format"),
        "content_b64": content if data.get("format") == "base64" else None,
    }
    if data.get("format") != "base64" and isinstance(content, str):
        out["text"] = content
        out["content_b64"] = base64.b64encode(content.encode()).decode()
    elif isinstance(content, str):
        try:
            decoded = base64.b64decode(content).decode("utf-8")
            if len(decoded) <= 200_000:
                out["text"] = decoded
        except Exception:  # noqa: BLE001 — binary stays b64-only
            pass
    return out


@server.tool(
    name="colab_list_runtimes",
    description=(
        "List the Colab runtimes assigned to the connected Google account, with "
        "accelerator specs and proxy-token expiry. Marks which runtime the "
        "bridge currently uses."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True),
)
async def colab_list_runtimes() -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    try:
        runtimes = await API.list_runtimes()
    except (AuthError, ColabError) as exc:
        raise ToolError(_auth_guidance(exc)) from exc
    current = RUNTIMES.runtime_id()
    return {
        "current_runtime_id": current,
        "runtimes": [
            {
                "id": runtime_id_from_name(rt.get("name")),
                "accelerator": (rt.get("runtimeSpec") or {}).get("accelerator"),
                "shape": (rt.get("runtimeSpec") or {}).get("shape"),
                "variant": (rt.get("runtimeSpec") or {}).get("variant"),
                "connected": bool(rt.get("connectionInfo")),
                "token_expires_at": (rt.get("connectionInfo") or {}).get("expireTime"),
            }
            for rt in runtimes
        ],
    }


@server.tool(
    name="colab_create_runtime",
    description=(
        "Assign (create) a new Colab runtime. accelerator is a model name like "
        "'T4', 'L4', 'A100' or 'NONE' for CPU; shape is STANDARD or HIGHMEM. "
        "Waits for the runtime to be ready (long-running operation) and falls "
        "back through eligible accelerators on capacity errors. Set "
        "make_default=true to switch the bridge to it."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
)
async def colab_create_runtime(
    accelerator: str = "T4",
    shape: str = "STANDARD",
    runtime_id: str = "",
    make_default: bool = True,
) -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    acc = (accelerator or "T4").strip().upper()
    shp = (shape or "STANDARD").strip().upper()
    if shp not in ("STANDARD", "HIGHMEM", "SHAPE_STANDARD", "SHAPE_HIGHMEM"):
        raise ToolError("shape must be STANDARD or HIGHMEM.")
    rid = (runtime_id or "").strip() or None
    try:
        runtime = await API.create_runtime(
            accelerator=acc, shape=shp, runtime_id=rid
        )
    except ColabError as exc:
        if exc.kind == "capacity":
            eligible = []
            try:
                specs = await API.runtime_specs()
                eligible = [
                    (s.get("key") or {}).get("accelerator")
                    for s in specs
                    if s.get("eligible") and (s.get("key") or {}).get("accelerator")
                ]
            except ColabError:
                pass
            raise ToolError(
                f"{exc} Eligible accelerators for this account: {eligible or 'unknown'}."
            ) from exc
        raise ToolError(_auth_guidance(exc)) from exc
    if make_default:
        RUNTIMES.adopt(runtime)
    spec = runtime.get("runtimeSpec") or {}
    info = runtime.get("connectionInfo") or {}
    return {
        "id": runtime_id_from_name(runtime.get("name")),
        "accelerator": spec.get("accelerator"),
        "shape": spec.get("shape"),
        "variant": spec.get("variant"),
        "is_default": make_default,
        "token_expires_at": info.get("expireTime"),
    }


@server.tool(
    name="colab_delete_runtime",
    description=(
        "Delete a Colab runtime (allowMissing semantics: deleting an already-"
        "gone runtime succeeds). With no runtime_id, deletes the bridge's "
        "current runtime. If the deleted runtime was active, the next tool call "
        "automatically creates a fresh one."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
)
async def colab_delete_runtime(runtime_id: str = "") -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    rid = (runtime_id or "").strip() or RUNTIMES.runtime_id()
    if not rid:
        raise ToolError("No runtime_id given and the bridge has no active runtime.")
    try:
        await API.delete_runtime(rid)
    except ColabError as exc:
        raise ToolError(_auth_guidance(exc)) from exc
    deleted_active = rid == RUNTIMES.runtime_id()
    if deleted_active:
        RUNTIMES.invalidate()
    return {
        "deleted": rid,
        "was_active": deleted_active,
        "note": (
            "The bridge will automatically create a fresh runtime on the next "
            "tool call if it needs one."
            if deleted_active
            else "Runtime deleted; the bridge's active runtime is unchanged."
        ),
    }


@server.tool(
    name="colab_subscription",
    description=(
        "Show the connected Google account's Colab subscription tier "
        "(FREE | PRO | PRO_PLUS) — determines which accelerators are eligible."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True),
)
async def colab_subscription() -> dict:
    if not AUTH.connected():
        return {"error": _connected_hint()}
    try:
        sub = await API.subscription()
    except (AuthError, ColabError) as exc:
        raise ToolError(_auth_guidance(exc)) from exc
    state = STORE.load_state()
    state["tier"] = sub.get("tier")
    STORE.save_state(state)
    return {"tier": sub.get("tier"), "raw": sub}


@server.tool(
    name="colab_status",
    description=(
        "Bridge + runtime snapshot: Google account connection state, active "
        "runtime (id, accelerator, token expiry), known sessions, running "
        "executions, recent reclaims. No code is executed."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
)
async def colab_status() -> dict:
    state = STORE.load_state()
    out: dict = {
        "connected": AUTH.connected(),
        "subscription_tier_cached": state.get("tier"),
        "active_runtime": RUNTIMES.runtime_summary(),
        "running_executions": REGISTRY.running_count(),
        "reclaimed_auto_recreates_since_boot": RUNTIMES.reclaimed_since_boot,
        "sessions": [],
    }
    if AUTH.connected() and RUNTIMES.runtime_id():
        try:
            sessions = await JUPYTER.list_sessions()
            out["sessions"] = sorted(
                [s.get("name") for s in sessions if s.get("name")]
            )
        except (ColabError, JupyterError, AuthError) as exc:
            out["sessions_note"] = f"sessions unavailable: {_auth_guidance(exc)}"
    return out


# --------------------------------------------------------------------------- #
# /connect — the one-time user consent flow (gcloud-style copy-paste)
# --------------------------------------------------------------------------- #

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Colab MCP Bridge — connect</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a2e}}
code,pre{{background:#f1f3f5;padding:2px 6px;border-radius:4px}}
pre{{padding:1rem;overflow-wrap:anywhere}}
a{{color:#0b66c3}}
.badge{{display:inline-block;background:#e7f5ff;color:#1864ab;border-radius:12px;padding:2px 10px;font-size:.85rem}}
form{{margin-top:1rem}} textarea{{width:100%;height:5rem}} button{{padding:.5rem 1.2rem}}
.ok{{background:#ebfbee;color:#2b8a3e;padding:.75rem 1rem;border-radius:6px}}
.err{{background:#fff0f0;color:#c92a2a;padding:.75rem 1rem;border-radius:6px}}
</style></head><body>
<h1>Colab MCP Bridge</h1>
{body}
</body></html>"""

_NOT_CONNECTED_BODY = """<p><span class="badge">option A</span> Open this Google consent URL in any browser (same one you use for Colab is fine):</p>
<p><a href="{consent_url}" rel="noopener noreferrer">Approve access on Google's consent page</a></p>
<pre>{consent_url}</pre>
<p>After approving, your browser will land on a <b>"can't connect" page at localhost:8080</b> — that is expected (nothing listens there). Copy the authorization <b>code from the browser's ADDRESS BAR</b>: the URL ends with <code>...&amp;code=4/0A...</code> — copy everything after <code>code=</code> (it starts with <code>4/0A</code>) and paste it below:</p>
<p><span class="badge">option B</span> Already ran <code>gcloud auth application-default login</code>? Paste the full ADC credential JSON instead (it starts with <code>{{"refresh_token"</code> ... and contains <code>"type": "authorized_user"</code>). The form auto-detects which payload you pasted:</p>
<form method="post" action="/connect">
<label for="code">Authorization code or credential JSON</label><br>
<textarea id="code" name="code" placeholder="4/0A... or the full ADC JSON"></textarea><br>
<button type="submit">Connect</button>
</form>
<p style="color:#666">The bridge exchanges the code server-side and stores the refresh token encrypted on its volume. The agent never sees Google credentials. (Using a different port? Set GOOGLE_OAUTH_REDIRECT_URI to match — Google's client accepts any localhost port.) You can also deploy with GOOGLE_CREDENTIALS_JSON set to skip this page entirely.</p>"""

_CONNECTED_BODY = """<p class="ok">Connected. The Google refresh token is stored encrypted on the bridge's volume.</p>
<p>MCP endpoint (keep it private): <code>https://YOUR-DOMAIN{mcp_path}</code></p>
<p>Health: <a href="/health">/health</a></p>
<p>To re-authorize (e.g. after revoking access): open the consent URL from the connect page again, and paste the <code>code=</code> value from the browser's address bar below:</p>
<form method="post" action="/connect">
<label for="code">Authorization code</label><br>
<textarea id="code" name="code"></textarea><br>
<button type="submit">Re-connect</button>
</form>"""

_ERROR_BODY = """<p class="err">{message}</p>
<p><a href="/connect">Back to the connect page</a></p>"""


def _page(body: str) -> str:
    return _PAGE.format(body=body)


@server.custom_route("/health", methods=["GET"])
async def health(request):  # noqa: ANN001 - Starlette handler
    creds_ok = False
    if AUTH.connected():
        try:
            creds_ok = AUTH.validate_stored()
        except AuthError:
            creds_ok = False
    return JSONResponse(
        {
            "status": "ok",
            "service": "colab-mcp-bridge",
            "version": VERSION,
            "transport": "streamable-http (stateless)",
            "mcp_path": MCP_PATH,
            "token_configured": AUTH.connected(),
            "token_valid": creds_ok,
            "bearer_gate": bool(MCP_TOKEN),
            "runtime_cached": RUNTIMES.runtime_id(),
            "quota_project": AUTH.quota_project() if AUTH.connected() else None,
            "data_dir": STORE.dir,
        }
    )


@server.custom_route("/connect", methods=["GET", "POST"])
async def connect(request):  # noqa: ANN001 - Starlette handler
    want_json = "format=json" in (request.url.query or "")
    if request.method == "GET":
        if want_json:
            return JSONResponse(
                {
                    "connected": AUTH.connected(),
                    "consent_url": AUTH.consent_url(),
                    "quota_project": AUTH.quota_project() if AUTH.connected() else None,
                }
            )
        body = _CONNECTED_BODY.format(mcp_path=MCP_PATH) if AUTH.connected() else _NOT_CONNECTED_BODY.format(
            consent_url=html.escape(AUTH.consent_url())
        )
        return _html(_page(body))

    # POST: paste the authorization code
    raw = await request.body()
    code = ""
    content_type = (request.headers.get("content-type") or "").lower()
    if "json" in content_type:
        try:
            code = (json.loads(raw or b"{}") or {}).get("code", "")
        except ValueError:
            code = ""
    else:
        try:
            code = parse_qs(raw.decode("utf-8", "replace")).get("code", [""])[0]
        except Exception:  # noqa: BLE001 — malformed form
            code = ""
    try:
        if code.startswith("{"):
            result = AUTH.connect_credentials_json(code)
            enable = await AUTH.enable_colab_api(force=True)
            result["colab_api"] = enable
        else:
            result = await AUTH.connect(code)
        _ = result  # page content is identical; result carries quota/enable info
    except AuthError as exc:
        if want_json:
            return JSONResponse({"connected": False, "error": str(exc)}, status_code=400)
        return _html(_page(_ERROR_BODY.format(message=html.escape(str(exc)))))
    if want_json:
        return JSONResponse({"connected": True, "quota_project": AUTH.quota_project()})
    return _html(_page(_CONNECTED_BODY.format(mcp_path=MCP_PATH)))


def _html(content: str) -> HTMLResponse:
    return HTMLResponse(content)


# --------------------------------------------------------------------------- #
# ASGI assembly: bearer gate on the secret path, 404 everywhere else
# --------------------------------------------------------------------------- #


class BearerTokenMiddleware:
    """Enforces 'Authorization: Bearer <MCP_TOKEN>' on the MCP path when
    MCP_TOKEN is set. /health and /connect stay open — the secret URL path is
    the primary gate."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and MCP_TOKEN and scope.get("path") == MCP_PATH:
            auth_header = ""
            for key, value in scope.get("headers", []):
                if key == b"authorization":
                    auth_header = value.decode("latin-1")
                    break
            if not hmac.compare_digest(auth_header, f"Bearer {MCP_TOKEN}"):
                response = JSONResponse(
                    {
                        "error": "unauthorized",
                        "hint": "Send 'Authorization: Bearer <MCP_TOKEN>' "
                        "(MCP_TOKEN is set on this deployment; delete the variable "
                        "if your MCP client cannot send headers).",
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


_ALLOWED_PATHS = {MCP_PATH, "/health", "/connect"}


class WrongPath404:
    """Everything that is not the secret MCP path, /health or /connect gets a
    clean 404 (no enumeration). Also lazily starts the background connection
    refresher on the first request."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._started = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] == "http":
            if scope.get("path") not in _ALLOWED_PATHS:
                body = b'{"error":"not found"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            self._maybe_start_refresher()
        await self._app(scope, receive, send)

    def _maybe_start_refresher(self) -> None:
        global _REFRESHER_TASK
        if not self._started:
            self._started = True
            _REFRESHER_TASK = asyncio.create_task(RUNTIMES.refresher_loop())


app = server.streamable_http_app(
    streamable_http_path=MCP_PATH,
    json_response=True,
    stateless_http=True,
    # The SDK auto-enables DNS-rebinding protection (localhost-only Host
    # allowlist) when the bind host looks local — on a public Railway domain
    # every request would 421 "Invalid Host header". Bind-host 0.0.0.0 keeps
    # the default protection off; the secret URL path + optional MCP_TOKEN
    # bearer remain the gates.
    host="0.0.0.0",
)
app = BearerTokenMiddleware(app)
app = WrongPath404(app)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
