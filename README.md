# Colab MCP Bridge

A self-hosted MCP server that lets **any AI agent** (Claude Code, Cursor, ChatGPT connectors, any streamable-HTTP MCP client) execute Python code on **Google Colab GPU runtimes** — including the free tier's T4. The bridge holds your Google OAuth tokens server-side, so the agent never sees Google credentials, and it never needs a client that can perform Google OAuth itself (Google's hosted Colab MCP requires exactly that, which is why ChatGPT and Cursor cannot connect to it directly).

- Repo: https://github.com/lNamelessl/colab-mcp-bridge
- One-click deploy: https://railway.com/deploy/colab-mcp-bridge

## How it works

```
AI agent ──streamable-HTTP──> Colab MCP Bridge (Railway) ──REST v1beta──> colaboratory.googleapis.com
                                │  secret-path + Bearer auth                │
                                │  encrypted refresh token on /data          └─ ConnectionInfo {url, proxy token}
                                └──── Jupyter API (WS channels, Contents) ────> your Colab runtime (T4/L4/A100/CPU)
```

The bridge implements the Colab toolset **directly over the official v1beta REST API + the standard Jupyter API** (the same pattern as Google's colab-vscode reference client): runtime lifecycle via REST, code execution via Jupyter kernel WebSocket channels, file transfer via the Jupyter Contents API. It is orchestration on your own Colab account — it never hosts anything from a runtime and never sends synthetic keep-alives.

## Quick start

1. Deploy to Railway (button above) — **zero deploy-form prompts**: `SECRET_PATH`, `TOKEN_SECRET` and `MCP_TOKEN` are generated per deployment and visible on the service's **Variables** tab.
2. Open `https://<your-domain>/connect` in any browser. Two paste paths, auto-detected:
   - **Consent code**: the page shows a consent URL bound to Google's own cloud-SDK OAuth client (no OAuth client registration needed). Approve the consent (scopes: `openid`, `cloud-platform`, `userinfo.email`, `colaboratory`); the browser lands on a "can't connect" page at `localhost:8080` — expected (nothing listens there). Copy the `code=4/0A...` value from the ADDRESS BAR and paste it.
   - **ADC credential JSON**: already ran `gcloud auth application-default login`? Paste the full authorized_user JSON instead.
3. The bridge exchanges the code (the token request echoes the exact loopback redirect_uri) or imports the JSON, stores the **refresh token encrypted** (Fernet, key derived from `TOKEN_SECRET`) at `/data/token.enc` on your volume, and auto-enables `colaboratory.googleapis.com` on the quota project. google-auth refreshes access tokens server-side forever.
4. Headless alternative: deploy with `GOOGLE_CREDENTIALS_JSON` set to a full authorized_user credential JSON — the bridge stores it at boot and `/connect` is never needed. ADC user credentials need a **quota project** for the Colab API (set `QUOTA_PROJECT_ID` to a GCP project that has the Colab API enabled; a credential's own `quota_project_id` field wins; the /connect flow auto-enables the API on it via ServiceUsage).
5. Point your MCP client at `https://<your-domain>/<SECRET_PATH>/mcp`.

## MCP client configuration

**Claude Code**

```bash
claude mcp add --transport http colab \
  "https://<your-domain>/<SECRET_PATH>/mcp" \
  --header "Authorization: Bearer <MCP_TOKEN>"
```

**Cursor** (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "colab": {
      "url": "https://<your-domain>/<SECRET_PATH>/mcp",
      "headers": { "Authorization": "Bearer <MCP_TOKEN>" }
    }
  }
}
```

**ChatGPT connectors**: Settings → Connectors → Create → MCP Server URL = `https://<your-domain>/<SECRET_PATH>/mcp`. ChatGPT cannot send custom headers, so **delete the `MCP_TOKEN` variable** on the Railway service (Railway redeploys automatically) — the unguessable secret URL path remains the gate.

Replace `<your-domain>`, `<SECRET_PATH>` and `<MCP_TOKEN>` with the values from your Railway service (domain under **Settings → Networking**, variables under **Variables**).

## Tools

| Tool | What it does |
|---|---|
| `colab_execute_code` | Runs Python on the runtime via a Jupyter kernel. Returns stdout/stderr/result. State persists per `session_id`; files persist across sessions. Long jobs: a single call waits up to `timeout_s` (default 240 s, max 280 s — kept below the platform edge's response timeout). A still-running job returns `status: running` plus the `execution_id` and KEEPS DRAINING server-side (2 h budget); re-call with the same `session_id` + `execution_id` to reattach and collect partial/final output until complete. Verified live with a 413 s GPU job past the edge cutoff. Executions on one session run serially. |
| `colab_file_put` | Writes a file via the Jupyter Contents API (base64, <50 MB per request). |
| `colab_file_get` | Reads a file via the Jupyter Contents API (base64 + text preview for text files). |
| `colab_list_runtimes` | Lists the account's runtimes with accelerator specs and proxy-token expiry. |
| `colab_create_runtime` | Assigns a new runtime (`accelerator`: T4/L4/A100/... or NONE for CPU; `shape`: STANDARD/HIGHMEM), waits for the long-running operation, falls back through eligible accelerators on capacity errors. |
| `colab_delete_runtime` | Deletes a runtime (allowMissing semantics). If it was the active one, the next tool call auto-recreates a fresh runtime. |
| `colab_subscription` | Shows the account tier (FREE / PRO / PRO_PLUS). |
| `colab_status` | Bridge + runtime snapshot: connection state, active runtime, token expiry, sessions, running executions. |

Example agent turn:

```
colab_execute_code {"code": "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"}
→ {"status": "ok", "stdout": "True Tesla T4\n", ...}
```

## Variables

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_PATH` | `${{secret(16)}}` (auto) | Secret URL segment of the MCP endpoint. |
| `TOKEN_SECRET` | `${{secret(32)}}` (auto) | Fernet key material for the encrypted token file. Rotating it invalidates `token.enc` (re-run /connect). |
| `MCP_TOKEN` | `${{secret(32)}}` (auto) | Optional Bearer gate on the MCP path. Delete the variable for headerless clients (ChatGPT connectors). |
| `RUNTIME_ID` | unset | Optional: pin a specific runtime id across redeploys. |
| `DEFAULT_ACCELERATOR` | `T4` | Accelerator used when the bridge auto-creates a runtime. |
| `DEFAULT_SHAPE` | `STANDARD` | Runtime shape (STANDARD / HIGHMEM). |
| `EXEC_TIMEOUT_DEFAULT` / `EXEC_TIMEOUT_MAX` | `240` / `280` | Single-call wait bound (edge-safe); longer jobs continue draining server-side and reattach via execution_id. |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` / `GOOGLE_OAUTH_REDIRECT_URI` | Google's public cloud-SDK client / `http://localhost:8080` | Override only if you have an allowlisted Colab API client with your own redirect URI. Google's client accepts any localhost port — set `GOOGLE_OAUTH_REDIRECT_URI` (e.g. `http://localhost:9933`) and the /connect page + token exchange stay in sync automatically. |
| `GOOGLE_CREDENTIALS_JSON` | unset | Headless connect: a full `authorized_user` credential JSON (from `gcloud auth application-default login`). Stored encrypted at boot when no token.enc exists. |
| `QUOTA_PROJECT_ID` | credential's `quota_project_id` or a default | GCP project used as the Colab API quota project (ADC user credentials 403 SERVICE_DISABLED without one). The /connect flow auto-enables the Colab API on it via ServiceUsage. |

## Honest limits

- **Runtime lifetime**: free tier runtimes live at most ~12 h (Pro+ up to 24 h); Google deletes idle runtimes by design. A deleted/reclaimed runtime surfaces as a 404 — the bridge auto-recreates a fresh runtime on the next tool call (session state resets; files on the old runtime are gone).
- **Per-session serial execution**: kernels run one execution at a time per session; concurrent calls queue.
- **File transfer**: <50 MB per Contents-API request (chunk or write via code for bigger payloads).
- **Interactive consent prompts are denied**: code like `drive.mount` triggers a Google auth prompt the bridge cannot complete — it auto-denies with an explanatory error instead of hanging.
- **GPU availability**: free tier T4 is usage-limited and often busy; the bridge falls back through eligible accelerators when Google reports capacity errors. Everything is billed to YOUR Colab account (free tier or your own Pro/Pro+ subscription) — the bridge never resells quota.
- **Colab ToS**: this bridge orchestrates runtimes via Google's official API on your own account (sanctioned batch-compute use). It never hosts live web services from a runtime and never sends synthetic keep-alives — idle runtimes are reclaimed by Google by design.
- **One Google account per bridge**: the OAuth token stored on the volume is account-wide; anyone who can reach the MCP endpoint can run code on YOUR account. Keep the secret path private, keep `MCP_TOKEN` set, rotate `MCP_TOKEN` if leaked, and revoke access via your Google account security page to disconnect.

## Local development

```bash
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt pytest
export SECRET_PATH=localtestpath TOKEN_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
python -m pytest tests -q
python -m app.main          # http://127.0.0.1:8000/connect
```

`DATA_DIR` defaults to `/data` in the container; locally it falls back to `./data`.

## License

Apache-2.0. This project is not affiliated with Google; "Colab" refers to Google Colaboratory.
