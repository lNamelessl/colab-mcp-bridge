# Deploy and Host

Deploy the Colab MCP Bridge with one click — zero deploy-form prompts (`SECRET_PATH`, `TOKEN_SECRET` and `MCP_TOKEN` are generated per deployment and visible on the service's Variables tab). After the deploy goes active:

1. Open `https://YOUR-DOMAIN.up.railway.app/connect` in any browser (domain under Settings → Networking).
2. Click the consent link — Google's consent page opens (bound to Google's own cloud-SDK OAuth client, so there is no OAuth client registration to do). Approve access for your Colab account (scopes: openid, cloud-platform, userinfo.email, colaboratory).
3. After approving, your browser lands on a "can't connect" page at localhost:8080 — that is expected (nothing listens there). Copy the authorization code from the browser's ADDRESS BAR: the URL ends with ...code=4/0A... — copy everything after code= and paste it into the /connect form. Done: the bridge exchanges the code (echoing the exact loopback redirect_uri) and stores the refresh token encrypted (Fernet, key derived from TOKEN_SECRET) on the /data volume. google-auth refreshes access tokens server-side forever.
4. Point your MCP client at `https://YOUR-DOMAIN.up.railway.app/SECRET_PATH/mcp`.

Client configuration:

- Claude Code: `claude mcp add --transport http colab "https://YOUR-DOMAIN.up.railway.app/SECRET_PATH/mcp" --header "Authorization: Bearer MCP_TOKEN"`
- Cursor (.cursor/mcp.json): a streamable-http entry with `url` set to the endpoint and `headers.Authorization` = `Bearer MCP_TOKEN`.
- ChatGPT connectors: create the connector with the endpoint URL, then delete the `MCP_TOKEN` variable on Railway (ChatGPT cannot send custom headers) — the unguessable secret URL path remains the gate.

Health check: `/health` (unauthenticated, no Google calls) reports token state and the cached runtime. Restart-safe: token and runtime bookkeeping live on the volume, so a redeploy keeps everything.

## About Hosting

One service, one volume. The bridge is a streamable-HTTP MCP server (mcp 2.3.0, stateless JSON mode) that implements Google Colab's toolset directly over the official v1beta REST API plus the standard Jupyter API — the same architecture as Google's colab-vscode reference client: runtime lifecycle via REST, code execution via Jupyter kernel WebSocket channels, file transfer via the Jupyter Contents API. It holds YOUR Google OAuth refresh token (encrypted on the volume) so agents that cannot perform Google OAuth (ChatGPT, Cursor, simple MCP clients) can still drive Colab. A background task refreshes the runtime proxy token ~5 minutes before its ~1 h expiry, exactly like the reference client. Runs as a non-root container; /data holds token.enc (Fernet-encrypted) plus runtime bookkeeping state.

## Why Deploy

Google's hosted Colab MCP endpoint requires Google OAuth at the client — and Google does not support Dynamic Client Registration on it, so ChatGPT, Cursor and most MCP clients cannot connect at all. This bridge converts Google OAuth into secret-path + optional Bearer auth that every MCP client speaks, while keeping your Google credentials server-side and encrypted (the agent never sees them). You get free-tier T4 GPUs (or your own Pro/Pro+ accelerators) behind a plain URL your agent can call, with runtime lifecycle handled automatically: capacity-aware creation with accelerator fallback, token refresh, and auto-recreation when Google reclaims an idle runtime. Per-second billing on Railway means the bridge costs pennies while idle and only works on YOUR account.

## Common Use Cases

- Give Claude Code or Cursor a GPU brain: run model inference, fine-tuning smoke tests, CUDA checks and data crunches on a T4 without owning one.
- Long-running jobs: kick off a training loop, disconnect, and reattach later with the same session + execution id to collect the complete output — the bridge drains the kernel past stream cutoffs.
- File pipelines: push datasets via colab_file_put, process on the runtime, pull results via colab_file_get (base64, under 50 MB chunks).
- Multi-runtime workflows: list runtimes, create a HIGHMEM CPU box for parsing and a T4 for training, delete them when done.
- Account status checks from agents: subscription tier and eligible accelerators via colab_subscription / colab_status.

## Dependencies for

The bridge talks to `colaboratory.googleapis.com` (REST v1beta) and to your Colab runtime's Jupyter server over HTTPS/WSS, so egress to Google is required. Your Google account needs access to Colab (free tier works; Pro/Pro+ unlock more accelerators). Python 3.12 slim base image; pip installs pinned mcp 2.3.0, fastapi, uvicorn, google-auth, httpx, websockets and cryptography at build time. The /data volume persists the encrypted token and runtime bookkeeping across redeploys.

### Deployment Dependencies

- Railway service built from this repo's Dockerfile (python:3.12-slim, non-root, PORT baked).
- Volume mounted at /data (auto-provisioned by the template).
- Generated secret variables at deploy time: SECRET_PATH (16-char url segment), TOKEN_SECRET (32-char Fernet key material), MCP_TOKEN (32-char Bearer gate, delete for headerless clients).
- Google account consent via /connect (the one manual step; refresh token stored encrypted on the volume).
- Runtime environment (created by Google on demand): Colab CPU/T4/L4/A100 runtimes per your tier's eligibility.

## Honest limits

- Runtime lifetime is Google-controlled: free tier ~12 h max, Pro+ up to 24 h; idle runtimes are reclaimed by design. A reclaimed runtime auto-recreates on the next tool call (session state resets, files on the old runtime are gone).
- Executions are serial per session (kernels run one cell at a time); concurrent calls queue.
- File transfer is under 50 MB per Contents-API request.
- Interactive Google consent prompts inside code (e.g. drive.mount) are auto-denied with an explanatory error instead of hanging.
- Colab usage is billed to your own Google account; the bridge is orchestration via the official API on your own account — it never hosts live services from a runtime and never sends synthetic keep-alives.
- The MCP endpoint executes code on your Google account — keep the secret path private and keep MCP_TOKEN set (rotate if leaked). Revoke the app in your Google account security page to disconnect.
