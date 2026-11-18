FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

# Defaults (overridable via Railway variables; Railway injects PORT at runtime):
# - DATA_DIR is the volume mount (/data) holding the encrypted Google token
# - DEFAULT_ACCELERATOR / DEFAULT_SHAPE are used when auto-creating a runtime
# - EXEC_TIMEOUT_DEFAULT / EXEC_TIMEOUT_MAX bound per-call waits for long jobs
ENV PORT=8000 \
    DATA_DIR=/data \
    DEFAULT_ACCELERATOR=T4 \
    DEFAULT_SHAPE=STANDARD \
    EXEC_TIMEOUT_DEFAULT=600 \
    EXEC_TIMEOUT_MAX=3600

# Non-root at runtime: the (root-owned) Railway volume is chowned to the app
# user before dropping privileges, so token.enc can be written to /data.
RUN useradd --create-home --uid 1000 bridge
EXPOSE 8000

CMD ["sh", "-c", "if [ -d /data ]; then chown 1000:1000 /data 2>/dev/null || true; fi; exec setpriv --reuid=1000 --regid=1000 --clear-groups python -m app.main"]
