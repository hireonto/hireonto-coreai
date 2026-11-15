# HireOnto AI — one image, two Railway services:
#   ai-api   : uvicorn (default CMD)               → railway.api.json
#   ai-agent : python -m agent.worker start (Tovia) → railway.agent.json
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
# Pre-download Silero VAD + turn-detector models so the voice worker starts fast.
RUN python -m agent.worker download-files
EXPOSE 8000
# Bind to :: so Railway private networking (IPv6) and the public proxy both reach it.
CMD ["sh", "-c", "uvicorn app.main:app --host :: --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
