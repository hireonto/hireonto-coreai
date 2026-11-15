#!/usr/bin/env bash
# VM startup script for ai-agent (runs as root on every boot). It pulls the image named in the
# `ai-image` metadata key and runs the LiveKit worker. Its env file is built from instance metadata
# (non-secret `ai-env`) plus Secret Manager, and is readable by root only.
set -euo pipefail
md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }

command -v docker >/dev/null || { apt-get update && apt-get install -y docker.io; }
IMAGE=$(md instance/attributes/ai-image)
PROJECT=$(md project/project-id)
gcloud auth configure-docker "${IMAGE%%/*}" --quiet

install -d -m 700 /etc/hireonto
umask 077
md instance/attributes/ai-env > /etc/hireonto/agent.env
echo >> /etc/hireonto/agent.env
for var in JWT_INTERNAL_SECRET S3_ACCESS_KEY S3_SECRET_KEY GEMINI_API_KEY GROQ_API_KEY ANTHROPIC_API_KEY \
           CARTESIA_API_KEY OPENAI_API_KEY LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  name=$(echo "$var" | tr 'A-Z_' 'a-z-')
  if value=$(gcloud secrets versions access latest --secret "$name" --project "$PROJECT" 2>/dev/null); then
    printf '%s=%s\n' "$var" "$value" >> /etc/hireonto/agent.env
  fi
done

docker pull "$IMAGE"
docker rm -f ai-agent 2>/dev/null || true
docker run -d --name ai-agent --restart always --env-file /etc/hireonto/agent.env \
  --log-driver gcplogs "$IMAGE" python -m agent.worker start
