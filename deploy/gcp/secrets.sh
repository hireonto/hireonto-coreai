#!/usr/bin/env bash
# Creates/updates Secret Manager secrets from your shell environment, without printing values.
#   export JWT_INTERNAL_SECRET=... S3_ACCESS_KEY=... (only the ones you use), then:  ./deploy/gcp/secrets.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

# Secret names are lower-kebab versions of the env var names.
VARS=(JWT_INTERNAL_SECRET S3_ACCESS_KEY S3_SECRET_KEY GEMINI_API_KEY GROQ_API_KEY ANTHROPIC_API_KEY
      CARTESIA_API_KEY OPENAI_API_KEY LIVEKIT_API_KEY LIVEKIT_API_SECRET)
for var in "${VARS[@]}"; do
  value="${!var:-}"
  [[ -z "$value" ]] && continue
  name=$(echo "$var" | tr 'A-Z_' 'a-z-')
  if ! gcloud secrets describe "$name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud secrets create "$name" --project "$PROJECT_ID" --replication-policy=automatic >/dev/null
  fi
  printf '%s' "$value" | gcloud secrets versions add "$name" --project "$PROJECT_ID" --data-file=- >/dev/null
  echo "stored $name"
done
