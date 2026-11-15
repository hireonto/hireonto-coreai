#!/usr/bin/env bash
# Builds the image with Cloud Build (no local Docker needed) and deploys ai-api to Cloud Run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/gcp/config.sh

TAG=$(git rev-parse --short HEAD)
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:$TAG"
gcloud builds submit --project "$PROJECT_ID" --region "$REGION" --tag "$IMAGE" .

# Map every secret that exists in Secret Manager to its env var.
SECRETS=()
for var in JWT_INTERNAL_SECRET S3_ACCESS_KEY S3_SECRET_KEY GEMINI_API_KEY GROQ_API_KEY ANTHROPIC_API_KEY \
           CARTESIA_API_KEY OPENAI_API_KEY LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  name=$(echo "$var" | tr 'A-Z_' 'a-z-')
  gcloud secrets describe "$name" --project "$PROJECT_ID" >/dev/null 2>&1 && SECRETS+=("$var=$name:latest")
done

# --allow-unauthenticated: Google IAM is open because the backend (outside GCP) authenticates with the
# internal JWT; every route except /health rejects requests without it.
gcloud run deploy "$API_SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE" \
  --allow-unauthenticated \
  --port 8080 --cpu 1 --memory 1Gi --cpu-boost \
  --concurrency 20 --timeout 300 --min-instances 0 --max-instances 5 \
  --env-vars-file deploy/gcp/api.env.yaml \
  --set-secrets "$(IFS=,; echo "${SECRETS[*]}")"

echo "$IMAGE" > deploy/gcp/.last-image
gcloud run services describe "$API_SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)'
