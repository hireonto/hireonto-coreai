#!/usr/bin/env bash
# Creates (first run) or updates the ai-agent VM to the image last built by deploy-api.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source deploy/gcp/config.sh
IMAGE=$(cat deploy/gcp/.last-image)
# Non-secret env: same keys as api.env.yaml, as KEY=value lines.
ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT
sed -n 's/^\([A-Z_][A-Z0-9_]*\): *\(.*\)$/\1=\2/p' deploy/gcp/api.env.yaml > "$ENV_FILE"

SA="$AGENT_VM@$PROJECT_ID.iam.gserviceaccount.com"
if ! gcloud compute instances describe "$AGENT_VM" --project "$PROJECT_ID" --zone "$ZONE" >/dev/null 2>&1; then
  gcloud iam service-accounts describe "$SA" --project "$PROJECT_ID" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "$AGENT_VM" --project "$PROJECT_ID" --display-name "ai-agent VM"
  for role in roles/secretmanager.secretAccessor roles/artifactregistry.reader roles/logging.logWriter; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$SA" --role "$role" --condition=None >/dev/null
  done
  # No inbound ports needed: the worker connects out to LiveKit Cloud and the backend.
  gcloud compute instances create "$AGENT_VM" \
    --project "$PROJECT_ID" --zone "$ZONE" --machine-type "$AGENT_MACHINE_TYPE" \
    --image-family debian-12 --image-project debian-cloud --boot-disk-size 30GB \
    --service-account "$SA" --scopes cloud-platform \
    --metadata "ai-image=$IMAGE" \
    --metadata-from-file "startup-script=deploy/gcp/agent-startup.sh,ai-env=$ENV_FILE"
else
  gcloud compute instances add-metadata "$AGENT_VM" --project "$PROJECT_ID" --zone "$ZONE" \
    --metadata "ai-image=$IMAGE" \
    --metadata-from-file "startup-script=deploy/gcp/agent-startup.sh,ai-env=$ENV_FILE"
  gcloud compute ssh "$AGENT_VM" --project "$PROJECT_ID" --zone "$ZONE" \
    --command 'sudo google_metadata_script_runner startup'
fi
