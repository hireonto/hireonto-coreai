# Copy to deploy/gcp/config.sh (git-ignored) and edit. No secrets here: those live in Secret Manager.
PROJECT_ID="your-gcp-project-id"
REGION="us-east1"                 # keep close to the backend and your LiveKit Cloud region
ZONE="us-east1-b"                 # for the agent VM
REPO="hireonto"                   # Artifact Registry Docker repository
IMAGE_NAME="hireonto-ai"
API_SERVICE="ai-api"
AGENT_VM="ai-agent"
AGENT_MACHINE_TYPE="e2-medium"    # 2 vCPU / 4 GB: VAD, turn detector and DTLN run in memory
