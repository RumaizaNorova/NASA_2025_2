#!/usr/bin/env bash
set -euo pipefail

# Render working dir is repo root. This script is invoked from the root.

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}" >&2
    exit 1
  fi
}

download_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  echo "Downloading $(basename "$dest")..."
  mkdir -p "$(dirname "$dest")"
  curl -fsSL -o "$dest" "$url"
}

# You will create a GitHub Release and upload these assets with these exact filenames.
# ARTIFACT_BASE_URL should look like:
#   https://github.com/<owner>/<repo>/releases/download/<tag>
need_env ARTIFACT_BASE_URL

MODEL_URL="${ARTIFACT_BASE_URL%/}/gradientboosting_model.pkl"
FEATURES_URL="${ARTIFACT_BASE_URL%/}/feature_names.pkl"
PERF_URL="${ARTIFACT_BASE_URL%/}/model_performance.json"
DATA_URL="${ARTIFACT_BASE_URL%/}/integrated_data_full.csv"

# Place files in paths that backend/app.py already searches.
download_if_missing "$MODEL_URL" "results_retrained/models/gradientboosting_model.pkl"
download_if_missing "$FEATURES_URL" "results_retrained/models/feature_names.pkl"
download_if_missing "$PERF_URL" "results_retrained/model_performance.json"
download_if_missing "$DATA_URL" "data/integrated_data_full.csv"

cd backend
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"

