#!/usr/bin/env bash
set -euo pipefail

BRANCH="$(git branch --show-current)"

if [[ "${BRANCH}" != research/* ]]; then
  echo "ERROR: current branch must begin with research/"
  echo "Current: ${BRANCH}"
  exit 2
fi

PROJECT_ROOT="$(cd ../../.. && pwd)"

"${PROJECT_ROOT}/.venv/bin/python" \
  research/proper_scoring_audit_2025.py \
  --project-root "${PROJECT_ROOT}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-5000}"
