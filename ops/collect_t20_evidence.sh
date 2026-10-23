#!/usr/bin/env bash
# Copy one slate's durable T-20 evidence into a directory for upload.
#
# The ledger, the preserved predictions and the kept receipts live under the
# production data root and survive the job, which is the whole point of them.
# This makes them readable from the workflow run as well, so a reviewer does
# not need access to the machine to see what a segment recorded.
#
# Read-only with respect to the data root, and never a reason for a job to
# fail: a slate with no evidence yet is the normal state of an off day.
#
#   collect_t20_evidence.sh <data-root> <slate-date> <destination>

set -euo pipefail

DATA_ROOT="${1:-}"
SLATE="${2:-}"
DEST="${3:-}"

if [ -z "$DATA_ROOT" ] || [ -z "$SLATE" ] || [ -z "$DEST" ]; then
  echo "usage: collect_t20_evidence.sh <data-root> <slate-date> <destination>"
  exit 0
fi

EVIDENCE="$DATA_ROOT/processed/t20_evidence"

mkdir -p "$DEST"

if [ -f "$EVIDENCE/$SLATE.jsonl" ]; then
  cp "$EVIDENCE/$SLATE.jsonl" "$DEST/"
  echo "collected the $SLATE ledger"
else
  echo "no ledger recorded for $SLATE"
fi

if [ -d "$EVIDENCE/$SLATE" ]; then
  cp -R "$EVIDENCE/$SLATE" "$DEST/"
  echo "collected the $SLATE receipts and preserved predictions"
else
  echo "no receipts or preserved predictions recorded for $SLATE"
fi
