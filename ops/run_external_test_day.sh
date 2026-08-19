#!/usr/bin/env bash
set -euo pipefail

# Frozen 2026-27 NBA external-test day runner.
#
# Usage:
#   ops/run_external_test_day.sh YYYY-MM-DD [tag]
#
# Optional environment variables:
#   COMBO_SIMULATIONS=20000
#   SKIP_MARKET_PRICING=0
#
# This wrapper is for official regular-season external-test captures.
# Preseason engineering captures should continue to use:
#   python ops/capture_external_test_day.py --mode engineering ...

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 YYYY-MM-DD [tag]" >&2
  exit 2
fi

DATE="$1"
TAG="${2:-morning}"
COMBO_SIMULATIONS="${COMBO_SIMULATIONS:-20000}"
SKIP_MARKET_PRICING="${SKIP_MARKET_PRICING:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: ${PYTHON}" >&2
  exit 2
fi

# Exact ISO date + same-day America/New_York guard.
"${PYTHON}" - "${DATE}" <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

text = sys.argv[1]

try:
    target = datetime.strptime(text, "%Y-%m-%d").date()
except ValueError as exc:
    raise SystemExit(
        f"ERROR: date must be exact YYYY-MM-DD; got {text!r}. {exc}"
    )

today_et = datetime.now(
    ZoneInfo("America/New_York")
).date()

if target != today_et:
    raise SystemExit(
        "ERROR: official external-test wrapper must be run on the "
        "actual target date in America/New_York. "
        f"target={target}, today={today_et}. "
        "Use capture_external_test_day.py --mode engineering for preseason."
    )
PY

echo "======================================================================================================================"
echo "NBA FROZEN EXTERNAL-TEST DAY"
echo "======================================================================================================================"
echo "Date:               ${DATE}"
echo "Tag:                ${TAG}"
echo "Combo simulations:  ${COMBO_SIMULATIONS}"
echo

echo "[1/4] Verify frozen deployment"
"${PYTHON}" scripts/verify_frozen_manifest.py

# Determine whether an official capture already exists for this slate date.
HAS_EXTERNAL_CAPTURE="$("${PYTHON}" - "${DATE}" <<'PY'
import json
import sys
from pathlib import Path

date = sys.argv[1]
root = Path.cwd()
capture_root = (
    root
    / "data/external_test/season=2026"
    / f"date={date}"
    / "captures"
)

found = False

if capture_root.exists():
    for manifest_path in capture_root.glob("*/capture_manifest.json"):
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if payload.get("mode") == "external":
            found = True
            break

print("1" if found else "0")
PY
)"

OPENING_DAY="2026-10-20"

if [[ "${HAS_EXTERNAL_CAPTURE}" == "1" ]]; then
  echo
  echo "[2/4] History refresh SKIPPED"
  echo "Reason: an official capture already exists for ${DATE}."
  echo "Same-day history refreshes are prohibited after the first capture."
elif [[ "${DATE}" == "${OPENING_DAY}" ]]; then
  echo
  echo "[2/4] History refresh SKIPPED"
  echo "Reason: opening-day exception (${OPENING_DAY}); no completed 2026-27 regular-season games exist yet."
else
  echo
  echo "[2/4] Refresh completed 2026-27 standard + advanced history"
  "${PYTHON}" scripts/01_ingest_history_resume.py \
    --start-season 2026 \
    --end-season 2026 \
    --include-advanced \
    --skip-players \
    --force
fi

echo
echo "[3/4] Validate frozen production contract"
"${PYTHON}" scripts/10a_validate_production_contract.py

echo
echo "[4/4] Create immutable contemporaneous capture"

CAPTURE_ARGS=(
  ops/capture_external_test_day.py
  --date "${DATE}"
  --mode external
  --tag "${TAG}"
  --combo-simulations "${COMBO_SIMULATIONS}"
)

if [[ "${SKIP_MARKET_PRICING}" == "1" ]]; then
  CAPTURE_ARGS+=(--skip-market-pricing)
fi

"${PYTHON}" "${CAPTURE_ARGS[@]}"

echo
echo "======================================================================================================================"
echo "EXTERNAL-TEST DAY COMPLETE"
echo "======================================================================================================================"
echo "No automatic betting threshold was used."
echo "Do not refresh historical data again on ${DATE}."
echo "Later same-day quote snapshots may be captured by rerunning this wrapper with a different tag."
