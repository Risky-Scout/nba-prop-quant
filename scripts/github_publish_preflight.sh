#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "GITHUB SOURCE SNAPSHOT PREFLIGHT"
echo "============================================================"

if [[ ! -d .git ]]; then
  echo "INFO: .git does not exist yet; running filesystem checks only."
fi

bad=0

for name in .env .env.local .env.production credentials.json secrets.json; do
  if find . -type f -name "$name" -print -quit | grep -q .; then
    echo "FAIL: secret-like file present: $name"
    bad=1
  fi
done

if find . -type f \( -path '*/data/raw/*' -o -path '*/data/processed/*' -o -path '*/data/external_test/*' \) -print -quit | grep -q .; then
  echo "FAIL: raw/processed/external-test data present in staged source repo."
  bad=1
fi

while IFS= read -r -d '' f; do
  bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [[ "$bytes" -ge 90000000 ]]; then
    echo "FAIL: staged source file is >= 90 MB: $f ($bytes bytes)"
    bad=1
  fi
done < <(find . -type f -print0)

if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
  if git status --porcelain | grep -E '(^|/)\.env($|\.)' >/dev/null 2>&1; then
    echo "FAIL: .env-like file is staged or untracked."
    bad=1
  fi
fi

if [[ "$bad" -ne 0 ]]; then
  echo
  echo "FAIL: GitHub preflight found blocking items."
  exit 1
fi

echo
echo "PASS: no obvious secrets, raw data, or oversized ordinary-Git files found."
