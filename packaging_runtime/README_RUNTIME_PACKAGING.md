# Runtime Packaging Kit v1

Expected branch: `packaging/runtime-bundle-v1`.

This kit builds a ready-to-run packaging layer for the existing frozen model.
It does not modify the statistical freeze or overwrite the existing production
Release.

Example:

```bash
PROJECT_ROOT="$(cd ../../.. && pwd)"
RUNTIME_OUT="$PROJECT_ROOT/dist/github_runtime/nba_prop_quant_20260818T205213Z_runtime_bundle_v1"

"$PROJECT_ROOT/.venv/bin/python"   packaging_runtime/build_runtime_bundle_v1.py   --package-root "$PKG_DIR"   --output-dir "$RUNTIME_OUT"
```
