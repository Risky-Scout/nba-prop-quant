# Branch Safety

The published deployment remains:

```text
nba_prop_quant_20260818T205213Z
```

Safe setup:

```bash
git switch main
git pull --ff-only
git switch -c research/proper-scoring-audit-2025
```

Add only the `research/` scripts on that branch.

Never retag, rewrite, or replace the published production Release as part of this audit.

The audit may motivate a future model version, but any material change requires a new freeze ID.
