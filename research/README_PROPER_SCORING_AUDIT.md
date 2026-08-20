# Frozen 2025 Proper-Scoring Research Audit

This research branch audits the already-frozen model. It does **not** retrain, recalibrate, change dependence, change thresholds, or alter the published deployment.

Primary branch:

```text
research/proper-scoring-audit-2025
```

Primary unit:

```text
game_id / player_id / prop_type / line_value
```

Pushes are excluded from binary Brier/log-loss. Negative paired deltas mean the selected calibrated model beats the de-vig market.

Outputs include:

- overall raw / selected / market Brier and log loss
- ECE and calibration curves
- per-prop scoring
- model-preferred Over vs Under
- model-market disagreement buckets
- market-probability buckets
- month and expected-minutes buckets
- quote-level vendor diagnostics
- game-cluster bootstrap confidence intervals

This remains a **retrospective 2025 development audit**, not external validation.

Run:

```bash
PROJECT_ROOT="$(cd ../../.. && pwd)"

"$PROJECT_ROOT/.venv/bin/python" \
  research/proper_scoring_audit_2025.py \
  --project-root "$PROJECT_ROOT" \
  --bootstrap-reps 5000
```
