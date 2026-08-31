# Known Limitations

## Prospective evidence is pending

This is deliberately a pre-prospective audit package.

The 2026-27 NBA season outcomes required for untouched external validation have not occurred yet.

No prospective claim of market superiority, profitability, or universal edge is made.

## Historical lineup timing

The 2025 historical lineup backfill is development-only.

Its original lineup-announcement timestamps are unavailable. It cannot be treated as untouched pre-tip evidence.

## Historical reuse

The 2025 season has been inspected during model development and certification. It is not an external holdout.

## Market evidence

Retrospective development evidence identifies steals and points+assists as market-superior under the locked development definition.

That finding must not be generalized to future markets until Gate 3 prospective evidence is accumulated.

## P+R uncertainty

Points+rebounds passed the preregistered Gate 2 survival rule, but its log-loss confidence interval versus frozen v1 crossed zero slightly. Its Brier interval excluded zero favorably.

## Betting policy

The system remains monitoring-only. `auto_bet=False`.

No production betting threshold has been certified.

## Data vendor

The live prospective protocol depends on captured BALLDONTLIE data and market availability. Missing or erroneous required T-20m components fail closed rather than being silently replaced with later information.
