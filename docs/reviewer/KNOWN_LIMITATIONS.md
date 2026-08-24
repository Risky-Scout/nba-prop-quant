# Known Limitations

The following limitations are intentionally disclosed for independent review.

1. **2025 line-level certification only.** Historical 2018-2024 line-level probability artifacts were not available in frozen form and were not reconstructed after preregistration.
2. **Market remains modestly better overall.** The selected model is close to market quality, but 2025 aggregate Brier and log loss modestly favor the market.
3. **Prop heterogeneity matters.** Steals outperforms market; assists and threes underperform; several combo markets show low calibration slopes.
4. **Mild high-confidence overconfidence.** Upper confidence buckets realize slightly below their stated confidence.
5. **Points and rebounds PIT departures.** Randomized-PIT diagnostics show modest distributional departures for these targets.
6. **Prospective performance is pending.** No claim of proven 2026-27 sportsbook profitability is made.
7. **Certified runtime scope.** The clean-room runtime bundle is certified for macOS arm64 / Python 3.14.5.
8. **Bet365 transport is not bound to an undocumented consumer-site interface.** The adapter remains `authorized_api_unbound` until an authorized API specification is supplied.
9. **No post-certification retuning.** These findings are frozen as evidence. Any statistical change requires a new model freeze.
