# WizardOfOdds NBA Feed v1

The WizardOfOdds adapter is a website/content feed, not a sportsbook-specific
market ingestion protocol.

Primary output:

```text
wizardofodds_nba_feed_v1.json
```

Each record contains the player, prop, line, selected probabilities, push
probability, fair American odds, fair decimal odds, and audit metadata.

The adapter intentionally does not assume a private WizardOfOdds HTTP endpoint.
If WizardOfOdds supplies a backend API contract later, bind that transport to
this payload without changing the canonical pricing engine.

For public website display, the recommended primary values are:

- selected non-push Over probability;
- selected non-push Under probability;
- push probability;
- fair American odds;
- fair decimal odds;
- expected minutes;
- projected mean;
- freeze ID / generated timestamp.

The website should fail closed on stale timestamps or an unexpected freeze ID.
