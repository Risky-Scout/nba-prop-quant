# Bet365 NBA Feed v1

## Purpose

This adapter is designed as a professional handoff boundary for an authorized
Bet365 integration.

It does not scrape, emulate, or automate the consumer website.

Primary output:

```text
bet365_nba_feed_v1.json
```

## External mapping fields

A Bet365 request/mapping file may add:

```text
request_id
bet365_event_id
bet365_player_id
bet365_market_id
bet365_over_selection_id
bet365_under_selection_id
```

Those fields pass through the canonical engine and appear in the Bet365
payload.

Use `--require-bet365-ids` when the authorized mapping is complete. The export
then fails closed if any required external ID is absent.

## Transport contract

The v1 adapter declares:

```text
transport_contract = authorized_api_unbound
```

This is deliberate. Authentication, endpoint paths, request signing, retries,
mTLS, streaming, or vendor-specific envelopes must be bound only after Bet365
supplies the authorized API specification.

The Bet365 transport layer must consume the fair probabilities produced by the
canonical engine without changing them.

## Pricing fields

Both Over and Under contain:

- fair unconditional probability;
- fair non-push probability;
- fair American odds;
- fair decimal odds;
- optional Bet365 selection ID.

No automatic bet instruction is emitted.
