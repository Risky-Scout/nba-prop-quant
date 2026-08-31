# Gate 3 Prospective Capture Protocol

The primary certification snapshot is T-20 minutes before scheduled tip.

There is no fallback window for the primary certification sample.

T-24h, T-8h, T-3h, T-90m, T-45m, and T-5m captures remain secondary monitoring evidence.

A primary record is eligible only when:

- the scheduled capture completed with zero recorded component errors;
- the capture occurred before scheduled tip;
- games, active players, injuries, lineups, and player props are tied to the same captured_at value;
- component SHA-256 hashes verify;
- the deterministic capture ID verifies;
- prediction and pricing use the same game-level capture ID;
- the frozen Gate 3 model candidate is used;
- no prospective outcome has been observed before prediction generation.

Direct live API prediction or pricing is not eligible for Gate 3 certification.

The candidate may not be retuned from Gate 3 outcomes.
