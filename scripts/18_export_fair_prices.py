from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prop_quant.fair_price_feed import (
    DEFAULT_BASE_SEED,
    DEFAULT_COMBO_SIMULATIONS,
    FairPriceEngine,
    build_surface_requests,
    sha256_file,
)
from nba_prop_quant.integrations.bet365 import (
    to_bet365_payload,
)
from nba_prop_quant.integrations.wizardofodds import (
    to_wizardofodds_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export market-independent NBA player-prop fair prices "
            "from the frozen production model."
        )
    )
    parser.add_argument(
        "--projections",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=None,
        help=(
            "CSV, parquet, JSON, or JSONL request file with "
            "game_id, player_id, prop_type, line_value."
        ),
    )
    parser.add_argument(
        "--surface",
        action="store_true",
        help=(
            "Generate a market-independent line surface instead "
            "of reading --requests."
        ),
    )
    parser.add_argument(
        "--surface-radius",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--surface-step",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--prop-types",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--combo-simulations",
        type=int,
        default=DEFAULT_COMBO_SIMULATIONS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BASE_SEED,
    )
    parser.add_argument(
        "--require-bet365-ids",
        action="store_true",
    )
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        data: Any = json.loads(
            path.read_text(encoding="utf-8")
        )
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        return pd.DataFrame(data)
    raise ValueError(
        f"Unsupported input format: {path}"
    )


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.surface == (args.requests is not None):
        raise SystemExit(
            "Use exactly one of --requests or --surface."
        )

    projections = read_frame(args.projections)

    engine = FairPriceEngine.from_model_dir(
        args.model_dir,
        manifest_path=args.manifest,
        combo_simulations=args.combo_simulations,
        base_seed=args.seed,
    )

    if args.surface:
        if (
            args.surface_radius is None
            or args.surface_step is None
        ):
            raise SystemExit(
                "--surface requires --surface-radius and "
                "--surface-step."
            )
        prop_types = (
            args.prop_types
            if args.prop_types
            else sorted(engine.artifacts.supported_props)
        )
        requests = build_surface_requests(
            projections,
            prop_types,
            radius=args.surface_radius,
            step=args.surface_step,
        )
    else:
        requests = read_frame(args.requests)

    canonical, rejected = engine.price_requests(
        projections,
        requests,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = (
        output_dir / "nba_fair_price_v1.parquet"
    )
    canonical.to_parquet(
        canonical_path,
        index=False,
    )
    canonical_jsonl_path = (
        output_dir / "nba_fair_price_v1.jsonl"
    )
    canonical.to_json(
        canonical_jsonl_path,
        orient="records",
        lines=True,
        date_format="iso",
    )

    rejected_path = output_dir / "rejected_requests.parquet"
    rejected.to_parquet(
        rejected_path,
        index=False,
    )

    wizard_payload = to_wizardofodds_payload(
        canonical
    )
    wizard_path = (
        output_dir / "wizardofodds_nba_feed_v1.json"
    )
    write_json(
        wizard_path,
        wizard_payload,
    )

    bet365_payload = to_bet365_payload(
        canonical,
        require_external_ids=(
            args.require_bet365_ids
        ),
    )
    bet365_path = (
        output_dir / "bet365_nba_feed_v1.json"
    )
    write_json(
        bet365_path,
        bet365_payload,
    )

    manifest = {
        "schema_version": 1,
        "model_freeze_id": (
            engine.artifacts.manifest["freeze_id"]
        ),
        "model_freeze_stage": (
            engine.artifacts.manifest["freeze_stage"]
        ),
        "canonical_rows": int(len(canonical)),
        "rejected_rows": int(len(rejected)),
        "combo_simulations": int(
            args.combo_simulations
        ),
        "base_seed": int(args.seed),
        "market_odds_used": False,
        "auto_bet": False,
        "outputs": {
            canonical_path.name: sha256_file(
                canonical_path
            ),
            canonical_jsonl_path.name: sha256_file(
                canonical_jsonl_path
            ),
            wizard_path.name: sha256_file(
                wizard_path
            ),
            bet365_path.name: sha256_file(
                bet365_path
            ),
            rejected_path.name: sha256_file(
                rejected_path
            ),
        },
    }
    write_json(
        output_dir / "FAIR_PRICE_FEED_MANIFEST.json",
        manifest,
    )

    print("=" * 100)
    print("NBA FAIR PRICE FEED EXPORT")
    print("=" * 100)
    print(
        f"Freeze:     "
        f"{engine.artifacts.manifest['freeze_id']}"
    )
    print(f"Canonical:  {len(canonical):,} rows")
    print(f"Rejected:   {len(rejected):,} rows")
    print(f"Output:     {output_dir}")
    print("Market odds used: NO")
    print("Auto bet:         NO")
    print(
        "PASS: canonical, WizardOfOdds, and Bet365 "
        "transport-neutral feeds exported."
    )


if __name__ == "__main__":
    main()
