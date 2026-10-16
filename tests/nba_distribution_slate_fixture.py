"""A served slate carrying every prop the frozen bundle can price.

The shared slate fixture quotes four prop types, which is all the suites it
was written for needed. The distribution contract is a claim about *every*
published target -- six single stats and the combinations -- so a parity suite
built on four of them would be asserting the contract on a subset and calling
it proven.

So this adds quotes. Nothing else: the roster, the history, the capture, the
schedule and the models are the shared fixture's and the frozen bundle's, and
the prices here are the same -110 both ways the shared fixture posts, because
a price has no effect on a distribution and a different one would only make
the two slates harder to compare.

WHAT THE FROZEN BUNDLE WILL AND WILL NOT PRICE
----------------------------------------------

``scripts/15_price_markets.py`` prices the intersection of what the model
supports with what the frozen market-probability calibration policy covers.
The installed bundle's policy covers ten prop types, and ``stocks`` -- which
``nba_prop_quant.pricing`` does support -- is not one of them. A quote on it
is therefore dropped by the pricing run, which means production publishes no
``stocks`` price and there is no ``stocks`` price for a distribution to sit
beside. :data:`UNPRICEABLE` records that, so a suite asserting coverage says
what it is not covering and why rather than quietly enumerating what it
found.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

for _relative in ("ops", "src", "tests"):
    if str(PROJECT / _relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / _relative))

import frozen_bundle_slate_fixture as slate_fixture  # noqa: E402
import nba_distribution_publication as distribution_stage  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_incumbent_production_serving as serving  # noqa: E402

from test_real_slate_serving_root import (  # noqa: E402
    installed_bundle,
    serve_the_fixture,
)

SLATE_DATE = slate_fixture.SLATE_DATE

#: Quotes on every prop type the frozen calibration policy covers that the
#: shared fixture does not already quote, spread over players with enough
#: history for the marginals to have a mean to work from. The lines straddle
#: the fixture's projections, so each one has mass on both sides of it and a
#: parity assertion is not comparing two zeros.
ADDED_PROPS: tuple[tuple[int, str, str], ...] = (
    (1101, "assists", "4.5"),
    (1101, "steals", "1.5"),
    (1101, "threes", "2.5"),
    (1102, "points_assists", "19.5"),
    (1102, "points_rebounds", "19.5"),
    (1102, "points_rebounds_assists", "24.5"),
    (1102, "rebounds_assists", "7.5"),
    (1103, "assists", "3.5"),
    (1103, "steals", "0.5"),
    (1103, "blocks", "0.5"),
    (1104, "points_rebounds", "16.5"),
    (1104, "rebounds_assists", "5.5"),
    (2201, "points_rebounds_assists", "28.5"),
    (2202, "points_assists", "16.5"),
    (2202, "rebounds_assists", "6.5"),
)

#: Props ``nba_prop_quant.pricing`` supports that the installed frozen
#: calibration policy does not cover, so the frozen serving path prices none
#: of them and no release can carry one.
UNPRICEABLE: tuple[str, ...] = ("stocks",)


@contextmanager
def extra_quotes() -> Iterator[None]:
    """The shared fixture's quote list with :data:`ADDED_PROPS` appended.

    Patched and restored rather than edited into the fixture, because every
    other suite's expected row counts are that list's length and this one has
    no business changing them.
    """
    original = slate_fixture.QUOTED_PROPS

    slate_fixture.QUOTED_PROPS = original + ADDED_PROPS

    try:
        yield
    finally:
        slate_fixture.QUOTED_PROPS = original


def serve(bundle: Path, root: Path) -> dict:
    """Serve the widened slate, then derive its distributions.

    Both steps through the real entry points and in the order the lifecycle
    runs them, so what comes back is a priced parquet the frozen pricing path
    wrote and a document the stage derived from it -- not two things built to
    match.
    """
    root = Path(root)

    with extra_quotes():
        served = serve_the_fixture(bundle, root)

    assert served["outcome"] == serving.OUTCOME_SERVED

    data_root = root / "data"

    priced_path = publisher.priced_markets_path(data_root, SLATE_DATE)

    priced = pd.read_parquet(priced_path)

    assert distribution_stage.main(
        [
            "--date",
            SLATE_DATE,
            "--data-root",
            str(data_root),
            "--model-dir",
            str(Path(bundle) / "models"),
        ]
    ) == 0

    document = distribution_stage.read_document(
        distribution_stage.distributions_path(data_root, SLATE_DATE)
    )

    return {
        "bundle": Path(bundle),
        "data_root": data_root,
        "document": document,
        "model_dir": Path(bundle) / "models",
        "priced": priced,
        "priced_path": priced_path,
        "root": root,
        "served": served,
    }


def skip_without_bundle():
    """The installed frozen bundle, or the reason there is nothing to assert."""
    import pytest

    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    return bundle


__all__ = [
    "ADDED_PROPS",
    "SLATE_DATE",
    "UNPRICEABLE",
    "extra_quotes",
    "serve",
    "skip_without_bundle",
]
