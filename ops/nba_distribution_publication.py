"""Expose the distributions the pricing engine already priced from.

Nothing here is a distribution. Every number this stage publishes comes out of
``nba_prop_quant.pricing``, through the same call the price in the same row
came out of, and the only work done here is arranging it into a record a
renderer can draw and a reader can read.

WHY IT IS A STAGE AND NOT PART OF PRICING
-----------------------------------------

The obvious place to produce these is inside ``scripts/15_price_markets.py``,
which already holds the loaded marginals, the loaded copula and the frozen
dependence policy. That script is byte-pinned in
``nba_prop_quant_v2_adaptive_serving_source_contract.json`` and every function
in it except ``parse_args`` and ``main`` is pinned syntax-tree-identical to
production, and a module it imports stops being declarable at all. Producing a
page feature there would mean relocking the serving source and moving a new
module into the undeclarable core, which is a large change to the frozen
pricing path in exchange for a display panel.

So this reads what pricing wrote. The priced parquet is the whole priced frame
-- ``write_parquet_atomic`` is handed it unselected -- so it carries the
canonical mean columns, the Gate 3 assists mean and every feature the fitted
marginals read. Everything the engines need to reproduce a distribution is in
the file, which means the distribution can be reproduced exactly without the
pricing path changing by one byte.

What this costs is that the stage has to name the same inputs pricing named,
rather than being handed them. Three of the four are facts in the file or in
the frozen policy. The fourth, the per-event simulation seed, is loaded out of
``scripts/15_price_markets.py`` itself rather than restated here, so there is
one definition of it in the repository: see :func:`load_pricing_entry_point`.

HOW IT AVOIDS BEING A SECOND OPINION
------------------------------------

This module holds no probability mathematics of its own. It calls
``_marginal_pmf_for_row`` for a single stat, ``independent_combo_pmf`` or
``simulate_combo_values`` for a combination, and ``_pmf_over_under_push`` and
``empirical_over_under_push`` when a test asks whether the arrangement still
agrees with the price. Those are the pricing engine's own functions, imported
under their own names, and a test pins that this module defines no others.

WHICH ENGINE A PROP COMES FROM
------------------------------

The choice is not made here either. ``pricing.price_combo_lines`` prices a
combination from an analytical convolution when its frozen dependence
coefficient is zero and from a seeded simulation when it is not, and this
module reads the same coefficient from the same frozen policy and follows it.
On the current frozen policy that makes four of the five combinations
analytical and ``rebounds_assists`` simulated, which is a fact about
``models/combo_dependence_policy.json`` rather than a choice made here. The
record says which it was, because a finite set of draws is not an analytical
distribution and a page that called it one would be claiming a precision the
number does not have.

THE TAIL
--------

A count distribution has no largest outcome, so a published record cannot
list every one. It lists outcomes upward from zero until the mass still above
them falls to ``TAIL_TOLERANCE`` and then states that remaining mass, exactly,
as one ``N+`` bucket. Nothing is discarded: the listed probabilities and the
bucket sum to the whole distribution. What the bucket cannot do is reproduce
the distribution's mean, because it says how much mass is above ``N`` and not
where it sits, which is why the record carries the mean computed from the
complete pre-bucket distribution and says so.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal
from nba_prop_quant.pricing import (
    COMBO_COMPONENTS,
    PROP_TO_TARGET,
    _marginal_pmf_for_row,
    _pmf_over_under_push,
    empirical_over_under_push,
    independent_combo_pmf,
    simulate_combo_values,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The pricing entry point, loaded for its seed function and nothing else.
PRICING_ENTRY_POINT = "scripts/15_price_markets.py"

#: ``pricing`` reads a component mean out of the row under this name, and
#: ``scripts/15_price_markets.py`` builds exactly this mapping before it
#: prices a combination. Restated rather than imported because importing it
#: would mean importing the script's ``main``, and it is one formatting rule
#: over a frozen target list rather than a choice: the combo parity test
#: reprices every tested line from the published distribution and requires the
#: probability the row already carries, so a mapping that named the wrong
#: column could not pass.
MU_COLUMN_TEMPLATE = "mu_selected_{target}"

#: The pricing script's own defaults for the two simulation arguments. A run
#: that passed something else has to say so here too, which is why the stage
#: takes them as arguments and records what it used.
DEFAULT_COMBO_SIMULATIONS = 20_000
DEFAULT_SEED = 73

#: How much mass may remain above the last individually listed outcome before
#: the record stops listing and states the remainder as one bucket. Chosen so
#: an ordinary scoring line lists into the forties and a blocks line lists
#: four or five outcomes, which is the range a reader can scan. It controls
#: only how many outcomes are listed separately: the bucket carries the exact
#: remainder either way, so the record's total mass does not depend on it.
TAIL_TOLERANCE = 1e-3

#: The hard ceiling on listed outcomes, so a pathological mean cannot produce
#: a record nobody can render. Reached only if the tolerance above has not
#: already stopped the listing, and the bucket still carries the exact
#: remainder when it is.
MAX_LISTED_OUTCOMES = 80

#: The mass the "most likely range" is required to contain. The range is the
#: narrowest run of consecutive outcomes holding at least this much, which is
#: a statement about the distribution rather than a word about confidence.
MOST_LIKELY_MASS = 0.80

#: What the page is allowed to call each kind of record. A simulated total is
#: labelled as one: ``pricing`` prices it from a finite set of draws, and the
#: empirical frequencies of those draws are the distribution, not an
#: approximation of a closed form that exists somewhere else.
KIND_MARGINAL = "analytical_marginal_pmf"
KIND_CONVOLUTION = "analytical_convolution_pmf"
KIND_SIMULATION = "simulation_empirical_pmf"

ANALYTICAL_LABEL = "Model Distribution"
SIMULATION_LABEL = "Model Distribution (simulation-based)"

DISTRIBUTION_LABEL: Mapping[str, str] = {
    KIND_MARGINAL: ANALYTICAL_LABEL,
    KIND_CONVOLUTION: ANALYTICAL_LABEL,
    KIND_SIMULATION: SIMULATION_LABEL,
}

ANALYTICAL_KINDS: tuple[str, ...] = (KIND_MARGINAL, KIND_CONVOLUTION)

#: Below this the frozen pricing path treats a combination as independent and
#: prices it by convolution. The same threshold ``pricing.price_combo_lines``
#: compares against, so the engine that priced a line is the engine whose
#: distribution is published beside it.
INDEPENDENCE_LAMBDA = 1e-12

#: The three fields a record is keyed by. One distribution belongs to a
#: player's stat in a game, not to a sportsbook's quote of it: six books
#: quoting six lines on the same player's points are six prices out of one
#: forecast.
DISTRIBUTION_KEY_FIELDS: tuple[str, str, str] = (
    "game_id",
    "player_id",
    "prop_type",
)

#: This document's own version, read for equality by everything downstream.
#: Independent of the publication contract: this file is an intermediate
#: artifact between two stages of one run, and what reaches the destination is
#: versioned there.
DOCUMENT_CONTRACT_VERSION = 1

#: The column the pricing path records the mean it priced a single stat
#: against. Read rather than re-derived: ``scripts/15_price_markets.py``
#: chooses per prop which mean column a prop is priced from, and a
#: distribution published beside a price has to be the one that price came
#: from.
PROJECTION_COLUMN = "model_projection"


class DistributionRefused(RuntimeError):
    """No record could be produced without inventing one."""


@dataclass(frozen=True)
class PublishedDistribution:
    """One prop's outcome distribution, as a reader will see it.

    ``support`` and ``probability`` are the individually listed outcomes.
    ``tail_outcome`` is the first outcome not listed and ``tail_probability``
    is the exact mass at or above it, so the two together are the whole
    distribution and ``mass`` is one.

    ``mean`` is computed from the complete distribution before any of it was
    bucketed. It is the only mean in the record and it is not recoverable from
    the listed rows: see the module docstring.
    """

    kind: str
    support: tuple[int, ...]
    probability: tuple[float, ...]
    tail_outcome: int | None
    tail_probability: float
    mean: float
    most_likely_low: int
    most_likely_high: int
    most_likely_mass: float
    dependence_lambda: float = 0.0
    simulation_draws: int | None = None
    simulation_seed: int | None = None

    @property
    def label(self) -> str:
        return DISTRIBUTION_LABEL[self.kind]

    @property
    def analytical(self) -> bool:
        return self.kind in ANALYTICAL_KINDS

    @property
    def mass(self) -> float:
        return float(sum(self.probability) + self.tail_probability)

    @property
    def tail_label(self) -> str | None:
        return None if self.tail_outcome is None else f"{self.tail_outcome}+"

    def payload(self) -> dict[str, Any]:
        """The record as it is published, under the names the contract uses."""
        return {
            "distribution_type": self.kind,
            "distribution_label": self.label,
            "distribution_analytical": self.analytical,
            "distribution_support": list(self.support),
            "distribution_probability": [float(p) for p in self.probability],
            "distribution_tail_outcome": self.tail_outcome,
            "distribution_tail_label": self.tail_label,
            "distribution_tail_probability": float(self.tail_probability),
            "distribution_mass": float(self.mass),
            "distribution_mean": float(self.mean),
            "distribution_most_likely_low": int(self.most_likely_low),
            "distribution_most_likely_high": int(self.most_likely_high),
            "distribution_most_likely_mass": float(self.most_likely_mass),
            "distribution_most_likely_target_mass": float(MOST_LIKELY_MASS),
            "dependence_lambda": float(self.dependence_lambda),
            "simulation_draws": self.simulation_draws,
            "simulation_seed": self.simulation_seed,
        }


def most_likely_range(
    probabilities: Sequence[float],
    *,
    mass: float = MOST_LIKELY_MASS,
) -> tuple[int, int, float]:
    """The narrowest run of consecutive outcomes holding ``mass`` or more.

    Ties broken toward the lower start, so the answer is a function of the
    distribution and not of iteration order. The whole support is returned
    when no run reaches the target, which happens only if the distribution
    itself carries less than ``mass``.
    """
    weights = np.asarray(probabilities, dtype=float)

    if weights.size == 0:
        raise DistributionRefused("an empty distribution has no range")

    best: tuple[int, int, float] | None = None
    total = 0.0
    low = 0

    for high in range(weights.size):
        total += float(weights[high])

        while low < high and total - float(weights[low]) >= mass:
            total -= float(weights[low])
            low += 1

        if total >= mass:
            width = high - low
            if best is None or width < best[1] - best[0]:
                best = (low, high, total)

    if best is None:
        return 0, int(weights.size - 1), float(weights.sum())

    return int(best[0]), int(best[1]), float(best[2])


def _bucketed(
    pmf: np.ndarray,
    *,
    kind: str,
    dependence_lambda: float = 0.0,
    simulation_draws: int | None = None,
    simulation_seed: int | None = None,
) -> PublishedDistribution:
    """One complete PMF, arranged for publication.

    The mean is taken here, from the complete array, before the tail is
    bucketed. Doing it afterwards would be computing the mean of the display.
    """
    pmf = np.asarray(pmf, dtype=float)

    if pmf.ndim != 1 or pmf.size == 0:
        raise DistributionRefused("a distribution must be a non-empty vector")

    if not np.all(np.isfinite(pmf)):
        raise DistributionRefused("a distribution must be finite everywhere")

    if np.any(pmf < 0.0):
        raise DistributionRefused("a distribution must be non-negative")

    total = float(pmf.sum())

    if not np.isfinite(total) or total <= 0.0:
        raise DistributionRefused("a distribution must carry positive mass")

    outcomes = np.arange(pmf.size, dtype=float)
    mean = float(np.dot(outcomes, pmf) / total)

    # Mass strictly above each outcome, so the first index whose remainder is
    # small enough is the first outcome not worth listing on its own.
    remaining = total - np.cumsum(pmf)
    listed = int(pmf.size)

    for index in range(pmf.size):
        if float(remaining[index]) <= TAIL_TOLERANCE * total:
            listed = index + 1
            break

    listed = min(listed, MAX_LISTED_OUTCOMES)

    if listed >= pmf.size:
        support = tuple(range(pmf.size))
        probability = tuple(float(p) for p in pmf)
        tail_outcome: int | None = None
        tail_probability = 0.0
    else:
        support = tuple(range(listed))
        probability = tuple(float(p) for p in pmf[:listed])
        tail_outcome = listed
        tail_probability = float(pmf[listed:].sum())

    ranged = list(probability)

    if tail_outcome is not None:
        ranged.append(tail_probability)

    low, high, held = most_likely_range(ranged)

    return PublishedDistribution(
        kind=kind,
        support=support,
        probability=probability,
        tail_outcome=tail_outcome,
        tail_probability=tail_probability,
        mean=mean,
        most_likely_low=low,
        most_likely_high=high,
        most_likely_mass=held,
        dependence_lambda=float(dependence_lambda),
        simulation_draws=simulation_draws,
        simulation_seed=simulation_seed,
    )


def single_prop_distribution(
    row: pd.Series,
    prop_type: str,
    marginals: Mapping[str, FittedMarginal],
    mu: float,
) -> PublishedDistribution:
    """The frozen marginal's own PMF for one player and one stat.

    ``mu`` is passed in rather than looked up because the pricing path chooses
    the mean column per prop -- assists is priced from the Gate 3 column --
    and the distribution beside a price must be the distribution that price
    came from. The caller that priced the row supplies the column it used.
    """
    if prop_type not in PROP_TO_TARGET:
        raise DistributionRefused(f"{prop_type} is not a single-stat prop")

    target = PROP_TO_TARGET[prop_type]

    if target not in marginals:
        raise DistributionRefused(f"no frozen marginal for {target}")

    pmf = _marginal_pmf_for_row(marginals[target], float(mu), row)

    return _bucketed(pmf, kind=KIND_MARGINAL)


def combo_distribution(
    row: pd.Series,
    prop_type: str,
    marginals: Mapping[str, FittedMarginal],
    mu_columns: Mapping[str, str],
    copula: GaussianCopula,
    dependence_lambda: float,
    *,
    simulations: int,
    seed: int,
) -> PublishedDistribution:
    """The combination total's distribution, from whichever engine priced it.

    The branch is ``pricing.price_combo_lines``'s branch, on the same
    threshold, so the record and the price in the row beside it are the same
    object. Under dependence the draws are regenerated from the same seed and
    the same count, which is what makes the empirical frequencies here equal
    the frequencies the line was priced from rather than merely resemble them.
    """
    if prop_type not in COMBO_COMPONENTS:
        raise DistributionRefused(f"{prop_type} is not a combination prop")

    dependence_lambda = float(dependence_lambda)

    if dependence_lambda <= INDEPENDENCE_LAMBDA:
        pmf = independent_combo_pmf(
            row,
            prop_type,
            dict(marginals),
            dict(mu_columns),
        )

        return _bucketed(
            pmf,
            kind=KIND_CONVOLUTION,
            dependence_lambda=dependence_lambda,
        )

    values = simulate_combo_values(
        row=row,
        prop_type=prop_type,
        marginals=dict(marginals),
        mu_columns=dict(mu_columns),
        copula=copula,
        dependence_lambda=dependence_lambda,
        simulations=int(simulations),
        seed=int(seed),
    )

    return empirical_distribution(
        values,
        dependence_lambda=dependence_lambda,
        seed=int(seed),
    )


def empirical_distribution(
    values: np.ndarray,
    *,
    dependence_lambda: float,
    seed: int,
) -> PublishedDistribution:
    """The frequencies of one set of simulated totals.

    Separated from ``combo_distribution`` so a test can hand it the very draws
    the price was computed from and require equality rather than agreement.
    """
    draws = np.asarray(values)

    if draws.ndim != 1 or draws.size == 0:
        raise DistributionRefused("a simulation must produce draws")

    if np.any(draws < 0):
        raise DistributionRefused("a count total cannot be negative")

    counts = np.bincount(draws.astype(int))

    return _bucketed(
        counts / float(draws.size),
        kind=KIND_SIMULATION,
        dependence_lambda=float(dependence_lambda),
        simulation_draws=int(draws.size),
        simulation_seed=int(seed),
    )


def line_probabilities(
    distribution: PublishedDistribution,
    line: float,
) -> tuple[float, float, float]:
    """Over, under and push read off the published record.

    What a reader could work out from the listed rows, computed the way the
    pricing engine computes it from a PMF -- the engine's own function, on the
    record's own numbers. Only usable while the line sits below the tail
    bucket, because the bucket says how much mass is above an outcome and not
    how it is distributed; a line inside it is refused rather than guessed.
    """
    if distribution.tail_outcome is not None:
        if float(line) >= float(distribution.tail_outcome) - 1.0:
            raise DistributionRefused(
                f"line {line} falls in or beside the {distribution.tail_label} "
                "bucket, whose internal shape the record does not state"
            )

    pmf = np.zeros(len(distribution.support) + 1, dtype=float)
    pmf[: len(distribution.probability)] = distribution.probability

    if distribution.tail_outcome is not None:
        pmf[distribution.tail_outcome] = distribution.tail_probability

    return _pmf_over_under_push(pmf, float(line))


def pricing_line_probabilities(
    values: np.ndarray,
    line: float,
) -> tuple[float, float, float]:
    """The simulated draws' own line probabilities, by the pricing function.

    Here so a parity test names the pricing engine's function rather than
    restating its arithmetic.
    """
    return empirical_over_under_push(np.asarray(values), float(line))


# ----------------------------------------------------------------------
# one priced slate's worth of records
# ----------------------------------------------------------------------


def distribution_key(game_id: Any, player_id: Any, prop_type: Any) -> str:
    """What one distribution is filed under.

    Three published identifiers, in a fixed order, joined by a character none
    of them contains. A string rather than a tuple because the record travels
    as JSON, where an object key is a string whatever it started as.
    """
    return "|".join(
        (
            str(int(game_id)),
            str(int(player_id)),
            str(prop_type),
        )
    )


def published_distribution_records(
    priced: pd.DataFrame,
    *,
    marginals: Mapping[str, FittedMarginal],
    mu_columns: Mapping[str, str],
    copula: GaussianCopula,
    dependence_lambdas: Mapping[str, float],
    simulations: int,
    combo_seed: Callable[[int, int, str], int],
) -> dict[str, dict[str, Any]]:
    """One record per player, stat and game on a priced slate.

    Grouped by the three fields a distribution belongs to rather than by quote,
    so a wager six sportsbooks price produces one distribution and not six.
    The first row of each group supplies the features the marginal reads and
    the mean the prop was priced against, which are properties of the player's
    game and are identical across that group's quotes.

    A prop whose distribution cannot be produced is omitted rather than
    approximated. The publisher then has no record for it, the page shows no
    panel for it, and nothing anywhere shows a distribution that was guessed.
    """
    missing = [
        name
        for name in (*DISTRIBUTION_KEY_FIELDS, PROJECTION_COLUMN)
        if name not in priced.columns
    ]

    if missing:
        raise DistributionRefused(
            "the priced frame cannot be keyed by distribution: missing "
            + ", ".join(missing)
        )

    records: dict[str, dict[str, Any]] = {}

    grouped = priced.groupby(list(DISTRIBUTION_KEY_FIELDS), sort=True)

    for (game_id, player_id, prop_type), group in grouped:
        row = group.iloc[0]
        prop_type = str(prop_type)

        if prop_type in PROP_TO_TARGET:
            mu = pd.to_numeric(row.get(PROJECTION_COLUMN), errors="coerce")

            if not np.isfinite(mu):
                continue

            distribution = single_prop_distribution(
                row,
                prop_type,
                marginals,
                float(mu),
            )

        elif prop_type in COMBO_COMPONENTS:
            if prop_type not in dependence_lambdas:
                raise DistributionRefused(
                    f"no frozen dependence coefficient for {prop_type}, so "
                    "which engine priced it is not established"
                )

            distribution = combo_distribution(
                row,
                prop_type,
                marginals,
                mu_columns,
                copula,
                float(dependence_lambdas[prop_type]),
                simulations=int(simulations),
                seed=int(
                    combo_seed(int(game_id), int(player_id), prop_type)
                ),
            )

        else:
            continue

        records[distribution_key(game_id, player_id, prop_type)] = (
            distribution.payload()
        )

    return records


# ----------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------


def load_pricing_entry_point(project_root: Path) -> ModuleType:
    """``scripts/15_price_markets.py``, loaded as a module.

    For ``stable_event_seed`` and nothing else. The seed decides which draws a
    dependent combination was priced from, so a second definition of it here
    would be a second answer to that question, and the two would agree until
    one of them was edited. Loading the file means there is one.

    Safe to load: the script's module body is imports, one ``Console()`` and
    an ``if __name__`` guard, so importing it runs no pricing and reads no
    model. A test pins that.
    """
    path = Path(project_root) / PRICING_ENTRY_POINT

    if not path.is_file():
        raise DistributionRefused(
            f"{path} does not exist, so the seed every dependent combination "
            "was priced from cannot be read from the script that priced it"
        )

    spec = importlib.util.spec_from_file_location(
        "nba_prop_quant_pricing_entry_point", path
    )

    if spec is None or spec.loader is None:
        raise DistributionRefused(f"{path} is not loadable as a module")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def mu_columns_for(targets: Sequence[str]) -> dict[str, str]:
    return {target: MU_COLUMN_TEMPLATE.format(target=target) for target in targets}


def distributions_path(data_root: Path, slate_date: str) -> Path:
    """Where this stage's output for one slate lives.

    Beside the priced markets rather than inside them: the priced parquet is
    the frozen serving path's own output, provenanced by digest in the serving
    receipt, and a stage that rewrote it would invalidate that receipt to add
    a display panel.
    """
    return (
        Path(data_root)
        / "processed"
        / "nba_distributions"
        / f"{slate_date}.json"
    )


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


def build_document(
    *,
    priced_path: Path,
    model_dir: Path,
    project_root: Path = PROJECT_ROOT,
    slate_date: str,
    simulations: int = DEFAULT_COMBO_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Every distribution behind one priced slate, and what it was derived from.

    ``priced_sha256`` is the whole point of the document having a header. The
    publisher is handed a serving receipt that digests the priced parquet, and
    it refuses a distributions document whose digest is not that one, so a
    stale file left over from an earlier run cannot be published beside a
    newer slate's prices.
    """
    priced_path = Path(priced_path)

    priced = pd.read_parquet(priced_path)

    marginals = joblib.load(Path(model_dir) / "marginals.joblib")

    copula = joblib.load(Path(model_dir) / "copula.joblib")

    policy = json.loads(
        (Path(model_dir) / "combo_dependence_policy.json").read_text(
            encoding="utf-8"
        )
    )

    entry_point = load_pricing_entry_point(project_root)

    records = published_distribution_records(
        priced,
        marginals=marginals,
        mu_columns=mu_columns_for(sorted(marginals)),
        copula=copula,
        dependence_lambdas={
            prop_type: float(entry["production_lambda"])
            for prop_type, entry in policy["combos"].items()
        },
        simulations=int(simulations),
        combo_seed=lambda game_id, player_id, prop_type: (
            entry_point.stable_event_seed(
                game_id, player_id, prop_type, int(seed)
            )
        ),
    )

    return {
        "combo_simulations": int(simulations),
        "contract_version": DOCUMENT_CONTRACT_VERSION,
        "distributions": records,
        "priced_relative": priced_path.name,
        "priced_sha256": sha256_file(priced_path),
        "seed": int(seed),
        "slate_date": str(slate_date),
    }


def read_document(path: Path) -> dict[str, Any]:
    """One distributions document, or a refusal naming what is wrong with it."""
    path = Path(path)

    if not path.is_file():
        raise DistributionRefused(f"{path} does not exist")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DistributionRefused(
            f"{path} is not readable JSON ({error})"
        ) from error

    if not isinstance(document, dict):
        raise DistributionRefused(f"{path} is not a JSON object")

    version = document.get("contract_version")

    if version != DOCUMENT_CONTRACT_VERSION:
        raise DistributionRefused(
            f"{path} declares distribution document contract {version!r} and "
            f"this reader was written against {DOCUMENT_CONTRACT_VERSION}"
        )

    records = document.get("distributions")

    if not isinstance(records, dict):
        raise DistributionRefused(f"{path} carries no distributions object")

    return document


def write_document(path: Path, document: Mapping[str, Any]) -> bytes:
    """The document on disk, byte-identical for two runs over the same slate."""
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    path.write_bytes(payload)

    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the outcome distribution behind every price on one "
            "priced slate. Reads the priced markets and the frozen model "
            "artifacts; writes one JSON document. Changes nothing it reads."
        )
    )

    parser.add_argument("--date", required=True)

    parser.add_argument("--data-root", type=Path, required=True)

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "the frozen bundle's models directory, which is what the pricing "
            "run loaded its marginals and copula from"
        ),
    )

    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    parser.add_argument(
        "--combo-simulations", type=int, default=DEFAULT_COMBO_SIMULATIONS
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    priced_path = (
        Path(args.data_root)
        / "processed"
        / "priced_markets"
        / f"{args.date}.parquet"
    )

    if not priced_path.is_file():
        raise DistributionRefused(
            f"{priced_path} does not exist, so no slate has been priced for "
            f"{args.date} and there is nothing to publish a distribution for"
        )

    document = build_document(
        priced_path=priced_path,
        model_dir=args.model_dir,
        project_root=args.project_root,
        slate_date=args.date,
        simulations=args.combo_simulations,
        seed=args.seed,
    )

    out_path = distributions_path(args.data_root, args.date)

    write_document(out_path, document)

    print(
        f"{len(document['distributions'])} distributions -> {out_path}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
