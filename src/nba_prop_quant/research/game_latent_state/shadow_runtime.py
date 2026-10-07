"""Controlled production shadow for the final same-game dependence model.

SHADOW ONLY. The incumbent remains the published authority.

What this module is for
-----------------------

The final model is frozen and graded, but it has never run beside production.
This is the harness that lets it: it consumes the *same* live marginals and the
*same* incumbent copula production serves from, builds the full-game latent
covariance, answers arbitrary same-game joint queries, and writes the candidate
and the incumbent probability side by side with immutable provenance attached.

What it deliberately cannot do
------------------------------

*   **It never serves the candidate.** :attr:`ShadowDecision.served_probability`
    is the incumbent's in every branch, including the branch where the
    candidate succeeded. Deciding which number ships is not this module's job
    and the type does not offer a way to express it.

*   **It fails closed.** Any failure anywhere in the candidate path -- a
    singular covariance, a missing marginal, an unseen player, an exception
    nobody predicted -- is recorded on the row and the incumbent is what comes
    back. A shadow that can break the thing it shadows is worse than no shadow.

*   **It has no promotion authority.** :func:`assert_no_promotion_authority`
    raises unconditionally. There is no argument that makes it return.

*   **It does not publish.** :func:`publish_shadow_probabilities` refuses while
    the switch is disabled, and the committed switch is disabled. The
    activation path exists so that turning it on is a reviewable change to a
    declared file rather than a surprise; see
    :func:`read_publishing_switch` for the three independent things enabling
    requires.

Grading
-------

:func:`grade_shadow_log` joins realized box scores onto logged rows and reports
Brier and log loss for the candidate, the incumbent and a cross-player
independence reference, per leg count and pooled. It reports; it does not
conclude. :func:`grade_shadow_log` raises if asked for a promotion verdict,
because the only safe interface for "should we promote" is one that does not
exist here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .artifacts import sha256_canonical, sha256_file
from .covariance import (
    GameCovariance,
    SharedFactorLoadings,
    implied_within_player_correlation,
    min_eigenvalue,
)
from .paths import DEFAULT_ARTIFACT_ROOT
from .query import JointQueryResult, PropLeg, evaluate_joint
from .simulator import SUPPORTED_STATS, GameRoster, GameSimulation, simulate_game
from .validation import (
    DEPENDENCE_BUCKETS,
    analytic_marginals,
    brier_score,
    bucket_values,
    log_loss,
    simulated_pair_moments,
)

__all__ = [
    "CANDIDATE",
    "INCUMBENT",
    "INDEPENDENCE",
    "PROMOTION_AUTHORITY",
    "PUBLISHING_DISABLED",
    "PUBLISHING_ENABLED",
    "PUBLISHING_SWITCH_NAME",
    "PUBLISHING_SWITCH_PATH",
    "PUBLISH_APPROVAL_ENV_VAR",
    "PUBLISH_ENV_VAR",
    "PublishingSwitch",
    "ShadowConfig",
    "ShadowDecision",
    "ShadowGameResult",
    "ShadowPromotionRefused",
    "ShadowProvenance",
    "ShadowPublishingDisabled",
    "assert_no_promotion_authority",
    "build_provenance",
    "dependence_diagnostics",
    "evaluate_shadow_game",
    "grade_shadow_log",
    "incumbent_game_simulation",
    "psd_diagnostics",
    "publish_shadow_probabilities",
    "read_publishing_switch",
    "shadow_log_frame",
]

CANDIDATE = "candidate"
INCUMBENT = "incumbent"
INDEPENDENCE = "independence"
MODELS = (CANDIDATE, INCUMBENT, INDEPENDENCE)

#: There is no value of any argument that makes the shadow promotable. The
#: string is here so a report can quote it rather than paraphrase it.
PROMOTION_AUTHORITY = "NONE__SHADOW_HAS_NO_AUTONOMOUS_PROMOTION_AUTHORITY"

PUBLISHING_SWITCH_NAME = "shadow_publishing_switch.json"

#: Where the declared switch lives in the repository, relative to its root.
#: Committed, so turning publication on is a reviewable diff to a tracked
#: file rather than an undocumented operator action.
PUBLISHING_SWITCH_PATH = DEFAULT_ARTIFACT_ROOT / "shadow" / PUBLISHING_SWITCH_NAME

PUBLISHING_DISABLED = "DISABLED"
PUBLISHING_ENABLED = "ENABLED"

#: Enabling publication needs all three of: the committed switch file set to
#: ENABLED, this environment variable set to the exact switch state, and a
#: separately supplied approval token that matches the switch's own. Three
#: independent things, so no single mistake turns it on.
PUBLISH_ENV_VAR = "NBA_PROP_SHADOW_PUBLISH"
PUBLISH_APPROVAL_ENV_VAR = "NBA_PROP_SHADOW_PUBLISH_APPROVAL"


class ShadowPromotionRefused(RuntimeError):
    """Raised whenever anything asks the shadow to promote itself."""


class ShadowPublishingDisabled(RuntimeError):
    """Raised when a publish is attempted while the switch is disabled."""


def assert_no_promotion_authority(context: str = "") -> None:
    """Refuse, always.

    The shadow grades itself against the incumbent and writes the numbers
    down. Acting on them is a separate decision made by people with a separate
    record, which is the only arrangement under which "fail closed to the
    incumbent" means anything.
    """
    detail = f" ({context})" if context else ""
    raise ShadowPromotionRefused(
        f"the production shadow has no promotion authority{detail}: "
        f"{PROMOTION_AUTHORITY}. Promotion is a separate, recorded decision; "
        "the incumbent remains the published authority until it is taken."
    )


# ======================================================================
# the publishing switch
# ======================================================================


@dataclass(frozen=True)
class PublishingSwitch:
    """Whether shadow probabilities may be published, and why not."""

    state: str
    reason: str
    approval_token: str | None = None
    environment_agrees: bool = False
    approval_supplied: bool = False
    source: str | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.state == PUBLISHING_ENABLED
            and self.environment_agrees
            and self.approval_supplied
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.state != PUBLISHING_ENABLED:
            out.append(f"switch file state is {self.state}")
        if not self.environment_agrees:
            out.append(f"{PUBLISH_ENV_VAR} does not match the switch state")
        if not self.approval_supplied:
            out.append(f"{PUBLISH_APPROVAL_ENV_VAR} does not match the approval token")
        return tuple(out)

    def payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "enabled": self.enabled,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "environment_agrees": self.environment_agrees,
            "approval_supplied": self.approval_supplied,
            "source": self.source,
            "environment_variable": PUBLISH_ENV_VAR,
            "approval_environment_variable": PUBLISH_APPROVAL_ENV_VAR,
        }


def read_publishing_switch(
    path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> PublishingSwitch:
    """Read the declared publishing switch and the two things that gate it.

    A missing file is read as disabled, not as permission. That is the whole
    reason the default is a file that has to exist and say so: the absence of
    a prohibition is not an approval.
    """
    environment = os.environ if environment is None else environment
    if path is None or not Path(path).exists():
        return PublishingSwitch(
            state=PUBLISHING_DISABLED,
            reason=(
                "no publishing switch was found; a missing switch is read as "
                "disabled rather than as permission"
            ),
            source=None if path is None else str(path),
        )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = str(payload.get("state", PUBLISHING_DISABLED))
    token = payload.get("approval_token")
    return PublishingSwitch(
        state=state,
        reason=str(payload.get("reason", "")),
        approval_token=None if token is None else str(token),
        environment_agrees=environment.get(PUBLISH_ENV_VAR) == state,
        approval_supplied=(
            token is not None
            and environment.get(PUBLISH_APPROVAL_ENV_VAR) == str(token)
        ),
        source=str(path),
    )


def publish_shadow_probabilities(
    decisions: Sequence[ShadowDecision],
    switch: PublishingSwitch,
) -> None:
    """Refuse to publish while the switch is disabled.

    The fallback path is not a separate code path to be tested in anger later:
    refusing here *is* the fallback, because nothing downstream of this
    function ever received a candidate probability in the first place. The
    incumbent feed is untouched whether this raises or not.
    """
    if not switch.enabled:
        raise ShadowPublishingDisabled(
            "shadow probability publishing is disabled: "
            + "; ".join(switch.blockers)
            + f". {len(decisions)} shadow row(s) were held back. The incumbent "
            "feed is unaffected."
        )
    raise ShadowPublishingDisabled(
        "the publishing switch reports enabled, but no shadow publisher has "
        "been built yet. Activation is a deliberate follow-up change, not a "
        "consequence of flipping this switch."
    )


# ======================================================================
# provenance
# ======================================================================


@dataclass(frozen=True)
class ShadowProvenance:
    """Immutable identity of everything that produced a shadow probability."""

    dependence_model_version: str
    factor_spec_hash: str
    factor_spec_sha256: str
    final_model_spec_sha256: str | None
    code_sha: str | None
    marginal_source: str
    marginal_source_sha256: str | None
    copula_source: str
    copula_source_sha256: str | None
    role_scale: Mapping[str, float]
    stats: tuple[str, ...]
    simulations: int
    seed: int
    built_at: str
    promotion_authority: str = PROMOTION_AUTHORITY
    published: bool = False

    def __post_init__(self) -> None:
        # ``frozen=True`` stops attribute rebinding but says nothing about the
        # contents of a mapping a caller handed in. Provenance that a caller
        # can edit after the fact is not provenance, so the mapping is copied
        # and made read-only here rather than relying on every construction
        # site to pass something immutable.
        object.__setattr__(
            self,
            "role_scale",
            MappingProxyType(
                {str(k): float(v) for k, v in dict(self.role_scale).items()}
            ),
        )
        object.__setattr__(self, "stats", tuple(self.stats))

    def payload(self) -> dict[str, object]:
        return {
            "dependence_model_version": self.dependence_model_version,
            "factor_spec_hash": self.factor_spec_hash,
            "factor_spec_sha256": self.factor_spec_sha256,
            "final_model_spec_sha256": self.final_model_spec_sha256,
            "code_sha": self.code_sha,
            "marginal_source": self.marginal_source,
            "marginal_source_sha256": self.marginal_source_sha256,
            "copula_source": self.copula_source,
            "copula_source_sha256": self.copula_source_sha256,
            "role_scale": {k: float(v) for k, v in sorted(self.role_scale.items())},
            "stats": list(self.stats),
            "simulations": int(self.simulations),
            "seed": int(self.seed),
            "built_at": self.built_at,
            "promotion_authority": self.promotion_authority,
            "published": bool(self.published),
        }

    @property
    def fingerprint(self) -> str:
        """One hash standing for the whole shadow configuration.

        Logged on every row. Two rows with the same fingerprint were produced
        by the same model, the same artifacts and the same code; two rows with
        different fingerprints are not comparable and the difference says so
        rather than hiding in a report header.
        """
        return sha256_canonical(self.payload())


def build_provenance(
    loadings: SharedFactorLoadings,
    factor_spec: Mapping[str, object],
    factor_spec_path: Path,
    marginal_source: str,
    copula_source: str,
    simulations: int,
    seed: int,
    final_model_spec_path: Path | None = None,
    marginal_source_path: Path | None = None,
    copula_source_path: Path | None = None,
    code_sha: str | None = None,
    built_at: str | None = None,
) -> ShadowProvenance:
    """Hash every input rather than naming it."""

    def digest(path: Path | None) -> str | None:
        return sha256_file(path) if path is not None and path.exists() else None

    return ShadowProvenance(
        dependence_model_version=str(
            factor_spec.get("dependence_model_version", "unknown")
        ),
        factor_spec_hash=str(factor_spec.get("spec_hash", "unknown")),
        factor_spec_sha256=sha256_file(factor_spec_path),
        final_model_spec_sha256=digest(final_model_spec_path),
        code_sha=code_sha,
        marginal_source=marginal_source,
        marginal_source_sha256=digest(marginal_source_path),
        copula_source=copula_source,
        copula_source_sha256=digest(copula_source_path),
        role_scale=MappingProxyType(
            {str(k): float(v) for k, v in dict(loadings.role_scale).items()}
        ),
        stats=tuple(loadings.stats),
        simulations=int(simulations),
        seed=int(seed),
        built_at=built_at
        or datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
    )


# ======================================================================
# configuration and results
# ======================================================================


@dataclass(frozen=True)
class ShadowConfig:
    """Everything the shadow needs that is not an artifact."""

    simulations: int = 20_000
    seed: int = 73
    stats: tuple[str, ...] = SUPPORTED_STATS
    role_column: str | None = "role_bucket"
    #: Whether to simulate the cross-player independence reference too. It is
    #: the arm that says how much of any candidate gain is dependence rather
    #: than marginals, so it is on by default.
    evaluate_independence: bool = True
    #: Cost ceiling for the per-game numerical diagnostics. The full spectrum
    #: of a 200-dimensional covariance is cheap; computing it per game for a
    #: full slate is not free, so it is a declared switch rather than a guess.
    collect_eigenvalues: bool = True


@dataclass(frozen=True)
class ShadowDecision:
    """One joint query, both models, and what was actually served.

    ``served_probability`` is the incumbent's. Always. There is no branch of
    :func:`evaluate_shadow_game` that puts a candidate number here.
    """

    event_id: str
    game_id: int
    legs: tuple[str, ...]
    candidate_probability: float | None
    candidate_standard_error: float | None
    incumbent_probability: float
    independence_probability: float | None
    candidate_independent_product: float | None
    push_fraction: float | None
    served_model: str
    served_probability: float
    fell_back: bool
    failure_reason: str | None
    provenance_fingerprint: str

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "game_id": self.game_id,
            "legs": list(self.legs),
            "leg_count": len(self.legs),
            "candidate_probability": self.candidate_probability,
            "candidate_standard_error": self.candidate_standard_error,
            "incumbent_probability": self.incumbent_probability,
            "independence_probability": self.independence_probability,
            "candidate_independent_product": self.candidate_independent_product,
            "push_fraction": self.push_fraction,
            "served_model": self.served_model,
            "served_probability": self.served_probability,
            "fell_back": self.fell_back,
            "failure_reason": self.failure_reason,
            "provenance_fingerprint": self.provenance_fingerprint,
        }


@dataclass
class ShadowGameResult:
    game_id: int
    decisions: list[ShadowDecision] = field(default_factory=list)
    numerical: dict[str, object] = field(default_factory=dict)
    dependence: dict[str, object] = field(default_factory=dict)
    fell_back: bool = False
    failure_reason: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "decisions": [decision.payload() for decision in self.decisions],
            "numerical_diagnostics": self.numerical,
            "dependence_diagnostics": self.dependence,
            "fell_back": self.fell_back,
            "failure_reason": self.failure_reason,
        }


# ======================================================================
# diagnostics
# ======================================================================


def psd_diagnostics(
    covariance: GameCovariance,
    within_player: Mapping[int, np.ndarray],
    collect_eigenvalues: bool = True,
) -> dict[str, object]:
    """PSD and same-player-contract readings off one assembled covariance.

    The same-player deviation is read back out of the assembled matrix rather
    than trusted: the role scale multiplies the shared cross-player factors,
    and the one way that could break the integration contract is by leaving a
    shared-factor echo inside a player's own block.
    """
    deviation = 0.0
    for player_id, block in within_player.items():
        induced = implied_within_player_correlation(covariance, player_id)
        deviation = max(deviation, float(np.max(np.abs(induced - block))))

    out: dict[str, object] = {
        "dimensions": int(covariance.size),
        "min_eigenvalue": float(covariance.min_eigenvalue),
        "min_residual_eigenvalue": float(covariance.min_residual_eigenvalue),
        "shared_shrink_engaged": bool(covariance.shrink_applied),
        "shared_shrink_min": (
            float(min(covariance.shared_shrink.values()))
            if covariance.shared_shrink
            else 1.0
        ),
        "same_player_max_block_deviation": deviation,
        "players_checked": len(within_player),
    }
    if collect_eigenvalues:
        out["correlation_min_eigenvalue_recomputed"] = float(
            min_eigenvalue(covariance.correlation)
        )
    return out


def dependence_diagnostics(
    simulations: Mapping[str, GameSimulation],
    loadings: SharedFactorLoadings,
    reference: Mapping[tuple[int, str], object],
) -> dict[str, object]:
    """What dependence the candidate is adding, in the declared buckets.

    Three readings per bucket: the model's own implied correlation, and the
    correlation each simulated arm actually realises. The implied and the
    simulated can differ -- the PSD projection and the discrete inverse CDF
    both bite -- so reporting only one of them would hide the gap.
    """
    implied = bucket_values(
        loadings.stats,
        loadings.same_team_correlation(),
        loadings.cross_team_correlation(),
    )
    simulated: dict[str, dict[str, float]] = {}
    for name, simulation in simulations.items():
        same, cross, same_pairs, cross_pairs = simulated_pair_moments(
            simulation, reference  # type: ignore[arg-type]
        )
        simulated[name] = {
            **bucket_values(simulation.stats, same, cross),
            "_same_team_pairs": float(same_pairs),
            "_cross_team_pairs": float(cross_pairs),
        }
    # Only the cross-player buckets. The same-player buckets in
    # DEPENDENCE_BUCKETS are the incumbent's pinned block by construction, so
    # reporting them here would be reporting the incumbent's own numbers as a
    # candidate diagnostic.
    return {
        "buckets": sorted(implied),
        "all_declared_buckets": [name for name, _, _ in DEPENDENCE_BUCKETS],
        "model_implied": implied,
        "simulated_by_model": simulated,
        "role_scale": {
            str(k): float(v) for k, v in dict(loadings.role_scale).items()
        },
    }


# ======================================================================
# the incumbent arm
# ======================================================================


def incumbent_game_simulation(
    roster: GameRoster,
    marginals: Mapping[str, object],
    copula,
    simulations: int,
    seed: int,
) -> GameSimulation:
    """The incumbent's implied joint game law: per-player copula, no coupling.

    Production draws each player through ``GaussianCopula.simulate`` on its
    own. Stacking independent per-player draws into one game object is what
    makes the incumbent's joint *game* law explicit instead of implicit --
    players independent of one another, each one's stats coupled by the
    incumbent copula.
    """
    mu_columns = {stat: f"mu_selected_{stat}" for stat in roster.stats}
    rows = [row for _, row in roster.frame.iterrows()]
    draws = np.empty((simulations, len(rows), len(roster.stats)), dtype=float)
    for player_index, row in enumerate(rows):
        frame = copula.simulate(
            row,
            marginals=dict(marginals),
            mu_columns=mu_columns,
            simulations=simulations,
            seed=seed + 1000 * player_index,
        )
        for stat_index, stat in enumerate(roster.stats):
            draws[:, player_index, stat_index] = frame[stat].to_numpy(dtype=float)

    return GameSimulation(
        game_id=int(roster.game_id),
        stats=tuple(roster.stats),
        player_ids=tuple(int(row["player_id"]) for row in rows),
        team_ids=tuple(int(row["team_id"]) for row in rows),
        draws=draws,
        simulations=int(simulations),
        seed=int(seed),
        covariance=None,
    )


# ======================================================================
# one game
# ======================================================================


def evaluate_shadow_game(
    roster: GameRoster,
    events: Mapping[str, Sequence[PropLeg]],
    marginals: Mapping[str, object],
    copula,
    loadings: SharedFactorLoadings,
    provenance: ShadowProvenance,
    config: ShadowConfig | None = None,
    within_player: Mapping[int, np.ndarray] | None = None,
) -> ShadowGameResult:
    """Price every event under both models, failing closed to the incumbent.

    ``events`` maps an event identifier to its legs, so an arbitrary same-game
    parlay is just a longer list and nothing about the shape of the query is
    baked in.

    The incumbent arm is computed first and separately. If it fails there is
    nothing to fall back *to*, so the exception propagates -- the shadow must
    not be able to invent an incumbent number. If the candidate arm fails, the
    reason is recorded on every row of the game and the incumbent is what each
    row carries.
    """
    from .factors import incumbent_within_player_blocks

    config = config or ShadowConfig()
    result = ShadowGameResult(game_id=int(roster.game_id))
    if not events:
        return result

    game_seed = int(config.seed) + int(roster.game_id)
    reference = analytic_marginals(roster, marginals)  # type: ignore[arg-type]

    # No try/except: a shadow that fabricates an incumbent probability when the
    # incumbent path breaks is not a shadow.
    incumbent = incumbent_game_simulation(
        roster,
        marginals=marginals,
        copula=copula,
        simulations=config.simulations,
        seed=game_seed,
    )

    if within_player is None:
        within_player = incumbent_within_player_blocks(
            copula, config.stats, roster.frame["player_id"].astype(int)
        )

    candidate: GameSimulation | None = None
    independence: GameSimulation | None = None
    failure: str | None = None
    try:
        candidate = simulate_game(
            roster,
            marginals=marginals,  # type: ignore[arg-type]
            loadings=loadings,
            within_player=within_player,
            simulations=config.simulations,
            seed=game_seed,
        )
        if candidate.covariance is not None:
            result.numerical = psd_diagnostics(
                candidate.covariance,
                within_player,
                collect_eigenvalues=config.collect_eigenvalues,
            )
        if config.evaluate_independence:
            independence = simulate_game(
                roster,
                marginals=marginals,  # type: ignore[arg-type]
                loadings=SharedFactorLoadings.independent(
                    config.stats, k_game=max(int(loadings.k_game), 1)
                ),
                within_player=within_player,
                simulations=config.simulations,
                seed=game_seed,
            )
    except Exception as error:  # noqa: BLE001 - this is the fail-closed branch
        failure = f"{type(error).__name__}: {error}"
        candidate = None
        independence = None

    result.fell_back = candidate is None
    result.failure_reason = failure

    if candidate is not None:
        arms = {CANDIDATE: candidate, INCUMBENT: incumbent}
        if independence is not None:
            arms[INDEPENDENCE] = independence
        result.dependence = dependence_diagnostics(arms, loadings, reference)

    for event_id, legs in events.items():
        legs = tuple(legs)
        incumbent_query = evaluate_joint(
            incumbent, legs, artifact_version=provenance.factor_spec_hash
        )
        candidate_query: JointQueryResult | None = None
        independence_query: JointQueryResult | None = None
        event_failure = failure
        if candidate is not None:
            try:
                candidate_query = evaluate_joint(
                    candidate, legs, artifact_version=provenance.factor_spec_hash
                )
                if independence is not None:
                    independence_query = evaluate_joint(
                        independence,
                        legs,
                        artifact_version=provenance.factor_spec_hash,
                    )
            except Exception as error:  # noqa: BLE001 - per-event fail-closed
                candidate_query = None
                independence_query = None
                event_failure = f"{type(error).__name__}: {error}"

        result.decisions.append(
            ShadowDecision(
                event_id=str(event_id),
                game_id=int(roster.game_id),
                legs=tuple(leg.describe() for leg in legs),
                candidate_probability=(
                    None if candidate_query is None else candidate_query.probability
                ),
                candidate_standard_error=(
                    None
                    if candidate_query is None
                    else candidate_query.standard_error
                ),
                incumbent_probability=incumbent_query.probability,
                independence_probability=(
                    None
                    if independence_query is None
                    else independence_query.probability
                ),
                candidate_independent_product=(
                    None
                    if candidate_query is None
                    else candidate_query.independent_product
                ),
                push_fraction=(
                    None if candidate_query is None else candidate_query.push_fraction
                ),
                # The incumbent is served in every branch, not only the failing
                # one. This is the whole contract.
                served_model=INCUMBENT,
                served_probability=incumbent_query.probability,
                fell_back=candidate_query is None,
                failure_reason=event_failure if candidate_query is None else None,
                provenance_fingerprint=provenance.fingerprint,
            )
        )
    return result


# ======================================================================
# logging and grading
# ======================================================================


def shadow_log_frame(
    results: Iterable[ShadowGameResult],
    provenance: ShadowProvenance,
) -> pd.DataFrame:
    """One row per event, with the provenance fingerprint on every row."""
    rows: list[dict[str, object]] = []
    for result in results:
        for decision in result.decisions:
            payload = decision.payload()
            payload["legs"] = "|".join(decision.legs)
            payload["game_fell_back"] = result.fell_back
            payload["min_eigenvalue"] = result.numerical.get("min_eigenvalue")
            payload["same_player_max_block_deviation"] = result.numerical.get(
                "same_player_max_block_deviation"
            )
            rows.append(payload)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["dependence_model_version"] = provenance.dependence_model_version
    frame["factor_spec_hash"] = provenance.factor_spec_hash
    frame["published"] = False
    return frame


def grade_shadow_log(
    log: pd.DataFrame,
    realized: Mapping[str, float] | pd.Series,
    floor: float = 1e-6,
    promotion_verdict: bool = False,
) -> dict[str, object]:
    """Brier and log loss for every arm, on the rows that actually settled.

    ``realized`` maps ``event_id`` to 1 or 0. Rows with no realized outcome
    are not graded and are counted, because an ungraded row is evidence about
    coverage and dropping it silently would hide that.

    A row where the candidate fell back is graded for the incumbent and the
    independence arm and *not* for the candidate. Grading a fallback row as if
    the candidate had answered would credit the candidate with the incumbent's
    score, which is the one way this report could lie.
    """
    if promotion_verdict:
        assert_no_promotion_authority("grade_shadow_log was asked for a verdict")

    if log.empty:
        return {
            "events_logged": 0,
            "events_graded": 0,
            "promotion_authority": PROMOTION_AUTHORITY,
            "by_model": {},
        }

    outcomes = pd.Series(dict(realized), dtype=float)
    graded = log.loc[log["event_id"].isin(outcomes.index)].copy()
    graded["outcome"] = graded["event_id"].map(outcomes).astype(float)

    columns = {
        CANDIDATE: "candidate_probability",
        INCUMBENT: "incumbent_probability",
        INDEPENDENCE: "independence_probability",
    }

    def score(frame: pd.DataFrame, column: str) -> dict[str, object] | None:
        usable = frame.loc[frame[column].notna()]
        if usable.empty:
            return None
        probability = usable[column].to_numpy(dtype=float)
        outcome = usable["outcome"].to_numpy(dtype=float)
        return {
            "events": len(usable),
            "brier": brier_score(probability, outcome),
            "log_loss": log_loss(probability, outcome, floor=floor),
            "mean_predicted": float(np.mean(probability)),
            "base_rate": float(np.mean(outcome)),
        }

    by_model = {
        name: score(graded, column)
        for name, column in columns.items()
        if column in graded.columns
    }
    by_legs: dict[str, object] = {}
    if "leg_count" in graded.columns:
        for leg_count, block in graded.groupby("leg_count"):
            by_legs[str(int(leg_count))] = {
                name: score(block, column)
                for name, column in columns.items()
                if column in block.columns
            }

    fell_back = (
        int(graded["fell_back"].sum()) if "fell_back" in graded.columns else 0
    )
    return {
        "events_logged": len(log),
        "events_graded": len(graded),
        "events_without_a_realized_outcome": int(len(log) - len(graded)),
        "events_where_the_candidate_fell_back": fell_back,
        "candidate_fallback_rate": (
            float(fell_back / len(graded)) if len(graded) else 0.0
        ),
        "served_model": INCUMBENT,
        "published": False,
        "promotion_authority": PROMOTION_AUTHORITY,
        "log_loss_floor": floor,
        "by_model": by_model,
        "by_legs": by_legs,
        "provenance_fingerprints": sorted(
            str(value) for value in log["provenance_fingerprint"].unique()
        )
        if "provenance_fingerprint" in log.columns
        else [],
        "note": (
            "fallback rows are graded for the incumbent and the independence "
            "reference but not for the candidate, so the candidate is never "
            "credited with a score it did not produce"
        ),
    }
