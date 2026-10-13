"""Which sportsbook a published wager sends a reader to, and why.

A product layer, downstream of everything. The incumbent decides what to bet
and what it is worth; this decides, among the books that already offer exactly
that bet, which one the reader is pointed at. Nothing here reads, derives or
rewrites a model number: a routing decision that could move an edge would make
the published opinion a function of who pays commission, which is the one thing
this must never be.

THE RULE IS FROZEN AND IT IS A TIER RULE
----------------------------------------

Tiers first, price second. Primary affiliates are preferred over secondary
affiliates, secondary over everybody else, and only inside the chosen tier does
the best price win. A non-affiliate book with a better number does not take the
click while an affiliate offers the same wager -- which is a commercial choice,
openly made, and the reason the page may not call an affiliate selection "best
available odds" unless it independently is. :func:`route` reports that as
``best_overall`` so the caller can label honestly rather than guess.

AN OFFER HAS TO BE THE SAME BET
-------------------------------

"Offers the exact wager" means the same game, the same player, the same prop,
the same line and the same side. Those five are
:data:`publish_incumbent_to_wizardofodds.LOGICAL_KEY` without ``vendor``, plus
the side the model picked, so the identity is the published schema's own and
nothing new was invented to express it. A book quoting the same player's points
at a different line is not a cheaper version of this bet, it is a different
bet, and it cannot be routed to.

Where the published rows have one row per quote, the offer universe for a wager
is simply every row that shares its contract, priced at that row's ``over_odds``
or ``under_odds`` according to the side.

A BOOK NAME IS NOT AN IDENTIFIER
--------------------------------

Tier membership and display name come from the product requirement.
Which sportsbook identifier a book appears under, and where a click goes, do
not: both are read from :data:`REGISTRY_RELATIVE`, which records them only
where site or data evidence proves them. Five of the eight books have neither,
so they cannot match an offer and cannot be given a link. That is not a gap to
be filled in by resemblance -- the site's redirect answers an unrecognised book
by sending it to Bovada, so a guessed identifier would not break visibly, it
would silently misattribute someone else's traffic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

import publish_incumbent_to_wizardofodds as publisher

#: Where the book facts live, relative to the repository root.
REGISTRY_RELATIVE = "ops/wizardofodds_affiliates.json"

TIER_PRIMARY = "primary"

TIER_SECONDARY = "secondary"

TIER_FALLBACK = "fallback"

#: No book offers this wager. Not a tier so much as the absence of one, named
#: so a caller cannot mistake it for a tier it could render a button for.
TIER_NONE = "none"

#: The order the tier rule tries, which is the whole of the rule.
TIER_ORDER: tuple[str, ...] = (TIER_PRIMARY, TIER_SECONDARY)

#: The tiers a registry entry may declare. ``fallback`` is not among them: a
#: book is in the fallback tier by not being an affiliate, never by being
#: recorded as one.
DECLARABLE_TIERS: frozenset[str] = frozenset({TIER_PRIMARY, TIER_SECONDARY})

SIDE_OVER = "over"

SIDE_UNDER = "under"

#: The published field each side's posted price is read from. The customer's
#: price, not the model's: ``calibrated_fair_*_american`` is what the model
#: thinks the bet is worth and is nobody's offer.
SIDE_PRICE_FIELD: Mapping[str, str] = MappingProxyType(
    {SIDE_OVER: "over_odds", SIDE_UNDER: "under_odds"}
)

#: The published fields that identify which bet a row is about.
CONTRACT_FIELDS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "prop_type",
    "line_value",
)


class RoutingRefused(RuntimeError):
    """The router declined to name a sportsbook."""


def normalise_vendor(vendor: Any) -> str:
    """A published vendor value in the form the registry records keys in.

    Case and surrounding space only. Deliberately not more: collapsing
    punctuation or dropping suffixes is how ``betanysports`` would come to
    match ``BetAnything``, and the whole point of the registry is that an
    identifier is matched exactly or not at all.
    """
    return str(vendor or "").strip().lower()


def payout_multiple(american: Any) -> float | None:
    """What one unit returns in profit at these American odds.

    The comparable form of a price. American odds are not ordered as numbers
    -- ``+100`` beats ``-110`` and ``-105`` beats ``-110``, so comparing the
    integers would rank every favourite below every underdog. The profit
    multiple is monotone in how good the price is for the customer, which is
    the only ordering "best price" can mean.

    ``None`` for a missing or unusable price, which is a book not offering a
    side rather than a book offering it badly.
    """
    if american is None or isinstance(american, bool):
        return None

    try:
        value = float(american)
    except (TypeError, ValueError):
        return None

    if value != value or value in (float("inf"), float("-inf")):
        return None

    if value > 0:
        return value / 100.0

    if value < 0:
        return 100.0 / abs(value)

    return None


@dataclass(frozen=True)
class Book:
    """One of the eight books this page routes, as the registry records it."""

    registry_id: str
    display_name: str
    tier: str
    vendor_keys: tuple[str, ...]
    redirect_book: str | None
    missing: tuple[str, ...]

    @property
    def routable(self) -> bool:
        """Whether an offer from this book could ever be selected.

        A book with no proven identifier cannot be recognised in the data, so
        it is not that it loses the comparison -- it never enters it.
        """
        return bool(self.vendor_keys)

    @property
    def destination_known(self) -> bool:
        return self.redirect_book is not None


@dataclass(frozen=True)
class Offer:
    """One book's posted price for one exact wager."""

    vendor: str
    american_odds: int | float
    quote_id: Any

    @property
    def payout(self) -> float | None:
        return payout_multiple(self.american_odds)


@dataclass(frozen=True)
class Wager:
    """The bet itself, identified the way the published schema identifies it."""

    game_id: Any
    player_id: Any
    prop_type: str
    line_value: Any
    side: str

    @property
    def contract(self) -> tuple[Any, ...]:
        """Everything but the side: the market, before a direction is chosen."""
        return (self.game_id, self.player_id, self.prop_type, self.line_value)


@dataclass(frozen=True)
class Routed:
    """Which book was chosen, out of what, and what may be claimed about it."""

    tier: str
    offer: Offer | None
    book: Book | None
    destination: str | None
    display_name: str
    best_overall: bool
    considered: tuple[Offer, ...]
    reason: str

    @property
    def has_affiliate_destination(self) -> bool:
        return self.destination is not None

    @property
    def is_affiliate(self) -> bool:
        return self.tier in DECLARABLE_TIERS


class AffiliateRegistry:
    """The book facts, and the only place a destination may come from.

    Constructed from the committed configuration in the normal case. Tests
    construct it directly, which is also how the identifier-dependent half of
    the tier rule can be exercised at all: five of the eight books have no
    proven identifier, and inventing one to make a test pass would be the
    defect the registry exists to prevent.
    """

    def __init__(
        self,
        books: tuple[Book, ...],
        *,
        redirect_path: str,
        redirect_parameter: str,
        non_affiliate_names: Mapping[str, str] | None = None,
        source: str = "",
    ) -> None:
        self.books = books

        self.redirect_path = redirect_path

        self.redirect_parameter = redirect_parameter

        self.non_affiliate_names = MappingProxyType(
            dict(non_affiliate_names or {})
        )

        self.source = source

        by_vendor: dict[str, Book] = {}

        for book in books:
            if book.tier not in DECLARABLE_TIERS:
                raise RoutingRefused(
                    f"{book.registry_id} declares tier {book.tier!r}, which is "
                    f"not one of {sorted(DECLARABLE_TIERS)}"
                )

            for key in book.vendor_keys:
                if key in by_vendor and by_vendor[key] is not book:
                    raise RoutingRefused(
                        f"sportsbook identifier {key!r} is claimed by both "
                        f"{by_vendor[key].registry_id} and {book.registry_id}, "
                        "so which book an offer came from is not settled"
                    )

                by_vendor[key] = book

        self._by_vendor = MappingProxyType(by_vendor)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        registry_path: Path | None = None,
    ) -> AffiliateRegistry:
        root = Path(project_root or PROJECT_ROOT)

        path = (
            Path(registry_path)
            if registry_path is not None
            else root / REGISTRY_RELATIVE
        )

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RoutingRefused(
                f"the affiliate configuration at {path} could not be read "
                f"({error}), so there is no proven destination for any book "
                "and none will be invented"
            ) from error

        redirect = document.get("redirect") or {}

        redirect_path = str(redirect.get("path") or "")

        redirect_parameter = str(redirect.get("parameter") or "")

        if not redirect_path or not redirect_parameter:
            raise RoutingRefused(
                f"{path} records no redirect path and parameter, so there is "
                "no way to address a destination"
            )

        books: list[Book] = []

        for registry_id, entry in sorted(
            (document.get("books") or {}).items()
        ):
            books.append(
                Book(
                    registry_id=str(registry_id),
                    display_name=str(entry.get("display_name") or registry_id),
                    tier=str(entry.get("tier") or ""),
                    vendor_keys=tuple(
                        normalise_vendor(key)
                        for key in (entry.get("vendor_keys") or [])
                        if normalise_vendor(key)
                    ),
                    redirect_book=(
                        str(entry["redirect_book"])
                        if entry.get("redirect_book")
                        else None
                    ),
                    missing=tuple(str(fact) for fact in entry.get("missing") or []),
                )
            )

        return cls(
            tuple(books),
            redirect_path=redirect_path,
            redirect_parameter=redirect_parameter,
            non_affiliate_names=document.get("non_affiliate_books") or {},
            source=str(path),
        )

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------

    def book_for(self, vendor: Any) -> Book | None:
        """The affiliate book a published vendor value is, or ``None``.

        ``None`` covers both "not an affiliate" and "an affiliate whose
        identifier is unproven", because in the data those are the same thing:
        an unrecognised string.
        """
        return self._by_vendor.get(normalise_vendor(vendor))

    def tier_of(self, vendor: Any) -> str:
        book = self.book_for(vendor)

        return book.tier if book is not None else TIER_FALLBACK

    def display_name_for(self, vendor: Any) -> str:
        """What to call a book on the page.

        Affiliate names come from the registry's affiliate entries, everybody
        else's from the non-affiliate name map, and a book in neither is shown
        under the identifier the data used. Showing the raw key is the honest
        fallback: it is what the source said, and it is not a name this
        repository made up.
        """
        book = self.book_for(vendor)

        if book is not None:
            return book.display_name

        key = normalise_vendor(vendor)

        return self.non_affiliate_names.get(key) or str(vendor or "")

    def destination_for(self, vendor: Any) -> str | None:
        """Where a click on this book goes, or ``None`` if nowhere proven.

        The only function in this module that returns a URL, and it returns
        one only for a registry entry carrying a proven ``redirect_book``.
        There is no other path to a destination and no default.
        """
        book = self.book_for(vendor)

        if book is None or book.redirect_book is None:
            return None

        return (
            f"{self.redirect_path}?{self.redirect_parameter}="
            f"{book.redirect_book}"
        )

    def books_in_tier(self, tier: str) -> tuple[Book, ...]:
        return tuple(book for book in self.books if book.tier == tier)

    def missing_facts(self) -> dict[str, tuple[str, ...]]:
        """What is still unknown, by book. Empty when the registry is complete."""
        return {
            book.display_name: book.missing
            for book in self.books
            if book.missing
        }


# ----------------------------------------------------------------------
# reading wagers and offers out of published rows
# ----------------------------------------------------------------------


def wager_of(row: Mapping[str, Any]) -> Wager:
    """The exact wager a published row's model opinion is about."""
    side = str(row.get("model_preferred_side") or "").strip().lower()

    if side not in SIDE_PRICE_FIELD:
        raise RoutingRefused(
            f"a published row names side {row.get('model_preferred_side')!r}, "
            f"which is not one of {sorted(SIDE_PRICE_FIELD)}, so which bet it "
            "is about is not settled"
        )

    missing = [name for name in CONTRACT_FIELDS if row.get(name) is None]

    if missing:
        raise RoutingRefused(
            "a published row is missing the fields that identify which wager "
            "it is about: " + ", ".join(missing)
        )

    return Wager(
        game_id=row["game_id"],
        player_id=row["player_id"],
        prop_type=str(row["prop_type"]),
        line_value=row["line_value"],
        side=side,
    )


def offer_in(row: Mapping[str, Any], *, side: str) -> Offer | None:
    """This row's book's price for ``side``, or ``None`` if it has none."""
    price = row.get(SIDE_PRICE_FIELD[side])

    if payout_multiple(price) is None:
        return None

    return Offer(
        vendor=normalise_vendor(row.get("vendor")),
        american_odds=price,
        quote_id=row.get("quote_id"),
    )


def offers_for(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    wager: Wager,
) -> tuple[Offer, ...]:
    """Every book's price for exactly this wager, in a stable order.

    Exactly: a row enters only if all four contract fields match and it quotes
    the side. A different line, player, prop or game is a different bet.
    """
    found: list[Offer] = []

    for row in rows:
        if tuple(row.get(name) for name in CONTRACT_FIELDS) != wager.contract:
            continue

        offer = offer_in(row, side=wager.side)

        if offer is not None:
            found.append(offer)

    return tuple(sorted(found, key=_offer_order))


def _offer_order(offer: Offer) -> tuple[str, str]:
    """Stable identifier ordering: vendor, then the quote's own id.

    The tie-break, and the sort every offer list is kept in. Vendor first
    because that is the stable sportsbook identifier; the quote id second so
    that two quotes from one book -- which the publisher's logical key makes
    impossible, but which this module does not get to assume -- still order
    deterministically rather than by list position.
    """
    return (offer.vendor, str(offer.quote_id))


def best_offer(offers: tuple[Offer, ...]) -> Offer | None:
    """The best customer price, ties broken by stable identifier ordering."""
    priced = [offer for offer in offers if offer.payout is not None]

    if not priced:
        return None

    return min(
        priced,
        key=lambda offer: (-(offer.payout or 0.0), *_offer_order(offer)),
    )


# ----------------------------------------------------------------------
# the rule
# ----------------------------------------------------------------------


def route(
    offers: tuple[Offer, ...],
    *,
    registry: AffiliateRegistry,
) -> Routed:
    """Apply the frozen tier rule to one wager's offer universe.

    Primary affiliates, then secondary affiliates, then everybody else; best
    price inside whichever tier was reached. No offers at all is answered with
    no selection, because the alternative would be a button that bets nothing.
    """
    considered = tuple(sorted(offers, key=_offer_order))

    if not considered:
        return Routed(
            tier=TIER_NONE,
            offer=None,
            book=None,
            destination=None,
            display_name="",
            best_overall=False,
            considered=(),
            reason="no sportsbook in the published data offers this exact wager",
        )

    overall = best_offer(considered)

    for tier in TIER_ORDER:
        eligible = tuple(
            offer
            for offer in considered
            if registry.tier_of(offer.vendor) == tier
        )

        selected = best_offer(eligible)

        if selected is not None:
            return _selected(
                selected,
                tier=tier,
                registry=registry,
                considered=considered,
                overall=overall,
                reason=(
                    f"the best price among {tier} affiliate books offering "
                    "this exact wager"
                ),
            )

    selected = best_offer(considered)

    if selected is None:
        return Routed(
            tier=TIER_NONE,
            offer=None,
            book=None,
            destination=None,
            display_name="",
            best_overall=False,
            considered=considered,
            reason=(
                "no sportsbook in the published data quotes a usable price "
                "for this exact wager"
            ),
        )

    return _selected(
        selected,
        tier=TIER_FALLBACK,
        registry=registry,
        considered=considered,
        overall=overall,
        reason=(
            "no primary or secondary affiliate book offers this exact wager, "
            "so the best available price from the remaining books"
        ),
    )


def _selected(
    offer: Offer,
    *,
    tier: str,
    registry: AffiliateRegistry,
    considered: tuple[Offer, ...],
    overall: Offer | None,
    reason: str,
) -> Routed:
    book = registry.book_for(offer.vendor)

    destination = (
        registry.destination_for(offer.vendor)
        if tier in DECLARABLE_TIERS
        else None
    )

    return Routed(
        tier=tier,
        offer=offer,
        book=book,
        destination=destination,
        display_name=registry.display_name_for(offer.vendor),
        best_overall=(
            overall is not None
            and offer.payout is not None
            and overall.payout is not None
            and offer.payout >= overall.payout
        ),
        considered=considered,
        reason=reason,
    )


def route_row(
    row: Mapping[str, Any],
    *,
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    registry: AffiliateRegistry,
) -> tuple[Wager, Routed]:
    """One published row's wager and where it routes within ``rows``."""
    wager = wager_of(row)

    return wager, route(offers_for(rows, wager), registry=registry)


__all__ = [
    "AffiliateRegistry",
    "Book",
    "CONTRACT_FIELDS",
    "DECLARABLE_TIERS",
    "Offer",
    "REGISTRY_RELATIVE",
    "Routed",
    "RoutingRefused",
    "SIDE_OVER",
    "SIDE_PRICE_FIELD",
    "SIDE_UNDER",
    "TIER_FALLBACK",
    "TIER_NONE",
    "TIER_ORDER",
    "TIER_PRIMARY",
    "TIER_SECONDARY",
    "Wager",
    "best_offer",
    "normalise_vendor",
    "offer_in",
    "offers_for",
    "payout_multiple",
    "publisher",
    "route",
    "route_row",
    "wager_of",
]
