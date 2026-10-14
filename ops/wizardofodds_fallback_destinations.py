"""Where a reader can reach a sportsbook that pays WizardOfOdds nothing.

A lookup and nothing else. The routing rule has already decided which
sportsbook a reader is being shown, and this answers one question about that
decision after the fact: is there a destination for it that the site has
established, or does the card show a price with nowhere to go.

WHY THIS IS NOT IN THE AFFILIATE REGISTRY
-----------------------------------------

Because a reader has to be able to tell the difference, and because a renderer
has to be unable to confuse them. The affiliate registry holds four books
WizardOfOdds is paid by, routed through the site's own redirect; this file
holds books it is not paid by, linked directly. Putting the two in one table
would mean one lookup returning both kinds of link, and the only thing keeping
a "sponsored" rel or an affiliate badge off a direct link would be a caller
remembering which kind it had just been handed.

Keeping them apart makes the separation structural. Nothing here can be
reached through :func:`wizardofodds_affiliate_routing.AffiliateRegistry
.destination_for`, nothing here carries a tracking parameter, and the
committed data says so in a ``link_policy`` block that
:func:`FallbackDestinations.load` enforces rather than merely documents.

WHAT IT MUST NEVER DO
---------------------

Influence a selection. The best available price is the best available price
whether or not WizardOfOdds happens to know a URL for the book posting it, so
this module is consulted strictly after :func:`wizardofodds_affiliate_routing
.route` has returned, and it has no way to express a preference: it maps a
sportsbook identifier to a string or to ``None``. A book with no destination
is still selected and still shown; it is shown without a button.
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

import wizardofodds_affiliate_routing as routing  # noqa: E402

#: Where the destinations live, relative to the repository root.
REGISTRY_RELATIVE = "ops/wizardofodds_fallback_destinations.json"

#: The contract this module was written against.
SUPPORTED_CONTRACT = 1

#: What a direct link must carry, and what it must not. ``sponsored`` says to a
#: search engine that a link was paid for; these were not, so claiming it would
#: be a false statement about a commercial relationship that does not exist.
REQUIRED_REL = "nofollow noopener"

FORBIDDEN_REL_TOKEN = "sponsored"

#: Substrings a direct destination may not contain. The redirect is the
#: affiliate path and belongs to the other registry; a query parameter that
#: looks like attribution would make a direct link into a referral.
FORBIDDEN_IN_URL: tuple[str, ...] = (
    "go.php",
    "btag=",
    "?aff=",
    "&aff=",
    "ref=WOO",
    "utm_",
    "clickid",
    "subid",
)


class DestinationRefused(RuntimeError):
    """The destination configuration was not usable, so nothing is linked."""


@dataclass(frozen=True)
class Destination:
    """One sportsbook, where it can be reached, and how that was established."""

    vendor_key: str
    display_name: str
    url: str
    verification_source: str
    verification_type: str


class FallbackDestinations:
    """Sportsbook identifier -> a direct destination, where one is proven.

    Constructed from the committed configuration in the normal case. Tests
    construct it directly where they need a book the committed file does not
    hold, rather than adding one to the committed file to make a test pass.
    """

    def __init__(
        self,
        destinations: tuple[Destination, ...],
        *,
        source: str = "",
    ) -> None:
        self.destinations = destinations

        self.source = source

        by_vendor: dict[str, Destination] = {}

        for destination in destinations:
            key = routing.normalise_vendor(destination.vendor_key)

            if not key:
                raise DestinationRefused(
                    "a destination entry names no sportsbook identifier, so "
                    "there is nothing it could be the destination of"
                )

            if key in by_vendor:
                raise DestinationRefused(
                    f"sportsbook identifier {key!r} is recorded twice, so "
                    "which destination it has is not settled"
                )

            _refuse_unusable(destination)

            by_vendor[key] = destination

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
    ) -> FallbackDestinations:
        root = Path(project_root or PROJECT_ROOT)

        path = (
            Path(registry_path)
            if registry_path is not None
            else root / REGISTRY_RELATIVE
        )

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DestinationRefused(
                f"the fallback destination configuration at {path} could not "
                f"be read ({error}), so no sportsbook has a proven direct "
                "destination and none will be invented"
            ) from error

        contract = document.get("contract_version")

        if contract != SUPPORTED_CONTRACT:
            raise DestinationRefused(
                f"{path} declares contract {contract!r} and this module was "
                f"written against {SUPPORTED_CONTRACT}, so the fields it "
                "would read are not the fields it would get"
            )

        _refuse_a_policy_that_permits_a_referral(document, path=path)

        entries = document.get("books")

        if not isinstance(entries, dict):
            raise DestinationRefused(
                f"{path} carries no books object, so it records no "
                "destinations"
            )

        destinations: list[Destination] = []

        for key, entry in sorted(entries.items()):
            if not isinstance(entry, dict):
                raise DestinationRefused(
                    f"{path} records {key!r} as {type(entry).__name__}, not "
                    "as a destination"
                )

            verification = entry.get("verification") or {}

            destinations.append(
                Destination(
                    vendor_key=str(key),
                    display_name=str(entry.get("display_name") or key),
                    url=str(entry.get("url") or ""),
                    verification_source=str(verification.get("source") or ""),
                    verification_type=str(verification.get("type") or ""),
                )
            )

        return cls(tuple(destinations), source=str(path))

    # ------------------------------------------------------------------
    # the one question this answers
    # ------------------------------------------------------------------

    def url_for(self, vendor: Any) -> str | None:
        """The direct destination for a sportsbook, or ``None``.

        ``None`` is an ordinary answer and not an error: most sportsbooks the
        market data carries have no established destination, and the page
        renders a price without a button for them. It is the only honest
        alternative to guessing a URL from a book's name.
        """
        destination = self._by_vendor.get(routing.normalise_vendor(vendor))

        return destination.url if destination is not None else None

    def display_name_for(self, vendor: Any) -> str | None:
        destination = self._by_vendor.get(routing.normalise_vendor(vendor))

        return destination.display_name if destination is not None else None

    def destination_for(self, vendor: Any) -> Destination | None:
        return self._by_vendor.get(routing.normalise_vendor(vendor))

    @property
    def vendor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_vendor))

    def linkable_books(self) -> Mapping[str, str]:
        """Display name -> destination, for a manifest a check can read."""
        return MappingProxyType(
            {
                destination.display_name: destination.url
                for destination in sorted(
                    self.destinations, key=lambda entry: entry.display_name
                )
            }
        )


def _refuse_unusable(destination: Destination) -> None:
    url = destination.url

    if not url.startswith("https://"):
        raise DestinationRefused(
            f"{destination.vendor_key} records {url!r}, which is not an "
            "https destination, so it is not something a reader is sent to"
        )

    for forbidden in FORBIDDEN_IN_URL:
        if forbidden in url:
            raise DestinationRefused(
                f"{destination.vendor_key} records a destination containing "
                f"{forbidden!r}. A direct link is not a referral: it carries "
                "no attribution and does not go through the affiliate "
                "redirect, so this entry is refused rather than rendered as "
                "though nobody were being paid"
            )

    if not destination.verification_source or not destination.verification_type:
        raise DestinationRefused(
            f"{destination.vendor_key} records a destination with no stated "
            "verification, and an unverified destination is a guess"
        )


def _refuse_a_policy_that_permits_a_referral(
    document: Mapping[str, Any], *, path: Path
) -> None:
    """The committed link policy, enforced rather than trusted.

    The file declares that nothing in it is paid for. That declaration decides
    how the page renders these links, so a file that had quietly relaxed it
    would change the page's claims about who pays whom, and is refused.
    """
    policy = document.get("link_policy") or {}

    if str(policy.get("rel") or "") != REQUIRED_REL:
        raise DestinationRefused(
            f"{path} declares rel {policy.get('rel')!r} for direct links and "
            f"{REQUIRED_REL!r} is required, so how these links would be "
            "rendered is not settled"
        )

    if FORBIDDEN_REL_TOKEN not in str(policy.get("rel_must_not_contain") or ""):
        raise DestinationRefused(
            f"{path} no longer forbids {FORBIDDEN_REL_TOKEN!r} on a direct "
            "link. That token states a link was paid for, and these were not"
        )

    for name in (
        "may_use_affiliate_redirect",
        "may_claim_affiliate",
        "may_append_tracking_parameters",
    ):
        if policy.get(name) is not False:
            raise DestinationRefused(
                f"{path} records {name} as {policy.get(name)!r}. A direct "
                "destination is not a referral and this file may not say "
                "otherwise"
            )


def fallback_direct_url(
    vendor: Any, *, project_root: Path | None = None
) -> str | None:
    """The direct destination for one sportsbook, read from the committed file.

    The convenience form, for a caller that has one identifier and no registry
    in hand. A caller rendering a page should load the registry once instead of
    reading the file per card.
    """
    return FallbackDestinations.load(project_root=project_root).url_for(vendor)


__all__ = [
    "Destination",
    "DestinationRefused",
    "FallbackDestinations",
    "FORBIDDEN_IN_URL",
    "REGISTRY_RELATIVE",
    "REQUIRED_REL",
    "SUPPORTED_CONTRACT",
    "fallback_direct_url",
]
