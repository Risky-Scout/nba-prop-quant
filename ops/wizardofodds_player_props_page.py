"""The WizardOfOdds NBA player-props page, built from a published release.

A renderer and nothing more. Every number on the page is copied out of the
incumbent's own publication; this module decides layout, ordering and which
sportsbook a reader is pointed at, and it computes no probability, no fair
price, no edge and no expected value. There is no second pricing system here
and no opinion of its own to have.

WHAT IT READS
-------------

One publication, through the publisher's own transport, in the order a
consumer would: ``current.json`` names a release, the release's manifest
describes it, and the manifest names the rows. Nothing is taken on trust --
the rows are hashed against the manifest, the manifest against the pointer,
and the authority fields are checked before a single row is rendered. A
destination mid-publication, or one serving a candidate, produces a refusal
rather than a page, because a page is a claim about what the model says and a
half-written release cannot support one.

WHAT IT WILL NOT PUT ON A PAGE
------------------------------

Provenance is rendered from an allow-list, not filtered by a deny-list. The
published manifest carries ``priced_market_artifact_path``, which is an
absolute path on whatever machine served the slate; it has no business in a
public document, and an allow-list means the next field like it is excluded by
default rather than by somebody remembering.

DETERMINISM
-----------

The output has no clock in it. "Generated" and "published" are the
publication's own ``served_at`` and ``promoted_at``, so building the same
release twice produces the same bytes and a rebuild that changed something is
a change in the release rather than in the hour.

NON-LIVE BY CONSTRUCTION
------------------------

The page canary destination is a canary the same way the feed canary is: every
path it can address resolves beneath its ``canary_subpath``, so it has no way
to express the live index one directory up, and publishing the live page is
not something this module can be asked to do by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402

#: What the page writes, beside each other, in the canary directory.
PAGE_NAME = "index.html"

PAGE_MANIFEST_NAME = "page_manifest.json"

#: The published document's contract this renderer was written against.
SUPPORTED_ROWS_CONTRACT = 1

#: The authority a publication must record to be renderable. A page is a
#: public claim about the incumbent, so a publication claiming anything else
#: is refused rather than relabelled.
REQUIRED_MODEL_AUTHORITY = publisher.MODEL_AUTHORITY

#: What the candidate's authority must still say. The shadow exists to be
#: measured, not published, and a destination that stopped saying so is not one
#: this page will read from.
REQUIRED_CANDIDATE_AUTHORITY = "shadow_only"

#: The published fields every row must carry for the page to render it. The
#: renderer's own requirement rather than the publisher's: a row missing one of
#: these would render as a blank where a number belongs.
REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "calibrated_q_over_nonpush",
    "calibrated_q_under_nonpush",
    "date",
    "freeze_id",
    "game_id",
    "line_value",
    "market_devig_q_over",
    "market_devig_q_under",
    "model_preferred_edge",
    "model_preferred_ev",
    "model_preferred_side",
    "opponent_id",
    "player_id",
    "player_name",
    "prop_type",
    "quote_id",
    "team_id",
    "vendor",
)

#: The provenance fields the page may show, and the label each is shown under.
#: An allow-list on purpose: see the module docstring.
RENDERED_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("slate_date", "Slate"),
    ("incumbent_version", "Incumbent version"),
    ("incumbent_authority", "Incumbent authority"),
    ("model_authority", "Published authority"),
    ("production_code_sha", "Production code SHA"),
    ("served_at", "Slate served"),
    ("input_state_fingerprint", "Input state fingerprint"),
)

#: Provenance fields that must never reach the page, whatever else changes.
#: Redundant beside the allow-list above and kept anyway, because a test that
#: names the field is a test that still fails if somebody turns the allow-list
#: into a deny-list.
WITHHELD_PROVENANCE: tuple[str, ...] = ("priced_market_artifact_path",)

#: Side -> the published probability field for that side.
SIDE_MODEL_PROBABILITY: Mapping[str, str] = {
    routing.SIDE_OVER: "calibrated_q_over_nonpush",
    routing.SIDE_UNDER: "calibrated_q_under_nonpush",
}

SIDE_MARKET_PROBABILITY: Mapping[str, str] = {
    routing.SIDE_OVER: "market_devig_q_over",
    routing.SIDE_UNDER: "market_devig_q_under",
}

#: How a selected offer may be described, by tier. A fallback selection is the
#: best price in the whole universe by construction -- it is only reached when
#: no affiliate offers the wager -- so it is the one case where "best available"
#: is a statement of fact rather than a sales line.
TIER_LABEL: Mapping[str, str] = {
    routing.TIER_PRIMARY: "Best Primary Affiliate Price",
    routing.TIER_SECONDARY: "Best Secondary Affiliate Price",
    routing.TIER_FALLBACK: "Best Available Price",
    routing.TIER_NONE: "No sportsbook offers this wager",
}

AFFILIATE_DISCLOSURE = (
    "Affiliate disclosure: links to Bovada, BetOnline and BetUS are paid "
    "affiliate links and WizardOfOdds may earn a commission if you open an "
    "account through them. Which sportsbook is featured is decided by the "
    "published routing rule, which prefers affiliate books over non-affiliate "
    "books at the same wager, so a featured price is not necessarily the best "
    "price available anywhere. Where a card says Best Available Price, no "
    "affiliate book offered that wager and the price shown is the best in the "
    "published data. Model numbers are never changed by this routing. Odds "
    "move; check the sportsbook before betting. 21+ where applicable, and "
    "nothing here is a guarantee of profit."
)


class PageRefused(RuntimeError):
    """The page builder declined to render anything."""


# ----------------------------------------------------------------------
# reading the publication
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Publication:
    """One release at a destination, read back and cross-checked."""

    release_id: str
    slate_date: str
    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_relative: str
    rows_relative: str
    rows_sha256: str
    promoted_at: str
    mode: str


def _pointer(transport: Any, relative: str, *, what: str) -> dict[str, Any]:
    if not transport.exists(relative):
        raise PageRefused(
            f"the destination has no {relative}, so there is no {what} to "
            "build a page from"
        )

    try:
        document = json.loads(transport.read_bytes(relative))
    except ValueError as error:
        raise PageRefused(
            f"{relative} is not readable JSON ({error}), so the {what} it "
            "describes cannot be rendered"
        ) from error

    if not isinstance(document, dict):
        raise PageRefused(
            f"{relative} is not a JSON object, so it does not describe a "
            f"{what}"
        )

    return document


def load_publication(transport: Any) -> Publication:
    """The release the destination says it is serving, verified.

    Read in the order a consumer reads it and checked at every hop, because
    each hop is a way the destination could be mid-publication: a pointer at a
    release whose manifest is not there yet, a manifest at rows that are not,
    or rows that are there but are not the ones the manifest describes.
    """
    current = _pointer(
        transport, publisher.CURRENT_POINTER, what="current release"
    )

    release_id = str(current.get("release_id") or "")

    manifest_relative = str(current.get("manifest_relative") or "")

    if not release_id or not manifest_relative:
        raise PageRefused(
            f"{publisher.CURRENT_POINTER} names no release and manifest, so "
            "what the destination is serving is not settled"
        )

    manifest = _pointer(transport, manifest_relative, what="release manifest")

    authority = str(manifest.get("model_authority") or "")

    if authority != REQUIRED_MODEL_AUTHORITY:
        raise PageRefused(
            f"{manifest_relative} records the model authority as "
            f"{authority or 'nothing'}, not {REQUIRED_MODEL_AUTHORITY!r}, so "
            "this is not an incumbent publication and no page is built from it"
        )

    candidate = str(manifest.get("candidate_authority") or "")

    if candidate != REQUIRED_CANDIDATE_AUTHORITY:
        raise PageRefused(
            f"{manifest_relative} records the candidate authority as "
            f"{candidate or 'nothing'}, not {REQUIRED_CANDIDATE_AUTHORITY!r}, "
            "so the shadow may no longer be shadow-only and no page is built "
            "from it"
        )

    for label, pointed, recorded in (
        ("release id", release_id, str(manifest.get("release_id") or "")),
        (
            "rows path",
            str(current.get("rows_relative") or ""),
            str(manifest.get("rows_relative") or ""),
        ),
        (
            "rows digest",
            str(current.get("rows_sha256") or ""),
            str(manifest.get("rows_sha256") or ""),
        ),
    ):
        if pointed != recorded:
            raise PageRefused(
                f"{publisher.CURRENT_POINTER} and {manifest_relative} "
                f"disagree on the {label} of the current release "
                f"({pointed or 'nothing'} against {recorded or 'nothing'}), "
                "so what the destination serves is not settled"
            )

    rows_relative = str(manifest.get("rows_relative") or "")

    rows_sha256 = str(manifest.get("rows_sha256") or "")

    if not rows_relative or not transport.exists(rows_relative):
        raise PageRefused(
            "the release manifest names rows at "
            f"{rows_relative or 'no path'}, which are not at the destination, "
            "so there is nothing to render"
        )

    rows_bytes = transport.read_bytes(rows_relative)

    served = publisher.digest(rows_bytes)

    if served != rows_sha256:
        raise PageRefused(
            f"{rows_relative} reads back as {served} but the manifest records "
            f"{rows_sha256}, so the destination is not serving the rows it "
            "says it is"
        )

    try:
        document = json.loads(rows_bytes)
    except ValueError as error:
        raise PageRefused(
            f"{rows_relative} is not readable JSON ({error})"
        ) from error

    if not isinstance(document, dict):
        raise PageRefused(f"{rows_relative} is not a JSON object")

    contract = document.get("contract_version")

    if contract != SUPPORTED_ROWS_CONTRACT:
        raise PageRefused(
            f"{rows_relative} declares rows contract {contract!r} and this "
            f"renderer was written against {SUPPORTED_ROWS_CONTRACT}, so the "
            "fields it would read are not the fields it would get"
        )

    rows = document.get("rows")

    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise PageRefused(
            f"{rows_relative} does not carry a list of row objects"
        )

    slate_date = str(manifest.get("slate_date") or "")

    approved_freeze = str(manifest.get("approved_freeze_id") or "")

    for index, row in enumerate(rows):
        missing = [name for name in REQUIRED_ROW_FIELDS if name not in row]

        if missing:
            raise PageRefused(
                f"row {index} of {rows_relative} is missing fields this page "
                "renders: " + ", ".join(missing)
            )

        if approved_freeze and str(row.get("freeze_id") or "") != approved_freeze:
            raise PageRefused(
                f"row {index} of {rows_relative} was produced by "
                f"{row.get('freeze_id')!r} and the release approves "
                f"{approved_freeze!r}, so it is not the incumbent's own output "
                "and is not rendered"
            )

    return Publication(
        release_id=release_id,
        slate_date=slate_date,
        rows=tuple(rows),
        provenance=dict(manifest.get("provenance") or {}),
        manifest=manifest,
        manifest_relative=manifest_relative,
        rows_relative=rows_relative,
        rows_sha256=rows_sha256,
        promoted_at=str(current.get("promoted_at") or ""),
        mode=str(manifest.get("mode") or current.get("mode") or ""),
    )


# ----------------------------------------------------------------------
# turning rows into what the page shows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PageRow:
    """One card: the model's opinion, and where to take it."""

    wager: routing.Wager
    routed: routing.Routed
    row: Mapping[str, Any] = field(repr=False)

    @property
    def side(self) -> str:
        return self.wager.side

    @property
    def model_probability(self) -> Any:
        return self.row[SIDE_MODEL_PROBABILITY[self.side]]

    @property
    def market_probability(self) -> Any:
        return self.row[SIDE_MARKET_PROBABILITY[self.side]]

    @property
    def edge(self) -> Any:
        return self.row["model_preferred_edge"]

    @property
    def expected_value(self) -> Any:
        return self.row["model_preferred_ev"]

    @property
    def matchup(self) -> str:
        """Who is playing, from the ids the publication actually carries.

        Team names are not published, so they are not shown. ``is_home``
        decides the direction, which is the one thing about the matchup the
        published fields do settle.
        """
        team = self.row.get("team_id")

        opponent = self.row.get("opponent_id")

        joiner = "vs" if _truthy(self.row.get("is_home")) else "@"

        return f"Team {team} {joiner} Team {opponent}"

    @property
    def game_order(self) -> tuple[str, str, str, str, str]:
        """Deterministic order for the game-time control.

        Tip-off times are not in the published feed, so this is the closest
        deterministic order over the fields that are: slate date, then game,
        then player, then prop, then line. The page says as much where the
        control is.
        """
        return (
            str(self.row.get("date") or ""),
            str(self.row.get("game_id") or ""),
            str(self.row.get("player_name") or ""),
            str(self.row.get("prop_type") or ""),
            str(self.row.get("line_value") or ""),
        )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}

    return bool(value)


def page_rows(
    publication: Publication, *, registry: routing.AffiliateRegistry
) -> tuple[PageRow, ...]:
    """Every published row, routed, strongest model edge first.

    The default order is the strongest edge, because that is what somebody
    opening the page is looking for. The sort is over the published
    ``model_preferred_edge`` and nothing derived from it: no ranking score was
    invented here.
    """
    built: list[PageRow] = []

    for row in publication.rows:
        wager, routed = routing.route_row(
            row, rows=publication.rows, registry=registry
        )

        built.append(PageRow(wager=wager, routed=routed, row=row))

    return tuple(
        sorted(
            built,
            key=lambda page_row: (
                -_number(page_row.edge),
                -_number(page_row.expected_value),
                page_row.game_order,
            ),
        )
    )


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------


def american(value: Any) -> str:
    """American odds the way a sportsbook writes them."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"

    return f"+{number}" if number > 0 else str(number)


def percent(value: Any, *, places: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def signed_percent(value: Any, *, places: int = 1) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return "—"

    return f"{number:+.{places}f}%"


def line_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{number:g}"


def prop_label(prop_type: Any) -> str:
    return str(prop_type or "").replace("_", " ").title()


def _json_for_html(payload: Any) -> str:
    """JSON safe to embed in a script element.

    ``<`` escaped rather than the whole string HTML-escaped, because the
    content of a ``script`` element is not parsed as HTML: escaping ampersands
    there would corrupt the data, while leaving ``</script`` intact would end
    the element early.
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


STYLE = """
:root{
  --bg:#0b0f16;--panel:#131a26;--panel-2:#182130;--line:#243040;
  --ink:#eaf0f8;--ink-dim:#93a3b8;--ink-faint:#64748b;
  --good:#22c55e;--bad:#ef4444;--warn:#f59e0b;
  --cta:#16a34a;--cta-hi:#22c55e;--cta-ink:#04140a;
  --aff:#f59e0b;--aff2:#38bdf8;
  --radius:14px;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
a{color:var(--aff2)}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
header.top{padding:4px 0 14px}
h1{margin:0 0 6px;font-size:1.5rem;letter-spacing:-.02em}
.sub{color:var(--ink-dim);font-size:.9rem;margin:0}
.notice{
  margin:14px 0 0;padding:10px 12px;border:1px solid var(--line);
  border-left:3px solid var(--warn);border-radius:8px;
  background:var(--panel);color:var(--ink-dim);font-size:.82rem;
}
.prov{
  margin:14px 0 0;padding:12px;border:1px solid var(--line);border-radius:var(--radius);
  background:var(--panel);display:grid;gap:10px;
  grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
}
.prov div{min-width:0}
.prov dt{margin:0;color:var(--ink-faint);font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}
.prov dd{margin:2px 0 0;font-size:.8rem;word-break:break-all;font-variant-numeric:tabular-nums}
.controls{
  position:sticky;top:0;z-index:5;margin:16px 0 0;padding:12px;
  border:1px solid var(--line);border-radius:var(--radius);
  background:rgba(19,26,38,.96);backdrop-filter:blur(8px);
  display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
}
.controls label{display:block;color:var(--ink-faint);font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.controls input,.controls select{
  width:100%;padding:10px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel-2);color:var(--ink);font-size:16px;
}
.controls .wide{grid-column:1/-1}
.count{padding:12px 2px 0;color:var(--ink-dim);font-size:.82rem}
.grid{
  display:grid;gap:14px;padding:12px 0 36px;
  grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
}
.card{
  border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);
  display:flex;flex-direction:column;overflow:hidden;
}
.card.aff{border-color:#3a3017;box-shadow:inset 0 0 0 1px rgba(245,158,11,.14)}
.card-head{padding:12px 14px 10px;border-bottom:1px solid var(--line)}
.who{font-weight:650;font-size:1.02rem;letter-spacing:-.01em}
.where{color:var(--ink-faint);font-size:.76rem;margin-top:2px}
.pick{
  display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;
  padding:12px 14px 0;
}
.pick .side{
  font-weight:750;font-size:1.22rem;letter-spacing:-.02em;
  padding:2px 9px;border-radius:8px;background:var(--panel-2);border:1px solid var(--line);
}
.pick .side.over{color:var(--good)}
.pick .side.under{color:var(--aff2)}
.pick .stat{color:var(--ink-dim);font-size:.92rem}
.nums{
  display:grid;grid-template-columns:repeat(2,1fr);gap:1px;
  margin:12px 0 0;background:var(--line);
}
.nums div{background:var(--panel);padding:9px 14px}
.nums dt{margin:0;color:var(--ink-faint);font-size:.66rem;text-transform:uppercase;letter-spacing:.06em}
.nums dd{margin:2px 0 0;font-size:1rem;font-variant-numeric:tabular-nums}
.pos{color:var(--good)}.neg{color:var(--bad)}
.offer{padding:12px 14px 14px;margin-top:auto;border-top:1px solid var(--line)}
.tier{
  display:inline-block;font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;
  padding:3px 8px;border-radius:999px;border:1px solid var(--line);color:var(--ink-dim);
}
.tier.primary{color:var(--cta-hi);border-color:#1d3b28;background:rgba(34,197,94,.08)}
.tier.secondary{color:var(--aff);border-color:#3a3017;background:rgba(245,158,11,.08)}
.book{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin:9px 0 0}
.book .name{font-weight:650}
.book .odds{font-size:1.3rem;font-weight:750;font-variant-numeric:tabular-nums}
.cta{
  display:flex;align-items:center;justify-content:center;gap:10px;
  margin:11px 0 0;padding:15px 16px;min-height:52px;
  border-radius:12px;border:0;text-decoration:none;
  background:linear-gradient(180deg,var(--cta-hi),var(--cta));
  color:var(--cta-ink);font-weight:800;font-size:1.02rem;letter-spacing:.01em;
  text-align:center;line-height:1.2;
}
.cta:active{transform:translateY(1px)}
.cta .amt{font-variant-numeric:tabular-nums}
.no-cta{
  margin:11px 0 0;padding:13px 14px;border-radius:12px;
  border:1px dashed var(--line);background:var(--panel-2);
  color:var(--ink-dim);font-size:.82rem;text-align:center;
}
.why{margin:8px 0 0;color:var(--ink-faint);font-size:.7rem}
.empty{
  padding:34px 16px;text-align:center;color:var(--ink-dim);
  border:1px dashed var(--line);border-radius:var(--radius);background:var(--panel);
}
footer.foot{padding:18px 0 40px;color:var(--ink-faint);font-size:.74rem;border-top:1px solid var(--line)}
@media (max-width:640px){
  .wrap{padding:12px}
  h1{font-size:1.25rem}
  .grid{grid-template-columns:1fr;gap:12px}
  .controls{grid-template-columns:1fr 1fr;position:static}
  .controls .wide{grid-column:1/-1}
  .cta{min-height:56px;font-size:1.06rem}
}
"""

SCRIPT = """
'use strict';
(function () {
  var cards = Array.prototype.slice.call(
    document.querySelectorAll('[data-card]')
  );
  var grid = document.getElementById('grid');
  var empty = document.getElementById('empty');
  var count = document.getElementById('count');
  var inputs = {
    search: document.getElementById('f-search'),
    game: document.getElementById('f-game'),
    prop: document.getElementById('f-prop'),
    book: document.getElementById('f-book'),
    sort: document.getElementById('f-sort')
  };

  function num(card, name) {
    var raw = card.getAttribute(name);
    var value = parseFloat(raw);
    return isNaN(value) ? -Infinity : value;
  }

  function apply() {
    var needle = (inputs.search.value || '').trim().toLowerCase();
    var game = inputs.game.value;
    var prop = inputs.prop.value;
    var book = inputs.book.value;
    var shown = 0;

    cards.forEach(function (card) {
      var ok =
        (!needle || card.getAttribute('data-search').indexOf(needle) !== -1) &&
        (game === 'all' || card.getAttribute('data-game') === game) &&
        (prop === 'all' || card.getAttribute('data-prop') === prop) &&
        (book === 'all' || card.getAttribute('data-book') === book);
      card.hidden = !ok;
      if (ok) shown += 1;
    });

    var mode = inputs.sort.value;
    var ordered = cards.slice().sort(function (a, b) {
      if (mode === 'ev') {
        return num(b, 'data-ev') - num(a, 'data-ev') ||
          a.getAttribute('data-order').localeCompare(b.getAttribute('data-order'));
      }
      if (mode === 'game') {
        return a.getAttribute('data-order').localeCompare(b.getAttribute('data-order'));
      }
      return num(b, 'data-edge') - num(a, 'data-edge') ||
        a.getAttribute('data-order').localeCompare(b.getAttribute('data-order'));
    });
    ordered.forEach(function (card) { grid.appendChild(card); });

    count.textContent =
      shown + (shown === 1 ? ' prediction' : ' predictions') +
      ' shown of ' + cards.length;
    if (empty) empty.hidden = shown !== 0 || cards.length === 0;
  }

  Object.keys(inputs).forEach(function (key) {
    if (!inputs[key]) return;
    inputs[key].addEventListener('input', apply);
    inputs[key].addEventListener('change', apply);
  });

  apply();
})();
"""


def _option(value: str, label: str) -> str:
    return f'<option value="{escape(value)}">{escape(label)}</option>'


def _select(
    element_id: str, label: str, options: Sequence[tuple[str, str]], *, wide: bool = False
) -> str:
    rendered = "".join(_option(value, text) for value, text in options)

    classes = ' class="wide"' if wide else ""

    return (
        f"<div{classes}><label for=\"{element_id}\">{escape(label)}</label>"
        f'<select id="{element_id}">{rendered}</select></div>'
    )


def _provenance_block(publication: Publication) -> str:
    cells: list[str] = []

    for name, label in RENDERED_PROVENANCE:
        value = publication.provenance.get(name)

        if value in (None, ""):
            continue

        cells.append(
            f"<div><dt>{escape(label)}</dt>"
            f"<dd>{escape(str(value))}</dd></div>"
        )

    cells.append(
        f"<div><dt>Release</dt><dd>{escape(publication.release_id)}</dd></div>"
    )

    cells.append(
        "<div><dt>Rows digest</dt>"
        f"<dd>{escape(publication.rows_sha256)}</dd></div>"
    )

    if publication.promoted_at:
        cells.append(
            "<div><dt>Published</dt>"
            f"<dd>{escape(publication.promoted_at)}</dd></div>"
        )

    cells.append(
        f"<div><dt>Mode</dt><dd>{escape(publication.mode)}</dd></div>"
    )

    return '<dl class="prov">' + "".join(cells) + "</dl>"


def _cta(page_row: PageRow) -> str:
    routed = page_row.routed

    if routed.offer is None:
        return (
            '<p class="no-cta">No sportsbook in the published data offers '
            "this exact wager, so there is nothing to bet here.</p>"
        )

    price = american(routed.offer.american_odds)

    name = routed.display_name or routed.offer.vendor

    if routed.destination is None:
        return (
            f'<p class="no-cta">Shown at <strong>{escape(name)}</strong> '
            f"{escape(price)}. WizardOfOdds has no affiliate link for this "
            "sportsbook, so this is a price, not a referral.</p>"
        )

    return (
        f'<a class="cta" href="{escape(routed.destination)}" '
        'target="_blank" rel="sponsored nofollow noopener">'
        f"<span>BET AT {escape(name.upper())}</span>"
        f'<span class="amt">{escape(price)}</span></a>'
    )


def _card(page_row: PageRow) -> str:
    row = page_row.row

    routed = page_row.routed

    side = page_row.side

    tier = routed.tier

    tier_label = TIER_LABEL.get(tier, tier)

    if tier == routing.TIER_PRIMARY and routed.best_overall:
        tier_label += " · also best overall"

    book_name = routed.display_name or routed.offer.vendor if routed.offer else ""

    search = " ".join(
        str(value).lower()
        for value in (
            row.get("player_name"),
            prop_label(row.get("prop_type")),
            page_row.matchup,
            book_name,
            side,
        )
        if value
    )

    nums = (
        ("Model probability", percent(page_row.model_probability), ""),
        ("Market probability", percent(page_row.market_probability), ""),
        (
            "Edge",
            signed_percent(page_row.edge),
            "pos" if _number(page_row.edge) > 0 else "neg",
        ),
        (
            "Expected value",
            signed_percent(page_row.expected_value),
            "pos" if _number(page_row.expected_value) > 0 else "neg",
        ),
    )

    numbers = "".join(
        f"<div><dt>{escape(label)}</dt>"
        f'<dd class="{klass}">{escape(text)}</dd></div>'
        for label, text, klass in nums
    )

    offer_bits = ""

    if routed.offer is not None:
        offer_bits = (
            '<div class="book">'
            f'<span class="name">{escape(book_name)}</span>'
            f'<span class="odds">{escape(american(routed.offer.american_odds))}</span>'
            "</div>"
        )

    considered = len(routed.considered)

    return (
        f'<article class="card{" aff" if routed.is_affiliate else ""}" data-card'
        f' data-game="{escape(str(row.get("game_id")))}"'
        f' data-prop="{escape(str(row.get("prop_type")))}"'
        f' data-book="{escape(routing.normalise_vendor(routed.offer.vendor) if routed.offer else "")}"'
        f' data-edge="{escape(str(page_row.edge))}"'
        f' data-ev="{escape(str(page_row.expected_value))}"'
        f' data-order="{escape("|".join(page_row.game_order))}"'
        f' data-tier="{escape(tier)}"'
        f' data-search="{escape(search)}">'
        '<div class="card-head">'
        f'<div class="who">{escape(str(row.get("player_name") or ""))}</div>'
        f'<div class="where">{escape(page_row.matchup)} · '
        f'{escape(str(row.get("date") or ""))}</div>'
        "</div>"
        '<div class="pick">'
        f'<span class="side {escape(side)}">{escape(side.upper())} '
        f'{escape(line_text(row.get("line_value")))}</span>'
        f'<span class="stat">{escape(prop_label(row.get("prop_type")))}</span>'
        "</div>"
        f'<dl class="nums">{numbers}</dl>'
        '<div class="offer">'
        f'<span class="tier {escape(tier)}">{escape(tier_label)}</span>'
        f"{offer_bits}"
        f"{_cta(page_row)}"
        f'<p class="why">{escape(routed.reason)}. '
        f"{considered} sportsbook offer{'' if considered == 1 else 's'} "
        "matched this exact wager in the published data.</p>"
        "</div>"
        "</article>"
    )


def render(
    publication: Publication,
    rows: tuple[PageRow, ...],
    *,
    registry: routing.AffiliateRegistry,
) -> str:
    """The page, as one self-contained document.

    Every card is in the HTML. The script filters and reorders nodes that are
    already there, so the predictions and their calls to action are readable
    without it, and no filter can remove a call to action from a card it
    leaves visible.
    """
    games = sorted({str(row.row.get("game_id")) for row in rows})

    props = sorted({str(row.row.get("prop_type")) for row in rows})

    books = sorted(
        {
            (
                routing.normalise_vendor(row.routed.offer.vendor),
                row.routed.display_name or row.routed.offer.vendor,
            )
            for row in rows
            if row.routed.offer is not None
        }
    )

    controls = "".join(
        (
            '<div class="wide"><label for="f-search">Player search</label>'
            '<input id="f-search" type="search" inputmode="search" '
            'placeholder="Search player, team, prop or sportsbook" '
            'autocomplete="off"></div>',
            _select(
                "f-game",
                "Game",
                [("all", "All games")] + [(game, f"Game {game}") for game in games],
            ),
            _select(
                "f-prop",
                "Prop",
                [("all", "All props")]
                + [(prop, prop_label(prop)) for prop in props],
            ),
            _select(
                "f-book",
                "Sportsbook",
                [("all", "All sportsbooks")] + [(key, name) for key, name in books],
            ),
            _select(
                "f-sort",
                "Sort",
                [
                    ("edge", "Model edge (strongest first)"),
                    ("ev", "Expected value (highest first)"),
                    ("game", "Game time"),
                ],
            ),
        )
    )

    cards = "".join(_card(row) for row in rows)

    body = (
        cards
        if cards
        else (
            '<p class="empty">The current release carries no published rows, '
            "so there is nothing to show for this slate. No predictions were "
            "withheld and none were invented.</p>"
        )
    )

    missing = registry.missing_facts()

    missing_note = ""

    if missing:
        listed = "; ".join(
            f"{escape(book)}" for book in sorted(missing)
        )

        missing_note = (
            '<p class="notice">WizardOfOdds has no verified sportsbook '
            f"identifier or affiliate destination for {listed}. Those books "
            "are therefore never shown and never linked, rather than being "
            "guessed at.</p>"
        )

    embedded = {
        "note": (
            "Every value here is copied from the published release named "
            "below. This page computes no probability, fair price, edge or "
            "expected value."
        ),
        "release_id": publication.release_id,
        "rows_sha256": publication.rows_sha256,
        "slate_date": publication.slate_date,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<meta name="referrer" content="no-referrer-when-downgrade">
<title>NBA Player Props — Model Edges | WizardOfOdds</title>
<meta name="description" content="Model-priced NBA player prop edges for {escape(publication.slate_date)}, with the sportsbook offering each exact wager.">
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
<header class="top">
<h1>NBA Player Props — Model Edges</h1>
<p class="sub">Slate {escape(publication.slate_date)} · {len(rows)} published prediction{'' if len(rows) == 1 else 's'} · every number below is copied from the incumbent model's published release, unchanged.</p>
<p class="notice">{escape(AFFILIATE_DISCLOSURE)}</p>
{missing_note}
<p class="notice">Tip-off times are not part of the published feed, so they are not shown. The <em>Game time</em> sort orders by slate date, then game, then player, then prop, then line.</p>
{_provenance_block(publication)}
</header>
<section class="controls" aria-label="Filter and sort">{controls}</section>
<p class="count" id="count" aria-live="polite"></p>
<main class="grid" id="grid">{body}</main>
<p class="empty" id="empty" hidden>No prediction matches these filters.</p>
<footer class="foot">
<p>Published authority: incumbent. The research candidate is shadow-only and never appears here.</p>
<p>Source release {escape(publication.release_id)}, rows {escape(publication.rows_sha256)}, production code {escape(str(publication.provenance.get('production_code_sha') or 'unknown'))}.</p>
<p>Odds change without notice. Nothing on this page is a guarantee of profit, and no offer here is risk-free unless the sportsbook itself says so in its own terms.</p>
</footer>
</div>
<script type="application/json" id="source-note">{_json_for_html(embedded)}</script>
<script>{SCRIPT}</script>
</body>
</html>
"""


# ----------------------------------------------------------------------
# building and publishing
# ----------------------------------------------------------------------


def page_manifest(
    publication: Publication,
    rows: tuple[PageRow, ...],
    *,
    registry: routing.AffiliateRegistry,
    page_sha256: str,
) -> dict[str, Any]:
    """What this build was, in a form a check can read rather than parse HTML."""
    by_tier: dict[str, int] = {
        routing.TIER_PRIMARY: 0,
        routing.TIER_SECONDARY: 0,
        routing.TIER_FALLBACK: 0,
        routing.TIER_NONE: 0,
    }

    for row in rows:
        by_tier[row.routed.tier] = by_tier.get(row.routed.tier, 0) + 1

    return {
        "affiliate_registry": {
            "missing_facts": {
                book: list(facts)
                for book, facts in sorted(registry.missing_facts().items())
            },
            "primary_books": [
                book.display_name
                for book in registry.books_in_tier(routing.TIER_PRIMARY)
            ],
            "redirect_path": registry.redirect_path,
            "routable_books": sorted(
                book.display_name for book in registry.books if book.routable
            ),
            "secondary_books": [
                book.display_name
                for book in registry.books_in_tier(routing.TIER_SECONDARY)
            ],
            "source": Path(registry.source).name if registry.source else "",
        },
        "contract_version": 1,
        "mode": publication.mode,
        "page_name": PAGE_NAME,
        "page_sha256": page_sha256,
        "provenance": {
            name: publication.provenance.get(name)
            for name, _ in RENDERED_PROVENANCE
        },
        "released": {
            "manifest_relative": publication.manifest_relative,
            "release_id": publication.release_id,
            "rows_relative": publication.rows_relative,
            "rows_sha256": publication.rows_sha256,
            "row_count": len(publication.rows),
            "slate_date": publication.slate_date,
        },
        "rendered": {
            "affiliate_cta_count": sum(
                1 for row in rows if row.routed.has_affiliate_destination
            ),
            "by_tier": by_tier,
            "prediction_count": len(rows),
        },
        "routed": [
            {
                "best_overall": row.routed.best_overall,
                "destination": row.routed.destination,
                "display_name": row.routed.display_name,
                "offers_considered": len(row.routed.considered),
                "selected_odds": (
                    row.routed.offer.american_odds
                    if row.routed.offer is not None
                    else None
                ),
                "selected_vendor": (
                    row.routed.offer.vendor
                    if row.routed.offer is not None
                    else None
                ),
                "tier": row.routed.tier,
                "wager": {
                    "game_id": row.wager.game_id,
                    "line_value": row.wager.line_value,
                    "player_id": row.wager.player_id,
                    "prop_type": row.wager.prop_type,
                    "side": row.wager.side,
                },
            }
            for row in rows
        ],
    }


def build_page(
    transport: Any, *, registry: routing.AffiliateRegistry
) -> dict[str, Any]:
    """Read one publication and render the page and its manifest.

    No clock is read, so the same release renders the same bytes every time.
    """
    publication = load_publication(transport)

    rows = page_rows(publication, registry=registry)

    html = render(publication, rows, registry=registry)

    page_bytes = html.encode("utf-8")

    manifest = page_manifest(
        publication,
        rows,
        registry=registry,
        page_sha256=publisher.digest(page_bytes),
    )

    return {
        "files": {
            PAGE_NAME: page_bytes,
            PAGE_MANIFEST_NAME: publisher.serialise(manifest),
        },
        "manifest": manifest,
        "publication": publication,
        "rows": rows,
    }


def publish_page(
    *,
    source_transport: Any,
    page_transport: Any,
    registry: routing.AffiliateRegistry,
) -> dict[str, Any]:
    """Build the page from one publication and write it to the page canary.

    Two transports on purpose. The release is read through one and the page
    written through another, so the destination that holds the feed is only
    ever read from here and the destination that holds the page is the only one
    written to.
    """
    built = build_page(source_transport, registry=registry)

    for name, payload in sorted(built["files"].items()):
        page_transport.write_bytes(name, payload)

    readback: dict[str, Any] = {}

    for name in sorted(built["files"]):
        served = page_transport.read_bytes(name)

        if served != built["files"][name]:
            raise PageRefused(
                f"{name} at the page destination does not read back as what "
                "was written, so the page is not published"
            )

        if hasattr(page_transport, "public_readback"):
            public = page_transport.public_readback(name)

            if public != built["files"][name]:
                raise PageRefused(
                    f"{name} fetched from "
                    f"{page_transport.public_url(name)} is not what was "
                    "published, so what a reader would get is unproven"
                )

            readback[name] = {
                "sha256": publisher.digest(public),
                "url": page_transport.public_url(name),
            }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": built["manifest"],
        "public_readback": readback,
        "written": sorted(built["files"]),
    }


# ----------------------------------------------------------------------
# the entry point
# ----------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the WizardOfOdds NBA player-props page from a published "
            "incumbent release. Canary only: this entry point has no live "
            "mode and the page destinations it accepts write beneath a canary "
            "subpath."
        )
    )

    parser.add_argument(
        "--source-destination-name",
        default="wizardofodds_nba_canary",
        help="named destination the published release is read from",
    )

    parser.add_argument(
        "--source-destination-root",
        default=None,
        help="read the release from a local directory instead",
    )

    parser.add_argument(
        "--page-destination-name",
        default="wizardofodds_nba_player_props_page_canary",
        help="named destination the page is written to",
    )

    parser.add_argument(
        "--page-destination-root",
        default=None,
        help="write the page to a local directory instead",
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="repository root the configuration is read from",
    )

    return parser


def _transport_for(
    *, name: str | None, root: str | None, project_root: Path
) -> Any:
    if root is not None:
        return publisher.LocalDirectoryTransport(Path(root))

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY,
        publisher.read_destinations(project_root=project_root),
        name=name,
    )

    return publisher.build_transport(destination)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    root = Path(arguments.project_root or PROJECT_ROOT)

    registry = routing.AffiliateRegistry.load(project_root=root)

    source = _transport_for(
        name=arguments.source_destination_name,
        root=arguments.source_destination_root,
        project_root=root,
    )

    page = _transport_for(
        name=arguments.page_destination_name,
        root=arguments.page_destination_root,
        project_root=root,
    )

    try:
        receipt = publish_page(
            source_transport=source,
            page_transport=page,
            registry=registry,
        )
    except (PageRefused, routing.RoutingRefused, publisher.PublishingRefused) as error:
        print(json.dumps({"outcome": "REFUSED", "reason": str(error)}, indent=2))

        return publisher.EXIT_REFUSED
    finally:
        source.close()

        page.close()

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return publisher.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
