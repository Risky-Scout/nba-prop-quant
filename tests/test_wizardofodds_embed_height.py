"""The published page, inside the frame the site puts it in.

The WizardOfOdds site does not serve a predictions page at its own URL: it
embeds one in an iframe and sizes that frame from a message the page sends it.
Every other embedded predictions page on the site -- the football index, the
NCAAF and NFL game pages, the world-cup pages -- carries the same handshake,
wraps its content in the same element and posts the same message key. A page
that does not is not broken in any way a reader or a screenshot can see; it is
simply stuck at whatever default height the frame was given, with its table
cut off or floating in empty space.

So the handshake is pinned here by executing it, in Node, against the page's
own markup -- see ``wizardofodds_embed_frame.js``. Three things are worth
saying out loud about what is asserted.

The message key and the measured element are the *site's*, not this page's.
They are asserted against the literal strings the site's own copies use, so
that renaming either in this repository fails here rather than in production,
where the symptom is a frame of the wrong size and no error anywhere.

The target origin is asserted to be a real origin and never ``*``. A mismatched
target origin is not an error -- the message is silently not delivered -- which
is the one failure mode of this contract that cannot be noticed by testing the
page on its own. Both origins the site uses are required to be addressed, for
the same reason: the copies on the site each name one, which is why the staging
copies had to be hand-edited, and why an embed on the other host sizes wrong.

Nothing here touches the model, the prices or the routing. The files that own
those are asserted unchanged by digest, as they are in the UI tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import wizardofodds_player_props_page as page  # noqa: E402

from test_wizardofodds_public_page_ui import (  # noqa: E402
    FROZEN_FILES,
    consumer,  # noqa: F401 - fixture
    consumer_html,  # noqa: F401 - fixture
    digest,
    proof_html,  # noqa: F401 - fixture
)

HARNESS = Path(__file__).resolve().parent / "wizardofodds_embed_frame.js"

#: The handshake as the site already runs it, quoted from the copies on the
#: server rather than from this repository. These are the terms the listener on
#: the other side was written against.
SITE_MESSAGE_KEY = "frameHeight"

SITE_WRAPPER_ID = "app-root"

#: The wrapper's opening tag, up to its own attributes. The element is the one
#: the site measures and has to stay that element; what else it carries is the
#: page's business -- the odds format a reader chose is set on it, because the
#: stylesheet needs one ancestor to switch every price at once -- and asserting
#: on the whole tag would turn that into an embed failure.
WRAPPER_OPEN = f'<div id="{SITE_WRAPPER_ID}"'

SITE_ORIGINS = ("https://wizardofodds.com", "https://dev.wizardofodds.com")


def embed_script(html: str) -> str:
    """The height handshake as published, and only it."""
    found = re.search(
        r'<script id="frame-height">(.*?)</script>', html, flags=re.S
    )

    assert found is not None, "the page carries no height handshake"

    return found.group(1)


def run_frame(html: str, tmp_path: Path, **options) -> dict:
    """What the page broadcasts, given a frame relationship and some heights."""
    if shutil.which("node") is None:  # pragma: no cover - CI has node
        pytest.fail("node is required to execute the page's own embed script")

    document = tmp_path / "page.html"

    document.write_text(html, encoding="utf-8")

    finished = subprocess.run(
        ["node", str(HARNESS), str(document), json.dumps(options)],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(finished.stdout)


# ======================================================================
# the contract itself
# ======================================================================


def test_the_page_carries_the_handshake_the_site_listens_for(consumer_html):
    """Not "a" height script: the one the site's receiver was written for."""
    script = embed_script(consumer_html)

    assert SITE_MESSAGE_KEY in script

    assert f"getElementById('{SITE_WRAPPER_ID}')" in script

    assert "postMessage" in script


def test_the_content_is_wrapped_in_the_element_the_site_measures(consumer_html):
    """The wrapper is what makes the broadcast height the content's height.

    Without it the script falls back to the body, whose height is the viewport
    in a frame that has not been sized yet -- so the first measurement would
    describe the frame rather than the page, and the frame would be sized to
    the size it already had.
    """
    assert WRAPPER_OPEN in consumer_html

    body = consumer_html.split('<body class="wf-body">', 1)[1]

    wrapper = body.split(WRAPPER_OPEN, 1)[1]

    content = wrapper.rsplit("</div>", 1)[0]

    for required in ("<h1 class=\"wf-h1\">", "<table", "<footer"):
        assert required in content, required


def test_the_wrapper_holds_the_whole_page_and_not_a_part_of_it(consumer_html):
    """A wrapper around half the page measures half the page."""
    opened = consumer_html.count(WRAPPER_OPEN)

    assert opened == 1

    wrapper = consumer_html.split(WRAPPER_OPEN, 1)[1]

    # Everything after the wrapper closes is script, not content.
    tail = wrapper.rsplit("</div>", 1)[1]

    assert "<table" not in tail

    assert "<footer" not in tail

    assert "<h1" not in tail


def test_the_manifest_states_the_embed_contract(consumer):
    """So a check can read the terms without parsing the page."""
    manifest = json.loads(
        consumer["files"][page.PAGE_MANIFEST_NAME].decode("utf-8")
    )

    assert manifest["embed"] == {
        "message_key": SITE_MESSAGE_KEY,
        "parent_origins": list(SITE_ORIGINS),
        "wrapper_id": SITE_WRAPPER_ID,
    }


# ======================================================================
# what it actually does, run
# ======================================================================


def test_an_embedded_page_broadcasts_its_measured_height(consumer_html, tmp_path):
    result = run_frame(
        consumer_html, tmp_path, framed=True, heights=[1480], wrapperId=SITE_WRAPPER_ID
    )

    assert result["at_load"], "an embedded page broadcast nothing at load"

    for entry in result["at_load"]:
        assert entry["message"] == {SITE_MESSAGE_KEY: 1480}


def test_a_page_opened_directly_broadcasts_nothing(consumer_html, tmp_path):
    """The published URL is reachable on its own, and reading it is not an embed.

    The guard matters beyond tidiness: ``window.parent`` is the window itself
    when the page is top-level, so a page without the guard would post messages
    to itself once a second forever.
    """
    result = run_frame(
        consumer_html, tmp_path, framed=False, heights=[1480, 1900]
    )

    assert result["posted_total"] == 0

    assert result["listener_names"] == {"window": [], "document": []}

    assert result["interval_ms"] == []


def test_the_height_goes_to_every_origin_the_site_serves_from(
    consumer_html, tmp_path
):
    """Both hosts, because a wrong target origin fails silently.

    Delivery is the browser's decision, not the page's: a message addressed to
    an origin that is not the parent's is dropped without an error on either
    side. Addressing both of the site's hosts means the real parent receives it
    exactly once wherever the page is embedded.
    """
    result = run_frame(
        consumer_html, tmp_path, framed=True, heights=[900], wrapperId=SITE_WRAPPER_ID
    )

    assert result["origins"] == list(SITE_ORIGINS)

    for origin in result["origins"]:
        assert origin.startswith("https://")

    assert "*" not in result["origins"]


def test_the_page_never_addresses_its_height_to_any_origin(consumer_html):
    """``*`` would hand the page's height to whatever site framed it."""
    script = embed_script(consumer_html)

    assert "'*'" not in script

    assert '"*"' not in script

    quoted = re.findall(r"https://[a-z0-9.\-]+", script)

    assert set(quoted) == set(SITE_ORIGINS)


def test_a_relayout_is_broadcast_again(consumer_html, tmp_path):
    """The filters change how tall the page is, so the frame has to be told."""
    result = run_frame(
        consumer_html,
        tmp_path,
        framed=True,
        heights=[1480, 620],
        wrapperId=SITE_WRAPPER_ID,
    )

    [relayout] = result["relayouts"]

    assert relayout["posted"], "the page got shorter and said nothing"

    for entry in relayout["posted"]:
        assert entry["message"] == {SITE_MESSAGE_KEY: 620}


def test_an_unchanged_height_is_not_rebroadcast(consumer_html, tmp_path):
    """Once a second, forever, is a lot of messages to send about nothing.

    The threshold is also what keeps the observers from turning a one-pixel
    rounding difference into a permanent conversation with the parent.
    """
    result = run_frame(
        consumer_html,
        tmp_path,
        framed=True,
        heights=[1480, 1480, 1481],
        wrapperId=SITE_WRAPPER_ID,
    )

    for relayout in result["relayouts"]:
        assert relayout["posted"] == [], relayout["height"]


def test_the_page_keeps_watching_after_it_has_loaded(consumer_html, tmp_path):
    """Load alone is not enough: the table is filtered after load, not before."""
    result = run_frame(
        consumer_html, tmp_path, framed=True, heights=[1480], wrapperId=SITE_WRAPPER_ID
    )

    assert "load" in result["listener_names"]["window"]

    assert "DOMContentLoaded" in result["listener_names"]["document"]

    assert "click" in result["listener_names"]["document"]

    assert result["click_handlers"] == 1

    assert result["click_delay_ms"] == 150

    assert result["interval_ms"] == [1000]

    kinds = sorted(entry["kind"] for entry in result["observed"])

    assert kinds == ["mutation", "resize"]

    for entry in result["observed"]:
        assert entry["isBody"] is True


def test_a_measurement_that_fails_does_not_break_the_page(consumer_html, tmp_path):
    """The handshake is an enhancement; the predictions are the page.

    Run with no wrapper to measure, which is the shape of the problem in a
    browser that resolves the element late: the script has to fall back rather
    than throw inside a listener the rest of the page shares.
    """
    result = run_frame(
        consumer_html, tmp_path, framed=True, heights=[1480], wrapperId="not-present"
    )

    assert result["warned"] == []

    assert result["at_load"], "the fallback measurement broadcast nothing"

    for entry in result["at_load"]:
        assert entry["message"] == {SITE_MESSAGE_KEY: 1480}


# ======================================================================
# it reaches the QA page too, and nothing else moved
# ======================================================================


def test_the_routing_proof_page_is_embeddable_on_the_same_terms(
    proof_html, tmp_path
):
    """One renderer, so the QA page cannot drift away from the real one."""
    result = run_frame(
        proof_html, tmp_path, framed=True, heights=[1200], wrapperId=SITE_WRAPPER_ID
    )

    assert result["origins"] == list(SITE_ORIGINS)

    assert WRAPPER_OPEN in proof_html

    for entry in result["at_load"]:
        assert entry["message"] == {SITE_MESSAGE_KEY: 1200}


def test_the_handshake_is_the_last_word_and_does_not_eat_the_controls(
    consumer_html,
):
    """Two scripts, each identified, so neither can be mistaken for the other.

    The filter script is found by id by its own harness. An unnamed script
    appended after it would be picked up instead, and the count tests would
    quietly start asserting things about the height handshake.
    """
    assert consumer_html.count('<script id="page-controls">') == 1

    assert consumer_html.count('<script id="frame-height">') == 1

    controls = consumer_html.index('<script id="page-controls">')

    handshake = consumer_html.index('<script id="frame-height">')

    assert controls < handshake


def test_the_handshake_is_identical_on_every_build(consumer_html, proof_html):
    """A shared constant rather than two copies that can drift."""
    assert embed_script(consumer_html) == embed_script(proof_html)

    assert embed_script(consumer_html).strip() == page.EMBED_SCRIPT.strip()


@pytest.mark.parametrize("relative", sorted(FROZEN_FILES))
def test_the_routing_and_affiliate_files_are_untouched(relative):
    """An embed change that reached the routing would change one of these."""
    assert digest(PROJECT / relative) == FROZEN_FILES[relative]
