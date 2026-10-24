"""The ssh_rsync transport, and the real WizardOfOdds NBA canary it reaches.

``test_wizardofodds_incumbent_publisher.py`` proves the publisher's rules
against a local directory, and those rules are transport-independent by
construction. This file is about the transport that is not: a release going to
a real web server over SSH and rsync, verified in a private staging directory
before it is served, read back over the public URL after it is, and promoted
only then.

TWO TIERS, AND WHY
------------------

The tier that runs everywhere does not stub the transport's logic. It replaces
exactly three seams -- the SSH invocation, the rsync target, and the public URL
-- with local equivalents, and leaves everything else real: the same shell
scripts, run by the same ``bash``; the same ``rsync`` binary; the same
``sha256sum`` the destination would answer with; the same staging-then-place
ordering; the same digest comparisons; and a genuine HTTP fetch of the
published bytes. What it cannot prove is that a real host authenticates, and
that is the only thing it is not asked to.

The tier that needs the real destination proves that, and skips without the
credential. A checkout never holds it: the key is supplied by the environment
and is not in this repository.

WHAT THE SEAMS ARE
------------------

``_run_remote`` becomes ``bash -c`` on this machine, so the destination's
paths are temporary directories and ``id -un`` answers with this user.
``_upload`` becomes a local rsync to the staging path rather than one over
``-e ssh``. ``public_url`` becomes a URL on a local HTTP server rooted at the
published directory. Three overrides, and the refusals under test are produced
by the publisher's own code every time.
"""

from __future__ import annotations

import ast
import functools
import getpass
import http.server
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

#: The name of the configured destination this change exists for. Read from
#: the committed configuration by the tests that use it, so a rename is a
#: failure here rather than a silently skipped file.
NBA_DESTINATION = "wizardofodds_nba_canary"

#: Set to ``1`` to run the tier that contacts the real host. Opt-in rather
#: than inferred from the credential being present, because a key in the
#: environment is not consent to write to a production web server from
#: whatever happens to be running the suite.
REAL_CANARY_ENV = "NBA_PROP_TEST_WOO_SSH_CANARY"

#: The only credential variable the NBA transport is configured to read, and
#: the one the real tier needs.
NBA_CREDENTIAL_ENV = "NBA_WOO_SSH_PRIVATE_KEY"

#: An unusable but well-formed key, for the tier that runs without a host. The
#: transport requires the variable to be non-empty before it does anything,
#: and nothing in the local tier presents it to an sshd.
PLACEHOLDER_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "bm90LWEta2V5LXRoaXMtdGllci1uZXZlci1hdXRoZW50aWNhdGVzLXdpdGg=\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)

#: The NBA publishing surface, for the two tests that are about what it does
#: not name. ``models/`` is excluded deliberately: the frozen manifests record
#: a ``git status`` line captured in August 2026 that happens to mention an
#: unrelated football file, and they are frozen artifacts whose bytes are
#: pinned by other tests. Scanning them would be testing history rather than
#: this transport.
TRANSPORT_SURFACE_DIRECTORIES: tuple[str, ...] = (
    "ops",
    "scripts",
    "src",
    ".github/workflows",
)

#: The football deploy account, and the football credential variables. Named
#: here so the ban is on the real strings.
FOOTBALL_DEPLOY_USER = "massey-deploy"
FOOTBALL_CREDENTIAL_PREFIX = "WIZARD_SSH_"


def requires(executable: str) -> Any:
    return pytest.mark.skipif(
        shutil.which(executable) is None,
        reason=f"{executable} is not on PATH",
    )


def real_canary_enabled() -> bool:
    return os.environ.get(REAL_CANARY_ENV) == "1" and bool(
        (os.environ.get(NBA_CREDENTIAL_ENV) or "").strip()
    )


real_canary = pytest.mark.skipif(
    not real_canary_enabled(),
    reason=(
        f"set {REAL_CANARY_ENV}=1 and supply {NBA_CREDENTIAL_ENV} to run the "
        "tier that publishes to the real WizardOfOdds NBA canary"
    ),
)


# ======================================================================
# the local tier's transport
# ======================================================================


def local_rsync_argv(
    transport: publisher.SshRsyncTransport, sent: Path, target: str
) -> list[str]:
    """The transport's own rsync command, addressed on this machine.

    Derived from :meth:`publisher.SshRsyncTransport._rsync_argv` rather than
    restated, so the local tier runs the option set production runs. A
    restated copy is how a portability defect survives its own fix: the
    option that the macOS runner's rsync rejected was in both, and a test
    carrying its own argv can keep passing while production cannot transfer.

    Two elements are replaced and nothing else is: the ``-e`` remote shell,
    which has no meaning for a transfer that does not leave this machine, and
    the ``user@host:path`` destination, which becomes the path alone.
    """
    argv = transport._rsync_argv(sent, target)

    shell = argv.index("-e")

    argv = argv[:shell] + argv[shell + 2 :]

    assert argv[-1].endswith(f":{target}"), argv

    argv[-1] = target

    return argv


class LoopbackSshRsync(publisher.SshRsyncTransport):
    """The real transport with its three outward seams pointed at this machine.

    The hooks are the defects the publisher is required to refuse, injected
    where they actually occur rather than where they are convenient. A partial
    upload is a prefix of the payload arriving, so ``truncate_to`` sends a
    prefix. A staging file that does not match what was sent is corruption
    between the transfer and the check, so ``corrupt_staging`` writes over it
    after rsync returns. A served file that does not match the staging one is
    corruption between the check and the placement, so ``corrupt_served``
    writes over it after the move.

    Each hook fires once and then clears itself, so a test can break one write
    of a release and leave the rest of the run genuine.
    """

    def __init__(self, destination: publisher.Destination, **keywords: Any) -> None:
        super().__init__(destination, **keywords)

        self.truncate_to: int | None = None
        self.corrupt_staging: bytes | None = None
        self.corrupt_served: bytes | None = None
        self.refuse_with: str | None = None
        #: Refuse only the remote commands whose script contains this, which
        #: is how a test breaks one step of a run and lets the rest happen.
        #: ``refuse_with`` breaks every command and cannot distinguish a
        #: destination that rejected the key from one that rejected a chmod.
        self.refuse_matching: str | None = None
        self.rsync_argv: list[list[str]] = []
        self.scripts: list[str] = []
        self.public_origin = ""

    # -- seam one: the SSH invocation ---------------------------------

    def _run_remote(self, script: str) -> str:
        if self.refuse_with is not None:
            raise publisher.RemoteCommandFailed(
                f"{self.remote.user}@{self.remote.host} refused a command "
                f"(exit 255): {self.refuse_with}"
            )

        if self.refuse_matching is not None and self.refuse_matching in script:
            self.scripts.append(script)

            raise publisher.RemoteCommandFailed(
                f"{self.remote.user}@{self.remote.host} refused a command "
                f"(exit 1): chmod: Operation not permitted"
            )

        self.scripts.append(script)

        completed = subprocess.run(
            ["bash", "-c", f"set -eu; {script}"],
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise publisher.RemoteCommandFailed(
                f"the loopback destination refused a command "
                f"(exit {completed.returncode}): "
                f"{(completed.stderr or completed.stdout).strip()}"
            )

        return completed.stdout

    # -- seam two: the rsync target -----------------------------------

    def _upload(self, safe: str, local: Path) -> None:
        target = self._staged(safe)

        self._run_remote(f"mkdir -p -- {shlex.quote(target.rsplit('/', 1)[0])}")

        sent = local

        if self.truncate_to is not None:
            sent = local.with_name(local.name + ".prefix")

            sent.write_bytes(local.read_bytes()[: self.truncate_to])

            self.truncate_to = None

        argv = local_rsync_argv(self, sent, target)

        self.rsync_argv.append(argv)

        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise publisher.RemoteCommandFailed(
                f"the loopback rsync of {safe} failed "
                f"(exit {completed.returncode}): {completed.stderr.strip()}"
            )

        # The transport's own step, run against this machine through the same
        # remote-shell seam. The option set above and the mode below are the
        # two halves of one arrangement, so the local tier exercises both
        # rather than asserting one and stubbing the other.
        self._set_staged_mode(target)

        if self.corrupt_staging is not None:
            Path(target).write_bytes(self.corrupt_staging)

            self.corrupt_staging = None

    def _place(self, safe: str) -> None:
        super()._place(safe)

        if self.corrupt_served is not None:
            Path(self._published(safe)).write_bytes(self.corrupt_served)

            self.corrupt_served = None

    # -- seam three: the public URL -----------------------------------

    def public_url(self, relative: str) -> str:
        return self.public_origin + self._relative(relative)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))

        return int(sock.getsockname()[1])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        return None


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    """An HTTP server over one directory, so the readback is a real fetch."""
    directory.mkdir(parents=True, exist_ok=True)

    port = free_port()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port),
        functools.partial(QuietHandler, directory=str(directory)),
    )

    threading.Thread(target=server.serve_forever, daemon=True).start()

    return server, f"http://127.0.0.1:{port}/"


@pytest.fixture
def loopback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A destination configured like the real one, addressed on this machine.

    Built by taking the committed NBA destination and replacing only its
    addresses, so the shape under test is the configured shape: the same
    transport, the same subpath confinement, the same retention and the same
    credential variable by name.
    """
    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    declared = dict(document[publisher.NAMED_DESTINATIONS_KEY][NBA_DESTINATION])

    declared["host"] = "127.0.0.1"
    declared["private_staging"] = str(tmp_path / "private")
    declared["public_target"] = str(tmp_path / "served")
    declared["public_base_url"] = "https://loopback.invalid/served/"
    declared["user"] = getpass.getuser()

    document[publisher.NAMED_DESTINATIONS_KEY] = {NBA_DESTINATION: declared}

    destinations = tmp_path / "destinations.json"

    destinations.write_text(json.dumps(document, indent=2), encoding="utf-8")

    monkeypatch.setenv(NBA_CREDENTIAL_ENV, PLACEHOLDER_KEY)

    document["source"] = str(destinations)

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY, document, name=NBA_DESTINATION
    )

    published = Path(destination.remote.published_root)

    server, origin = serve(published)

    transports: list[LoopbackSshRsync] = []

    #: Applied to every transport this fixture builds, which is how a test
    #: injects a defect into a run it otherwise lets happen normally. A list
    #: rather than a replaced factory so the fixture stays the only thing that
    #: knows how to build one.
    prepares: list[Any] = []

    def build(
        resolved: publisher.Destination, **keywords: Any
    ) -> LoopbackSshRsync:
        # The destination the publisher resolved, not the one this fixture
        # closed over. They are the same for an ordinary run -- the fixture's
        # own configuration file is what gets resolved -- and a test that
        # edits that file is testing resolution, so the edit has to reach here.
        transport = LoopbackSshRsync(resolved, **keywords)

        transport.public_origin = origin

        for prepare in prepares:
            prepare(transport)

        transports.append(transport)

        return transport

    monkeypatch.setattr(publisher, "build_transport", build)

    try:
        yield {
            "build": build,
            "destination": destination,
            "destinations_path": destinations,
            "prepare": prepares,
            "public_origin": origin,
            "published": published,
            "staging": Path(destination.remote.staging_root),
            "transports": transports,
        }
    finally:
        for transport in transports:
            transport.close()

        server.shutdown()

        server.server_close()


def publish(built: dict, loopback: dict, **overrides: Any) -> dict:
    arguments: dict[str, Any] = {
        "mode": publisher.MODE_CANARY,
        "slate_date": built["slate_date"],
        "data_root": built["data_root"],
        "project_root": PROJECT,
        "destinations_path": loopback["destinations_path"],
        "destination_name": NBA_DESTINATION,
    }

    arguments.update(overrides)

    return publisher.publish(**arguments)


def fresh(loopback: dict) -> LoopbackSshRsync:
    """A transport for an operation of its own.

    ``publish`` closes the transport it built, because a key on disk outlives
    its use for exactly as long as somebody leaves it there. So a rollback is
    a second operation with a second transport, which is also how the entry
    point does it.
    """
    return loopback["build"](loopback["destination"])


# ======================================================================
# 1 and 5. the transport publishes, and the canary is promoted
# ======================================================================


@requires("rsync")
@requires("bash")
def test_ssh_rsync_publishes_a_release_and_promotes_it(
    tmp_path: Path, loopback: dict
):
    """1 and 5. The whole flow, in the order the contract requires it.

    Staged before served, verified before promoted, and the pointer written
    last. Asserted on the artefacts rather than on the receipt alone: a
    receipt is the publisher's account of what it did, and the files are what
    it did.
    """
    built = fixture.build(tmp_path / "source")

    receipt = publish(built, loopback)

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["destination"]["transport"] == publisher.TRANSPORT_SSH_RSYNC
    assert receipt["destination"]["name"] == NBA_DESTINATION
    assert receipt["verification"]["verified"] is True
    assert receipt["promotion"]["promoted"] is True
    # The state the switch records, reported as a fact about the file rather
    # than as a claim about this run, which published a canary.
    assert receipt["live_publishing"] == publisher.STATE_ENABLED
    assert receipt["mode"] == publisher.MODE_CANARY
    assert receipt["model_authority"] == publisher.MODEL_AUTHORITY

    release = receipt["release"]

    rows = loopback["published"] / release["rows_relative"]

    assert rows.is_file()
    assert publisher.sha256_file(rows) == release["rows_sha256"]

    # Staged first, and the staging copy is still where the upload landed.
    staged = loopback["staging"] / release["rows_relative"]

    assert staged.is_file()
    assert publisher.sha256_file(staged) == release["rows_sha256"]

    # The pointer names the verified release.
    current = json.loads(
        (loopback["published"] / publisher.CURRENT_POINTER).read_text(
            encoding="utf-8"
        )
    )

    assert current["release_id"] == release["release_id"]
    assert current["rows_sha256"] == release["rows_sha256"]
    assert current["mode"] == publisher.MODE_CANARY

    # The staging root is not inside the served tree, so nothing unverified is
    # ever reachable through the destination's public URL.
    assert not str(loopback["staging"]).startswith(str(loopback["published"]))


@requires("rsync")
@requires("bash")
def test_the_served_tree_holds_only_the_canary_subpath(
    tmp_path: Path, loopback: dict
):
    """Everything written, enumerated, and all of it below the subpath.

    The served root on the real host also holds the legacy NBA pages. This is
    the property that keeps them out of reach: the destination addresses one
    directory, and the directory is not the served root.
    """
    built = fixture.build(tmp_path / "source")

    receipt = publish(built, loopback)

    release = receipt["release"]["release_id"]

    public_target = Path(loopback["destination"].remote.public_target)

    written = sorted(
        path.relative_to(public_target).as_posix()
        for path in public_target.rglob("*")
        if path.is_file()
    )

    subpath = loopback["destination"].remote.canary_subpath

    assert written == [
        f"{subpath}/{publisher.AUDIT_LOG_NAME}",
        f"{subpath}/{publisher.ATTESTATION_NAME}",
        f"{subpath}/{publisher.CURRENT_POINTER}",
        f"{subpath}/{publisher.RELEASES_DIRECTORY}/{release}/"
        f"{publisher.MANIFEST_NAME}",
        f"{subpath}/{publisher.RELEASES_DIRECTORY}/{release}/{publisher.ROWS_NAME}",
    ]

    assert all(name.startswith(f"{subpath}/") for name in written)


# ======================================================================
# 2. authentication failure
# ======================================================================


@requires("rsync")
@requires("bash")
def test_an_authentication_failure_refuses_and_promotes_nothing(
    tmp_path: Path, loopback: dict
):
    """2. The destination rejects the key, and nothing is written or promoted.

    The refusal has to carry what the destination said. An unauthorised key, a
    wrong user and an unreachable host are three different problems with three
    different fixes, and a publisher that reported them all the same way would
    make the operator guess.
    """
    built = fixture.build(tmp_path / "source")

    loopback["prepare"].append(
        lambda transport: setattr(
            transport, "refuse_with", "Permission denied (publickey)."
        )
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    assert "Permission denied (publickey)" in str(refusal.value)

    assert not (loopback["published"] / publisher.CURRENT_POINTER).exists()
    assert not (loopback["published"] / publisher.ATTESTATION_NAME).exists()
    assert list(loopback["published"].rglob("*")) == []


@requires("rsync")
@requires("bash")
def test_a_missing_credential_refuses_before_anything_is_addressed(
    tmp_path: Path, loopback: dict, monkeypatch: pytest.MonkeyPatch
):
    """An empty credential variable is a refusal, not an anonymous attempt."""
    built = fixture.build(tmp_path / "source")

    monkeypatch.setenv(NBA_CREDENTIAL_ENV, "   ")

    with pytest.raises(publisher.PublishingRefused, match="carries no private key"):
        publish(built, loopback)

    assert list(loopback["published"].rglob("*")) == []


# ======================================================================
# 3 and 4. a partial upload, and a staging file that does not match
# ======================================================================


@requires("rsync")
@requires("bash")
def test_a_partial_upload_refuses_and_serves_nothing(
    tmp_path: Path, loopback: dict
):
    """3. A prefix of the payload arrives, and the staging check catches it.

    The check that catches it is a digest the destination computed, not one
    this process remembers, which is the only kind that can see a transfer
    that ended early. Nothing reaches the served tree: the placement happens
    after this check, so a partial upload is refused while it is still
    unreachable.
    """
    built = fixture.build(tmp_path / "source")

    loopback["prepare"].append(
        lambda transport: setattr(transport, "truncate_to", 64)
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    message = str(refusal.value)

    assert "is partial or altered" in message
    assert str(loopback["staging"]) in message

    assert list(loopback["published"].rglob("*")) == []

    # The short file is in staging, where nobody can read it, and nowhere else.
    staged = [
        path
        for path in loopback["staging"].rglob("*")
        if path.is_file()
    ]

    assert len(staged) == 1
    assert staged[0].stat().st_size == 64


@requires("rsync")
@requires("bash")
def test_a_staging_checksum_mismatch_refuses(tmp_path: Path, loopback: dict):
    """4. The upload completed and the bytes are not the bytes.

    Distinct from a partial upload: the transfer succeeded, so rsync has
    nothing to report, and only comparing digests can tell. Refused before the
    placement for the same reason.
    """
    built = fixture.build(tmp_path / "source")

    loopback["prepare"].append(
        lambda transport: setattr(
            transport, "corrupt_staging", b'{"rows": [], "tampered": true}\n'
        )
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    assert "is partial or altered" in str(refusal.value)

    assert list(loopback["published"].rglob("*")) == []


@requires("rsync")
@requires("bash")
def test_a_served_copy_that_does_not_match_the_staged_one_refuses(
    tmp_path: Path, loopback: dict
):
    """The digest that matters is of what is served, so it is checked there too.

    A staging file can verify and the copy into the served tree still land
    wrong. Checking only the upload would make the publisher's guarantee one
    about its staging directory rather than about its destination.
    """
    built = fixture.build(tmp_path / "source")

    loopback["prepare"].append(
        lambda transport: setattr(
            transport, "corrupt_served", b'{"rows": [], "tampered": true}\n'
        )
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    message = str(refusal.value)

    assert "does not serve what was verified" in message

    assert not (loopback["published"] / publisher.CURRENT_POINTER).exists()


# ======================================================================
# 6. the public readback
# ======================================================================


@requires("rsync")
@requires("bash")
def test_the_public_readback_matches_the_source_row_for_row(
    tmp_path: Path, loopback: dict
):
    """6. Fetched over HTTP, parsed, and compared to the priced parquet.

    The readback through the transport proves the destination holds the
    release. This proves a consumer receives it, which is a different claim: a
    file can be correct on disk and unreachable, misaddressed, or shadowed by
    a web server rule. Compared against the priced source rather than against
    the payload the publisher built, so the chain the website depends on is
    checked end to end.
    """
    built = fixture.build(tmp_path / "source")

    receipt = publish(built, loopback)

    public = receipt["verification"]["public"]

    assert public["checked"] is True
    assert public["rows_sha256"] == receipt["release"]["rows_sha256"]
    assert public["manifest_sha256"] == receipt["release"]["manifest_sha256"]

    with urllib.request.urlopen(public["rows_url"], timeout=30) as response:
        body = response.read()

    assert publisher.digest(body) == receipt["release"]["rows_sha256"]

    served = json.loads(body)["rows"]

    source = pd.read_parquet(built["priced_path"])

    assert len(served) == len(source) == public["readback_rows"]

    by_quote = {int(row["quote_id"]): row for row in served}

    for _, row in source.iterrows():
        out = by_quote[int(row["id"])]

        for field in (
            *publisher.PUBLISHED_PROBABILITY_FIELDS,
            "calibrated_ev_over",
            "calibrated_ev_under",
            "calibrated_edge_over",
            "calibrated_edge_under",
            "over_odds",
            "under_odds",
            "calibrated_fair_over_american",
            "calibrated_fair_under_american",
            "line_value",
        ):
            assert float(out[field]) == float(row[field]), field


@requires("rsync")
@requires("bash")
def test_a_release_that_is_not_publicly_readable_is_not_promoted(
    tmp_path: Path, loopback: dict
):
    """A served file nobody can fetch is not a publication.

    The public origin is pointed at a port with nothing behind it, which is
    what an unreachable or misconfigured destination looks like from here. The
    bytes are correct at the destination and the pointer must still not move.
    """
    built = fixture.build(tmp_path / "source")

    loopback["prepare"].append(
        lambda transport: setattr(
            transport, "public_origin", f"http://127.0.0.1:{free_port()}/"
        )
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    assert "could not be read back" in str(refusal.value)

    assert not (loopback["published"] / publisher.CURRENT_POINTER).exists()


# ======================================================================
# 7. a rerun
# ======================================================================


@requires("rsync")
@requires("bash")
def test_a_rerun_over_the_same_slate_changes_nothing(
    tmp_path: Path, loopback: dict
):
    """7. The same rows land on the same release, and no pointer rotates.

    The release id is the payload digest, so this has to hold. It matters more
    on a remote destination than on a local one: a rerun that rotated the
    pointers would move the rollback target of a live web server, and the
    audit log would record a publication that published nothing new.
    """
    built = fixture.build(tmp_path / "source")

    first = publish(built, loopback)

    before = {
        path.relative_to(loopback["published"]).as_posix(): path.read_bytes()
        for path in loopback["published"].rglob("*")
        if path.is_file()
    }

    second = publish(built, loopback)

    assert second["release"]["release_id"] == first["release"]["release_id"]
    assert second["release"]["rows_sha256"] == first["release"]["rows_sha256"]
    assert second["promotion"]["promoted"] is False
    assert second["verification"]["public"]["checked"] is True

    after = {
        path.relative_to(loopback["published"]).as_posix(): path.read_bytes()
        for path in loopback["published"].rglob("*")
        if path.is_file()
    }

    assert not (loopback["published"] / publisher.PREVIOUS_POINTER).exists()

    # The audit log gained one line recording a no-op, and nothing else moved.
    assert sorted(after) == sorted(before)

    for name, payload in before.items():
        if name == publisher.AUDIT_LOG_NAME:
            continue

        assert after[name] == payload, name

    entries = [
        json.loads(line)
        for line in after[publisher.AUDIT_LOG_NAME]
        .decode("utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [entry["action"] for entry in entries] == [
        publisher.ACTION_PUBLISHED,
        publisher.ACTION_REPUBLISHED,
    ]


# ======================================================================
# 8. rollback, and retention
# ======================================================================


@requires("rsync")
@requires("bash")
def test_rollback_restores_the_previous_canary_release(
    tmp_path: Path, loopback: dict
):
    """8. Two releases, then back to the first, verified at the destination.

    Restoration rather than reconstruction: the previous release's bytes are
    still served, so the rollback is a pointer moving back. Checked against
    what the public URL serves as well as against what the destination holds,
    because a pointer restored to bytes nobody can fetch is a rollback to
    nothing.
    """
    first_built = fixture.build(tmp_path / "first")

    first = publish(first_built, loopback)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4712

    second_built = fixture.build(tmp_path / "second", frame=frame)

    second = publish(second_built, loopback)

    assert second["release"]["release_id"] != first["release"]["release_id"]
    assert second["promotion"]["promoted"] is True
    assert second["promotion"]["rollback_target"] == first["release"]["release_id"]

    transport = fresh(loopback)

    try:
        restored = publisher.rollback(transport)
    finally:
        transport.close()

    assert restored["outcome"] == publisher.OUTCOME_ROLLED_BACK
    assert restored["restored_release_id"] == first["release"]["release_id"]
    assert restored["superseded_release_id"] == second["release"]["release_id"]
    assert restored["rows_sha256"] == first["release"]["rows_sha256"]
    assert restored["public_rows_sha256"] == first["release"]["rows_sha256"]

    current = json.loads(
        (loopback["published"] / publisher.CURRENT_POINTER).read_text(
            encoding="utf-8"
        )
    )

    assert current["release_id"] == first["release"]["release_id"]

    # Both releases' bytes are still there: retention keeps the one served and
    # the one before it, which is what made this rollback possible.
    for receipt in (first, second):
        assert (
            loopback["published"] / receipt["release"]["rows_relative"]
        ).is_file()

    entries = [
        json.loads(line)
        for line in (loopback["published"] / publisher.AUDIT_LOG_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert entries[-1]["action"] == publisher.ACTION_ROLLED_BACK
    assert entries[-1]["release_id"] == first["release"]["release_id"]
    assert entries[-1]["model_authority"] == publisher.MODEL_AUTHORITY


@requires("rsync")
@requires("bash")
def test_retention_keeps_the_served_release_the_previous_one_and_the_receipts(
    tmp_path: Path, loopback: dict
):
    """A third release prunes the first, and never the rollback target.

    The legacy NBA output accumulated a dated snapshot per day until nobody
    could say what it was serving. This destination keeps two payloads and the
    records of everything, so the audit trail outlives the bytes.
    """
    published: list[str] = []

    for index, over in enumerate((0.41, 0.42, 0.43)):
        frame = fixture.priced_frame()

        frame.loc[0, "calibrated_q_over_nonpush"] = over

        built = fixture.build(tmp_path / f"slate{index}", frame=frame)

        published.append(publish(built, loopback)["release"]["release_id"])

    releases = sorted(
        path.name
        for path in (
            loopback["published"] / publisher.RELEASES_DIRECTORY
        ).iterdir()
    )

    assert releases == sorted(published[1:]), "retention kept the wrong releases"

    assert len(releases) == loopback["destination"].retained_releases

    # The receipts describe all three, including the one whose bytes are gone.
    entries = [
        json.loads(line)
        for line in (loopback["published"] / publisher.AUDIT_LOG_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [entry["release_id"] for entry in entries] == published

    # And the rollback target is one of the two kept, so rolling back works.
    transport = fresh(loopback)

    try:
        restored = publisher.rollback(transport)
    finally:
        transport.close()

    assert restored["restored_release_id"] == published[1]


# ======================================================================
# 9 and 10. live mode, and live approval
# ======================================================================


def test_live_mode_refuses_while_the_approval_is_not_carried(
    tmp_path: Path, loopback: dict
):
    """9. The real destination existing is not an argument for publishing live.

    Refused on the committed switch, before any destination is resolved and
    before the transport is built, so no key is read and nothing is
    contacted. The switch's state is ENABLED and that is not what decides
    this: the two approval variables are blockers until they are set, and the
    refusal names them.
    """
    built = fixture.build(tmp_path / "source")

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(
            built, loopback, mode=publisher.MODE_LIVE, environment={}
        )

    message = str(refusal.value)

    for variable in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
    ):
        assert variable in message

    assert list(loopback["published"].rglob("*")) == []


def test_live_mode_refuses_without_its_approval_even_when_enabled(
    tmp_path: Path, loopback: dict
):
    """10. The switch alone is not the approval.

    Enabling the committed state still leaves the environment variable and the
    token, which are the two conditions a commit cannot satisfy by itself.
    """
    built = fixture.build(tmp_path / "source")

    switch = tmp_path / "switch.json"

    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    document["modes"][publisher.MODE_LIVE] = {
        "state": publisher.STATE_ENABLED,
        "approval_token": "the-real-token",
        "reason": "enabled for this test",
    }

    switch.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(
            built,
            loopback,
            mode=publisher.MODE_LIVE,
            switch_path=switch,
            environment={},
        )

    message = str(refusal.value)

    assert "does not carry its approval" in message
    assert publisher.PUBLISH_ENV_VAR in message
    assert publisher.PUBLISH_APPROVAL_ENV_VAR in message

    assert list(loopback["published"].rglob("*")) == []


def test_the_named_canary_destination_cannot_be_used_as_a_live_one(
    tmp_path: Path, loopback: dict
):
    """A destination built for one mode is not reinterpreted as another.

    Checked with every switch condition satisfied, so the refusal is the
    mode-to-name disagreement rather than the approval. Without this, the one
    configured remote destination would be reachable in live mode the moment
    an approval existed.
    """
    built = fixture.build(tmp_path / "source")

    switch = tmp_path / "switch.json"

    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    document["modes"][publisher.MODE_LIVE] = {
        "state": publisher.STATE_ENABLED,
        "approval_token": "the-real-token",
        "reason": "enabled for this test",
    }

    switch.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(
            built,
            loopback,
            mode=publisher.MODE_LIVE,
            switch_path=switch,
            environment={
                publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
                publisher.PUBLISH_APPROVAL_ENV_VAR: "the-real-token",
            },
        )

    assert "is not reinterpreted as another" in str(refusal.value)

    assert list(loopback["published"].rglob("*")) == []


# ======================================================================
# 11. a candidate or shadow artifact
# ======================================================================


@requires("rsync")
@requires("bash")
def test_a_shadow_candidate_frame_refuses_over_this_transport_too(
    tmp_path: Path, loopback: dict
):
    """11. A candidate probability never reaches the real destination.

    The refusal is the publisher's, not the transport's, and it happens before
    a release is built -- so the test's value here is that adding a remote
    transport did not create a path around it.
    """
    frame = fixture.priced_frame()

    frame["candidate_probability"] = 0.5
    frame["served_model"] = "candidate"

    built = fixture.build(tmp_path / "source", frame=frame)

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    message = str(refusal.value)

    assert "shadow-candidate columns" in message
    assert "candidate_probability" in message

    assert list(loopback["published"].rglob("*")) == []


@requires("rsync")
@requires("bash")
def test_a_non_incumbent_receipt_refuses_over_this_transport_too(
    tmp_path: Path, loopback: dict
):
    """Only the incumbent is published, whatever the destination."""
    built = fixture.build(
        tmp_path / "source", receipt_changes={"model_authority": "candidate"}
    )

    with pytest.raises(publisher.PublishingRefused, match="model authority"):
        publish(built, loopback)

    assert list(loopback["published"].rglob("*")) == []


# ======================================================================
# 12. the SSH user
# ======================================================================


@requires("rsync")
@requires("bash")
def test_a_destination_that_authenticates_as_another_user_refuses(
    tmp_path: Path, loopback: dict
):
    """12. The key worked and the account is not the configured one.

    The quietest way a deployment goes wrong. Authenticating as somebody else
    succeeds, writes files that account owns, and looks like a working
    publication until the ownership matters -- so the user the destination
    reports is checked against the user it is configured for, before anything
    is written.
    """
    built = fixture.build(tmp_path / "source")

    document = json.loads(
        loopback["destinations_path"].read_text(encoding="utf-8")
    )

    document[publisher.NAMED_DESTINATIONS_KEY][NBA_DESTINATION]["user"] = (
        "somebody-else"
    )

    loopback["destinations_path"].write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(built, loopback)

    message = str(refusal.value)

    assert "authenticated this key as" in message
    assert "somebody-else" in message

    assert list(loopback["published"].rglob("*")) == []


def test_the_configured_nba_user_is_the_nba_deploy_account():
    """The account the destination is configured for, read off the config."""
    declared = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )[publisher.NAMED_DESTINATIONS_KEY][NBA_DESTINATION]

    assert declared["user"] == "nba-prop-deploy"
    assert declared["port"] == 22
    assert declared["transport"] == publisher.TRANSPORT_SSH_RSYNC
    assert declared["mode"] == publisher.MODE_CANARY
    assert declared["credential_env"] == NBA_CREDENTIAL_ENV
    assert declared["private_staging"] == "/var/www/.nba-prop-quant-deploy"
    assert declared["public_target"] == (
        "/var/www/sportsodds/tools/odds-scanner/predictions/nba"
    )
    assert declared["public_base_url"].endswith(
        "/tools/odds-scanner/predictions/nba/"
    )
    assert declared["public_base_url"].startswith("https://")


# ======================================================================
# 13 and 14. what the NBA transport does not name
# ======================================================================


def surface_files() -> list[Path]:
    paths: list[Path] = []

    for relative in TRANSPORT_SURFACE_DIRECTORIES:
        directory = PROJECT / relative

        if not directory.is_dir():
            continue

        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".json", ".yml", ".yaml", ".sh")
        )

    return paths


def test_the_nba_transport_never_references_the_football_deploy_account():
    """13. ``massey-deploy`` appears nowhere on the NBA publishing surface.

    A separate sport, a separate account and separate recurring publishing.
    The two deployments share a web server, which is exactly why the account
    this one uses has to be the one it was given rather than the one that was
    already there.
    """
    offenders = [
        path.relative_to(PROJECT).as_posix()
        for path in surface_files()
        if FOOTBALL_DEPLOY_USER in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert offenders == [], f"the football deploy account is named in: {offenders}"


def test_the_nba_transport_never_references_the_football_credentials():
    """14. No ``WIZARD_SSH_*`` variable is read anywhere on that surface.

    The NBA transport reads one credential variable and it is the NBA one.
    Pinned both ways: the football prefix appears nowhere, and the configured
    destination names the NBA variable.
    """
    offenders = [
        path.relative_to(PROJECT).as_posix()
        for path in surface_files()
        if FOOTBALL_CREDENTIAL_PREFIX
        in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert offenders == [], f"football credentials are named in: {offenders}"

    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    named = document[publisher.NAMED_DESTINATIONS_KEY]

    assert {
        declared["credential_env"] for declared in named.values()
    } == {NBA_CREDENTIAL_ENV}


def test_no_credential_material_is_committed_with_the_destination():
    """The configuration names a variable; it does not hold a key.

    The one way a remote transport turns a configuration file into a secret,
    closed by reading the committed bytes rather than by intending not to.
    """
    text = (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")

    for marker in (
        "PRIVATE KEY",
        "ssh-rsa ",
        "ssh-ed25519 ",
        "BEGIN RSA",
        "BEGIN OPENSSH",
    ):
        assert marker not in text, f"the destination configuration holds {marker}"


# ======================================================================
# the upload's permissions, and their portability
# ======================================================================


class HeldRsync:
    """Stands in for the module the transport runs rsync through.

    Narrower than patching ``subprocess.run`` itself: the only call it can
    answer is the transfer, and the transport's remote-shell seam is
    overridden separately, so a test that reaches anything else fails loudly
    rather than silently running a real command.
    """

    def __init__(self, transport: RecordedUpload) -> None:
        self.transport = transport

    def run(self, argv: list[str], **_: Any) -> subprocess.CompletedProcess:
        self.transport.argv.append(list(argv))

        return subprocess.CompletedProcess(
            argv,
            self.transport.rsync_returncode,
            "",
            self.transport.rsync_stderr,
        )


class RecordedUpload(publisher.SshRsyncTransport):
    """The real ``_upload``, with both of its outward calls held.

    These tests are about the two commands the transport builds and the order
    it runs them in, so the commands are captured rather than performed. The
    option set is proven against a real rsync by the local tier above, which
    runs the same argv; what cannot be proven there is what happens when the
    transfer fails, because a transfer to this machine does not.
    """

    def __init__(
        self, destination: publisher.Destination, **keywords: Any
    ) -> None:
        super().__init__(destination, **keywords)

        self.argv: list[list[str]] = []
        self.scripts: list[str] = []
        self.rsync_returncode = 0
        self.rsync_stderr = ""
        self.refuse_matching: str | None = None

    def _run_remote(self, script: str) -> str:
        self.scripts.append(script)

        if self.refuse_matching is not None and self.refuse_matching in script:
            raise publisher.RemoteCommandFailed(
                f"{self.remote.user}@{self.remote.host} refused a command "
                "(exit 1): chmod: changing permissions: Operation not "
                "permitted"
            )

        return ""


@pytest.fixture
def recorded(loopback: dict, monkeypatch: pytest.MonkeyPatch):
    transport = RecordedUpload(loopback["destination"])

    monkeypatch.setattr(publisher, "subprocess", HeldRsync(transport))

    try:
        yield transport
    finally:
        transport.close()


def chmod_scripts(transport: RecordedUpload) -> list[str]:
    return [script for script in transport.scripts if "chmod" in script]


@requires("rsync")
def test_the_upload_command_names_no_rsync_chmod_option(recorded: RecordedUpload):
    """The option the macOS runner rejected is gone, and nothing replaced it.

    ``--chmod`` is asserted absent in every spelling rather than only as the
    ``F600`` that failed: ``--chmod=600`` parses on the runner today and is
    the same bet on one implementation's option table, which is the bet that
    cost a slate.
    """
    recorded._upload("releases/r/rows.json", Path("/dev/null"))

    assert len(recorded.argv) == 1

    argv = recorded.argv[0]

    assert not any(item.startswith("--chmod") for item in argv), argv
    assert "F600" not in " ".join(argv)
    assert not any(item.startswith("--no-perms") for item in argv), argv
    assert not any(item.startswith("--perms") for item in argv), argv
    assert "-p" not in argv


@requires("rsync")
def test_the_upload_still_transfers_over_the_configured_ssh_destination(
    recorded: RecordedUpload,
):
    """Removing the option changed the permissions, not the transport.

    The remote shell is the transport's own ``ssh`` argv -- its key, port,
    batch mode and known-hosts file -- and the destination is still the
    staging path at the configured account and host.
    """
    recorded._upload("releases/r/rows.json", Path("/dev/null"))

    argv = recorded.argv[0]

    assert argv[0] == "rsync"
    assert "--checksum" in argv
    assert "--times" in argv

    assert argv[argv.index("-e") + 1] == shlex.join(recorded._ssh_argv())

    remote = recorded.remote

    assert argv[-1] == (
        f"{remote.user}@{remote.host}:"
        f"{recorded._staged('releases/r/rows.json')}"
    )

    # ``--`` before the operands, so a path is never read as an option.
    assert argv[-3] == "--"


@requires("rsync")
def test_a_successful_transfer_is_followed_by_an_exact_remote_chmod(
    recorded: RecordedUpload,
):
    """One file, by name, after the transfer and not before it.

    The mode is the one ``--chmod=F600`` expressed, so the file the
    destination holds is the file it held before this fix. ``-R`` is asserted
    absent because a recursive chmod reaching up from this path would widen
    the staging directory that keeps an unverified upload unreadable.
    """
    safe = "releases/r/rows.json"

    recorded._upload(safe, Path("/dev/null"))

    staged = recorded._staged(safe)

    assert chmod_scripts(recorded) == [
        f"[ -f {shlex.quote(staged)} ]; "
        f"[ ! -L {shlex.quote(staged)} ]; "
        f"chmod 600 -- {shlex.quote(staged)}"
    ]

    assert publisher.STAGED_FILE_MODE == "600"

    script = chmod_scripts(recorded)[0]

    assert " -R" not in script
    assert "--recursive" not in script
    assert script.count("chmod") == 1

    # The mkdir, then the transfer, then the mode. A chmod before the
    # transfer would be a permission for a file that is not there yet.
    assert recorded.scripts.index(script) == len(recorded.scripts) - 1
    assert len(recorded.argv) == 1


@requires("rsync")
def test_a_failed_transfer_refuses_before_any_chmod_is_attempted(
    recorded: RecordedUpload,
):
    """rsync failing is the end of the upload, not a step on the way through.

    This is the case that was failing in production: the runner's rsync exits
    non-zero while parsing its own arguments. The refusal has to carry what
    rsync said, and nothing may run afterwards -- a chmod that reported
    success here would make a transfer that never happened look like one that
    did.
    """
    recorded.rsync_returncode = 1
    recorded.rsync_stderr = "rsync: --chmod=F600: invalid argument"

    with pytest.raises(publisher.RemoteCommandFailed) as refusal:
        recorded._upload("releases/r/rows.json", Path("/dev/null"))

    assert "invalid argument" in str(refusal.value)

    assert chmod_scripts(recorded) == []


@requires("rsync")
def test_a_refused_remote_chmod_fails_the_upload_closed(
    recorded: RecordedUpload,
):
    """A mode that could not be set is a failed upload, not a warning.

    The staging file exists at this point and has whatever mode the transfer
    left it. Continuing would place a release whose permissions are unknown,
    so the refusal propagates and the caller never reaches the placement.
    """
    recorded.refuse_matching = "chmod"

    with pytest.raises(publisher.RemoteCommandFailed) as refusal:
        recorded._upload("releases/r/rows.json", Path("/dev/null"))

    assert "Operation not permitted" in str(refusal.value)

    assert len(chmod_scripts(recorded)) == 1


@requires("rsync")
@requires("bash")
def test_the_staged_file_is_mode_600_and_the_served_copy_is_readable(
    tmp_path: Path, loopback: dict
):
    """The modes the destination ends up with, measured rather than argued.

    600 in private staging is what ``--chmod=F600`` was for and is unchanged.
    644 on the served copy is set where it always was, as the file is placed,
    and is deliberately not 600: the public readback fetches it over HTTPS,
    and a release only the deploy account can read is a release a reader gets
    a 403 for. ``_place`` is byte-identical to production, which the
    lifecycle's numerical pin holds it to; this measures what that means.
    """
    built = fixture.build(tmp_path / "source")

    receipt = publish(built, loopback)

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["promotion"]["promoted"] is True

    staged = sorted(
        path for path in loopback["staging"].rglob("*") if path.is_file()
    )

    assert staged

    for path in staged:
        assert path.stat().st_mode & 0o777 == 0o600, path

    served = sorted(
        path for path in loopback["published"].rglob("*") if path.is_file()
    )

    assert served

    for path in served:
        assert path.stat().st_mode & 0o777 == 0o644, path


@requires("rsync")
@requires("bash")
def test_a_shell_sensitive_remote_path_is_quoted_and_interpolates_nothing(
    tmp_path: Path, loopback: dict, monkeypatch: pytest.MonkeyPatch
):
    """A path that is a command if unquoted is only ever a path.

    The relative-path check refuses a path that could climb out of the
    subpath; it does not refuse spaces, quotes or ``$(...)``, because those
    address a file below the subpath perfectly well and the defence against
    them is quoting rather than rejection. So this writes one, through every
    command the transport builds -- the mkdir, the transfer, the new chmod,
    the copy, the rename and both digests -- with the working directory moved
    somewhere empty. An interpolated command substitution would run there and
    leave a file behind, and there is none.
    """
    sandbox = tmp_path / "cwd"

    sandbox.mkdir()

    monkeypatch.chdir(sandbox)

    relative = "releases/odd $(touch injected) 'name'/rows.json"

    payload = b'{"rows": []}\n'

    transport = LoopbackSshRsync(loopback["destination"])

    try:
        transport.write_bytes(relative, payload)

        assert transport.read_bytes(relative) == payload
        assert transport.exists(relative) is True
    finally:
        transport.close()

    staged = loopback["staging"] / relative

    assert staged.is_file()
    assert staged.stat().st_mode & 0o777 == 0o600

    served = loopback["published"] / relative

    assert served.is_file()
    assert served.read_bytes() == payload
    assert served.stat().st_mode & 0o777 == 0o644

    assert not (sandbox / "injected").exists()
    assert sorted(path.name for path in sandbox.iterdir()) == []


def test_no_production_command_names_an_rsync_chmod_option():
    """The defect is fixed everywhere it could be, not only where it fired.

    Feed and page publication are the same transport -- the page builder and
    the publication entry point both reach it through
    ``publisher.build_transport`` -- so one call site is the whole surface.
    Asserted as a scan anyway, because "there is only one" is the claim most
    worth having a test for.

    Scanned as string literals that *are* the option rather than as text
    containing it, so the prose explaining why the option is gone does not
    read as the option coming back.
    """
    offenders = []

    for directory in ("ops", "scripts", "src"):
        for path in sorted((PROJECT / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(
                    node.value, str
                ):
                    if node.value.startswith("--chmod"):
                        offenders.append(
                            f"{path.relative_to(PROJECT).as_posix()}:"
                            f"{node.lineno}"
                        )

    assert offenders == []


def test_the_switch_still_puts_the_authority_where_it_was():
    """A transport fix, so the publishing authority is where it was.

    The live state has since been set in a reviewed commit, and nothing about
    the authority moved with it: the incumbent publishes and the candidate is
    shadow only, both before and after. The gate is still shut from here --
    a switch read with nothing in the environment is not enabled -- which is
    the claim this test was always making.
    """
    switch = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    modes = switch["modes"]

    assert modes[publisher.MODE_CANARY]["state"] == publisher.STATE_ENABLED
    assert switch["published_authority"] == publisher.MODEL_AUTHORITY
    assert switch["candidate_authority"] == "shadow_only"

    assert (
        publisher.read_publishing_switch(
            publisher.MODE_LIVE, project_root=PROJECT, environment={}
        ).enabled
        is False
    )


# ======================================================================
# the transport's own confinement
# ======================================================================


@pytest.mark.parametrize(
    "relative",
    [
        "../escape.json",
        "releases/../../escape.json",
        "/etc/passwd",
        "",
        "./rows.json",
        "releases//rows.json",
        "-rf",
    ],
)
@requires("rsync")
@requires("bash")
def test_the_transport_refuses_to_address_anything_outside_its_subpath(
    loopback: dict, relative: str
):
    """Confinement as a property of the transport, not of its callers.

    The served root on the real host holds the legacy NBA pages, so a relative
    path that could climb out of the canary directory is the one defect that
    would let this publisher touch them. Refused on the way in, which is why
    no caller has to be careful.
    """
    transport = LoopbackSshRsync(loopback["destination"])

    try:
        for call in (
            lambda: transport.write_bytes(relative, b"{}\n"),
            lambda: transport.read_bytes(relative),
            lambda: transport.exists(relative),
            lambda: transport.public_url(relative),
        ):
            with pytest.raises(publisher.PublishingRefused):
                call()
    finally:
        transport.close()

    assert list(loopback["published"].rglob("*")) == []


@pytest.mark.parametrize(
    ("subpath", "expected"),
    [
        ("", "canary subpath"),
        ("/", "canary subpath"),
        ("..", "canary subpath"),
        ("canary/../..", "canary subpath"),
        ("/absolute", "canary subpath"),
    ],
)
def test_a_destination_without_a_usable_subpath_is_refused(
    subpath: str, expected: str
):
    """The subpath is the confinement, so there is no permissive reading of it.

    Absent and unusable are both refused, and now in the same terms. The
    distinction they used to be refused in -- a required field nobody filled
    in, against a field describing somewhere it must not reach -- stopped
    being available when the field became required of a canary rather than of
    every remote destination, because a live destination is the one that
    correctly has none. Which defect it is matters less than that neither
    produces a canary: an empty subpath would resolve this destination to the
    served root, and that is the whole thing confinement rules out.
    """
    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    document[publisher.NAMED_DESTINATIONS_KEY][NBA_DESTINATION][
        "canary_subpath"
    ] = subpath

    with pytest.raises(publisher.PublishingRefused, match=expected):
        publisher.resolve_destination(
            publisher.MODE_CANARY, document, name=NBA_DESTINATION
        )


def test_the_credential_file_is_removed_when_the_transport_closes(
    loopback: dict,
):
    """The key is written mode 600 outside the checkout, and then deleted."""
    transport = LoopbackSshRsync(loopback["destination"])

    key = transport._key_path

    assert key.is_file()
    assert key.stat().st_mode & 0o777 == 0o600
    assert PROJECT not in key.parents

    transport.close()

    assert not key.exists()
    assert not key.parent.exists()


def test_an_unnamed_canary_cannot_ask_for_a_remote_transport():
    """Reaching a host is never what a run that named no destination does."""
    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    document["destinations"][publisher.MODE_CANARY]["transport"] = (
        publisher.TRANSPORT_SSH_RSYNC
    )

    with pytest.raises(publisher.PublishingRefused, match="addressed by name"):
        publisher.resolve_destination(
            publisher.MODE_CANARY, document, root=Path("/tmp/anywhere")
        )


def test_only_two_transports_are_implemented():
    """Exactly one was added, and it is the one the destination speaks.

    Named here because the alternative to a specific list is a publisher that
    grows a transport per destination somebody once had.
    """
    assert publisher.TRANSPORT_SSH_RSYNC == "ssh_rsync"

    implemented = {
        publisher.TRANSPORT_LOCAL_DIRECTORY,
        publisher.TRANSPORT_SSH_RSYNC,
    }

    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    assert set(document["supported_transports"]) == implemented

    tree = ast.parse(
        (PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py").read_text(
            encoding="utf-8"
        )
    )

    # Read off the syntax tree rather than the text: the module's prose names
    # the protocols it deliberately does not speak, and a substring search
    # cannot tell a refusal to implement FTP from an implementation of it.
    transports = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Transport")
    )

    assert transports == ["LocalDirectoryTransport", "SshRsyncTransport"]

    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert imported & {
        "boto3",
        "botocore",
        "ftplib",
        "google",
        "paramiko",
        "pysftp",
        "requests",
    } == set()


# ======================================================================
# the real destination
# ======================================================================


@real_canary
@requires("rsync")
def test_the_real_destination_authenticates_as_the_nba_deploy_account():
    """The configured account, confirmed by the host itself.

    The one fact in this file that only the real destination can answer. Read
    through the transport rather than by running ssh beside it, so what is
    confirmed is the transport's own authentication.
    """
    document = publisher.read_destinations(project_root=PROJECT)

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY, document, name=NBA_DESTINATION
    )

    transport = publisher.build_transport(destination)

    try:
        identity = transport.verify_identity()
    finally:
        transport.close()

    assert identity["user"] == "nba-prop-deploy"
    assert identity["private_staging"] == "/var/www/.nba-prop-quant-deploy"
    assert identity["published_root"].endswith("/predictions/nba/canary")


@real_canary
@requires("rsync")
def test_the_real_destination_refuses_an_unusable_credential():
    """2, against the host. A well-formed key it does not know is refused."""
    document = publisher.read_destinations(project_root=PROJECT)

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY, document, name=NBA_DESTINATION
    )

    transport = publisher.build_transport(
        destination,
        environment={NBA_CREDENTIAL_ENV: _unusable_key()},
    )

    try:
        with pytest.raises(publisher.RemoteCommandFailed) as refusal:
            transport.verify_identity()
    finally:
        transport.close()

    assert "refused a command" in str(refusal.value)


@real_canary
@requires("rsync")
def test_the_real_destination_refuses_a_wrong_ssh_user():
    """12, against the host. The key is good and the account is not configured."""
    document = publisher.read_destinations(project_root=PROJECT)

    declared = dict(
        document[publisher.NAMED_DESTINATIONS_KEY][NBA_DESTINATION]
    )

    declared["user"] = "nobody-this-host-knows"

    document[publisher.NAMED_DESTINATIONS_KEY] = {NBA_DESTINATION: declared}

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY, document, name=NBA_DESTINATION
    )

    transport = publisher.build_transport(destination)

    try:
        with pytest.raises(publisher.PublishingRefused):
            transport.verify_identity()
    finally:
        transport.close()


#: The legacy NBA pages and feeds the website already publishes, beside the
#: served root this publisher writes below. Hashed before and after the real
#: canary: they are owned by somebody else's cron, and the whole reason the
#: destination is confined to a subdirectory is that nothing here may touch
#: them.
LEGACY_NBA_FILES: tuple[str, ...] = (
    "nba-props.html",
    "nba-live-props.html",
    "nba-pmf-research.html",
    "nba_props_today.json",
    "pmf_research.json",
)


def _ssh_beside_the_transport(destination: publisher.Destination, script: str) -> str:
    """Run a command at the destination without going through the transport.

    Deliberately not a transport method. The transport cannot address the
    legacy files -- that is the property under test -- so checking they are
    untouched has to be done by something that can, which makes this an
    independent observer rather than the subject marking its own work.
    """
    remote = destination.remote

    with tempfile.TemporaryDirectory() as directory:
        key = Path(directory) / "key"

        key.touch(mode=0o600)

        material = os.environ[NBA_CREDENTIAL_ENV]

        key.write_text(
            material if material.endswith("\n") else material + "\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "ssh",
                "-i",
                str(key),
                "-p",
                str(remote.port),
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={Path(directory) / 'known_hosts'}",
                f"{remote.user}@{remote.host}",
                f"set -eu; {script}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return completed.stdout


def _legacy_digests(destination: publisher.Destination) -> dict[str, str]:
    parent = Path(destination.remote.public_target).parent.as_posix()

    output = _ssh_beside_the_transport(
        destination,
        f"cd {shlex.quote(parent)} && sha256sum -- "
        + " ".join(shlex.quote(name) for name in LEGACY_NBA_FILES),
    )

    digests: dict[str, str] = {}

    for line in output.splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)

            digests[name.strip()] = value

    return digests


@real_canary
@requires("rsync")
def test_the_real_canary_publishes_the_frozen_models_slate_and_rolls_back(
    tmp_path: Path,
):
    """The whole thing, against the real destination, in one sequence.

    The frozen models score the serving fixture, the publisher puts the result
    on the configured host under its canary subdirectory, a consumer fetching
    the public URL gets exactly the priced numbers back, and a rollback moves
    the destination to the release before it and then forward again.

    One test rather than several because it is one ordered sequence against
    one real destination, and tests that depend on each other's leftovers
    through the suite's ordering are worse than a sequence that says so.

    The rollback target is a release of the publisher's own fixture, published
    first. A rollback needs two releases to be a rollback, the two have to
    differ, and the alternative -- altering the frozen models' numbers to
    manufacture a second one -- would mean publishing something that is not
    the incumbent's output in order to test the incumbent's publisher. The
    sequence ends on the frozen models' release, so that is what the canary is
    left serving.
    """
    import frozen_bundle_slate_fixture as slate_fixture
    from test_real_slate_serving_root import installed_bundle
    from test_wizardofodds_canary_end_to_end import (
        checkout_sha,
        serve_and_provenance,
    )

    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    document = publisher.read_destinations(project_root=PROJECT)

    destination = publisher.resolve_destination(
        publisher.MODE_CANARY, document, name=NBA_DESTINATION
    )

    legacy_before = _legacy_digests(destination)

    assert sorted(legacy_before) == sorted(LEGACY_NBA_FILES)

    # ---- the rollback target, published first ----------------------

    seed = fixture.build(tmp_path / "seed")

    seeded = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=seed["slate_date"],
        data_root=seed["data_root"],
        project_root=PROJECT,
        destination_name=NBA_DESTINATION,
    )

    assert seeded["outcome"] == publisher.OUTCOME_PUBLISHED

    # ---- the frozen models' own slate ------------------------------

    root = tmp_path / "run"

    provenanced = serve_and_provenance(
        bundle, root, production_sha=checkout_sha()
    )

    data_root = root / "data"

    projections = pd.read_parquet(
        data_root
        / "processed"
        / "projections"
        / f"{slate_fixture.SLATE_DATE}.parquet"
    )

    priced = pd.read_parquet(
        publisher.priced_markets_path(data_root, slate_fixture.SLATE_DATE)
    )

    assert len(projections) == len(slate_fixture.ROSTER) > 0
    assert len(priced) == len(slate_fixture.QUOTED_PROPS) > 0

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT,
        destination_name=NBA_DESTINATION,
    )

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["mode"] == publisher.MODE_CANARY
    assert receipt["model_authority"] == publisher.MODEL_AUTHORITY
    assert receipt["live_publishing"] == publisher.STATE_DISABLED
    assert receipt["promotion"]["promoted"] is True
    assert receipt["promotion"]["rollback_target"] == seeded["release"]["release_id"]
    assert receipt["destination"]["transport"] == publisher.TRANSPORT_SSH_RSYNC
    assert receipt["destination"]["authenticated_as"]["user"] == "nba-prop-deploy"
    assert receipt["release"]["row_count"] == len(priced)

    verification = receipt["verification"]

    assert verification["verified"] is True
    assert verification["readback_rows"] == len(priced)
    assert verification["public"]["checked"] is True
    assert verification["public"]["readback_rows"] == len(priced)
    assert verification["public"]["rows_sha256"] == receipt["release"]["rows_sha256"]

    provenance = receipt["provenance"]

    served_receipt = provenanced["receipt"]

    assert provenance["model_authority"] == publisher.MODEL_AUTHORITY
    assert provenance["production_code_sha"] == served_receipt["production_code_sha"]
    assert provenance["production_code_sha"] == checkout_sha()
    assert (
        provenance["priced_market_artifact_sha256"]
        == served_receipt["priced_market_artifact"]["sha256"]
    )
    assert "incumbent_fit_id" in provenance

    # ---- the provenance a consumer can read off the destination ----
    #
    # The receipt is what this process believes it published. The manifest
    # fetched over its public URL is what the destination will tell anybody
    # else, and the code SHA is the field that has to survive that trip: a
    # publication whose manifest names a commit nobody can check provenances
    # the numbers to nothing.
    with urllib.request.urlopen(
        verification["public"]["manifest_url"], timeout=60
    ) as response:
        published_manifest = json.loads(response.read())

    assert (
        published_manifest["provenance"]["production_code_sha"]
        == served_receipt["production_code_sha"]
    )
    assert published_manifest["provenance"]["production_code_sha"] == checkout_sha()

    # ---- what a consumer actually receives -------------------------

    with urllib.request.urlopen(
        verification["public"]["rows_url"], timeout=60
    ) as response:
        body = response.read()

    assert publisher.digest(body) == receipt["release"]["rows_sha256"]

    served = json.loads(body)["rows"]

    assert len(served) == len(priced)

    by_quote = {int(row["quote_id"]): row for row in served}

    assert sorted(by_quote) == sorted(int(value) for value in priced["id"])

    for _, source in priced.iterrows():
        out = by_quote[int(source["id"])]

        for column in (
            # the model's probabilities, and the market's
            *publisher.PUBLISHED_PROBABILITY_FIELDS,
            # the prices
            "over_odds",
            "under_odds",
            "calibrated_fair_over_american",
            "calibrated_fair_under_american",
            "line_value",
            # the edges and the EV
            "calibrated_edge_over",
            "calibrated_edge_under",
            "calibrated_ev_over",
            "calibrated_ev_under",
            "model_preferred_edge",
            "model_preferred_ev",
        ):
            assert float(out[column]) == float(source[column]), (
                f"{column} on quote {source['id']} is not the served value"
            )

        assert out["freeze_id"] == source["freeze_id"]
        assert out["model_preferred_side"] == source["model_preferred_side"]

    assert set(served[0]) == {name for _, name in publisher.PUBLISHED_FIELDS}

    # ---- the rollback ----------------------------------------------

    transport = publisher.build_transport(destination)

    try:
        restored = publisher.rollback(transport)
    finally:
        transport.close()

    assert restored["outcome"] == publisher.OUTCOME_ROLLED_BACK
    assert restored["restored_release_id"] == seeded["release"]["release_id"]
    assert restored["superseded_release_id"] == receipt["release"]["release_id"]
    assert restored["rows_sha256"] == seeded["release"]["rows_sha256"]
    assert restored["public_rows_sha256"] == seeded["release"]["rows_sha256"]

    # ---- and forward again, so the canary serves the frozen slate --

    republished = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT,
        destination_name=NBA_DESTINATION,
    )

    assert (
        republished["release"]["release_id"] == receipt["release"]["release_id"]
    )
    assert republished["verification"]["public"]["readback_rows"] == len(priced)

    observer = publisher.build_transport(destination)

    try:
        current = publisher.read_pointer(observer, publisher.CURRENT_POINTER)

        listing = observer.listing()
    finally:
        observer.close()

    assert current["release_id"] == receipt["release"]["release_id"]

    # Nothing was written at the served root itself, so no live or current
    # pointer for the website to read into existence was created.
    served_root = Path(destination.remote.public_target)

    at_the_root = _ssh_beside_the_transport(
        destination,
        f"find {shlex.quote(served_root.as_posix())} -maxdepth 1 -type f "
        "-printf '%P\\n' || true",
    ).split()

    assert at_the_root == [], f"the canary wrote beside the legacy feed: {at_the_root}"

    assert all(
        name.startswith(
            (
                publisher.RELEASES_DIRECTORY,
                publisher.AUDIT_LOG_NAME,
                publisher.ATTESTATION_NAME,
                publisher.CURRENT_POINTER,
                publisher.PREVIOUS_POINTER,
            )
        )
        for name in listing
    ), listing

    # ---- the legacy feed, byte for byte ----------------------------

    assert _legacy_digests(destination) == legacy_before, (
        "the canary changed a legacy NBA file"
    )


def _unusable_key() -> str:
    """A syntactically valid ed25519 key the destination has never seen."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "key"

        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "not-the-nba-deploy-key",
                "-f",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

        return path.read_text(encoding="utf-8")
