#!/usr/bin/env python3
"""Install the already-frozen production model binaries where serving can read them.

THE DEFECT
----------

``models/frozen_manifests/LATEST.json`` records twelve ``.joblib`` files that
prediction and pricing load on every run. ``.gitignore`` excludes
``models/**/*.joblib``, and ``docs/GITHUB_DISTRIBUTION.md`` says so
deliberately: the monolithic frozen model package is distributed as a release
asset rather than committed, because GitHub refuses normal objects over
100 MiB and a code repository is not a place to version model binaries.

So the twelve files the frozen manifest requires have never existed in a
clean checkout, and ``actions/checkout`` cleans ignored files out of the
runner workspace anyway. Verifying the frozen bundle against the checkout was
therefore asking a question the checkout cannot answer, and the answer it
gave -- twelve required artifacts missing -- was correct.

WHAT THIS DOES INSTEAD
----------------------

One narrow preparation operation: it takes the frozen package that already
exists, extracts the model tree it already contains into a runtime directory
outside the Git checkout, and verifies every artifact against the hashes the
frozen manifest already records. Nothing is built, regenerated, retrained or
re-frozen; no manifest is written; no model byte is produced here that was not
produced in August 2026 and published then.

The resulting directory is what the incumbent serving step is handed
explicitly, so the serving path reads the frozen binaries from a verified
bundle rather than from whatever a checkout's ``models/`` happens to hold.

IDENTITY AND REUSE
------------------

The install location is keyed to the identity of what is installed: the freeze
id, the digest of the frozen manifest that describes it, and the digest of the
package it came from. Two different frozen packages can therefore never share
a directory, and re-running against an already verified copy re-verifies it
and reuses it rather than downloading 386 MiB again.

A partial extraction is never a valid bundle. Extraction happens in a staging
directory under the runtime root, verification happens there, the readiness
marker is written there, and only then is the whole directory moved into place
by a single rename. An existing directory without a matching readiness marker,
or one that fails re-verification, is discarded and rebuilt.

OBTAINING THE PACKAGE
---------------------

The release is immutable, so a package whose bytes are not the published bytes
is always a statement about the transfer. Getting that distinction right is
what the download path is for. A 386 MiB body over a residential link ends
early often enough to be ordinary, and an early end is invisible by default:
``shutil.copyfileobj`` reads to EOF, a dropped connection *is* an EOF, and the
copy returns normally having written a prefix. So the byte count is compared
against what the response and the release both declare, a short body is named
as a truncated transfer, and the transfer is retried a bounded number of times
before the install refuses. Each attempt is verified whole and a rejected one
leaves nothing behind, so retrying can only retry the transfer.

The expected digest comes from the release's own ``SHA256SUMS.txt`` and is
established before any bytes are accepted, including the cache's. A cached
package is therefore reused because it verified rather than because it is
there, and a release that publishes no digest for its own asset is a refusal
rather than a reason to skip verification.

WHAT IT CANNOT DO
-----------------

It cannot fit, refit, recalibrate, promote or publish. It reads a published
immutable release asset and the repository's own frozen manifest, writes into
a runtime directory outside the checkout, and reports what it verified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.production import (  # noqa: E402
    load_json,
    load_verified_manifest_metadata,
    sha256_file,
)

#: Where a serving artifact tree carries the manifest that describes it.
MANIFEST_RELATIVE = Path("frozen_manifests") / "LATEST.json"

#: The manifest group holding the artifacts serving actually loads. The
#: manifest also records source files, scripts, configs and the 2025 audit
#: parquets; those are historical provenance for the freeze, not runtime
#: inputs, and this module neither installs nor verifies them.
MODEL_ARTIFACT_GROUP = "model_artifacts"

#: Every runtime artifact record is relative to the project root and therefore
#: begins with this component. Stripping it is what makes a bundle relocatable:
#: the artifacts are resolved inside the model directory wherever it lives.
MODEL_PREFIX = "models"

#: The model binaries prediction and pricing load. Named explicitly so a
#: frozen manifest that had quietly stopped recording one of them is refused
#: rather than satisfied.
REQUIRED_MODEL_BINARIES: tuple[str, ...] = (
    "experience_curves.joblib",
    "marginals.joblib",
    "marginals_pre2025.joblib",
    "copula.joblib",
    "copula_pre2025.joblib",
    "minutes.joblib",
    "pts.joblib",
    "reb.joblib",
    "ast.joblib",
    "stl.joblib",
    "blk.joblib",
    "fg3m.joblib",
)

#: ``ops/build_full_model_package.py`` copies every file the frozen manifest
#: names under this prefix inside the package, having first verified it against
#: the manifest's own SHA-256. The package layout is therefore
#: ``<package name>/frozen_project/<manifest relative path>``.
PACKAGE_PROJECT_PREFIX = "frozen_project"

#: Production pricing requires the deployment freeze, exactly as the
#: repository's own verifier requires it.
REQUIRED_FREEZE_STAGE = "external_test_deployment"

#: Subdirectories of the runtime root. The downloaded package is cached beside
#: the bundles it is extracted into so that a second run costs nothing.
BUNDLES_RELATIVE = Path("frozen_runtime_bundles")
PACKAGES_RELATIVE = Path("frozen_packages")
STAGING_RELATIVE = BUNDLES_RELATIVE / ".staging"

#: Written last, inside the staging directory, so its presence at the final
#: location means the rename completed over a fully verified tree.
READY_RELATIVE = Path("FROZEN_RUNTIME_BUNDLE.json")

#: The canonical asset name and tag, from docs/GITHUB_DISTRIBUTION.md. The tag
#: is the freeze id, so neither is guessed: both are derived from the frozen
#: manifest being installed.
ASSET_TEMPLATE = "{freeze_id}_production_model_package.zip"
CHECKSUM_ASSET = "SHA256SUMS.txt"

API_ROOT = "https://api.github.com"

TOKEN_VARIABLES: tuple[str, ...] = ("GITHUB_TOKEN", "GH_TOKEN")

#: How many times the published package is fetched before the install refuses.
#: A 386 MiB transfer over a residential link is long enough that a dropped
#: connection is an ordinary event rather than an exceptional one, and a
#: truncated body is not evidence that the immutable release is wrong. Each
#: attempt is verified in full and a failed one leaves nothing behind, so
#: retrying can only ever retry the transfer.
DOWNLOAD_ATTEMPTS = 4

#: Seconds before each retry, so a transient outage is not hammered.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (4.0, 8.0, 16.0)

#: The environment variable the install location is published under, for the
#: serving step that is handed it.
MODEL_DIR_VARIABLE = "FROZEN_RUNTIME_MODEL_DIR"

#: The manifest resolution root the serving scripts verify against. Published
#: alongside the model directory because the two are a pair: the model
#: directory says which artifacts to load and the bundle root says which tree
#: the frozen manifest describes, and serving needs both to be explicit.
BUNDLE_ROOT_VARIABLE = "FROZEN_RUNTIME_BUNDLE_ROOT"

OUTCOME_INSTALLED = "INSTALLED"
OUTCOME_REUSED = "REUSED"
OUTCOME_REFUSED = "INSTALL_REFUSED"

EXIT_OK = 0
EXIT_FAILED = 1


class BundleRefused(RuntimeError):
    """The frozen runtime bundle could not be established, so nothing serves."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ----------------------------------------------------------------------
# what the frozen manifest requires at serving time
# ----------------------------------------------------------------------


def manifest_identity(manifest_path: Path) -> dict[str, str]:
    """The frozen manifest's own identity, with its stage gate applied."""
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing frozen manifest: {path}")

    manifest = load_json(path)

    stage = str(manifest.get("freeze_stage", ""))

    if stage != REQUIRED_FREEZE_STAGE:
        raise BundleRefused(
            f"the frozen manifest at {path} is at stage {stage!r} rather "
            f"than {REQUIRED_FREEZE_STAGE!r}, so it does not describe a "
            "production serving bundle"
        )

    return {
        "freeze_id": str(manifest["freeze_id"]),
        "freeze_stage": stage,
        "manifest_sha256": sha256_file(path),
    }


def runtime_artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The manifest's runtime model artifacts, resolved inside the model tree.

    A record outside ``models/`` is a refusal rather than a skip: it would
    mean the manifest calls something a runtime model artifact that cannot be
    resolved inside a relocatable bundle, and silently ignoring it would be
    the one thing this module must not do.
    """
    records = (manifest.get("files") or {}).get(MODEL_ARTIFACT_GROUP) or []

    if not records:
        raise BundleRefused(
            f"the frozen manifest records no {MODEL_ARTIFACT_GROUP!r}, so "
            "there is no runtime model artifact set to verify"
        )

    resolved: list[dict[str, Any]] = []

    for record in records:
        relative = Path(str(record["path"]))

        if relative.parts[:1] != (MODEL_PREFIX,):
            raise BundleRefused(
                f"{relative} is recorded as a runtime model artifact but "
                f"does not live under {MODEL_PREFIX}/, so it cannot be "
                "resolved inside a relocatable runtime bundle"
            )

        resolved.append(
            {
                "relative": relative.as_posix(),
                "sha256": str(record["sha256"]),
                "within_model_dir": Path(*relative.parts[1:]),
            }
        )

    return resolved


def verify_runtime_model_artifacts(model_dir: Path) -> dict[str, Any]:
    """Re-hash every runtime model artifact the frozen manifest names.

    This is the serving path's integrity check and it is a verification, not a
    read: a missing or drifted artifact raises. Its scope is the runtime model
    artifacts alone, resolved inside ``model_dir``. The source files, scripts
    and configs the same manifest records are the frozen freeze's historical
    provenance, and the integrity of the code actually running is established
    by the authoritative production SHA the lifecycle checks out and the CI
    that reported on it, not by hashing a checkout against an August manifest.
    Their hashes stay in the manifest; they are simply not what decides whether
    a model binary is the one that was frozen.
    """
    directory = Path(model_dir)

    manifest_path = directory / MANIFEST_RELATIVE

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")

    manifest = load_json(manifest_path)

    stage = str(manifest.get("freeze_stage", ""))

    if stage != REQUIRED_FREEZE_STAGE:
        raise RuntimeError(
            "Production serving requires an "
            f"{REQUIRED_FREEZE_STAGE} freeze. "
            f"The bundle at {directory} is at stage {stage!r}."
        )

    records = runtime_artifact_records(manifest)

    missing: list[str] = []
    drifted: list[str] = []

    for record in records:
        path = directory / record["within_model_dir"]

        if not path.exists():
            missing.append(record["relative"])
            continue

        if sha256_file(path) != record["sha256"]:
            drifted.append(record["relative"])

    if missing or drifted:
        parts = []

        if missing:
            parts.append("missing=" + ", ".join(missing))

        if drifted:
            parts.append("hash_mismatch=" + ", ".join(drifted))

        raise RuntimeError(
            "Frozen runtime manifest verification failed: " + " | ".join(parts)
        )

    return {
        "freeze_id": str(manifest["freeze_id"]),
        "freeze_stage": stage,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_artifacts": len(records),
    }


def verify_frozen_bundle(bundle_root: Path) -> dict[str, Any]:
    """Verify the whole frozen manifest against an installed bundle root.

    This is the repository's own verifier, called with the bundle root as the
    resolution root: the exact check ``scripts/10_predict_slate.py`` and
    ``scripts/15_price_markets.py`` perform before they load a model. It is
    called here, at install time, so that a bundle the serving scripts would
    reject is never handed to them in the first place.

    Imported rather than reimplemented. A second implementation of "does this
    tree match the freeze" could disagree with the first, and the one that
    decides whether production prices tonight is the scripts'.
    """
    root = Path(bundle_root)

    metadata = load_verified_manifest_metadata(
        model_dir=root / MODEL_PREFIX,
        project_root=root,
    )

    manifest = load_json(root / MODEL_PREFIX / MANIFEST_RELATIVE)

    return {
        **metadata,
        "verified_manifest_entries": sum(
            len(records) for records in (manifest.get("files") or {}).values()
        ),
    }


def unrecorded_model_binaries(records: list[dict[str, Any]]) -> list[str]:
    """Model binaries serving loads that the manifest does not record."""
    recorded = {record["within_model_dir"].as_posix() for record in records}

    return [name for name in REQUIRED_MODEL_BINARIES if name not in recorded]


# ----------------------------------------------------------------------
# obtaining the published frozen package
# ----------------------------------------------------------------------


def _token() -> str | None:
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name)

        if value:
            return value

    return None


def _request(url: str, *, accept: str, token: str | None) -> Any:
    request = urllib.request.Request(url)

    request.add_header("Accept", accept)
    request.add_header("X-GitHub-Api-Version", "2022-11-28")

    if token:
        request.add_header("Authorization", f"Bearer {token}")

    return urllib.request.urlopen(request, timeout=300)


def release_assets(
    *, repository: str, tag: str, token: str | None
) -> dict[str, dict[str, Any]]:
    """Asset name to download URL and published byte count for one release.

    The size is carried because it is the only thing that distinguishes a
    truncated transfer from a complete one before hashing: a short body is a
    well-formed HTTP response and reading it to EOF raises nothing.
    """
    url = f"{API_ROOT}/repos/{repository}/releases/tags/{tag}"

    try:
        with _request(
            url, accept="application/vnd.github+json", token=token
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        raise BundleRefused(
            f"the release {tag!r} in {repository} could not be read "
            f"(HTTP {error.code}), so the frozen package could not be "
            "located"
        ) from error

    except (urllib.error.URLError, OSError) as error:
        raise BundleRefused(
            f"the release {tag!r} in {repository} could not be reached: "
            f"{error}"
        ) from error

    assets: dict[str, dict[str, Any]] = {}

    for asset in payload.get("assets") or []:
        size = asset.get("size")

        assets[str(asset["name"])] = {
            "url": str(asset["url"]),
            "size": int(size) if isinstance(size, int) else None,
        }

    return assets


def download_asset(
    *,
    url: str,
    destination: Path,
    token: str | None,
    expected_bytes: int | None = None,
) -> int:
    """Stream one release asset into place, completed or not at all.

    Completeness is asserted rather than assumed. ``shutil.copyfileobj`` reads
    to EOF, and a connection that drops mid-body produces an EOF: the copy
    returns normally having written a prefix of the asset. So the byte count is
    compared against what the response and the release both say it should be,
    and a short body is refused here, where it is recognisable as a failed
    transfer, rather than surfacing later as an unexplained digest mismatch
    against an immutable published artifact.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle, staged = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}."
    )

    os.close(handle)

    staged_path = Path(staged)

    try:
        with _request(
            url, accept="application/octet-stream", token=token
        ) as response:
            declared = response.headers.get("Content-Length")

            with staged_path.open("wb") as sink:
                shutil.copyfileobj(response, sink, length=1024 * 1024)

        written = staged_path.stat().st_size

        for label, count in (
            ("the response's Content-Length", declared),
            ("the release's published size", expected_bytes),
        ):
            if count is None:
                continue

            if written != int(count):
                raise BundleRefused(
                    f"the transfer of {destination.name} ended after "
                    f"{written} bytes but {label} is {int(count)}, so the "
                    "body was truncated in flight"
                )

        os.replace(staged_path, destination)

        return written

    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise


def recorded_package_digest(text: str, asset: str) -> str | None:
    """The digest a ``SHA256SUMS.txt`` body records for one asset."""
    for line in text.splitlines():
        fields = line.split()

        if len(fields) == 2 and fields[1].lstrip("*") == asset:
            return fields[0]

    return None


def _fetch_until_verified(
    *,
    url: str,
    destination: Path,
    token: str | None,
    expected_digest: str,
    expected_bytes: int | None,
) -> str:
    """Download one asset repeatedly until its bytes are the published bytes.

    Verification is per attempt and a rejected attempt leaves nothing behind,
    so this cannot accumulate state or accept a partially good transfer. It
    exists because the alternative -- refusing production for the day on the
    first dropped connection -- treats a flaky link as though the immutable
    release had changed.
    """
    failures: list[str] = []

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_asset(
                url=url,
                destination=destination,
                token=token,
                expected_bytes=expected_bytes,
            )

            digest = sha256_file(destination)

            if digest == expected_digest:
                return digest

            destination.unlink(missing_ok=True)

            failures.append(
                f"attempt {attempt} hashed to {digest}, not the published "
                f"{expected_digest}"
            )

        except (BundleRefused, urllib.error.URLError, OSError) as error:
            destination.unlink(missing_ok=True)

            failures.append(f"attempt {attempt} failed: {error}")

        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(
                RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                ]
            )

    raise BundleRefused(
        f"{destination.name} could not be obtained intact in "
        f"{DOWNLOAD_ATTEMPTS} attempts: " + "; ".join(failures)
    )


def published_package_digest(
    *,
    runtime_root: Path,
    assets: dict[str, dict[str, Any]],
    asset: str,
    tag: str,
    token: str | None,
) -> str:
    """The digest the release itself publishes for one asset.

    Established before any package bytes are trusted, because it is what makes
    "verified" mean anything: without it the only available check is that some
    bytes arrived. A release that publishes no digest for its own asset is a
    refusal rather than a reason to skip verification.
    """
    if CHECKSUM_ASSET not in assets:
        raise BundleRefused(
            f"the release {tag!r} publishes no {CHECKSUM_ASSET}, so the "
            f"expected digest of {asset} could not be established and the "
            "package cannot be verified"
        )

    sums = Path(runtime_root) / PACKAGES_RELATIVE / f"{tag}.{CHECKSUM_ASSET}"

    failures: list[str] = []

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_asset(
                url=assets[CHECKSUM_ASSET]["url"],
                destination=sums,
                token=token,
                expected_bytes=assets[CHECKSUM_ASSET]["size"],
            )

            recorded = recorded_package_digest(
                sums.read_text(encoding="utf-8"), asset
            )

            if recorded:
                return recorded.lower()

            raise BundleRefused(
                f"{CHECKSUM_ASSET} records no digest for {asset}"
            )

        except (BundleRefused, urllib.error.URLError, OSError) as error:
            sums.unlink(missing_ok=True)

            failures.append(f"attempt {attempt} failed: {error}")

        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(
                RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                ]
            )

    raise BundleRefused(
        f"the published digest of {asset} could not be read from "
        f"{CHECKSUM_ASSET}: " + "; ".join(failures)
    )


def resolve_package(
    *,
    package: Path | None,
    runtime_root: Path,
    freeze_id: str,
    repository: str | None,
    release_tag: str | None,
    asset: str | None,
    package_sha256: str | None,
    allow_download: bool,
) -> dict[str, Any]:
    """The frozen package to install from, and the digest it was accepted at.

    Resolution order is the documented distribution mechanism: an explicitly
    supplied package, then the runtime root's cache, then the published release
    asset. Nothing is ever built.

    The expected digest is established before any cached or downloaded bytes
    are accepted, so the cache is reused because it was verified rather than
    because it exists. A cached file that does not match is debris from an
    interrupted transfer and is discarded.
    """
    expected = package_sha256.lower() if package_sha256 else None

    if package is not None:
        path = Path(package).resolve()

        if not path.exists():
            raise BundleRefused(f"the frozen package {path} does not exist")

        digest = sha256_file(path)

        if expected and digest != expected:
            raise BundleRefused(
                f"the frozen package {path} hashes to {digest}, not the "
                f"expected {expected}"
            )

        return {
            "asset": path.name,
            "package_sha256": digest,
            "path": path,
            "source": "supplied",
            "expected_digest_source": "argument" if expected else "none",
        }

    name = asset or ASSET_TEMPLATE.format(freeze_id=freeze_id)

    cached = Path(runtime_root) / PACKAGES_RELATIVE / name

    digest_source = "argument" if expected else "none"

    tag = release_tag or freeze_id

    token = _token()

    assets: dict[str, dict[str, Any]] = {}

    if allow_download and repository:
        assets = release_assets(repository=repository, tag=tag, token=token)

        if name not in assets:
            raise BundleRefused(
                f"the release {tag!r} in {repository} publishes no asset "
                f"named {name!r}; it publishes {sorted(assets) or 'nothing'}"
            )

        if expected is None:
            expected = published_package_digest(
                runtime_root=runtime_root,
                assets=assets,
                asset=name,
                tag=tag,
                token=token,
            )

            digest_source = CHECKSUM_ASSET

    if expected is None:
        raise BundleRefused(
            f"the expected digest of {name} could not be established -- no "
            "--package-sha256 was given and the published checksum could not "
            "be read -- so no package may be accepted as the frozen one"
        )

    if cached.exists():
        digest = sha256_file(cached)

        if digest == expected:
            return {
                "asset": name,
                "package_sha256": digest,
                "path": cached,
                "source": "cache",
                "expected_digest_source": digest_source,
            }

        cached.unlink()

    if not allow_download:
        raise BundleRefused(
            f"the frozen package {name} is not present and verified at "
            f"{cached} and downloading was refused, so the published release "
            "asset could not be obtained"
        )

    if not repository:
        raise BundleRefused(
            "no repository was given and GITHUB_REPOSITORY is unset, so the "
            "release holding the frozen package could not be identified"
        )

    digest = _fetch_until_verified(
        url=assets[name]["url"],
        destination=cached,
        token=token,
        expected_digest=expected,
        expected_bytes=assets[name]["size"],
    )

    return {
        "asset": name,
        "package_sha256": digest,
        "path": cached,
        "source": "release",
        "expected_digest_source": digest_source,
    }


# ----------------------------------------------------------------------
# hydrating the runtime bundle
# ----------------------------------------------------------------------


def extract_frozen_project(package: Path, destination: Path) -> int:
    """Copy the frozen package's whole frozen project tree into ``destination``.

    The whole tree rather than only ``models/``, because the frozen manifest
    names 62 files across five groups and the serving scripts verify all of
    them. Resolved against a root that holds only the model artifacts, 43 of
    those records are unsatisfiable; resolved against this tree, every one of
    them is exactly the file the freeze recorded.

    The tree is a *verification* root, not an execution root. The code that
    runs is the production checkout's, whose integrity comes from the
    authoritative production SHA and the CI that reported on it. Nothing here
    is ever imported or executed.
    """
    with zipfile.ZipFile(package) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

        roots = sorted({name.split("/")[0] for name in names})

        if len(roots) != 1:
            raise BundleRefused(
                f"the frozen package {package.name} has {len(roots)} top "
                "level entries, so its frozen project tree cannot be located"
            )

        prefix = f"{roots[0]}/{PACKAGE_PROJECT_PREFIX}/"

        members = [name for name in names if name.startswith(prefix)]

        if not members:
            raise BundleRefused(
                f"the frozen package {package.name} contains no {prefix} "
                "tree, so it is not a frozen project package"
            )

        destination.mkdir(parents=True, exist_ok=True)

        for name in members:
            relative = Path(name[len(prefix) :])

            if relative.is_absolute() or ".." in relative.parts:
                raise BundleRefused(
                    f"the frozen package {package.name} names {name!r}, "
                    "which escapes the frozen project tree"
                )

            target = destination / relative

            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(name) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)

    return len(members)


def bundle_key(
    *, freeze_id: str, manifest_sha256: str, package_sha256: str
) -> str:
    """The install location's name, derived from what is installed."""
    return f"{freeze_id}__{manifest_sha256[:16]}__{package_sha256[:16]}"


def ready_marker(bundle: Path) -> dict[str, Any] | None:
    path = Path(bundle) / READY_RELATIVE

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))

    except (json.JSONDecodeError, OSError):
        return None


def installed_bundle_is_valid(bundle: Path, expected: dict[str, str]) -> bool:
    """Whether an existing directory really is the bundle it claims to be.

    An interrupted install cannot produce one of these: the marker is written
    inside the staging directory and the directory is moved into place by a
    single rename, so a directory without a matching marker is debris and is
    rebuilt rather than served.
    """
    marker = ready_marker(bundle)

    if marker is None:
        return False

    if any(marker.get(key) != value for key, value in expected.items()):
        return False

    try:
        verify_frozen_bundle(Path(bundle))

    except (FileNotFoundError, RuntimeError, BundleRefused):
        return False

    return True


def install(
    *,
    runtime_root: Path,
    manifest_path: Path,
    package: Path | None = None,
    repository: str | None = None,
    release_tag: str | None = None,
    asset: str | None = None,
    package_sha256: str | None = None,
    allow_download: bool = True,
) -> dict[str, Any]:
    """Establish a verified frozen runtime bundle and report what it holds."""
    root = Path(runtime_root).resolve()

    identity = manifest_identity(manifest_path)

    manifest = load_json(Path(manifest_path))

    records = runtime_artifact_records(manifest)

    unrecorded = unrecorded_model_binaries(records)

    if unrecorded:
        raise BundleRefused(
            "the frozen manifest does not record every model binary serving "
            f"loads: {', '.join(unrecorded)}"
        )

    resolved = resolve_package(
        package=package,
        runtime_root=root,
        freeze_id=identity["freeze_id"],
        repository=repository,
        release_tag=release_tag,
        asset=asset,
        package_sha256=package_sha256,
        allow_download=allow_download,
    )

    key = bundle_key(
        freeze_id=identity["freeze_id"],
        manifest_sha256=identity["manifest_sha256"],
        package_sha256=resolved["package_sha256"],
    )

    expected = {
        "bundle_key": key,
        "freeze_id": identity["freeze_id"],
        "manifest_sha256": identity["manifest_sha256"],
        "package_sha256": resolved["package_sha256"],
    }

    bundle = root / BUNDLES_RELATIVE / key

    model_dir = bundle / MODEL_PREFIX

    if installed_bundle_is_valid(bundle, expected):
        return _receipt(
            outcome=OUTCOME_REUSED,
            bundle=bundle,
            model_dir=model_dir,
            identity=identity,
            resolved=resolved,
            key=key,
            metadata={
                **verify_runtime_model_artifacts(model_dir),
                **verify_frozen_bundle(bundle),
            },
            reason=(
                "a verified frozen runtime bundle for this freeze and package "
                "was already installed, so it was re-verified and reused"
            ),
        )

    if bundle.exists():
        shutil.rmtree(bundle)

    staging = root / STAGING_RELATIVE

    staging.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(dir=str(staging), prefix="install-"))

    try:
        staged_models = work / MODEL_PREFIX

        extract_frozen_project(resolved["path"], work)

        staged_manifest = staged_models / MANIFEST_RELATIVE

        if not staged_manifest.exists():
            raise BundleRefused(
                f"the frozen package {resolved['asset']} carries no "
                f"{MODEL_PREFIX}/{MANIFEST_RELATIVE.as_posix()}, so the "
                "installed bundle would describe nothing"
            )

        packaged = sha256_file(staged_manifest)

        if packaged != identity["manifest_sha256"]:
            raise BundleRefused(
                f"the frozen manifest inside {resolved['asset']} hashes to "
                f"{packaged}, not the repository's {identity['manifest_sha256']}, "
                "so the package does not describe this freeze"
            )

        verify_runtime_model_artifacts(staged_models)

        verify_frozen_bundle(work)

        (work / READY_RELATIVE).write_text(
            json.dumps(
                {
                    **expected,
                    "asset": resolved["asset"],
                    "freeze_stage": identity["freeze_stage"],
                    "installed_at": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        bundle.parent.mkdir(parents=True, exist_ok=True)

        os.replace(work, bundle)

    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    return _receipt(
        outcome=OUTCOME_INSTALLED,
        bundle=bundle,
        model_dir=model_dir,
        identity=identity,
        resolved=resolved,
        key=key,
        metadata={
            **verify_runtime_model_artifacts(model_dir),
            **verify_frozen_bundle(bundle),
        },
        reason=(
            "the frozen project tree was installed from the published package "
            "and the whole frozen manifest verified against it"
        ),
    )


def _receipt(
    *,
    outcome: str,
    bundle: Path,
    model_dir: Path,
    identity: dict[str, str],
    resolved: dict[str, Any],
    key: str,
    metadata: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "bundle_key": key,
        "bundle_root": str(bundle),
        "frozen_package": {
            "asset": resolved["asset"],
            "expected_digest_source": resolved["expected_digest_source"],
            "sha256": resolved["package_sha256"],
            "source": resolved["source"],
        },
        "freeze_id": identity["freeze_id"],
        "freeze_stage": identity["freeze_stage"],
        "generated_at": _utc_now(),
        "manifest_sha256": identity["manifest_sha256"],
        "model_dir": str(model_dir),
        "outcome": outcome,
        "reason": reason,
        "required_model_binaries": len(REQUIRED_MODEL_BINARIES),
        "verified_artifacts": metadata["verified_artifacts"],
        "verified_manifest_entries": metadata["verified_manifest_entries"],
    }


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def render(receipt: dict[str, Any]) -> str:
    rows = [
        ("outcome", receipt["outcome"]),
        ("freeze id", receipt["freeze_id"]),
        ("freeze stage", receipt["freeze_stage"]),
        ("package asset", receipt["frozen_package"]["asset"]),
        ("package sha256", receipt["frozen_package"]["sha256"][:16]),
        ("package source", receipt["frozen_package"]["source"]),
        ("manifest sha256", receipt["manifest_sha256"][:16]),
        ("verified artifacts", receipt["verified_artifacts"]),
        ("verified manifest entries", receipt["verified_manifest_entries"]),
        ("model dir", receipt["model_dir"]),
        ("bundle root", receipt["bundle_root"]),
    ]

    lines = [
        "## Frozen runtime bundle",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", receipt["reason"]]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install_frozen_model_artifacts",
        description=(
            "Install the already-frozen production model artifacts into a "
            "verified runtime directory outside the Git checkout. Builds "
            "nothing, trains nothing, promotes nothing."
        ),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="the durable production work root the bundle is installed under. "
        "Must be outside the repository: the model binaries are ignored in "
        "Git by design and are never copied into the checkout.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / MODEL_PREFIX / MANIFEST_RELATIVE,
        help="the authoritative frozen manifest whose hashes every installed "
        "artifact must match.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="an already obtained frozen package. Omitted, the runtime root's "
        "cache is used, and failing that the published release asset.",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name of the repository publishing the frozen package.",
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help="the release holding the frozen package. Defaults to the freeze "
        "id, which is the documented tag.",
    )
    parser.add_argument(
        "--asset",
        default=None,
        help="the package asset name. Defaults to the documented "
        f"{ASSET_TEMPLATE!r}.",
    )
    parser.add_argument(
        "--package-sha256",
        default=None,
        help="the digest the package must hash to. Omitted, the release's "
        f"{CHECKSUM_ASSET} is read when it is published.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="refuse rather than contacting GitHub for the package.",
    )
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--github-env",
        type=Path,
        default=None,
        help=f"append {MODEL_DIR_VARIABLE} and {BUNDLE_ROOT_VARIABLE} to this "
        "file, so the serving step is handed the verified model directory and "
        "the verified manifest resolution root explicitly.",
    )

    return parser.parse_args(argv)


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = install(
            runtime_root=args.runtime_root,
            manifest_path=args.manifest,
            package=args.package,
            repository=args.repository,
            release_tag=args.release_tag,
            asset=args.asset,
            package_sha256=args.package_sha256,
            allow_download=not args.no_download,
        )

    except (
        BundleRefused,
        FileNotFoundError,
        RuntimeError,
        OSError,
        zipfile.BadZipFile,
    ) as error:
        failure = {
            "error": type(error).__name__,
            "message": str(error),
            "outcome": OUTCOME_REFUSED,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            "## Frozen runtime bundle\n\n" f"REFUSED: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_FAILED

    _write(
        args.receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    _write(args.summary_path, render(receipt), append=True)

    _write(
        args.github_env,
        f"{MODEL_DIR_VARIABLE}={receipt['model_dir']}\n"
        f"{BUNDLE_ROOT_VARIABLE}={receipt['bundle_root']}\n",
        append=True,
    )

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
