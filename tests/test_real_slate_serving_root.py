"""The real-slate serving path: which root the frozen manifest resolves against.

The frozen manifest records 62 files in five groups. Both serving entry points
verify all 62 before they load a model, and they resolved them against the
process working directory. A Git checkout can satisfy neither the twelve
ignored model binaries nor the six 2025 audit outputs, and four of the source
records are approved divergences, so installing a verified frozen runtime
bundle got the scripts past nothing: the first non-empty REFRESHED slate would
have failed inside ``scripts/10_predict_slate.py`` at the verification, before
any model was loaded.

Nothing caught it because no test ever ran the two scripts against a verified
bundle and a slate with rows in it. Every existing test either replaced both
scripts with stubs or built one of their inputs in isolation. So the tests here
run the real scripts, as subprocesses, the way the lifecycle runs them.

Two tiers. The ones that only need a tree the manifest describes build a
synthetic bundle and run everywhere, including CI. The ones that need the
frozen models to actually score a slate require an installed bundle and skip
when there is not one, because the model binaries are a release asset by
design and a checkout never holds them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

import frozen_bundle_slate_fixture as fixture  # noqa: E402

PREDICT_SCRIPT = "scripts/10_predict_slate.py"
PRICE_SCRIPT = "scripts/15_price_markets.py"

MANIFEST_RELATIVE = Path("models/frozen_manifests/LATEST.json")

#: The twelve binaries serving loads, read from the installer rather than
#: restated so the two cannot disagree about what serving needs.
installer_spec = importlib.util.spec_from_file_location(
    "install_frozen_model_artifacts_for_root_tests",
    PROJECT / "ops" / "install_frozen_model_artifacts.py",
)
assert installer_spec is not None and installer_spec.loader is not None
installer = importlib.util.module_from_spec(installer_spec)
sys.modules[installer_spec.name] = installer
installer_spec.loader.exec_module(installer)

serving_spec = importlib.util.spec_from_file_location(
    "run_incumbent_production_serving_for_root_tests",
    PROJECT / "ops" / "run_incumbent_production_serving.py",
)
assert serving_spec is not None and serving_spec.loader is not None
serving = importlib.util.module_from_spec(serving_spec)
sys.modules[serving_spec.name] = serving
serving_spec.loader.exec_module(serving)

REQUIRED_MODEL_BINARIES = installer.REQUIRED_MODEL_BINARIES

#: The 2025 out-of-fold and market-backtest outputs the freeze recorded. They
#: are generated artifacts: no serving code reads them, and no checkout has
#: them, and the verifier resolves them anyway.
AUDIT_ONLY_DATA_FILES = tuple(
    record["path"]
    for record in json.loads(
        (PROJECT / MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )["files"]["data_audit_files"]
)


# ----------------------------------------------------------------------
# a synthetic bundle: a tree the frozen manifest describes, without models
# ----------------------------------------------------------------------


def synthetic_bundle(root: Path, *, groups: dict[str, tuple[str, ...]] | None = None) -> Path:
    """A bundle root holding every file its own manifest records.

    Shaped like the real one: a manifest at the external_test_deployment stage
    under ``models/frozen_manifests/``, recording files across several groups,
    some of which live outside ``models/``. That last part is the whole point
    -- a manifest naming only ``models/`` paths would resolve against a
    checkout too, and could not tell the two roots apart.
    """
    members = groups or {
        "model_artifacts": tuple(
            f"models/{name}" for name in REQUIRED_MODEL_BINARIES
        ),
        "data_audit_files": AUDIT_ONLY_DATA_FILES,
        "source_files": ("src/nba_prop_quant/production.py",),
        "scripts": (PREDICT_SCRIPT, PRICE_SCRIPT),
    }

    records: dict[str, list[dict[str, str]]] = {}

    for group, relatives in members.items():
        records[group] = []

        for relative in relatives:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"synthetic-{relative}".encode("utf-8")
            path.write_bytes(payload)
            records[group].append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    manifest = {
        "created_utc": "2026-10-10T00:00:00+00:00",
        "files": records,
        "freeze_id": "nba_prop_quant_20261010T000000Z",
        "freeze_stage": installer.REQUIRED_FREEZE_STAGE,
    }

    manifest_path = root / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return root


def run_script(
    script: str,
    *,
    bundle_root: Path | None,
    model_dir: Path,
    data_root: Path,
    extra: tuple[str, ...] = (),
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a serving script the way the lifecycle does: as a subprocess."""
    environment = dict(os.environ)
    environment.update(
        {
            "BDL_API_KEY": "unused-in-snapshot-mode",
            "ODDS_API_KEY": "unused-in-snapshot-mode",
            "NBA_PROP_DATA_DIR": str(data_root),
            "NBA_PROP_MODEL_DIR": str(model_dir),
            "NBA_PROP_GATE3_SNAPSHOT_DIR": str(data_root / "snapshots"),
            "PYTHONPATH": str(PROJECT / "src"),
        }
    )

    arguments = [
        sys.executable,
        str(PROJECT / script),
        "--date",
        fixture.SLATE_DATE,
        "--input-source",
        "snapshot",
    ]

    if bundle_root is not None:
        arguments += ["--frozen-bundle-root", str(bundle_root)]

    return subprocess.run(
        [*arguments, *extra],
        cwd=str(cwd or PROJECT),
        env=environment,
        capture_output=True,
        text=True,
    )


VERIFICATION_FAILED = "Frozen manifest verification failed"


# ----------------------------------------------------------------------
# 1. the checkout holds none of the binaries serving needs
# ----------------------------------------------------------------------


def test_the_checkout_contains_none_of_the_required_joblibs():
    """The fact the whole resolution-root question rests on.

    If a checkout could hold these, resolving the manifest against it would
    work and none of this would be necessary. It cannot: ``.gitignore``
    excludes them because the frozen package is a release asset, and
    ``actions/checkout`` cleans ignored files out of the workspace anyway.
    """
    ignore = (PROJECT / ".gitignore").read_text(encoding="utf-8")

    assert "models/**/*.joblib" in ignore

    present = [
        name
        for name in REQUIRED_MODEL_BINARIES
        if (PROJECT / "models" / name).exists()
    ]

    assert present == [], (
        "this suite states what a clean checkout looks like; a checkout "
        f"carrying model binaries is not the case under test: {present}"
    )


# ----------------------------------------------------------------------
# 2. a verified bundle holds every file its manifest records
# ----------------------------------------------------------------------


def test_a_verified_bundle_holds_everything_its_manifest_records(tmp_path: Path):
    """What makes the bundle a resolution root and the checkout not one."""
    bundle = synthetic_bundle(tmp_path / "bundle")

    verified = installer.verify_frozen_bundle(bundle)

    manifest = json.loads(
        (bundle / MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )

    recorded = sum(len(group) for group in manifest["files"].values())

    assert verified["verified_manifest_entries"] == recorded
    assert verified["freeze_stage"] == installer.REQUIRED_FREEZE_STAGE

    for group in manifest["files"].values():
        for record in group:
            assert (bundle / record["path"]).exists(), record["path"]


# ----------------------------------------------------------------------
# 3 and 4. each script resolves the manifest against the root it is given
# ----------------------------------------------------------------------


@pytest.mark.parametrize("script", [PREDICT_SCRIPT, PRICE_SCRIPT])
def test_a_serving_script_verifies_against_the_bundle_root_not_the_cwd(
    script: str, tmp_path: Path
):
    """The defect and the fix, stated on each script separately.

    Run from the checkout with a verified model directory and no bundle root,
    both scripts fail at the verification: the manifest names files the
    checkout does not have, so neither reaches its own work at all. Given the
    bundle root, the same manifest verifies and each script proceeds past it,
    which is what this is about -- what it then does with an empty data root
    is the next test's business.
    """
    bundle = synthetic_bundle(tmp_path / "bundle")

    data_root = tmp_path / "data"
    data_root.mkdir()

    without = run_script(
        script,
        bundle_root=None,
        model_dir=bundle / "models",
        data_root=data_root,
    )

    assert without.returncode != 0
    assert VERIFICATION_FAILED in without.stdout + without.stderr

    with_root = run_script(
        script,
        bundle_root=bundle,
        model_dir=bundle / "models",
        data_root=data_root,
    )

    assert VERIFICATION_FAILED not in with_root.stdout + with_root.stderr


# ----------------------------------------------------------------------
# 7. a corrupt runtime model binary fails closed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("script", [PREDICT_SCRIPT, PRICE_SCRIPT])
def test_a_corrupt_runtime_model_binary_fails_closed(script: str, tmp_path: Path):
    """Resolving against the bundle is not trusting the bundle.

    The bundle root is where the manifest's records are looked up; it is not
    an assertion that whatever is there is right. A tampered binary inside the
    bundle still fails the same verification.
    """
    bundle = synthetic_bundle(tmp_path / "bundle")

    (bundle / "models" / "marginals.joblib").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match=VERIFICATION_FAILED):
        installer.verify_frozen_bundle(bundle)

    data_root = tmp_path / "data"
    data_root.mkdir()

    completed = run_script(
        script,
        bundle_root=bundle,
        model_dir=bundle / "models",
        data_root=data_root,
    )

    assert completed.returncode != 0

    output = completed.stdout + completed.stderr

    assert VERIFICATION_FAILED in output
    assert "marginals.joblib" in output


# ----------------------------------------------------------------------
# 8. the wrong bundle root fails closed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("script", [PREDICT_SCRIPT, PRICE_SCRIPT])
def test_the_wrong_bundle_root_fails_closed(script: str, tmp_path: Path):
    """An explicit root is not an unchecked one.

    Pointing at a tree that is not the bundle has to fail, or the argument
    would be a way to turn verification off rather than a way to aim it.
    """
    bundle = synthetic_bundle(tmp_path / "bundle")

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    data_root = tmp_path / "data"
    data_root.mkdir()

    completed = run_script(
        script,
        bundle_root=unrelated,
        model_dir=bundle / "models",
        data_root=data_root,
    )

    assert completed.returncode != 0
    assert VERIFICATION_FAILED in completed.stdout + completed.stderr


def test_a_bundle_root_from_another_freeze_is_refused(tmp_path: Path):
    """Two verified bundles are not interchangeable halves.

    Both roots verify on their own here, so nothing downstream would have
    noticed. The caller refuses, because serving one freeze's models while
    verifying another freeze's manifest is not serving either of them.
    """
    first = synthetic_bundle(tmp_path / "first")
    second = synthetic_bundle(tmp_path / "second")

    manifest_path = second / MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["freeze_id"] = "nba_prop_quant_20261011T000000Z"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(serving.ServingRefused, match="not the same frozen bundle"):
        serving._bundle_identity(first / "models", second)


# ----------------------------------------------------------------------
# 9. the six audit-only outputs are not a serving prerequisite
# ----------------------------------------------------------------------


def test_the_missing_audit_outputs_do_not_block_serving(tmp_path: Path):
    """Absent from the checkout, present in the bundle, unchanged in both.

    This is the precise shape of the original mistake. "No serving code reads
    these" is true and was taken to mean "serving does not need these", and
    the verifier resolves every group the manifest records regardless of what
    reads what. The fix is the root they resolve against, not an exemption:
    their hashes stay in the manifest and all six are still verified.
    """
    assert len(AUDIT_ONLY_DATA_FILES) == 6

    for relative in AUDIT_ONLY_DATA_FILES:
        assert not (PROJECT / relative).exists(), (
            f"{relative} is a generated audit output and absent from a "
            "checkout; a checkout carrying it is not the case under test"
        )

    bundle = synthetic_bundle(tmp_path / "bundle")

    for relative in AUDIT_ONLY_DATA_FILES:
        assert (bundle / relative).exists()

    verified = installer.verify_frozen_bundle(bundle)

    assert verified["verified_manifest_entries"] == len(
        REQUIRED_MODEL_BINARIES
    ) + len(AUDIT_ONLY_DATA_FILES) + 3

    # And removing one from the bundle is a refusal, not a shrug.
    (bundle / AUDIT_ONLY_DATA_FILES[0]).unlink()

    with pytest.raises(RuntimeError, match=VERIFICATION_FAILED):
        installer.verify_frozen_bundle(bundle)


# ----------------------------------------------------------------------
# 12. the incumbent is still resolved from promotion state alone
# ----------------------------------------------------------------------


def test_an_unpromoted_candidate_is_never_served_through_the_new_roots(
    tmp_path: Path,
):
    """Adding a root argument must not add an authority.

    The bundle root says where to verify. It does not say what to serve, and a
    registry with a registered but unpromoted candidate still resolves to the
    frozen bundle.
    """
    bundle = synthetic_bundle(tmp_path / "bundle")

    registry = serving.FitRegistry(
        root=tmp_path / "registry", project_root=PROJECT
    )

    assert registry.current()["current_good_fit_id"] is None

    incumbent = serving.resolve_incumbent(
        registry=registry, model_dir=bundle / "models", bundle_root=bundle
    )

    assert incumbent["authority"] == serving.AUTHORITY_FROZEN_BUNDLE
    assert incumbent["fit_id"] is None
    assert incumbent["bundle_root"] == str(bundle)
    assert incumbent["frozen_bundle"]["verified_manifest_entries"] > 0


# ----------------------------------------------------------------------
# 13. nothing here publishes or promotes
# ----------------------------------------------------------------------


def test_nothing_in_the_corrected_serving_path_publishes_or_promotes():
    """The change is a path, not an authority."""
    changed = (
        PREDICT_SCRIPT,
        PRICE_SCRIPT,
        "ops/run_incumbent_production_serving.py",
        "ops/install_frozen_model_artifacts.py",
    )

    # The same tokens the existing serving contract test uses: a call or an
    # import, not a word. These files explain in prose that they cannot
    # publish or promote, and that prose must not be what the check reads.
    forbidden = (
        ".promote(",
        ".rollback(",
        "publish_",
        "_publish",
        "wizardofodds",
        "WizardOfOdds",
    )

    for relative in changed:
        source = (PROJECT / relative).read_text(encoding="utf-8")

        for token in forbidden:
            assert token not in source, f"{relative} names {token}"


def test_the_serving_step_passes_both_verified_roots():
    """What the lifecycle actually hands the caller."""
    import yaml

    workflow = yaml.safe_load(
        (PROJECT / ".github/workflows/nba_production_lifecycle.yml").read_text(
            encoding="utf-8"
        )
    )

    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "run_incumbent_production_serving.py" in str(step.get("run", ""))
    ]

    assert steps, "no lifecycle step invokes the incumbent serving caller"

    for step in steps:
        run = str(step["run"])
        assert f"--model-dir \"${installer.MODEL_DIR_VARIABLE}\"" in run
        assert (
            f"--frozen-bundle-root \"${installer.BUNDLE_ROOT_VARIABLE}\"" in run
        )


def test_the_install_step_publishes_both_roots():
    """The two variables the serving step reads are the two it writes."""
    source = (PROJECT / "ops" / "install_frozen_model_artifacts.py").read_text(
        encoding="utf-8"
    )

    assert installer.MODEL_DIR_VARIABLE in source
    assert installer.BUNDLE_ROOT_VARIABLE in source
    assert installer.BUNDLE_ROOT_VARIABLE == "FROZEN_RUNTIME_BUNDLE_ROOT"


# ----------------------------------------------------------------------
# 5, 6, 10 and 11. the frozen models actually scoring a non-empty slate
# ----------------------------------------------------------------------


def installed_bundle() -> Path | None:
    """An installed frozen runtime bundle, if this machine has one.

    Discovered rather than built. The twelve model binaries total 386MB and
    are published as a release asset precisely so they are not in Git, so
    these tests describe an operator's machine or a runner that has run the
    install step, and skip on one that has not.
    """
    explicit = os.environ.get("NBA_PROP_TEST_FROZEN_BUNDLE_ROOT")

    if explicit:
        return Path(explicit)

    work = os.environ.get("NBA_PROP_WORK_DIR")

    if not work:
        return None

    bundles = Path(work) / installer.BUNDLES_RELATIVE

    if not bundles.is_dir():
        return None

    for candidate in sorted(bundles.iterdir()):
        if (candidate / installer.READY_RELATIVE).exists():
            return candidate

    return None


@pytest.fixture(scope="module")
def frozen_bundle() -> Path:
    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    installer.verify_frozen_bundle(bundle)

    return bundle


def serve_the_fixture(bundle: Path, root: Path) -> dict:
    """The fixture slate, through the real caller and the real scripts."""
    fixture.build(root)

    (root / "registry").mkdir(exist_ok=True)

    registry = serving.FitRegistry(
        root=root / "registry", project_root=PROJECT
    )

    incumbent = serving.resolve_incumbent(
        registry=registry,
        model_dir=bundle / "models",
        bundle_root=bundle,
    )

    saved = dict(os.environ)
    os.environ.update(
        {
            "BDL_API_KEY": "unused-in-snapshot-mode",
            "ODDS_API_KEY": "unused-in-snapshot-mode",
            "NBA_PROP_GATE3_SNAPSHOT_DIR": str(root / "data" / "snapshots"),
            "PYTHONPATH": str(PROJECT / "src"),
        }
    )

    try:
        return serving.serve(
            incumbent=incumbent,
            slate_date=fixture.SLATE_DATE,
            data_root=root / "data",
            project_root=PROJECT,
            # The one concession, and it is to a different defect: the slate
            # builder emits none of the five game-context flags the frozen
            # models were trained on, so the feature contract refuses any
            # non-empty slate. That is reported, not fixed here, and
            # test_the_slate_builder_still_omits_the_game_context_flags pins
            # it so the day it is fixed this override can come out.
            predict_arguments=["--allow-missing-model-features"],
        )
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_a_non_empty_slate_produces_predictions(frozen_bundle: Path, tmp_path: Path):
    """predictions > 0, through the bundle, with no checkout artifacts."""
    served = serve_the_fixture(frozen_bundle, tmp_path / "run")

    assert served["outcome"] == serving.OUTCOME_SERVED

    projections = pd.read_parquet(
        tmp_path / "run" / "data" / "processed" / "projections"
        / f"{fixture.SLATE_DATE}.parquet"
    )

    assert len(projections) == len(fixture.ROSTER)
    assert projections["freeze_id"].nunique() == 1

    for target in ("pts", "reb", "ast", "stl", "blk", "fg3m"):
        column = f"mu_selected_{target}"
        assert column in projections.columns
        assert projections[column].notna().all()
        assert (projections[column] >= 0).all()


def test_a_non_empty_slate_produces_priced_markets(
    frozen_bundle: Path, tmp_path: Path
):
    """priced rows > 0, from the same run."""
    served = serve_the_fixture(frozen_bundle, tmp_path / "run")

    assert served["outcome"] == serving.OUTCOME_SERVED

    priced = pd.read_parquet(
        tmp_path / "run" / "data" / "processed" / "priced_markets"
        / f"{fixture.SLATE_DATE}.parquet"
    )

    assert len(priced) > 0
    assert len(priced) == fixture.QUOTED_PROPS.__len__()

    for column in (
        "calibrated_q_over_nonpush",
        "market_devig_q_over",
        "model_preferred_side",
        "model_preferred_edge",
    ):
        assert column in priced.columns
        assert priced[column].notna().all()


@pytest.mark.parametrize(
    "artifact,key",
    [
        ("projections", "mu_selected_pts"),
        ("priced_markets", "calibrated_q_over_nonpush"),
    ],
)
def test_the_same_frozen_artifacts_produce_the_same_numbers(
    frozen_bundle: Path, tmp_path: Path, artifact: str, key: str
):
    """The resolution root decides nothing about the output.

    Two runs of the same fixture against the same bundle, in different
    directories. If the root the manifest resolves against could reach the
    numbers, these would differ.
    """
    frames = []

    for label in ("first", "second"):
        root = tmp_path / label
        serve_the_fixture(frozen_bundle, root)
        frames.append(
            pd.read_parquet(
                root / "data" / "processed" / artifact
                / f"{fixture.SLATE_DATE}.parquet"
            )
        )

    first, second = frames

    assert len(first) == len(second)
    assert key in first.columns

    pd.testing.assert_series_equal(
        first[key].reset_index(drop=True), second[key].reset_index(drop=True)
    )


def test_the_slate_builder_still_omits_the_game_context_flags(
    frozen_bundle: Path, tmp_path: Path
):
    """A second defect, recorded rather than fixed.

    ``build_upcoming_slate_features`` emits none of the five game-context
    flags the frozen models were trained on, and nothing between it and the
    feature contract supplies them, so a strict non-empty slate is refused
    after the manifest verifies. It is a separate defect in a different file
    and fixing it is not this change.

    Pinned here because it is the next thing a real slate hits, and because
    the override the tests above pass should be removed the day it is fixed --
    a test that fails when a defect is repaired is how that gets noticed.
    """
    root = tmp_path / "run"

    fixture.build(root)

    completed = run_script(
        PREDICT_SCRIPT,
        bundle_root=frozen_bundle,
        model_dir=frozen_bundle / "models",
        data_root=root / "data",
    )

    assert completed.returncode != 0

    output = completed.stdout + completed.stderr

    assert VERIFICATION_FAILED not in output, (
        "the manifest must verify against the bundle; this test is about what "
        "happens after it does"
    )

    assert "missing 5 trained feature(s)" in output

    for flag in (
        "is_nba_cup",
        "is_nba_cup_championship",
        "is_play_in",
        "is_playoffs",
        "is_regular_season",
    ):
        assert flag in output, flag
