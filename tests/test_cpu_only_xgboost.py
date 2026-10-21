"""Tests that this project needs only the CPU XGBoost build.

There is no GPU production path. The model fits with ``tree_method="hist"``
on the self-hosted Mac and in CI, and nothing here has ever trained
distributed. But the default ``xgboost`` wheel declares
``nvidia-nccl-cu13`` on Linux -- a 291 MB CUDA collective-communication
library for multi-GPU training -- so every Linux environment downloaded and
installed it to run a CPU fit.

``xgboost-cpu`` is the same upstream release with that requirement removed and
the same ``xgboost`` import package, so the swap is a packaging change and not
a modelling one. These tests are what makes that claim checkable: that the
import still resolves at the pinned version, that it is the CPU distribution
providing it on Linux, that NCCL is genuinely absent, that no source file
names an API the CPU build does not have, and that a fit still trains,
predicts and reloads to the same numbers.

The round trip trains 24 trees on 240 synthetic rows. It is a serialization
and API check, not a model check: it asserts determinism and identity across
a save/load, never an accuracy number. Nothing here reads production data,
loads the frozen bundle, or writes outside ``tmp_path``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost

from nba_prop_quant.model import (
    DEFAULT_XGB_PARAMS,
    ModelBundle,
    fit_minutes_model,
)

REPO = Path(__file__).resolve().parents[1]

CONSTRAINTS = REPO / "constraints" / "py312-production.txt"

#: The distribution that must provide ``xgboost`` on Linux, and the one that
#: must not. ``xgboost-cpu`` publishes no macOS wheel, which is why the pin is
#: a marker rather than a substitution: the production runner is a Mac, where
#: the default wheel never carried the NVIDIA payload in the first place.
CPU_DISTRIBUTION = "xgboost-cpu"

DEFAULT_DISTRIBUTION = "xgboost"

NVIDIA_DISTRIBUTION = "nvidia-nccl-cu13"

#: XGBoost API surface that exists only with a CUDA build, or only for the
#: distributed training this project does not do. ``nccl`` and ``rabit`` are
#: the collective layer the NVIDIA dependency exists to serve; the rest select
#: a GPU device or a GPU tree builder.
GPU_ONLY_TOKENS: tuple[str, ...] = (
    "gpu_hist",
    "gpu_approx",
    "gpu_predictor",
    "gpu_id",
    "cuda",
    "nccl",
    "rabit",
    "xgboost.dask",
    "xgboost.spark",
    "xgboost.collective",
)

#: The tree builders that run without a GPU. ``approx`` and ``exact`` are
#: here because they are CPU-valid, not because anything uses them: the point
#: is that whatever the model names must be one of these.
CPU_TREE_METHODS = ("auto", "exact", "approx", "hist")

SEARCHED_DIRECTORIES = ("src", "ops", "scripts", "research")


def pinned_xgboost_version() -> str:
    """The version the constraint file allows on this platform.

    Read from the file rather than written out here, so there is one place a
    version bump has to happen and this test cannot drift away from the
    environment production builds.
    """
    pattern = re.compile(
        r"^(xgboost|xgboost-cpu)==(?P<version>[^\s;]+)", re.MULTILINE
    )

    versions = {
        match.group("version")
        for match in pattern.finditer(
            CONSTRAINTS.read_text(encoding="utf-8")
        )
    }

    assert len(versions) == 1, (
        "the two XGBoost pins name different versions, so Linux and macOS "
        f"would run different builds: {sorted(versions)}"
    )

    return versions.pop()


def installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def providers_of_xgboost() -> list[str]:
    return sorted(metadata.packages_distributions().get("xgboost") or [])


# ----------------------------------------------------------------------
# the environment this suite is running in
# ----------------------------------------------------------------------


def test_the_import_is_the_pinned_version():
    assert xgboost.__version__ == pinned_xgboost_version()


def test_linux_gets_xgboost_from_the_cpu_distribution():
    if sys.platform != "linux":
        pytest.skip(f"{CPU_DISTRIBUTION} publishes no wheel for this platform")

    assert providers_of_xgboost() == [CPU_DISTRIBUTION]

    assert installed(DEFAULT_DISTRIBUTION) is None, (
        f"both {DEFAULT_DISTRIBUTION} and {CPU_DISTRIBUTION} are installed, "
        "so which one provides the import depends on install order"
    )


def test_the_nvidia_collective_library_is_not_installed():
    if sys.platform != "linux":
        pytest.skip(f"{NVIDIA_DISTRIBUTION} is a Linux-only requirement")

    assert installed(NVIDIA_DISTRIBUTION) is None, (
        f"{NVIDIA_DISTRIBUTION} is installed at "
        f"{installed(NVIDIA_DISTRIBUTION)}; it is a CUDA collective library "
        "for multi-GPU training and this project fits on CPU"
    )


def test_the_cpu_distribution_requires_nothing_from_nvidia():
    """Not just absent now -- not reachable.

    An absent package could be absent because an install step happened to
    skip it. This reads the metadata the resolver reads, so it fails if a
    future version of the CPU build starts declaring the dependency again.
    """
    if sys.platform != "linux":
        pytest.skip(f"{CPU_DISTRIBUTION} publishes no wheel for this platform")

    requires = metadata.distribution(CPU_DISTRIBUTION).requires or []

    assert [name for name in requires if "nvidia" in name.lower()] == []


# ----------------------------------------------------------------------
# the source names no API the CPU build lacks
# ----------------------------------------------------------------------


def python_files() -> list[Path]:
    return sorted(
        path
        for directory in SEARCHED_DIRECTORIES
        for path in (REPO / directory).rglob("*.py")
    )


def executable_source(path: Path) -> str:
    """The file's code with its prose removed.

    Scanning raw text cannot tell ``tree_method="gpu_hist"`` from a comment
    explaining why this project does not use one -- and the module that
    declares why the NVIDIA dependency was dropped is full of the second.
    Comments and docstrings are the prose; string literals are not, because a
    GPU tree method is selected by one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        body = node.body

        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]

    return ast.unparse(ast.fix_missing_locations(tree))


def test_no_source_file_names_a_gpu_only_api():
    found: list[str] = []

    for path in python_files():
        lowered = executable_source(path).lower()

        for token in GPU_ONLY_TOKENS:
            if token in lowered:
                found.append(f"{path.relative_to(REPO)}: {token}")

    assert found == [], (
        "these name XGBoost surface the CPU build does not provide, so the "
        f"CPU-only dependency would stop being enough: {found}"
    )


def test_the_default_parameters_select_a_cpu_tree_builder():
    assert DEFAULT_XGB_PARAMS["tree_method"] in CPU_TREE_METHODS

    assert "device" not in DEFAULT_XGB_PARAMS

    assert "predictor" not in DEFAULT_XGB_PARAMS


def test_the_installed_build_reports_no_gpu_support():
    """The build config, not just the parameters.

    ``XGBRegressor`` would accept ``device="cuda"`` and fail at fit time, so
    the question worth asking the library is what it was compiled with.
    """
    info = xgboost.build_info()

    for key in ("USE_CUDA", "USE_NCCL", "USE_DLOPEN_NCCL", "USE_FEDERATED"):
        assert info.get(key) in (False, None), f"{key}={info.get(key)!r}"


# ----------------------------------------------------------------------
# a fit still trains, predicts and reloads
# ----------------------------------------------------------------------

#: Small enough to be a unit test, large enough that 24 trees have something
#: to split on. The relationship is deterministic because the test asserts
#: reproducibility, not fit quality.
ROWS = 240

SEED = 7

#: A deliberately cheap fit. The production parameters are not this test's
#: business and are asserted unchanged elsewhere; overriding them here keeps
#: the test under a second without touching what production trains with.
ROUND_TRIP_PARAMS = {"n_estimators": 24, "max_depth": 3, "n_jobs": 1}


@pytest.fixture
def training_frame() -> pd.DataFrame:
    generator = np.random.default_rng(SEED)

    frame = pd.DataFrame(
        {
            "is_home": generator.integers(0, 2, ROWS).astype(float),
            "days_rest": generator.integers(0, 4, ROWS).astype(float),
            "prior_minutes_mean": generator.uniform(8.0, 36.0, ROWS),
            "prior_points_mean": generator.uniform(2.0, 28.0, ROWS),
        }
    )

    frame["minutes"] = (
        6.0
        + 0.7 * frame["prior_minutes_mean"]
        + 0.8 * frame["is_home"]
        + 0.3 * frame["days_rest"]
    )

    return frame


def test_a_cpu_fit_trains_predicts_and_reloads_identically(
    training_frame, tmp_path
):
    bundle = fit_minutes_model(training_frame, params=ROUND_TRIP_PARAMS)

    before = bundle.predict(training_frame)

    assert before.shape == (ROWS,)

    assert np.isfinite(before).all()

    path = tmp_path / "minutes.joblib"

    bundle.save(path)

    reloaded = ModelBundle.load(path)

    assert reloaded.feature_names == bundle.feature_names

    assert reloaded.target_name == bundle.target_name

    assert reloaded.objective == bundle.objective

    # Bit-for-bit, because a predictor that reloads to approximately itself is
    # one that prices a slate differently after a restart.
    assert np.array_equal(reloaded.predict(training_frame), before)


def test_the_booster_round_trips_through_the_native_format(
    training_frame, tmp_path
):
    """The serialization half, independent of pickle.

    ``joblib`` round trips the Python object; this round trips the model the
    CPU build actually produced, which is the part that would break if the
    wheel were built differently.
    """
    bundle = fit_minutes_model(training_frame, params=ROUND_TRIP_PARAMS)

    booster = bundle.model.get_booster()

    assert booster.num_boosted_rounds() == ROUND_TRIP_PARAMS["n_estimators"]

    path = tmp_path / "minutes.ubj"

    booster.save_model(str(path))

    reloaded = xgboost.Booster()

    reloaded.load_model(str(path))

    matrix = xgboost.DMatrix(
        training_frame.reindex(columns=bundle.feature_names)
    )

    assert np.array_equal(
        reloaded.predict(matrix), booster.predict(matrix)
    )


def test_two_fits_of_the_same_rows_agree(training_frame):
    """The seed does what it says under the CPU build.

    A fit whose trees depend on thread scheduling would make every downstream
    probability a function of how busy the runner was.
    """
    first = fit_minutes_model(training_frame, params=ROUND_TRIP_PARAMS)

    second = fit_minutes_model(training_frame, params=ROUND_TRIP_PARAMS)

    assert np.array_equal(
        first.predict(training_frame), second.predict(training_frame)
    )


def test_a_fit_in_a_fresh_interpreter_predicts_the_same_numbers():
    """Determinism across processes, not just within one.

    Production fits in a per-run virtual environment, so the numbers that
    matter are the ones a new interpreter produces.
    """
    script = f"""
import numpy as np
import pandas as pd

from nba_prop_quant.model import fit_minutes_model

generator = np.random.default_rng({SEED})

frame = pd.DataFrame(
    {{
        "is_home": generator.integers(0, 2, {ROWS}).astype(float),
        "days_rest": generator.integers(0, 4, {ROWS}).astype(float),
        "prior_minutes_mean": generator.uniform(8.0, 36.0, {ROWS}),
        "prior_points_mean": generator.uniform(2.0, 28.0, {ROWS}),
    }}
)

frame["minutes"] = (
    6.0
    + 0.7 * frame["prior_minutes_mean"]
    + 0.8 * frame["is_home"]
    + 0.3 * frame["days_rest"]
)

bundle = fit_minutes_model(frame, params={ROUND_TRIP_PARAMS!r})

print(bundle.predict(frame).tobytes().hex())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=True,
    )

    generator = np.random.default_rng(SEED)

    frame = pd.DataFrame(
        {
            "is_home": generator.integers(0, 2, ROWS).astype(float),
            "days_rest": generator.integers(0, 4, ROWS).astype(float),
            "prior_minutes_mean": generator.uniform(8.0, 36.0, ROWS),
            "prior_points_mean": generator.uniform(2.0, 28.0, ROWS),
        }
    )

    frame["minutes"] = (
        6.0
        + 0.7 * frame["prior_minutes_mean"]
        + 0.8 * frame["is_home"]
        + 0.3 * frame["days_rest"]
    )

    here = fit_minutes_model(frame, params=ROUND_TRIP_PARAMS)

    assert completed.stdout.strip() == (
        here.predict(frame).tobytes().hex()
    )
