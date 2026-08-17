"""Guards what a release actually ships (the hatchling packaging contract).

setuptools shipped only the ``.py`` files it could discover as packages, so the
tracked test-fixture videos under ``theremin/tests/testing_data/`` never reached
a user. hatchling ships **every** non-VCS-ignored file under ``theremin/``, so
the build-backend migration swept them in: the wheel went from 51 KB to 56.5 MB
and every ``pip install theremin`` would have downloaded ~52 MB of test videos.

Nothing failed. The suite was green, the wheel built, the metadata was correct;
the only symptom lived in a number nobody looks at. So this module looks at it.

Deliberately *not* checked against a built artifact: that would need a build
backend and tens of seconds per run. Instead the shipped payload is computed
from the excludes declared in ``pyproject.toml``, which is the thing that can
regress -- deleting them makes the numbers below jump by ~52 MB.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR_NAME = 'theremin'

# Uncompressed, which is what a source tree can be measured for without a build.
# The payload today is the gesture-recognizer model (8.4 MB) plus ~0.3 MB of code
# -- 6.1 MB once zipped into the wheel. The headroom is for a deliberate asset
# addition; a build that needs more should raise this in a commit that says why.
MAX_SHIPPED_PAYLOAD_BYTES = 12 * 1024 * 1024

# Ships on purpose: theremin/video_features.py reads it from the package data
# directory at runtime, and earlier releases were broken without it.
MUST_SHIP = 'theremin/data/gesture_recognizer.task'

# Never ships: 52 MB of recorded video used only by the test suite.
MUST_NOT_SHIP_DIR = 'theremin/tests/testing_data'

# Not tracked by git, so hatchling never sees them either.
UNTRACKED_NAMES = {'__pycache__', '.DS_Store'}


def _build_target_excludes(target):
    """The `exclude` patterns declared for a `[tool.hatch.build.targets.<target>]`."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        tomllib = pytest.importorskip(
            'tomli', reason='needs tomllib (3.11+) or tomli to parse pyproject.toml'
        )
    pyproject = REPO_ROOT / 'pyproject.toml'
    if not pyproject.is_file():  # pragma: no cover - installed, not a source tree
        pytest.skip('pyproject.toml not available (not a source checkout)')
    config = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    targets = config.get('tool', {}).get('hatch', {}).get('build', {}).get('targets', {})
    assert target in targets, f'no [tool.hatch.build.targets.{target}] section'
    return targets[target].get('exclude', [])


def _is_excluded(relative_path, excludes):
    """True when `relative_path` falls under one of the declared exclude patterns.

    Only the plain directory-prefix form used by this project is understood. A
    glob pattern would be read as a literal here and so would *under*-exclude,
    which shows up as a failure rather than as a false pass.
    """
    posix_path = relative_path.as_posix()
    return any(
        posix_path == pattern or posix_path.startswith(f'{pattern}/')
        for pattern in excludes
    )


def _shipped_files(excludes):
    """Files under the package directory that the declared excludes let through."""
    package_dir = REPO_ROOT / PACKAGE_DIR_NAME
    for path in sorted(package_dir.rglob('*')):
        if not path.is_file():
            continue
        relative_path = path.relative_to(REPO_ROOT)
        if UNTRACKED_NAMES & set(relative_path.parts) or path.suffix == '.pyc':
            continue
        if not _is_excluded(relative_path, excludes):
            yield relative_path


@pytest.mark.parametrize('target', ['wheel', 'sdist'])
def test_shipped_payload_stays_small(target):
    excludes = _build_target_excludes(target)
    shipped = list(_shipped_files(excludes))
    total_bytes = sum((REPO_ROOT / path).stat().st_size for path in shipped)
    heaviest = sorted(shipped, key=lambda p: -(REPO_ROOT / p).stat().st_size)[:5]
    assert total_bytes <= MAX_SHIPPED_PAYLOAD_BYTES, (
        f'the {target} would ship {total_bytes / 1e6:.1f} MB uncompressed, over '
        f'the {MAX_SHIPPED_PAYLOAD_BYTES / 1e6:.1f} MB budget. Heaviest entries: '
        + ', '.join(
            f'{p.as_posix()} ({(REPO_ROOT / p).stat().st_size / 1e6:.1f} MB)'
            for p in heaviest
        )
    )


@pytest.mark.parametrize('target', ['wheel', 'sdist'])
def test_test_fixture_videos_never_ship(target):
    excludes = _build_target_excludes(target)
    shipped = {path.as_posix() for path in _shipped_files(excludes)}
    leaked = sorted(p for p in shipped if p.startswith(f'{MUST_NOT_SHIP_DIR}/'))
    assert not leaked, f'test fixtures would ship in the {target}: {leaked}'


@pytest.mark.parametrize('target', ['wheel', 'sdist'])
def test_the_gesture_recognizer_model_still_ships(target):
    """The size budget must not be met by dropping the data file users need."""
    assert (REPO_ROOT / MUST_SHIP).is_file(), f'{MUST_SHIP} is missing from the repo'
    excludes = _build_target_excludes(target)
    shipped = {path.as_posix() for path in _shipped_files(excludes)}
    assert MUST_SHIP in shipped
