"""Guards the property that keeps CI green: import-safety without optional engines.

Issue #7. The wads test action always appends ``--doctest-modules`` and passes no
explicit path, so pytest falls back to ``[tool.pytest.ini_options] testpaths`` --
which means every path listed there is *imported* at collection time. A module
that reaches ``import pyo`` (the real-time audio engine, an opt-in extra with no
universal wheels) therefore does not merely fail its own tests: it kills
collection for the whole run.

That is exactly how the legacy CI died. It ran ``pytest -v --doctest-modules``
over every source file, so ``theremin/audio.py`` -> ``from hum import Synth`` ->
``hum.pyo_util`` -> ``from pyo import PyoObject`` -> ``ModuleNotFoundError``.

These tests make that regression loud and local instead of red and remote. Each
candidate module is imported in a **subprocess** with the optional engines forced
to look uninstalled, so the check is meaningful on a developer machine that has
them installed -- otherwise it would silently pass for the wrong reason.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Optional/system-dependent imports that must never be needed at module-import
# time by anything pytest collects. `pyo` is an opt-in extra (`theremin[audio]`);
# `cv2`/`mediapipe` are hard dependencies but need system libraries (e.g. libGL)
# that a headless CI runner does not necessarily have.
BLOCKED_AT_IMPORT = ('pyo', 'cv2', 'mediapipe')

# The modules that must stay importable for the CLI entry point to work at all.
# Checked unconditionally, on every Python version, independently of pyproject
# parsing (which needs tomllib, and so is skipped on Python 3.10).
CORE_IMPORT_SAFE_MODULES = (
    'theremin',
    'theremin.util',
    'theremin.control_events',
    'theremin.script_utils',
    'theremin.main',
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a subprocess: install a meta-path finder that makes `blocked` top-level
# packages raise ModuleNotFoundError even when they are installed, then import the
# target module and assert none of them leaked into sys.modules.
_BLOCKED_IMPORT_SCRIPT = '''
import importlib
import sys

blocked = %(blocked)r
target = %(target)r


class BlockedFinder:
    """Makes `blocked` top-level packages look uninstalled."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('No module named ' + repr(fullname), name=fullname)
        return None


sys.meta_path.insert(0, BlockedFinder())
for name in [n for n in sys.modules if n.split('.')[0] in blocked]:
    del sys.modules[name]

importlib.import_module(target)

leaked = sorted(n for n in sys.modules if n.split('.')[0] in blocked)
if leaked:
    raise AssertionError('blocked modules were imported: ' + ', '.join(leaked))
'''


def _import_in_subprocess(module_name, *, blocked=BLOCKED_AT_IMPORT):
    """Import `module_name` with `blocked` packages masked; return the CompletedProcess."""
    script = _BLOCKED_IMPORT_SCRIPT % {'blocked': tuple(blocked), 'target': module_name}
    return subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _testpaths_from_pyproject():
    """`[tool.pytest.ini_options] testpaths`, or None when it cannot be read."""
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
    return config.get('tool', {}).get('pytest', {}).get('ini_options', {}).get(
        'testpaths'
    )


@pytest.mark.parametrize('module_name', CORE_IMPORT_SAFE_MODULES)
def test_core_modules_import_without_optional_engines(module_name):
    """The CLI entry path imports with pyo/cv2/mediapipe all unavailable."""
    result = _import_in_subprocess(module_name)
    assert result.returncode == 0, (
        f'{module_name} failed to import without {BLOCKED_AT_IMPORT}:\n'
        f'{result.stderr}'
    )


def test_every_testpath_module_is_import_safe():
    """Every module pytest collects must import without the optional engines.

    This is the self-maintaining half of the guard: adding a pyo-dependent module
    to `testpaths` fails here, at the exact place the reason is written down,
    rather than as a collection error in CI.
    """
    testpaths = _testpaths_from_pyproject()
    assert testpaths, 'testpaths must be set: it is what bounds doctest collection'

    module_paths = [Path(p) for p in testpaths if p.endswith('.py')]
    assert module_paths, 'expected at least one module listed in testpaths'

    for path in module_paths:
        assert (REPO_ROOT / path).is_file(), f'testpaths entry does not exist: {path}'
        module_name = '.'.join(path.with_suffix('').parts)
        result = _import_in_subprocess(module_name)
        assert result.returncode == 0, (
            f'testpaths lists {path}, but {module_name} cannot be imported without '
            f'{BLOCKED_AT_IMPORT} -- this would break CI collection:\n{result.stderr}'
        )


def test_importing_theremin_does_not_pull_in_the_audio_engine():
    """`import theremin` must not transitively import pyo, even where pyo exists.

    The package's __init__ is documentation only. Making it import the synthesis
    modules would reintroduce issue #7 for every consumer, not just for CI.
    """
    result = _import_in_subprocess('theremin')
    assert result.returncode == 0, result.stderr
