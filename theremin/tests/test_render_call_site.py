"""Guards *where* a recording may be rendered from -- the fix site of issue #4.

The bug was never in a helper. It was one line in `run_theremin`'s teardown
handing hum's renderer the raw recording, settings and all. Testing the helper
that fixes it does not test that the fix is wired in: with every
`control_events` unit test passing, restoring that one line brings the crash
straight back, and `run_theremin`'s `except` clause turns it into a log line
while the recording is silently lost -- precisely the reported symptom.

`run_theremin` cannot be executed here (it opens a camera and an audio server,
and imports pyo/cv2/mediapipe, none of which exist on a CI runner), so the wiring
is checked structurally instead, by reading the package's own syntax tree:

1. a synth's ``render_events`` is called from exactly one place --
   `control_events.render_recording`, which filters first;
2. `run_theremin` delegates to `render_recording`, so the recording is still
   written at all.

What this cannot prove is that the delegation runs at runtime; that is the price
of a call site which needs a camera. What it does prove is that nobody has
reintroduced the shape of the bug.
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# The one function permitted to call a synth's `render_events`, as
# (module path relative to the package root, enclosing function name).
SANCTIONED_RENDER_CALL_SITE = ('control_events.py', 'render_recording')

# Not part of the shipped runtime, and excluded from ruff/pytest for the same
# reason (see pyproject.toml): the tests themselves legitimately call the
# renderer on fakes.
UNSCANNED_DIRS = {'tests', '__pycache__'}


def _package_modules():
    """Yield (relative_path, syntax_tree) for every scanned module in the package."""
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        relative_path = path.relative_to(PACKAGE_ROOT)
        if UNSCANNED_DIRS & set(relative_path.parts):
            continue
        yield relative_path, ast.parse(path.read_text(encoding='utf-8'))


def _enclosing_function_of(tree):
    """Return a function mapping a node to the name of its innermost enclosing def."""
    parent_of = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }

    def enclosing_function_name(node):
        while node is not None:
            node = parent_of.get(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return None

    return enclosing_function_name


def _method_call_sites(method_name):
    """Every ``<something>.method_name(...)`` call, as {(module, function), ...}."""
    call_sites = set()
    for relative_path, tree in _package_modules():
        enclosing_function_name = _enclosing_function_of(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method_name
            ):
                call_sites.add(
                    (relative_path.as_posix(), enclosing_function_name(node))
                )
    return call_sites


def _function_named(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == name
        ):
            return node
    return None


def _called_names_in(node):
    """Names of every plain ``name(...)`` call under `node` (attribute calls aside)."""
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def test_render_events_is_called_from_exactly_one_sanctioned_place():
    assert _method_call_sites('render_events') == {SANCTIONED_RENDER_CALL_SITE}


def test_run_theremin_delegates_to_render_recording():
    script_utils = ast.parse(
        (PACKAGE_ROOT / 'script_utils.py').read_text(encoding='utf-8')
    )
    run_theremin = _function_named(script_utils, 'run_theremin')
    assert run_theremin is not None, 'run_theremin has been renamed or removed'
    assert 'render_recording' in _called_names_in(run_theremin), (
        'run_theremin no longer calls render_recording: a run would stop writing '
        'its recording, or would be writing it through an unfiltered path'
    )


def test_the_sanctioned_call_site_exists():
    """Guards the guard: a typo'd expectation would make the firewall vacuous."""
    module_path, function_name = SANCTIONED_RENDER_CALL_SITE
    tree = ast.parse((PACKAGE_ROOT / module_path).read_text(encoding='utf-8'))
    assert _function_named(tree, function_name) is not None
