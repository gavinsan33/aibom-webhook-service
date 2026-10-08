"""runtime_detector._redact_git_url and postprocess.redact_git_url must stay identical.

They can't share a module: runtime_detector.py is mounted alone as
sitecustomize.py into arbitrary user images, and postprocess.py's image only
COPYs postprocess.py + k8s_api.py. So this fails if either copy drifts.
"""
import ast
import inspect

import postprocess as pp
import runtime_detector as rd


def _normalized(fn, name):
    # Compare logic only: ignore the name, docstring and import statements
    # (runtime_detector imports lazily inside the function, postprocess at
    # module level).
    node = ast.parse(inspect.getsource(fn)).body[0]
    body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if ast.get_docstring(node):
        body = body[1:] if body and isinstance(body[0], ast.Expr) else body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_redact_git_url_implementations_have_identical_logic():
    assert _normalized(rd._redact_git_url, "rd") == _normalized(pp.redact_git_url, "pp"), (
        "redact_git_url (postprocess.py) and _redact_git_url (runtime_detector.py) "
        "have diverged; update both."
    )
