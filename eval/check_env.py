from __future__ import annotations
import os
import pathlib

def _is_path_under(p: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        p.relative_to(parent)
        return True
    except Exception:
        return False

def _has_git_dir(p: pathlib.Path) -> bool:
    for a in (p,) + tuple(p.parents):
        if (a / ".git").exists():
            return True
    return False

def get_env_name() -> str:
    """Detect whether the current environment is 'dit' or 'vlm'.

    Returns:
        'dit' if flashinfer is loaded from a local source tree or submodule
        (path contains DiTEvalKit/kernels/3rdparty/flashinfer, or a .git
        directory exists upstream, or FLASHINF_DIR points to it).
        'vlm' otherwise (i.e. flashinfer installed in site-packages).

    Raises:
        ImportError: if flashinfer cannot be imported at all.
    """
    import flashinfer

    mod_file = getattr(flashinfer, "__file__", None)
    if not mod_file:
        return "vlm"

    p = pathlib.Path(mod_file).resolve()

    if "site-packages" in p.parts or "dist-packages" in p.parts:
        return "vlm"

    flashinf_dir = os.environ.get("FLASHINF_DIR")
    if flashinf_dir:
        try:
            base = pathlib.Path(flashinf_dir).resolve()
            if _is_path_under(p, base):
                return "dit"
        except Exception:
            pass

    if {"eval", "DiTEvalKit", "kernels", "3rdparty", "flashinfer"}.issubset(set(p.parts)):
        return "dit"

    if _has_git_dir(p):
        return "dit"

    return "vlm"

if __name__ == "__main__":
    print(get_env_name())
