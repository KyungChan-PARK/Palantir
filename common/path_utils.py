from pathlib import Path
import os


def get_palantir_root() -> Path:
    """Return the project root directory.

    The root can be configured via the ``PALANTIR_ROOT`` environment variable.
    If not set, the function walks up parent directories from this file until a
    directory containing ``config`` and ``README.md`` is found.
    """
    env_root = os.getenv("PALANTIR_ROOT")
    if env_root:
        return Path(env_root)

    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config").is_dir() and (parent / "README.md").exists():
            return parent
    # Fallback
    return current.parents[2]
