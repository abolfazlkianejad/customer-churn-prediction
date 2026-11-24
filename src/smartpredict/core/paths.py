"""Stable path definitions for the project tree."""

from __future__ import annotations

from pathlib import Path

# src/smartpredict/core/paths.py -> project root is three levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_DIR: Path = PROJECT_ROOT / "src"
PACKAGE_DIR: Path = SRC_DIR / "smartpredict"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
EXPLAINABILITY_DIR: Path = ARTIFACTS_DIR / "explainability"
OPTIMIZATION_DIR: Path = ARTIFACTS_DIR / "optimization"


def get_project_root() -> Path:
    """Return the absolute project root directory.

    Returns:
        Absolute path to the directory containing ``pyproject.toml``.

    """
    return PROJECT_ROOT


def resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve a filesystem path relative to the project root.

    Args:
        path: Absolute path or path relative to ``base_dir``.
        base_dir: Base directory for relative paths. Defaults to the project root.

    Returns:
        The resolved, absolute path.

    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir or PROJECT_ROOT) / candidate


def ensure_directory(path: str | Path, *, base_dir: Path | None = None) -> Path:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Absolute path or path relative to ``base_dir``.
        base_dir: Base directory for relative paths. Defaults to the project root.

    Returns:
        The absolute path of the existing directory.

    """
    directory = resolve_path(path, base_dir=base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
