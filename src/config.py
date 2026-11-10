"""Load config.json and resolve project-relative paths."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.json"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve(path: str | Path) -> Path:
    """Paths in config.json are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def raw_files(cfg: dict) -> dict[str, Path]:
    """Map each data file key (nodes, edges, metagraph) to its local path."""
    raw_dir = resolve(cfg["data"]["raw_dir"])
    return {key: raw_dir / Path(rel).name for key, rel in cfg["data"]["files"].items()}
