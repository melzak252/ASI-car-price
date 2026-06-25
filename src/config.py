from pathlib import Path

import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path
