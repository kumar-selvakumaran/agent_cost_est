"""Shared helpers: config, seeding, IO."""
from __future__ import annotations

import hashlib
import os

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_for(cfg: dict, name: str) -> int:
    """SEED + hash(component_name) % 10_000 (stable hash, spec §2.3)."""
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return int(cfg["seed"]) + h % 10_000


def ensure_dirs():
    for d in ["results/tables", "results/figures", "results/logs", "results/predictions", "data_cache"]:
        os.makedirs(d, exist_ok=True)


def config_hash(path: str = "config.yaml") -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]
