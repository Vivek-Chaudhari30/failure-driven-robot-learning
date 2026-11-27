"""Shared command-line handling: ``--config a.yaml --config b.yaml key=value ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from fdrl.config import Config, load_config

DEFAULT_CONFIGS = [
    "configs/env/lift_cube.yaml",
    "configs/expert/scripted.yaml",
    "configs/policy/bc_mlp.yaml",
    "configs/strategy/default.yaml",
]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", action="append", default=[], help="YAML config (repeatable, later wins)")
    p.add_argument("--no-defaults", action="store_true", help="do not load the default config stack")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. sim.backend=toy train.steps=2000")
    return p


def config_from_args(args: argparse.Namespace, repo_root: Path | None = None) -> Config:
    root = repo_root or Path(__file__).resolve().parents[2]
    paths = [] if args.no_defaults else [root / p for p in DEFAULT_CONFIGS if (root / p).exists()]
    paths += [Path(p) for p in args.config]
    return load_config(*paths, overrides=args.overrides)
