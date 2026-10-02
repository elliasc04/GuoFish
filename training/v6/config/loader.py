"""YAML -> resolved Config: `extends:` inheritance, `--set` overrides, hashing (§4).

    cfg = load_config("training/v6/config/configs/base.yaml", ["model.n_layers=10"])

Inheritance: `extends: <file>` (relative to the child) names one parent,
resolved recursively. Mappings merge leaf by leaf; lists and scalars replace.
Overrides are `a.b.c=<yaml value>` and apply last. The resolved config hashes
as sha256 over canonical JSON (sorted keys, shortest-repr floats).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from core.guofish_net.strict import from_dict_strict
from training.v6.config.schema import Config

# Keys a resume may change without the config-hash check refusing it (§10.2).
RESUME_WHITELIST = {"run.out_root", "system.log_every", "data.workers",
                    "data.prefetch_factor", "schedule.total_samples"}


def _read_yaml(path: Path, seen: tuple) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"extends cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise TypeError(f"{path}: extends must be one file path")
    return merge(_read_yaml(path.parent / parent, (*seen, path)), raw)


def merge(base: dict, child: dict) -> dict:
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_override(d: dict, spec: str) -> dict:
    if "=" not in spec:
        raise ValueError(f"override {spec!r} is not key.path=value")
    key, _, text = spec.partition("=")
    parts = key.strip().split(".")
    if not all(parts):
        raise ValueError(f"override {spec!r} has an empty key segment")
    value = yaml.safe_load(text)
    patch = value
    for p in reversed(parts):
        patch = {p: patch}
    return merge(d, patch)


def build_config(d: dict) -> Config:
    return from_dict_strict(Config, d, "")


def load_config(path, overrides=()) -> Config:
    d = _read_yaml(Path(path), ())
    for spec in overrides:
        d = apply_override(d, spec)
    d.pop("prod", None)         # the production driver's own section (tools/prod.py), not trainer config
    return build_config(d)


def to_plain(cfg: Config) -> dict:
    """JSON-safe dict (tuples become lists)."""
    return json.loads(canonical_json(cfg))


def canonical_json(cfg: Config) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def dump_yaml(cfg: Config) -> str:
    return yaml.safe_dump(to_plain(cfg), sort_keys=True)


def diff_paths(a: dict, b: dict, prefix: str = "") -> set:
    """Dotted leaf paths whose values differ between two plain config dicts."""
    out = set()
    for k in set(a) | set(b):
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out |= diff_paths(a[k], b[k], p)
        elif a.get(k) != b.get(k):
            out.add(p)
    return out
