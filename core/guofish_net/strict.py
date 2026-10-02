"""Strict nested frozen-dataclass construction from plain dicts.

Shared by ModelConfig.from_dict (checkpoints, export files) and the v6 config
loader (YAML). Unknown keys and wrong types are errors naming the full key
path. Numeric strings are parsed for int/float fields because PyYAML reads
`360e6` and `7.2e6` as strings (YAML 1.1 floats need a dot and a signed
exponent); an int field accepts only integral values.
"""
from __future__ import annotations

import types
import typing
from dataclasses import fields, is_dataclass


def from_dict_strict(cls, d, path: str):
    if isinstance(d, cls):
        return d
    if not isinstance(d, dict):
        raise TypeError(f"{path}: expected a mapping, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    join = (lambda k: f"{path}.{k}") if path else (lambda k: k)
    unknown = sorted(set(d) - known)
    if unknown:
        raise KeyError(f"unknown key(s): {', '.join(join(k) for k in unknown)}")
    kw = {k: coerce(hints[k], v, join(k)) for k, v in d.items()}
    try:
        return cls(**kw)
    except (ValueError, TypeError, KeyError) as e:
        msg = e.args[0] if e.args else str(e)
        raise type(e)(f"{path}: {msg}" if path else msg) from e


def coerce(tp, v, path: str):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if v is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"{path}: unsupported union {tp}")
        return coerce(inner[0], v, path)
    if origin is typing.Literal:
        if v not in typing.get_args(tp):
            raise ValueError(f"{path}={v!r}; expected one of {typing.get_args(tp)}")
        return v
    if origin in (tuple, list):
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {type(v).__name__}")
        args = typing.get_args(tp)
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(v):
                raise ValueError(f"{path}: expected {len(args)} items, got {len(v)}")
            return tuple(coerce(a, x, f"{path}[{i}]") for i, (a, x) in enumerate(zip(args, v)))
        elem = args[0] if args else typing.Any
        return tuple(coerce(elem, x, f"{path}[{i}]") for i, x in enumerate(v))
    if origin is dict:
        if not isinstance(v, dict):
            raise TypeError(f"{path}: expected a mapping, got {type(v).__name__}")
        _, vt = typing.get_args(tp) or (typing.Any, typing.Any)
        return {k: coerce(vt, x, f"{path}.{k}") for k, x in v.items()}
    if is_dataclass(tp):
        return from_dict_strict(tp, v, path)
    if tp is typing.Any:
        return v
    if tp is bool:
        if not isinstance(v, bool):
            raise TypeError(f"{path}: expected a bool, got {v!r}")
        return v
    if tp is int or tp is float:
        if isinstance(v, bool):
            raise TypeError(f"{path}: expected a number, got {v!r}")
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                raise TypeError(f"{path}: expected a number, got {v!r}") from None
        if not isinstance(v, (int, float)):
            raise TypeError(f"{path}: expected a number, got {type(v).__name__}")
        if tp is float:
            return float(v)
        if float(v) != int(v):
            raise ValueError(f"{path}: expected an integer, got {v!r}")
        return int(v)
    if tp is str:
        if not isinstance(v, str):
            raise TypeError(f"{path}: expected a string, got {v!r}")
        return v
    raise TypeError(f"{path}: unsupported field type {tp}")
