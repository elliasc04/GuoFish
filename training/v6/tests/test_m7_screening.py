"""Screening-pass harness (brief of 2026-09-26): H1 keyed init, H2 loader-worker
BLAS threads, H3 v2 eval sets, H5 forward-cost conversion, H4 queue decisions.
CPU only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from core.guofish_net import ModelConfig, build_model
from training.v6.config import load_config
from training.v6.data.formats import REPO

CFG = REPO / "training/v6/config/configs"


# ---------------------------------------------------------------- H1

def _d384x10(**kw) -> ModelConfig:
    return ModelConfig.from_dict({**load_config(CFG / "shapes/d384x10.yaml").model.to_dict(), **kw})


def test_keyed_init_shares_every_common_parameter():
    """Plain, static-bias, smolgen (and flatten-head) d384x10 models built with
    one init_seed hold identical weights under every shared name and shape."""
    plain = build_model(_d384x10(), init_seed=20260802).state_dict()
    assert sum(v.numel() for v in plain.values()) > 17_000_000          # it is the d384x10
    for variant in ({"attn_bias": "static"}, {"attn_bias": "smolgen"},
                    {"value_head": {"pool": "flatten"}}):
        other = build_model(_d384x10(**variant), init_seed=20260802).state_dict()
        shared = [k for k in plain if k in other and plain[k].shape == other[k].shape]
        assert len(shared) >= len(plain) - 3, variant        # flatten reshapes value_fc1/fc2
        for k in shared:
            assert torch.equal(plain[k], other[k]), (variant, k)
        assert set(other) - set(plain), variant              # the variant did add something

    again = build_model(_d384x10(), init_seed=20260802).state_dict()
    assert all(torch.equal(plain[k], again[k]) for k in plain)
    reseeded = build_model(_d384x10(), init_seed=20260925).state_dict()
    assert not torch.equal(plain["blocks.3.qkv.weight"], reseeded["blocks.3.qkv.weight"])
    assert not torch.equal(plain["blocks.3.qkv.weight"], plain["blocks.4.qkv.weight"])


def test_keyed_init_leaves_the_global_rng_alone():
    """The dropout stream starts where torch.manual_seed put it, whatever the
    architecture; v5_deepcopy still draws its init from the global RNG."""
    torch.manual_seed(5)
    want = torch.rand(4)
    for kw in ({}, {"attn_bias": "smolgen"}):
        torch.manual_seed(5)
        build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, **kw), init_seed=1)
        assert torch.equal(torch.rand(4), want)
    torch.manual_seed(5)
    build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, init="v5_deepcopy"))
    assert not torch.equal(torch.rand(4), want)


def test_seed_split_reaches_sampler_and_init():
    c = load_config(CFG / "base.yaml", ["run.data_seed=1", "run.init_seed=2"])
    assert (c.run.data_seed, c.run.init_seed) == (1, 2)
    with pytest.raises(Exception, match="seed"):
        load_config(CFG / "base.yaml", ["run.seed=1"])
