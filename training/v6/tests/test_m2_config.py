"""M2: inheritance, overrides, validation errors, hash stability, v5 compat."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from training.v6.config import build_config, config_hash, load_config, to_plain
from training.v6.data.formats import REPO

CFG = REPO / "training/v6/config/configs"


def test_base_loads_with_doc_values():
    c = load_config(CFG / "base.yaml")
    assert (c.model.d_model, c.model.n_layers, c.model.n_heads, c.model.d_ff) == (384, 6, 6, 1536)
    assert c.schedule.total_samples == 360_000_000          # "360e6" parsed
    assert c.schedule.warmup_samples == 7_200_000           # "7.2e6" parsed
    assert c.ema.half_life_samples == 10e6
    assert c.optim.betas == (0.9, 0.999)
    assert c.mixture.groups == "natural"


def test_inheritance_leaf_by_leaf():
    c = load_config(CFG / "shapes/d448x6_smolgen.yaml")
    assert (c.model.d_model, c.model.n_heads, c.model.d_ff, c.model.attn_bias) == (448, 7, 1792, "smolgen")
    assert c.model.n_layers == 6 and c.model.smolgen.gen == 128   # from the parent
    assert c.run.name == "v6-d448x6-sg" and c.run.seed == 20260924


def test_lists_replace_and_extends_chain(tmp_path):
    (tmp_path / "a.yaml").write_text("optim: {betas: [0.9, 0.95], lr: 1.0e-3}\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\noptim: {betas: [0.8, 0.99]}\n")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\nrun: {name: chain}\n")
    c = load_config(tmp_path / "c.yaml")
    assert c.optim.betas == (0.8, 0.99) and c.optim.lr == 1e-3 and c.run.name == "chain"
    (tmp_path / "x.yaml").write_text("extends: y.yaml\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\n")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config(tmp_path / "x.yaml")


def test_overrides_apply_last():
    c = load_config(CFG / "shapes/d448x6_smolgen.yaml",
                    ["model.n_layers=10", "optim.lr=1e-3", "model.smolgen.gen=64",
                     "optim.betas=[0.9, 0.95]", "targets.policy_soft.temperature=null"])
    assert c.model.n_layers == 10 and c.optim.lr == 1e-3 and c.model.smolgen.gen == 64
    assert c.optim.betas == (0.9, 0.95) and c.model.d_model == 448


@pytest.mark.parametrize("override, exc, pattern", [
    ("optim.lrr=1", KeyError, r"optim\.lrr"),
    ("model.smolgen.bogus=1", KeyError, r"model\.smolgen\.bogus"),
    ("nosuchsection.x=1", KeyError, r"nosuchsection"),
    ("model.attn_bias=alibi", ValueError, r"model: attn_bias='alibi'"),
    ("model.n_layers=2.5", ValueError, r"model\.n_layers: expected an integer"),
    ("model.d_model=abc", TypeError, r"model\.d_model: expected a number"),
    ("system.compile=1", TypeError, r"system\.compile: expected a bool"),
    ("optim.betas=[0.9]", ValueError, r"optim\.betas: expected 2 items"),
    ("targets.policy_soft.source=pv_score", ValueError, r"temperature is required"),
    ("model.token_scheme=canonical_65", ValueError, r"mirror_prob must be 0"),
    ("targets.policy_hard.head=aux", ValueError, r"aux_policy_head"),
    ("ckpt.every_samples=1000", ValueError, r"multiple of the effective batch"),
    ("mixture.groups=[{name: p, where: {label: multipv}, share: 1.5}]", ValueError, r"mixture\.groups\[0\]: .*share"),
    ("mixture.groups=[{name: p, where: {lable: multipv}, share: 0.5}]", KeyError, r"mixture\.groups\[0\]: .*where\.lable"),
    ("mixture.groups=[{name: p, where: {label: multi}, share: 0.5}]", ValueError, r"where\.label='multi'"),
])
def test_validation_errors_name_the_path(override, exc, pattern):
    with pytest.raises(exc, match=pattern):
        load_config(CFG / "base.yaml", [override])


def test_hash_stable_and_sensitive(tmp_path):
    a = load_config(CFG / "base.yaml")
    b = load_config(CFG / "base.yaml")
    assert config_hash(a) == config_hash(b)
    # same values spelled differently: key order, float spelling, override to same value
    (tmp_path / "r.yaml").write_text(
        "extends: " + str((CFG / "base.yaml").as_posix()) + "\n"
        "optim: {accum: 2, lr: 0.00035}\nschedule: {total_samples: 360000000}\n")
    assert config_hash(load_config(tmp_path / "r.yaml")) == config_hash(a)
    assert config_hash(load_config(CFG / "base.yaml", ["model.n_layers=6"])) == config_hash(a)
    # any real change moves the hash
    assert config_hash(load_config(CFG / "base.yaml", ["optim.lr=3.6e-4"])) != config_hash(a)
    # a fresh interpreter (different PYTHONHASHSEED) gets the same digest
    out = subprocess.run(
        [sys.executable, "-c", "from training.v6.config import load_config, config_hash;"
         f"print(config_hash(load_config(r'{CFG / 'base.yaml'}')))"],
        capture_output=True, text=True, cwd=REPO, check=True,
        env={"PYTHONHASHSEED": "12345", "CUDA_VISIBLE_DEVICES": "",
             "SYSTEMROOT": __import__("os").environ["SYSTEMROOT"],
             "PATH": __import__("os").environ["PATH"]})
    assert out.stdout.strip() == config_hash(a)
    # the resolved dict rebuilds to the same config
    assert build_config(to_plain(a)) == a


def test_v5_compat_matches_v5_config():
    """Every hyperparameter in v5's corpus90m.yaml that has a v6 counterpart."""
    v5 = yaml.safe_load((REPO / "training/v5_multiPV/configs/corpus90m.yaml").read_text())
    man = json.loads((REPO / v5["manifest"]).read_text())
    c = load_config(CFG / "v5_compat.yaml")
    m, o, s = c.model, c.optim, c.schedule

    assert (m.d_model, m.n_layers, m.n_heads, m.d_ff, m.head_dim) == (
        v5["d_model"], v5["num_layers"], v5["nhead"], v5["dim_feedforward"], v5["head_dim"])
    assert (m.vocab_size, m.seq_len) == (v5["vocab_size"], v5["seq_len"])
    assert m.dropout == v5["dropout"] and m.final_norm == v5["final_norm"]
    assert v5["activation"] == "gelu" and v5["smolgen"] is False
    assert m.init == "v5_deepcopy" and m.attn_bias == "none"
    assert (m.value_head.pool, m.value_repr.kind, m.token_scheme) == ("cls_mlp", "scalar", "v5_68")

    assert (o.micro_batch, o.accum) == (v5["micro_batch"], v5["accum_steps"])
    assert o.lr == v5["max_lr"] and o.weight_decay == v5["weight_decay"]
    assert o.betas == (v5["beta1"], v5["beta2"]) and o.eps == v5["adam_eps"]
    assert o.grad_clip == v5["grad_clip"] and o.decay_embedding is True
    assert o.name == "adamw"

    assert s.kind == "onecycle"
    oc = s.onecycle
    assert (oc.pct_start, oc.div_factor, oc.final_div_factor) == (
        v5["pct_start"], v5["div_factor"], v5["final_div_factor"])
    # torch OneCycleLR defaults, which v5 inherited (recon H4)
    assert (oc.cycle_momentum, oc.base_momentum, oc.max_momentum) == (True, 0.85, 0.95)
    epoch = (man["records_train"] // v5["micro_batch"]) * v5["micro_batch"]   # drop_last
    assert s.total_samples == v5["epochs"] * epoch == 360_302_592

    assert c.targets.mirror_prob == v5["mirror_prob"]
    assert v5["temperature"] is None and c.targets.policy_soft.source == "stored"
    assert c.targets.policy_soft.epsilon == man["epsilon"]
    assert (c.targets.policy_soft.weight, c.targets.value.weight) == (
        v5["policy_weight"], v5["value_weight"])
    assert c.targets.policy_hard.weight == 0.0 and c.mixture.groups == "natural"
    assert c.ema.enabled is False
    assert c.run.seed == v5["seed"]
    assert (c.data.corpus, c.data.manifest) == (v5["shards"], v5["manifest"])

    eff = v5["micro_batch"] * v5["accum_steps"]
    assert c.eval.quick_every_samples == v5["val_every"] * eff
    assert c.ckpt.every_samples == v5["ckpt_every"] * eff and c.ckpt.keep_last == v5["keep_step_ckpts"]
    assert c.system.log_every == v5["log_every"]
    assert (c.system.precision, c.system.compile, c.system.tf32) == (v5["amp"], v5["compile"], v5["tf32"])
