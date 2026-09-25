# v6 harness — decisions and deviations

Deviations from *GuoFish v6 Training Stack — Design* (`docs/capacity/training_stack.md`,
"the doc") and judgment calls it leaves open. Each entry: what, why.

---

## M0 — frozen val

- **Manifest lives in the shard directory** (`val_frozen_90m_v1/manifest.json`) and uses
  Pass B's `record_dtype` spelling, so the v6 reader identifies the format from it the
  same way it does for the corpus manifests.

## M1 — model package (`core/guofish_net/`)

- **`init: v5_deepcopy` matches v5's distribution, not its RNG stream.** It applies
  PyTorch-default init (Embedding N(0,1), Linear Kaiming-uniform, `in_proj` Xavier-uniform
  with zero biases, the `out_proj` bias zeroed, positions randn×0.02) and copies block 0 into every block.
  The same seed does not produce v5's exact initial weights. S2 is judged against a seed-noise band,
  so bit-identical init buys nothing there.
- **Static bias is passed to SDPA as a 4-D tensor, detached when grad is off.** Measured:
  CPU `scaled_dot_product_attention` leaves its fused kernel when the mask is 3-D *or*
  has `requires_grad=True`, even under `no_grad`. Either one alone gave a 5.2e-6
  policy difference at zero bias, which failed S8. With both fixed, S8 is exact (0.0) for static and
  smolgen.
- **Smolgen details the doc leaves open.** The second LayerNorm normalizes the whole
  `heads × 128` vector, before the reshape; the doc lists LN before the reshape. The shared
  128→4096 projection has no bias, so a zero weight gives a zero bias exactly.
- **`aux_policy_head` is a `ModelConfig` field.** The doc puts `policy_hard.head: main | aux`
  under `targets`, but the builder has to know whether to create the head. The config
  loader requires the two to agree (`targets.policy_hard.head == "aux"` ⇔
  `model.aux_policy_head`).
- **hlgauss constants are code constants, not config:** 101 bins over [−1, 1],
  σ = 0.75 × bin width. They are part of the architecture (`arch_version`), not a tunable.
- **`forward` returns fp32 even for a float64 model.** The doc's contract is fp32
  outputs, cast once after the forward. The float64 parity test accounts for this cast.
- **`strict.py` (strict nested frozen-dataclass builder) lives in `core/`** because
  `ModelConfig.from_dict` needs it, and the trainer's config loader reuses it rather than
  keeping a second copy. It parses numeric strings for int/float fields, because
  PyYAML reads `360e6` and `7.2e6` as strings.
- **`load_for_inference` accepts export files only**, not training checkpoints. One format
  reaches the engine.
- **The module keeps a `seq_length` attribute**, which the v5 engine reads off a loaded
  module.

## M3 — data (reader)

- **v1 upcast sentinels.** v1 records carry neither `value_depth` nor `src_line`. The reader
  reports `value_depth = 0` (a real depth is ≥ 20) and `src_line = 0xFFFFFFFF`, alongside the
  doc's `hard_move = −1`, `origin = 0`. v1 `pv_score` (float16 of an int) converts to
  int16 exactly, and the reader checks this on every read.
