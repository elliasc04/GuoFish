"""Shared v6 model package: imported by the trainer and the engine evaluator."""
from core.guofish_net.model import (  # noqa: F401
    ARCH_VERSION, HLGAUSS_BINS, POLICY_SIZE, TOKEN_SCHEMES, GuoFishNet, ModelConfig,
    SmolgenConfig, ValueHeadConfig, ValueReprConfig, build_model, hlgauss_centers,
    hlgauss_target, load_for_inference,
)
