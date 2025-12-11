"""
Simple registry describing how to call different generative models for sampling.

Note:
- This module only defines *metadata* and lightweight helper functions.
- The actual heavy model code (e.g. BondFlow.models.mymodel.MySampler) should
  be imported and run inside the *generation* environment / script, not here
  in the generic evaluation environment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelEntry:
    """Metadata for a generative model used in evaluation."""

    name: str
    # Human-readable description
    description: str
    # Optional conda env name used for *generation* (can be None if handled externally)
    gen_env: Optional[str] = None
    # Python module path for the runner implementation (in generation env)
    runner_module: Optional[str] = None
    # Class or function name inside runner_module implementing the sampling logic
    runner_callable: Optional[str] = None
    # Default config path (e.g. YAML) for this model, if any
    default_config: Optional[str] = None
    # Extra default kwargs to pass to the runner
    default_kwargs: Optional[Dict[str, Any]] = None


# Minimal in-repo registry example.
# In the evaluation environment we mainly need "name" to aggregate results.
MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "bondflow_cyclize": ModelEntry(
        name="bondflow_cyclize",
        description="BondFlow MySampler with cyclize configuration.",
        gen_env=None,  # fill in if you want to drive `conda run` from eval side
        runner_module="BondFlow.experiment.model_eval.runners.bondflow_sampler_runner",
        runner_callable="BondFlowSamplerRunner",
        default_config="BondFlow/config/cyclize.yaml",
        default_kwargs={"device": "cuda:0"},
    ),
}


def list_models() -> List[str]:
    """Return available model names."""

    return list(MODEL_REGISTRY.keys())


def get_model_entry(name: str) -> ModelEntry:
    """Get registry entry by name, raising KeyError if missing."""

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name '{name}'. Known models: {list_models()}")
    return MODEL_REGISTRY[name]





