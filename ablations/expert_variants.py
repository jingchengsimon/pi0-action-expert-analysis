"""Action Expert variant experiments.

Tests custom Action Expert configurations by modifying the gemma architecture
parameters (depth, width, num_heads) and measuring downstream task performance.

This module complements A1 (a1_expert_scale.yaml) by providing the
model configuration overrides needed to create smaller/larger Action Experts.

Usage::

    python ablations/expert_variants.py \\
        --variant shallow_9 \\
        --config pi0_libero_low_mem_finetune \\
        --data-dir data/libero_lerobot/libero_spatial \\
        --output-dir results/ablations/expert_variants
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action Expert architecture variants
# ---------------------------------------------------------------------------

# Default: gemma_300m → width=1024, depth=18, num_heads=8
EXPERT_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {
        "description": "gemma_300m (width=1024, depth=18, heads=8)",
        "width": 1024,
        "depth": 18,
        "num_heads": 8,
    },
    "shallow_9": {
        "description": "Half depth (width=1024, depth=9, heads=8)",
        "width": 1024,
        "depth": 9,
        "num_heads": 8,
    },
    "shallow_6": {
        "description": "Third depth (width=1024, depth=6, heads=8)",
        "width": 1024,
        "depth": 6,
        "num_heads": 8,
    },
    "narrow_512": {
        "description": "Half width (width=512, depth=18, heads=8)",
        "width": 512,
        "depth": 18,
        "num_heads": 8,
    },
    "narrow_256": {
        "description": "Quarter width (width=256, depth=18, heads=4)",
        "width": 256,
        "depth": 18,
        "num_heads": 4,
    },
    "tiny": {
        "description": "Tiny expert (width=256, depth=6, heads=4)",
        "width": 256,
        "depth": 6,
        "num_heads": 4,
    },
    "wide_2048": {
        "description": "Double width (width=2048, depth=18, heads=16)",
        "width": 2048,
        "depth": 18,
        "num_heads": 16,
    },
}


def get_variant_config(variant_name: str) -> dict[str, Any]:
    """Get architecture config for a named variant."""
    if variant_name not in EXPERT_VARIANTS:
        available = list(EXPERT_VARIANTS.keys())
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")
    return EXPERT_VARIANTS[variant_name]


def apply_variant_to_env(variant_config: dict[str, Any]) -> None:
    """Set environment variables that openpi training reads for AE overrides.

    The openpi training script checks for ``OPENPI_AE_*`` env vars
    to override the default gemma_300m architecture.
    """
    os.environ["OPENPI_AE_WIDTH"] = str(variant_config["width"])
    os.environ["OPENPI_AE_DEPTH"] = str(variant_config["depth"])
    os.environ["OPENPI_AE_NUM_HEADS"] = str(variant_config["num_heads"])
    logger.info(
        "Applied AE variant: width=%d, depth=%d, heads=%d",
        variant_config["width"],
        variant_config["depth"],
        variant_config["num_heads"],
    )


def estimate_params(variant_config: dict[str, Any]) -> dict[str, float]:
    """Estimate parameter count for a given AE variant.

    Approximate formula for a Gemma-like transformer:
      params ≈ depth * (4 * width^2 + 3 * width * width + 2 * width * vocab)
    For the Action Expert, vocab is replaced by action_dim * action_horizon.
    """
    w = variant_config["width"]
    d = variant_config["depth"]
    # Rough approximation: ~6 * width^2 per layer for self-attention + FFN.
    per_layer = 6 * w * w
    total = d * per_layer
    return {
        "width": w,
        "depth": d,
        "per_layer_params": per_layer,
        "total_params_approx": total,
        "total_params_M": total / 1e6,
    }


def compare_variants() -> dict[str, dict[str, Any]]:
    """Compare all variants and return a summary table."""
    results = {}
    for name, cfg in EXPERT_VARIANTS.items():
        est = estimate_params(cfg)
        results[name] = {
            **cfg,
            **est,
        }
    return results


def run_variant_experiment(
    variant_name: str,
    config_name: str = "pi0_libero_low_mem_finetune",
    data_dir: str = "data/libero_lerobot/libero_spatial",
    output_dir: str = "results/ablations/expert_variants",
    num_steps: int = 30000,
) -> pathlib.Path:
    """Train and evaluate a specific AE variant.

    This sets environment variables and delegates to finetune/train.py.
    """
    variant_config = get_variant_config(variant_name)
    apply_variant_to_env(variant_config)

    out = pathlib.Path(output_dir) / variant_name
    out.mkdir(parents=True, exist_ok=True)

    # Save variant metadata.
    import json
    meta = {
        "variant": variant_name,
        "config": variant_config,
        "estimate": estimate_params(variant_config),
        "train_config": config_name,
        "num_steps": num_steps,
    }
    with open(out / "variant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Delegate to training.
    from finetune.train import train_jax
    logger.info("Training variant: %s", variant_name)
    train_jax(
        config_name=config_name,
        data_dir=data_dir,
        output_dir=str(out / "checkpoints"),
        num_steps=num_steps,
    )

    logger.info("Variant %s training complete. Output: %s", variant_name, out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Action Expert variant experiments")
    parser.add_argument("--variant", type=str, default="default",
                        choices=list(EXPERT_VARIANTS.keys()),
                        help="AE variant name.")
    parser.add_argument("--config", type=str, default="pi0_libero_low_mem_finetune")
    parser.add_argument("--data-dir", type=str, default="data/libero_lerobot/libero_spatial")
    parser.add_argument("--output-dir", type=str, default="results/ablations/expert_variants")
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table and exit.")
    args = parser.parse_args(argv)

    if args.compare:
        import json
        results = compare_variants()
        print(json.dumps(results, indent=2))
        return

    run_variant_experiment(
        variant_name=args.variant,
        config_name=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
