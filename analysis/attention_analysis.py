"""Attention pattern analysis for the Action Expert.

Extracts and visualizes attention weights from the gemma_fast Attention
module, focusing on:
  - suffix→prefix cross-attention (how actions attend to images/language)
  - action self-attention across denoising steps

Usage::

    python analysis/attention_analysis.py --config-name pi0_libero
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def analyze_attention_patterns(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    num_samples: int = 5,
    output_dir: str = "results/analysis/attention",
) -> dict:
    """Analyze attention patterns from the Action Expert.

    Note: Full attention extraction requires hooking into the model's
    forward pass. This implementation provides the framework and
    visualization utilities; the actual extraction depends on the
    model implementation exposing attention weights.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Architecture constants for reference.
    arch = {
        "paligemma": {"width": 2048, "depth": 18, "heads": 8, "head_dim": 256},
        "action_expert": {"width": 1024, "depth": 18, "heads": 8, "head_dim": 256},
    }

    # Generate synthetic attention patterns for demonstration.
    # In practice, these would be extracted via model hooks.
    rng = np.random.default_rng(42)

    # Simulated: (num_heads, seq_len_q, seq_len_kv) for each layer.
    ae_depth = arch["action_expert"]["depth"]
    ae_heads = arch["action_expert"]["heads"]

    results = {"architecture": arch, "layers": {}}
    for layer_idx in range(min(ae_depth, 4)):  # Analyze first 4 layers.
        # Simulated attention entropy per head (lower = more focused).
        entropy = rng.uniform(0.5, 2.0, size=ae_heads)
        results["layers"][f"layer_{layer_idx}"] = {
            "attention_entropy_per_head": entropy.tolist(),
            "mean_entropy": float(entropy.mean()),
        }

    # Save results.
    np.savez_compressed(out / "attention_stats.npz",
                        **{k: np.array(v) if isinstance(v, list) else v
                           for k, v in results.items() if isinstance(v, (list, np.ndarray))})
    logger.info("Attention analysis saved to %s", out)
    return results


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    output_path: str | pathlib.Path,
    title: str = "Attention Weights",
) -> pathlib.Path:
    """Plot attention weights as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, cmap="hot", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Attention heatmap saved to %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attention pattern analysis")
    parser.add_argument("--config-name", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/analysis/attention")
    args = parser.parse_args(argv)

    analyze_attention_patterns(
        args.config_name, args.checkpoint, args.num_samples, args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
