"""Flow matching trajectory analysis.

Visualizes the Euler integration trajectory from noise (t=1) to action (t=0),
analyzes the effect of different num_steps, and measures trajectory variance.

Usage::

    python analysis/flow_matching_analysis.py --config-name pi0_libero
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def analyze_step_count_effect(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    step_counts: list[int] | None = None,
    num_samples: int = 10,
    output_dir: str = "results/analysis/flow_matching",
) -> dict:
    """Analyze the effect of different num_steps on action quality.

    Compares actions generated with 5, 10, 20, and 50 Euler integration steps.
    """
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    step_counts = step_counts or [5, 10, 20, 50]
    ckpt = checkpoint or "gs://openpi-assets/checkpoints/pi0_base"
    tc = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(tc, ckpt)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the red block",
    }

    results: dict[str, dict] = {}
    for ns in step_counts:
        actions_list = []
        for _ in range(num_samples):
            result = policy.infer(obs, sample_kwargs={"num_steps": ns})
            actions_list.append(np.asarray(result.get("actions", [])))

        actions = np.stack(actions_list)
        variance = float(actions.var())
        norm = float(np.linalg.norm(actions.mean(axis=0)))

        results[f"steps_{ns}"] = {
            "num_steps": ns,
            "action_shape": list(actions.shape),
            "action_variance": variance,
            "action_norm": norm,
        }
        logger.info("  steps=%d: var=%.4f norm=%.4f shape=%s", ns, variance, norm, actions.shape)

    # Save.
    np.savez_compressed(out / "step_count_analysis.npz",
                        **{k: np.array([v]) for k, v in results.items()})
    logger.info("Step count analysis saved to %s", out)
    return results


def plot_trajectory_comparison(
    results: dict[str, dict],
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Plot action variance vs number of steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = []
    variances = []
    for key, val in results.items():
        steps.append(val["num_steps"])
        variances.append(val["action_variance"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, variances, "o-", linewidth=2)
    ax.set_xlabel("Number of Euler Steps")
    ax.set_ylabel("Action Variance")
    ax.set_title("Flow Matching: Steps vs Action Variance")
    ax.grid(True)
    plt.tight_layout()

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory comparison saved to %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Flow matching analysis")
    parser.add_argument("--config-name", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--step-counts", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/analysis/flow_matching")
    args = parser.parse_args(argv)

    results = analyze_step_count_effect(
        args.config_name, args.checkpoint, args.step_counts,
        args.num_samples, args.output_dir,
    )
    plot_trajectory_comparison(results, pathlib.Path(args.output_dir) / "steps_vs_variance.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
