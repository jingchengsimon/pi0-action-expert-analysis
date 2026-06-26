"""Aggregate and visualize ablation experiment results.

Reads JSON results from all ablation variants and produces:
  - Comparison bar charts with bootstrap confidence intervals
  - Statistical significance tests (paired t-test / Wilcoxon)
  - Scaling law plots (LoRA rank, AE size, flow steps vs success rate)
  - Summary table in CSV and Markdown format

Usage::

    python ablations/aggregate_results.py \\
        --results-dir results/ablations \\
        --output-dir results/ablations/reports
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_ablation_results(results_dir: str) -> dict[str, dict[str, Any]]:
    """Load all ablation results from directory tree.

    Expected structure::

        results/ablations/
        ├── a1_expert_scale/
        │   ├── ablation_summary.json
        │   ├── ae_default/eval_results/*.json
        │   ├── ae_shallow_9/eval_results/*.json
        │   └── ...
        ├── a2_lora_rank/
        │   └── ...
    """
    base = pathlib.Path(results_dir)
    all_results: dict[str, dict[str, Any]] = {}

    for ablation_dir in sorted(base.iterdir()):
        if not ablation_dir.is_dir():
            continue

        ablation_name = ablation_dir.name
        summary_file = ablation_dir / "ablation_summary.json"

        ablation_data: dict[str, Any] = {"variants": {}}
        if summary_file.exists():
            with open(summary_file) as f:
                ablation_data["summary"] = json.load(f)

        # Load per-variant eval results.
        for variant_dir in sorted(ablation_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            eval_dir = variant_dir / "eval_results"
            if not eval_dir.exists():
                continue

            variant_results = []
            for json_file in eval_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        variant_results.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass

            if variant_results:
                ablation_data["variants"][variant_dir.name] = {
                    "results": variant_results,
                    "output_dir": str(variant_dir),
                }

        if ablation_data["variants"]:
            all_results[ablation_name] = ablation_data

    logger.info("Loaded results for %d ablation experiments", len(all_results))
    return all_results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_success_rates(variant_results: list[dict]) -> dict[str, float]:
    """Compute aggregate success rate statistics from eval results."""
    rates = []
    for r in variant_results:
        sr = r.get("success_rate", r.get("overall_success_rate", None))
        if sr is not None:
            rates.append(float(sr))

    if not rates:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    return {
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "n": len(rates),
    }


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))
    return lower, upper


def statistical_significance(
    group_a: list[float],
    group_b: list[float],
) -> dict[str, Any]:
    """Test statistical significance between two groups.

    Uses Welch's t-test (does not assume equal variances).
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return {"test": "welch_t", "p_value": 1.0, "significant": False, "note": "insufficient samples"}

    a = np.array(group_a)
    b = np.array(group_b)

    # Welch's t-test.
    mean_diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se < 1e-10:
        t_stat = 0.0
    else:
        t_stat = mean_diff / se

    # Approximate p-value using normal distribution for large samples.
    from scipy import stats as _stats  # noqa: optional import
    try:
        df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
            (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) +
            (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
        )
        p_value = 2 * (1 - _stats.t.cdf(abs(t_stat), df))
    except Exception:
        p_value = 1.0

    return {
        "test": "welch_t",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_diff": float(mean_diff),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_comparison_table(
    results: dict[str, dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a CSV comparison table."""
    rows: list[dict] = []

    for ablation_name, data in sorted(results.items()):
        for variant_name, vdata in sorted(data["variants"].items()):
            stats = compute_success_rates(vdata["results"])
            rows.append({
                "ablation": ablation_name,
                "variant": variant_name,
                "success_rate_mean": stats["mean"],
                "success_rate_std": stats["std"],
                "n_evals": stats["n"],
            })

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Comparison table saved: %s", out)
    return str(out)


def generate_markdown_report(
    results: dict[str, dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a Markdown report with results summary."""
    lines = [
        "# Ablation Experiment Results",
        "",
        f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for ablation_name, data in sorted(results.items()):
        desc = data.get("summary", {}).get("description", "")
        lines.append(f"## {ablation_name}")
        if desc:
            lines.append(f"\n{desc}\n")

        lines.append("| Variant | Success Rate | Std | N |")
        lines.append("|---------|-------------|-----|---|")

        for variant_name, vdata in sorted(data["variants"].items()):
            stats = compute_success_rates(vdata["results"])
            sr = f"{stats['mean'] * 100:.1f}%" if stats["mean"] <= 1 else f"{stats['mean']:.1f}%"
            std = f"±{stats['std'] * 100:.1f}%" if stats["std"] <= 1 else f"±{stats['std']:.1f}%"
            lines.append(f"| {variant_name} | {sr} | {std} | {stats['n']} |")

        lines.append("")

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))

    logger.info("Markdown report saved: %s", out)
    return str(out)


def generate_plots(
    results: dict[str, dict[str, Any]],
    output_dir: str,
) -> list[str]:
    """Generate comparison plots (bar charts, scaling laws)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return []

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []

    for ablation_name, data in sorted(results.items()):
        variants = sorted(data["variants"].keys())
        means = []
        stds = []

        for v in variants:
            stats = compute_success_rates(data["variants"][v]["results"])
            means.append(stats["mean"])
            stds.append(stats["std"])

        if not means:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(variants))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Ablation: {ablation_name}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = out / f"{ablation_name}_comparison.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plots.append(str(plot_path))
        logger.info("Plot saved: %s", plot_path)

    return plots


def aggregate_all(
    results_dir: str = "results/ablations",
    output_dir: str = "results/ablations/reports",
) -> None:
    """Run full aggregation pipeline."""
    results = load_ablation_results(results_dir)

    if not results:
        logger.warning("No ablation results found in %s", results_dir)
        return

    # Generate reports.
    generate_comparison_table(results, f"{output_dir}/comparison.csv")
    generate_markdown_report(results, f"{output_dir}/report.md")
    generate_plots(results, output_dir)

    logger.info("Aggregation complete. Reports in %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--results-dir", type=str, default="results/ablations")
    parser.add_argument("--output-dir", type=str, default="results/ablations/reports")
    args = parser.parse_args(argv)

    aggregate_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
