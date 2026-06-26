"""Unified analysis entry point.

Runs all analysis modules selectively and aggregates results.

Usage::

    python analysis/run_analysis.py --config-name pi0_libero --output-dir results/analysis
"""

from __future__ import annotations

import argparse
import logging
import pathlib

logger = logging.getLogger(__name__)

ANALYSIS_MODULES = [
    "extract_activations",
    "representation_similarity",
    "attention_analysis",
    "flow_matching_analysis",
    "lora_comparison",
]


def run_all_analyses(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    output_dir: str = "results/analysis",
    modules: list[str] | None = None,
    num_samples: int = 10,
) -> None:
    """Run selected analysis modules."""
    modules = modules or ANALYSIS_MODULES
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "extract_activations" in modules:
        logger.info("=== 1/5 Extracting activations ===")
        from analysis.extract_activations import main as extract_main
        extract_main([
            "--config-name", config_name,
            "--output", str(out / "activations"),
            "--num-samples", str(num_samples),
        ] + (["--checkpoint", checkpoint] if checkpoint else []))

    if "representation_similarity" in modules:
        logger.info("=== 2/5 Computing CKA similarity ===")
        from analysis.representation_similarity import main as cka_main
        cka_main([
            "--activations-dir", str(out / "activations"),
            "--output", str(out / "cka_matrix.png"),
        ])

    if "attention_analysis" in modules:
        logger.info("=== 3/5 Analyzing attention patterns ===")
        from analysis.attention_analysis import main as attn_main
        attn_main([
            "--config-name", config_name,
            "--output-dir", str(out / "attention"),
            "--num-samples", str(min(num_samples, 5)),
        ] + (["--checkpoint", checkpoint] if checkpoint else []))

    if "flow_matching_analysis" in modules:
        logger.info("=== 4/5 Analyzing flow matching ===")
        from analysis.flow_matching_analysis import main as fm_main
        fm_main([
            "--config-name", config_name,
            "--output-dir", str(out / "flow_matching"),
            "--num-samples", str(min(num_samples, 10)),
        ] + (["--checkpoint", checkpoint] if checkpoint else []))

    if "lora_comparison" in modules:
        logger.info("=== 5/5 LoRA comparison ===")
        from analysis.lora_comparison import main as lora_main
        lora_main([
            "--output-dir", str(out / "lora"),
        ] + (["--lora-checkpoint", checkpoint] if checkpoint else []))

    logger.info("All analyses complete. Results in %s", out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified analysis runner")
    parser.add_argument("--config-name", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    parser.add_argument("--modules", nargs="+", default=None,
                        choices=ANALYSIS_MODULES, help="Specific modules to run.")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args(argv)

    run_all_analyses(
        config_name=args.config_name,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        modules=args.modules,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
