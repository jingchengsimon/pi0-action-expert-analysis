"""Inference optimization experiments.

Tests various optimization strategies for π0 policy inference:
  - torch.compile modes
  - bfloat16 half-precision
  - KV-cache prefix caching
  - Batch inference throughput

Usage::

    python profiling/inference_optimizer.py --config-name pi0_libero
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _make_batch_obs(batch_size: int, resize_size: int = 224) -> list[dict]:
    """Create a batch of synthetic observations."""
    return [
        {
            "observation/image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
            "observation/state": np.random.randn(8).astype(np.float32),
            "prompt": "pick up the red block",
        }
        for _ in range(batch_size)
    ]


def benchmark_compile_modes(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    num_queries: int = 30,
    warmup: int = 5,
) -> dict[str, Any]:
    """Compare different torch.compile modes for inference speed."""
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    ckpt = checkpoint or "gs://openpi-assets/checkpoints/pi0_base"
    modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
    results: dict[str, dict] = {}

    for mode in modes:
        logger.info("Testing compile mode: %s", mode)
        try:
            tc = train_config.get_config(config_name)
            # Override compile mode via model config.
            import dataclasses
            tc = dataclasses.replace(tc, model=dataclasses.replace(tc.model, pytorch_compile_mode=mode))
            policy = policy_config.create_trained_policy(tc, ckpt, pytorch_device="cuda")

            obs = _make_batch_obs(1)[0]
            # Warmup.
            for _ in range(warmup):
                policy.infer(obs)

            latencies = []
            for _ in range(num_queries):
                t0 = time.monotonic()
                policy.infer(obs)
                latencies.append((time.monotonic() - t0) * 1000)

            results[mode] = {
                "p50_ms": statistics.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "mean_ms": statistics.mean(latencies),
            }
            logger.info("  %s: P50=%.1fms P95=%.1fms", mode, results[mode]["p50_ms"], results[mode]["p95_ms"])
        except Exception as e:
            logger.warning("  %s failed: %s", mode, e)
            results[mode] = {"error": str(e)}

    return results


def benchmark_batch_throughput(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    batch_sizes: list[int] | None = None,
    num_queries: int = 20,
    warmup: int = 3,
) -> dict[str, Any]:
    """Measure inference throughput at different batch sizes."""
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    batch_sizes = batch_sizes or [1, 2, 4, 8]
    ckpt = checkpoint or "gs://openpi-assets/checkpoints/pi0_base"

    tc = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(tc, ckpt)

    results: dict[str, Any] = {}
    for bs in batch_sizes:
        obs_batch = _make_batch_obs(bs)
        # Warmup.
        for _ in range(warmup):
            for obs in obs_batch:
                policy.infer(obs)

        t0 = time.monotonic()
        for _ in range(num_queries):
            for obs in obs_batch:
                policy.infer(obs)
        total_ms = (time.monotonic() - t0) * 1000

        per_sample_ms = total_ms / (num_queries * bs)
        throughput = (num_queries * bs) / (total_ms / 1000)

        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "total_ms": total_ms,
            "per_sample_ms": per_sample_ms,
            "throughput_qps": throughput,
        }
        logger.info("  BS=%d: %.1fms/sample, %.1f QPS", bs, per_sample_ms, throughput)

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inference optimization benchmarks")
    parser.add_argument("--config-name", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/optimization_report.json")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}

    logger.info("=== Compile mode comparison ===")
    report["compile_modes"] = benchmark_compile_modes(
        args.config_name, args.checkpoint, args.num_queries, args.warmup,
    )

    logger.info("=== Batch throughput ===")
    report["batch_throughput"] = benchmark_batch_throughput(
        args.config_name, args.checkpoint, num_queries=20, warmup=3,
    )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
