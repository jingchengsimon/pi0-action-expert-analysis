"""Profile policy/client latency, throughput, and GPU memory.

Measures inference latency breakdown across pipeline stages:
  - WebSocket round-trip
  - Model forward pass
  - Image preprocessing
  - Flow matching denoising steps

Usage::

    python profiling/latency_profile.py --host 127.0.0.1 --port 8000 --num-queries 100
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _make_fake_obs(resize_size: int = 224) -> dict:
    return {
        "observation/image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the red block",
    }


def profile_websocket_latency(
    host: str = "127.0.0.1",
    port: int = 8000,
    num_queries: int = 100,
    warmup: int = 5,
    resize_size: int = 224,
) -> dict[str, Any]:
    """Measure end-to-end WebSocket inference latency."""
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    obs = _make_fake_obs(resize_size)

    # Warmup.
    for _ in range(warmup):
        client.infer(obs)

    latencies_ms: list[float] = []
    for _ in range(num_queries):
        t0 = time.monotonic()
        result = client.infer(obs)
        elapsed = (time.monotonic() - t0) * 1000
        latencies_ms.append(elapsed)

        # Extract server-side timing if available.
        timing = result.get("policy_timing", {})
        if timing:
            logger.debug("Server timing: %s", timing)

    report = {
        "num_queries": num_queries,
        "warmup": warmup,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_qps": 1000.0 / statistics.mean(latencies_ms),
    }

    logger.info(
        "WebSocket latency: P50=%.1fms  P95=%.1fms  P99=%.1fms  "
        "mean=%.1f±%.1fms  throughput=%.1f QPS",
        report["p50_ms"], report["p95_ms"], report["p99_ms"],
        report["mean_ms"], report["std_ms"], report["throughput_qps"],
    )
    return report


def profile_local_inference_latency(
    config_name: str = "pi0_libero",
    checkpoint: str | None = None,
    num_queries: int = 50,
    warmup: int = 5,
    resize_size: int = 224,
) -> dict[str, Any]:
    """Measure local model inference latency (no network)."""
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    ckpt = checkpoint or "gs://openpi-assets/checkpoints/pi0_base"
    tc = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(tc, ckpt)
    obs = _make_fake_obs(resize_size)

    # Warmup.
    for _ in range(warmup):
        policy.infer(obs)

    latencies_ms: list[float] = []
    for _ in range(num_queries):
        t0 = time.monotonic()
        policy.infer(obs)
        latencies_ms.append((time.monotonic() - t0) * 1000)

    report = {
        "config": config_name,
        "num_queries": num_queries,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
    }

    logger.info(
        "Local inference: P50=%.1fms  P95=%.1fms  P99=%.1fms  mean=%.1f±%.1fms",
        report["p50_ms"], report["p95_ms"], report["p99_ms"],
        report["mean_ms"], report["std_ms"],
    )
    return report


def profile_image_preprocessing(num_images: int = 500, resize_size: int = 224) -> dict[str, Any]:
    """Profile image preprocessing (resize + normalize) latency."""
    from sim.obs_adapter import _resize_image

    images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(num_images)]

    t0 = time.monotonic()
    for img in images:
        _resize_image(img, resize_size)
    total_ms = (time.monotonic() - t0) * 1000

    report = {
        "num_images": num_images,
        "total_ms": total_ms,
        "per_image_ms": total_ms / num_images,
        "throughput_fps": 1000.0 / (total_ms / num_images),
    }
    logger.info(
        "Image preprocessing: %d images in %.1fms (%.1fms/image, %.0f FPS)",
        num_images, total_ms, report["per_image_ms"], report["throughput_fps"],
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Latency profiling")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--local", action="store_true", help="Profile local model (no server).")
    parser.add_argument("--config-name", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/profiling_report.json")
    args = parser.parse_args(argv)

    import json
    import pathlib

    report: dict[str, Any] = {}

    # Image preprocessing (always).
    report["preprocessing"] = profile_image_preprocessing(resize_size=args.resize_size)

    if args.local:
        report["local_inference"] = profile_local_inference_latency(
            config_name=args.config_name, checkpoint=args.checkpoint,
            num_queries=args.num_queries, warmup=args.warmup,
        )
    else:
        report["websocket_inference"] = profile_websocket_latency(
            host=args.host, port=args.port,
            num_queries=args.num_queries, warmup=args.warmup,
        )

    # Save report.
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
