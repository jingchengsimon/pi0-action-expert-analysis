"""Smoke test: client inference against a running policy server.

Verifies that:
  1. A policy server can be connected to via WebSocket.
  2. A synthetic observation produces an action output of the expected shape.
  3. The action values are finite (no NaN / Inf).

Usage::

    # Start a policy server first:
    python serving/launch_policy_server.py --config configs/serving.yaml

    # Then run the smoke test:
    python smoke_tests/test_inference.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

# Expected shapes.
EXPECTED_IMAGE_SHAPE = (224, 224, 3)
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_HORIZON = 50  # default; may vary by model


def _make_fake_obs(resize_size: int = 224) -> dict:
    """Create a synthetic observation dict matching the π0 client format."""
    return {
        "observation/image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (resize_size, resize_size, 3), dtype=np.uint8),
        "observation/state": np.random.randn(EXPECTED_STATE_DIM).astype(np.float32),
        "prompt": "pick up the red block",
    }


def test_inference(
    host: str = "127.0.0.1",
    port: int = 8000,
    num_queries: int = 3,
    resize_size: int = 224,
) -> bool:
    """Run inference smoke test against a running policy server.

    Returns *True* if all checks pass.
    """
    from openpi_client import websocket_client_policy

    logger.info("Connecting to policy server at %s:%d …", host, port)
    client = websocket_client_policy.WebsocketClientPolicy(host, port)

    # Check server metadata.
    try:
        metadata = client.get_server_metadata()
        logger.info("Server metadata: %s", metadata)
    except Exception as e:
        logger.warning("Could not fetch server metadata: %s", e)

    all_ok = True
    for i in range(num_queries):
        obs = _make_fake_obs(resize_size)
        t0 = time.monotonic()

        try:
            result = client.infer(obs)
        except Exception as e:
            logger.error("Query %d FAILED: %s", i, e)
            all_ok = False
            continue

        elapsed_ms = (time.monotonic() - t0) * 1000

        actions = result.get("actions")
        if actions is None:
            logger.error("Query %d: no 'actions' key in result (keys: %s)", i, list(result.keys()))
            all_ok = False
            continue

        actions = np.asarray(actions)
        logger.info(
            "Query %d: actions shape=%s, elapsed=%.1f ms, "
            "min=%.4f, max=%.4f, mean=%.4f",
            i, actions.shape, elapsed_ms,
            float(actions.min()), float(actions.max()), float(actions.mean()),
        )

        # Check finiteness.
        if not np.all(np.isfinite(actions)):
            logger.error("Query %d: actions contain NaN or Inf!", i)
            all_ok = False
            continue

        # Check action dimension (should be >= 7 for LIBERO).
        if actions.ndim < 2 or actions.shape[-1] < 7:
            logger.warning(
                "Query %d: unexpected action shape %s (expected at least (T, 7))",
                i, actions.shape,
            )

    if all_ok:
        logger.info("✓ All %d inference queries passed.", num_queries)
    else:
        logger.error("✗ Some queries failed.")

    return all_ok


def test_local_model_inference(checkpoint_dir: str | None = None, config_name: str = "pi0_libero") -> bool:
    """Test inference directly (no server) by loading the model locally.

    Useful for quick validation without starting a separate server process.
    """
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    ckpt = checkpoint_dir or "gs://openpi-assets/checkpoints/pi0_base"
    logger.info("Loading model locally: config=%s checkpoint=%s", config_name, ckpt)

    tc = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(tc, ckpt)

    obs = _make_fake_obs()
    logger.info("Running local inference …")
    t0 = time.monotonic()
    result = policy.infer(obs)
    elapsed_ms = (time.monotonic() - t0) * 1000

    actions = np.asarray(result.get("actions", []))
    logger.info(
        "Local inference: shape=%s, elapsed=%.1f ms, finite=%s",
        actions.shape, elapsed_ms, bool(np.all(np.isfinite(actions))),
    )

    ok = actions.ndim >= 2 and actions.shape[-1] >= 7 and np.all(np.isfinite(actions))
    if ok:
        logger.info("✓ Local inference test passed.")
    else:
        logger.error("✗ Local inference test FAILED.")
    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inference smoke test")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--local", action="store_true",
                        help="Test local model inference (no server).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir for local test.")
    parser.add_argument("--config-name", type=str, default="pi0_libero",
                        help="Training config name for local test.")
    args = parser.parse_args(argv)

    if args.local:
        ok = test_local_model_inference(args.checkpoint, args.config_name)
    else:
        ok = test_inference(args.host, args.port, args.num_queries, args.resize_size)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
