"""Launch the pi0 policy server (wraps openpi serving).

Reads ``configs/serving.yaml`` to determine which checkpoint / model variant
to load, then starts a :class:`WebsocketPolicyServer` that blocks until
killed (Ctrl-C).

Usage::

    python serving/launch_policy_server.py --config configs/serving.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import socket
import sys
from typing import Any

import yaml

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# ---------------------------------------------------------------------------
# Default checkpoints per model variant
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINTS: dict[str, dict[str, str]] = {
    "pi0": {
        "config": "pi0_libero",
        "dir": "gs://openpi-assets/checkpoints/pi0_base",
    },
    "pi0-fast": {
        "config": "pi0_fast_libero",
        "dir": "gs://openpi-assets/checkpoints/pi0_fast_base",
    },
    "pi0.5": {
        "config": "pi05_libero",
        "dir": "gs://openpi-assets/checkpoints/pi05_base",
    },
    "pi05_libero": {
        "config": "pi05_libero",
        "dir": "gs://openpi-assets/checkpoints/pi05_libero",
    },
    "pi0_libero_low_mem_finetune": {
        "config": "pi0_libero_low_mem_finetune",
        "dir": "gs://openpi-assets/checkpoints/pi0_base",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_policy(
    *,
    model: str = "pi0",
    checkpoint: str | None = None,
    config_name: str | None = None,
    default_prompt: str | None = None,
    norm_stats: dict | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from config parameters.

    Parameters
    ----------
    model:
        Model variant name (``pi0``, ``pi0-fast``, ``pi0.5``, or a training
        config name like ``pi0_libero``).
    checkpoint:
        Explicit checkpoint directory or GCS path.  When *None* the default
        checkpoint for *model* is used.
    config_name:
        Override the training config name.  When *None* it is inferred from
        *model* via :data:`DEFAULT_CHECKPOINTS`.
    default_prompt:
        Default language prompt injected when the observation dict has no
        ``prompt`` key.
    norm_stats:
        Optional normalisation statistics; loaded from checkpoint when *None*.
    pytorch_device:
        Device for PyTorch models (e.g. ``"cuda"``).
    """
    if config_name is not None and checkpoint is not None:
        train_config = _config.get_config(config_name)
        ckpt_dir = checkpoint
    else:
        entry = DEFAULT_CHECKPOINTS.get(model)
        if entry is None:
            # Fall back: treat *model* as a training-config name directly.
            train_config = _config.get_config(model)
            ckpt_dir = checkpoint or entry["dir"] if entry else ""
        else:
            train_config = _config.get_config(entry["config"])
            ckpt_dir = checkpoint or entry["dir"]

    return _policy_config.create_trained_policy(
        train_config,
        ckpt_dir,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
        pytorch_device=pytorch_device,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch pi0 policy server")
    parser.add_argument("--config", type=str, default="configs/serving.yaml",
                        help="Path to serving YAML config.")
    parser.add_argument("--host", type=str, default=None,
                        help="Override host (default: from YAML or 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port (default: from YAML or 8000).")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model variant.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Override training config name.")
    parser.add_argument("--default-prompt", type=str, default=None,
                        help="Default language prompt.")
    parser.add_argument("--record", action="store_true",
                        help="Record policy I/O for debugging.")
    args = parser.parse_args(argv)

    # Merge YAML config with CLI overrides (CLI wins).
    yaml_cfg = _load_yaml(args.config)
    policy_cfg = yaml_cfg.get("policy", {})
    server_cfg = yaml_cfg.get("server", {})

    model = args.model or policy_cfg.get("model", "pi0")
    checkpoint = args.checkpoint or policy_cfg.get("checkpoint")
    config_name = args.config_name or policy_cfg.get("config_name")
    default_prompt = args.default_prompt or policy_cfg.get("default_prompt")

    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8000)

    logging.info("Loading policy: model=%s checkpoint=%s", model, checkpoint)
    policy = create_policy(
        model=model,
        checkpoint=checkpoint,
        config_name=config_name,
        default_prompt=default_prompt,
    )
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy_metadata,
    )
    logging.info("Server starting on %s:%d — press Ctrl-C to stop.", host, port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
