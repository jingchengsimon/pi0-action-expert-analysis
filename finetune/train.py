"""LoRA / full fine-tuning entry point.

Wraps the openpi training scripts to provide a unified CLI for:
  - LoRA fine-tuning (single GPU, JAX)
  - Full fine-tuning (multi-GPU FSDP, JAX)
  - PyTorch DDP training (multi-GPU)

Usage::

    # LoRA fine-tuning (single GPU):
    python finetune/train.py --config pi0_libero_low_mem_finetune \\
        --data-dir data/libero_lerobot/libero_spatial \\
        --output-dir results/finetune/lora_both

    # Full fine-tuning (multi-GPU FSDP):
    python finetune/train.py --config pi0_libero \\
        --fsdp-devices 2 \\
        --data-dir data/libero_lerobot/libero_spatial
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_train_config(config_name: str):
    """Resolve a training config by name from the openpi registry."""
    from openpi.training import config as _config
    try:
        return _config.get_config(config_name)
    except KeyError:
        available = [c.name for c in _config.CONFIGS]
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {available}"
        ) from None


def train_jax(
    *,
    config_name: str,
    data_dir: str | None = None,
    output_dir: str = "results/finetune",
    num_steps: int | None = None,
    batch_size: int | None = None,
    fsdp_devices: int = 1,
    save_interval: int | None = None,
    resume: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> pathlib.Path:
    """Run JAX-based training (LoRA or full fine-tune).

    This delegates to ``third_party/openpi/scripts/train.py`` with
    appropriate configuration overrides.
    """
    train_config = _resolve_train_config(config_name)

    if num_steps is not None:
        train_config = dataclasses.replace(train_config, num_train_steps=num_steps)
    if batch_size is not None:
        train_config = dataclasses.replace(train_config, batch_size=batch_size)
    if fsdp_devices > 1:
        train_config = dataclasses.replace(train_config, fsdp_devices=fsdp_devices)
    if save_interval is not None:
        train_config = dataclasses.replace(train_config, save_interval=save_interval)

    # Set up data directory override via environment variable.
    if data_dir is not None:
        os.environ["OPENPI_DATA_DIR"] = str(data_dir)

    # Set up output directory.
    output_path = pathlib.Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure wandb.
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name

    logger.info("Training config: %s", config_name)
    logger.info("  num_train_steps: %d", train_config.num_train_steps)
    logger.info("  batch_size: %d", train_config.batch_size)
    logger.info("  fsdp_devices: %d", train_config.fsdp_devices)
    logger.info("  output: %s", output_path)
    logger.info("  freeze_filter: %s", train_config.freeze_filter)

    # Delegate to openpi's training script.
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party" / "openpi" / "scripts"))
        import train as _train_script

        # Build CLI args that the openpi train script expects.
        cli_args = [
            "train.py",
            config_name,
            f"--exp-name={config_name}",
            f"--output-dir={output_path}",
        ]
        if resume:
            cli_args.append("--resume")

        logger.info("Launching openpi JAX training: %s", " ".join(cli_args))
        _train_script.main(cli_args)
    except SystemExit as e:
        if e.code != 0 and e.code is not None:
            raise
    except ImportError:
        logger.error(
            "Could not import openpi training script. "
            "Make sure the submodule is initialised: "
            "git submodule update --init && pip install -e third_party/openpi"
        )
        raise

    return output_path


def train_pytorch(
    *,
    config_name: str,
    data_dir: str | None = None,
    output_dir: str = "results/finetune",
    num_steps: int | None = None,
    batch_size: int | None = None,
    num_gpus: int = 1,
    gradient_checkpointing: bool = True,
    resume: bool = False,
    wandb_project: str | None = None,
) -> pathlib.Path:
    """Run PyTorch DDP training.

    Must be launched via ``torchrun``::

        torchrun --nproc_per_node=2 finetune/train.py --backend pytorch ...
    """
    train_config = _resolve_train_config(config_name)

    if data_dir is not None:
        os.environ["OPENPI_DATA_DIR"] = str(data_dir)

    output_path = pathlib.Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    logger.info("PyTorch DDP training: config=%s gpus=%d", config_name, num_gpus)

    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party" / "openpi" / "scripts"))
        import train_pytorch as _pt_train

        cli_args = [
            "train_pytorch.py",
            config_name,
            f"--exp-name={config_name}",
            f"--output-dir={output_path}",
        ]
        if gradient_checkpointing:
            cli_args.append("--gradient-checkpointing")
        if resume:
            cli_args.append("--resume")

        _pt_train.main(cli_args)
    except SystemExit as e:
        if e.code != 0 and e.code is not None:
            raise
    except ImportError:
        logger.error("PyTorch training script not found. Ensure openpi is installed.")
        raise

    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="pi0 fine-tuning entry point")
    parser.add_argument("--config", type=str, default="pi0_libero_low_mem_finetune",
                        help="Training config name.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to LeRobot dataset.")
    parser.add_argument("--output-dir", type=str, default="results/finetune",
                        help="Output directory for checkpoints and logs.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override number of training steps.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--fsdp-devices", type=int, default=1,
                        help="Number of FSDP devices (JAX path).")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Checkpoint save interval (steps).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint.")
    parser.add_argument("--backend", type=str, default="jax",
                        choices=["jax", "pytorch"],
                        help="Training backend.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for PyTorch DDP.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name.")
    args = parser.parse_args(argv)

    if args.backend == "jax":
        output = train_jax(
            config_name=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            fsdp_devices=args.fsdp_devices,
            save_interval=args.save_interval,
            resume=args.resume,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
    else:
        output = train_pytorch(
            config_name=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            resume=args.resume,
            wandb_project=args.wandb_project,
        )

    logger.info("Training complete. Output: %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
