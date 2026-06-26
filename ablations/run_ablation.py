"""Declarative ablation framework for pi0 experiments.

Reads YAML ablation configs that define experiment variants, then
automatically runs: train → eval → collect for each variant.

Supports wandb integration for experiment tracking.

Usage::

    python ablations/run_ablation.py \\
        --config ablations/configs/a1_expert_scale.yaml \\
        --output-dir results/ablations \\
        --dry-run  # print plan without executing

Each YAML config defines a list of *variants* that differ in a small number
of hyper-parameters (e.g. action expert size, LoRA rank, flow matching steps).
The runner iterates over variants, trains each, evaluates on LIBERO, and
writes structured JSON results for downstream aggregation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_ablation_config(path: str) -> dict[str, Any]:
    """Load and validate an ablation YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = ["name", "description", "variants"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in ablation config")

    for i, v in enumerate(cfg["variants"]):
        if "id" not in v:
            v["id"] = f"variant_{i}"
    return cfg


# ---------------------------------------------------------------------------
# Variant → training config override mapping
# ---------------------------------------------------------------------------

def build_train_overrides(variant: dict[str, Any]) -> list[str]:
    """Convert variant overrides dict into CLI arguments for finetune/train.py."""
    args: list[str] = []

    if "config" in variant:
        args.extend(["--config", variant["config"]])
    if "data_dir" in variant:
        args.extend(["--data-dir", variant["data_dir"]])
    if "num_steps" in variant:
        args.extend(["--num-steps", str(variant["num_steps"])])
    if "batch_size" in variant:
        args.extend(["--batch-size", str(variant["batch_size"])])
    if "fsdp_devices" in variant:
        args.extend(["--fsdp-devices", str(variant["fsdp_devices"])])
    if "backend" in variant:
        args.extend(["--backend", variant["backend"]])
    if "save_interval" in variant:
        args.extend(["--save-interval", str(variant["save_interval"])])

    # LoRA-specific environment overrides.
    env_overrides: dict[str, str] = {}
    if "lora_rank_vlm" in variant:
        env_overrides["OPENPI_LORA_RANK_VLM"] = str(variant["lora_rank_vlm"])
    if "lora_rank_ae" in variant:
        env_overrides["OPENPI_LORA_RANK_AE"] = str(variant["lora_rank_ae"])
    if "freeze_filter" in variant:
        env_overrides["OPENPI_FREEZE_FILTER"] = variant["freeze_filter"]

    return args, env_overrides


def build_eval_overrides(variant: dict[str, Any], checkpoint_dir: str) -> list[str]:
    """Build CLI arguments for eval_harness/run_eval.py."""
    args = [
        "--config", "configs/eval_libero.yaml",
        "--checkpoint", checkpoint_dir,
    ]
    if "eval_suites" in variant:
        args.extend(["--suites"] + variant["eval_suites"])
    if "num_steps_flow" in variant:
        args.extend(["--num-steps", str(variant["num_steps_flow"])])
    if "action_horizon" in variant:
        args.extend(["--action-horizon", str(variant["action_horizon"])])
    return args


# ---------------------------------------------------------------------------
# Job submission helpers
# ---------------------------------------------------------------------------

def _submit_slurm(
    script: str,
    job_name: str,
    *,
    extra_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    partition: str = "gpu",
    gpu: int = 1,
    cpus: int = 8,
    mem: str = "48G",
    time_limit: str = "24:00:00",
    dependency: str | None = None,
    output_dir: str = "results/slurm",
) -> str:
    """Submit a Slurm job and return the job ID."""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time_limit}",
        f"--output={output_dir}/%j.out",
    ]
    if gpu > 0:
        cmd.append(f"--gres=gpu:{gpu}")
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")

    # Build environment setup string.
    env_str = "source ~/.bashrc && conda activate pi0"
    if env_vars:
        for k, v in env_vars.items():
            env_str += f" && export {k}={v}"

    cmd.append("--wrap")
    inner = f"{env_str} && python {script}"
    if extra_args:
        inner += " " + " ".join(extra_args)
    cmd.append(inner)

    logger.info("Submitting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split()[-1]
    logger.info("Job %s: %s", job_id, job_name)
    return job_id


def _run_local(
    script: str,
    extra_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> int:
    """Run a script locally (for debugging / smoke testing)."""
    import os
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running locally: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    config_path: str,
    output_dir: str = "results/ablations",
    *,
    dry_run: bool = False,
    local: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    """Run all variants defined in an ablation config.

    Returns a summary dict with variant IDs → job IDs / status.
    """
    cfg = load_ablation_config(config_path)
    ablation_name = cfg["name"]
    out = pathlib.Path(output_dir) / ablation_name
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Ablation: %s", ablation_name)
    logger.info("Description: %s", cfg.get("description", ""))
    logger.info("Variants: %d", len(cfg["variants"]))
    logger.info("=" * 60)

    # Shared defaults from config level.
    shared = cfg.get("shared", {})
    default_data_dir = shared.get("data_dir", "data/libero_lerobot/libero_spatial")
    default_num_steps = shared.get("num_steps", 30000)
    default_eval_suites = shared.get("eval_suites", ["libero_spatial"])

    summary: dict[str, Any] = {
        "ablation": ablation_name,
        "description": cfg.get("description", ""),
        "num_variants": len(cfg["variants"]),
        "variants": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for variant in cfg["variants"]:
        vid = variant["id"]
        logger.info("--- Variant: %s ---", vid)

        # Merge shared defaults with variant-specific overrides.
        merged = {**shared, **variant}
        merged.setdefault("data_dir", default_data_dir)
        merged.setdefault("num_steps", default_num_steps)
        merged.setdefault("eval_suites", default_eval_suites)

        variant_out = out / vid
        variant_out.mkdir(parents=True, exist_ok=True)

        train_args, env_overrides = build_train_overrides(merged)
        train_args.extend(["--output-dir", str(variant_out / "checkpoints")])
        train_args.extend(["--data-dir", merged["data_dir"]])

        if wandb_project:
            train_args.extend(["--wandb-project", wandb_project,
                               "--wandb-run-name", f"{ablation_name}_{vid}"])

        variant_info: dict[str, Any] = {
            "id": vid,
            "overrides": {k: v for k, v in merged.items() if k not in shared},
            "output_dir": str(variant_out),
        }

        if dry_run:
            logger.info("  [DRY RUN] train args: %s", train_args)
            variant_info["status"] = "dry_run"
            variant_info["train_args"] = train_args
        else:
            # Step 1: Train.
            if not skip_train:
                if local:
                    rc = _run_local("finetune/train.py", train_args, env_overrides)
                    variant_info["train_status"] = "success" if rc == 0 else "failed"
                else:
                    train_job = _submit_slurm(
                        "finetune/train.py",
                        f"{ablation_name}_{vid}_train",
                        extra_args=train_args,
                        env_vars=env_overrides,
                        gpu=merged.get("fsdp_devices", 1),
                        cpus=merged.get("cpus", 8),
                        mem=merged.get("mem", "48G"),
                        time_limit=merged.get("train_time", "24:00:00"),
                    )
                    variant_info["train_job"] = train_job
            else:
                variant_info["train_status"] = "skipped"

            # Step 2: Evaluate.
            if not skip_eval:
                ckpt_dir = str(variant_out / "checkpoints")
                eval_args = build_eval_overrides(merged, ckpt_dir)
                eval_args.extend(["--results-dir", str(variant_out / "eval_results")])

                if local:
                    rc = _run_local("eval_harness/run_eval.py", eval_args)
                    variant_info["eval_status"] = "success" if rc == 0 else "failed"
                else:
                    dep = variant_info.get("train_job")
                    eval_job = _submit_slurm(
                        "eval_harness/run_eval.py",
                        f"{ablation_name}_{vid}_eval",
                        extra_args=eval_args,
                        dependency=dep,
                        time_limit=merged.get("eval_time", "12:00:00"),
                    )
                    variant_info["eval_job"] = eval_job
            else:
                variant_info["eval_status"] = "skipped"

        summary["variants"][vid] = variant_info

    # Save summary.
    summary_path = out / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Ablation summary saved to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Declarative ablation runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to ablation YAML config.")
    parser.add_argument("--output-dir", type=str, default="results/ablations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing.")
    parser.add_argument("--local", action="store_true",
                        help="Run locally instead of submitting to Slurm.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args(argv)

    run_ablation(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        local=args.local,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
