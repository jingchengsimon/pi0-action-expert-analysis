"""Automated experiment pipeline orchestrator.

Defines a DAG of experiment steps and submits them as chained Slurm jobs::

    data_prep → train → norm_stats → serve → eval → collect

Usage::

    python pipeline/run_experiment.py \\
        --variant lora_both \\
        --data-dir data/libero_lerobot/libero_spatial \\
        --output-dir results/experiments
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _submit_job(
    script: str,
    job_name: str,
    *,
    partition: str = "gpu-redhat",
    gpu: int = 1,
    cpus: int = 8,
    mem: str = "48G",
    time_limit: str = "08:00:00",
    dependency: str | None = None,
    extra_args: list[str] | None = None,
    output_dir: str = "results/slurm",
) -> str:
    """Submit a Slurm job and return the job ID."""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--gres=gpu:{gpu}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time_limit}",
        f"--output={output_dir}/%j.out",
    ]

    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")

    cmd.append("--wrap")
    wrap_cmd = f"source ~/.bashrc && conda activate pi0 && python {script}"
    if extra_args:
        wrap_cmd += " " + " ".join(extra_args)
    cmd.append(wrap_cmd)

    logger.info("Submitting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse job ID from sbatch output: "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    logger.info("Submitted job %s: %s", job_id, job_name)
    return job_id


def run_pipeline(
    *,
    variant: str = "lora_both",
    data_dir: str = "data/libero_lerobot/libero_spatial",
    output_dir: str = "results/experiments",
    num_steps: int = 30000,
    eval_suites: list[str] | None = None,
    eval_episodes: int = 50,
    eval_seeds: list[int] | None = None,
    skip_train: bool = False,
    skip_eval: bool = False,
) -> dict[str, str]:
    """Run the full experiment pipeline.

    Returns a dict mapping step names to Slurm job IDs.
    """
    eval_suites = eval_suites or ["libero_spatial"]
    eval_seeds = eval_seeds or [0, 1, 2]
    job_ids: dict[str, str] = {}

    out = pathlib.Path(output_dir) / variant
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Data prep (CPU only).
    logger.info("=== Step 1: Data preparation ===")
    data_job = _submit_job(
        "finetune/convert_to_lerobot.py",
        f"{variant}_data",
        partition="main",
        gpu=0,
        cpus=8,
        mem="64G",
        time_limit="04:00:00",
        extra_args=["--suite", variant.split("_")[-1] if "libero" in variant else "libero_spatial",
                     "--output-dir", data_dir],
    )
    job_ids["data_prep"] = data_job

    # Step 2: Training.
    if not skip_train:
        logger.info("=== Step 2: Training ===")
        train_job = _submit_job(
            "finetune/train.py",
            f"{variant}_train",
            partition="gpu",
            gpu=1,
            cpus=8,
            mem="48G",
            time_limit="24:00:00",
            dependency=data_job,
            extra_args=[
                "--config", "pi0_libero_low_mem_finetune",
                "--data-dir", data_dir,
                "--output-dir", str(out / "checkpoints"),
                "--num-steps", str(num_steps),
            ],
        )
        job_ids["train"] = train_job
    else:
        train_job = data_job

    # Step 3: Compute norm stats.
    logger.info("=== Step 3: Norm stats ===")
    norm_job = _submit_job(
        "finetune/compute_norm_stats.py",
        f"{variant}_norm",
        partition="main",
        gpu=0,
        cpus=4,
        mem="32G",
        time_limit="01:00:00",
        dependency=train_job,
        extra_args=["--dataset", data_dir, "--output", str(out / "norm_stats.json")],
    )
    job_ids["norm_stats"] = norm_job

    # Step 4: Evaluation.
    if not skip_eval:
        logger.info("=== Step 4: Evaluation ===")
        for suite in eval_suites:
            eval_job = _submit_job(
                "eval_harness/run_eval.py",
                f"{variant}_eval_{suite}",
                partition="gpu",
                gpu=1,
                cpus=8,
                mem="48G",
                time_limit="12:00:00",
                dependency=norm_job,
                extra_args=[
                    "--config", "configs/eval_libero.yaml",
                    "--suites", suite,
                    "--results-dir", str(out / "eval_results"),
                ],
            )
            job_ids[f"eval_{suite}"] = eval_job

    # Save pipeline manifest.
    manifest = {
        "variant": variant,
        "job_ids": job_ids,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out),
    }
    with open(out / "pipeline_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Pipeline manifest saved to %s", out / "pipeline_manifest.json")

    return job_ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment pipeline orchestrator")
    parser.add_argument("--variant", type=str, default="lora_both")
    parser.add_argument("--data-dir", type=str, default="data/libero_lerobot/libero_spatial")
    parser.add_argument("--output-dir", type=str, default="results/experiments")
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--eval-suites", nargs="+", default=["libero_spatial"])
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    run_pipeline(
        variant=args.variant,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        eval_suites=args.eval_suites,
        eval_episodes=args.eval_episodes,
        eval_seeds=args.eval_seeds,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
