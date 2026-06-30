"""Formal multi-task / multi-seed evaluation harness.

Orchestrates the full evaluation pipeline:
  1. Start policy server (subprocess)
  2. Wait for server health
  3. Run closed-loop evaluation via :mod:`sim.libero_runner`
  4. Collect & save results
  5. Shut down server

Supports batch evaluation across multiple task suites and checkpoints.

Usage::

    python eval_harness/run_eval.py --config configs/eval_libero.yaml \\
        --serving-config configs/serving.yaml
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

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _wait_for_server(host: str, port: int, timeout: float = 120.0, interval: float = 2.0) -> bool:
    """Poll the policy server's /healthz endpoint until it responds."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/healthz"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("Server is healthy at %s", url)
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)
    logger.error("Server did not become healthy within %.0fs", timeout)
    return False


def run_eval_harness(
    *,
    eval_config: dict[str, Any],
    serving_config: dict[str, Any],
    task_suites: list[str] | None = None,
    checkpoints: list[dict[str, str]] | None = None,
    results_dir: str = "results",
    start_server: bool = True,
) -> dict[str, Any]:
    """Run the complete evaluation harness.

    Parameters
    ----------
    eval_config:
        Evaluation parameters (from ``configs/eval_libero.yaml``).
    serving_config:
        Serving parameters (from ``configs/serving.yaml``).
    task_suites:
        Override list of task suites.  Defaults to ``[eval_config["task_suite"]]``.
    checkpoints:
        List of ``{"config": ..., "dir": ...}`` dicts.  When provided, runs
        evaluation for each checkpoint × suite combination.
    results_dir:
        Directory to save JSON results.
    start_server:
        Whether to launch the policy server as a subprocess.  Set to *False*
        if the server is already running.

    Returns
    -------
    dict
        Aggregated results across all checkpoints and suites.
    """
    from sim.libero_runner import run_eval

    task_suites = task_suites or [eval_config.get("task_suite", "libero_spatial")]
    seeds = eval_config.get("seeds", [7])
    num_episodes = eval_config.get("num_episodes", 50)

    server_cfg = serving_config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    results_path = pathlib.Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    server_proc: subprocess.Popen | None = None

    try:
        # --- Start server ---
        if start_server:
            cmd = [
                sys.executable, "serving/launch_policy_server.py",
                "--config", "configs/serving.yaml",
                "--port", str(port),
            ]
            policy_cfg = serving_config.get("policy", {})
            if policy_cfg.get("model"):
                cmd.extend(["--model", policy_cfg["model"]])
            if policy_cfg.get("checkpoint"):
                cmd.extend(["--checkpoint", policy_cfg["checkpoint"]])

            logger.info("Starting policy server: %s", " ".join(cmd))
            server_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )

            if not _wait_for_server("127.0.0.1", port):
                server_proc.kill()
                stdout_data = server_proc.stdout.read().decode(errors="replace") if server_proc.stdout else ""
                logger.error("Policy server output:\n%s", stdout_data)
                raise RuntimeError("Policy server failed to start. See output above.")

        # --- Run evaluations ---
        all_summaries: dict[str, Any] = {"checkpoints": []}

        ckpt_list = checkpoints or [
            {"config": serving_config.get("policy", {}).get("config_name") or "default",
             "dir": serving_config.get("policy", {}).get("checkpoint") or "default"}
        ]

        for ckpt in ckpt_list:
            ckpt_label = ckpt.get("config", "default")
            ckpt_results: dict[str, Any] = {"checkpoint": ckpt_label, "suites": {}}

            for suite in task_suites:
                logger.info("=== Evaluating %s on %s ===", ckpt_label, suite)
                summary = run_eval(
                    task_suite=suite,
                    num_episodes=num_episodes,
                    seeds=seeds,
                    action_horizon=eval_config.get("action_horizon", 10),
                    replan_steps=eval_config.get("replan_steps", 5),
                    resize_size=eval_config.get("resize_size", 224),
                    num_steps_wait=eval_config.get("num_steps_wait", 10),
                    render=eval_config.get("render", True),
                    video_out=str(results_path / "rollouts" / ckpt_label / suite),
                    host="127.0.0.1",
                    port=port,
                )
                ckpt_results["suites"][suite] = {
                    "success_rate": summary["success_rate"],
                    "total_episodes": summary["total_episodes"],
                    "total_successes": summary["total_successes"],
                }

                # Save per-suite results
                suite_file = results_path / f"{ckpt_label}_{suite}.json"
                with open(suite_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                logger.info("Saved: %s", suite_file)

            all_summaries["checkpoints"].append(ckpt_results)

        # Save aggregate
        agg_file = results_path / "eval_summary.json"
        with open(agg_file, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        logger.info("Aggregate results saved to %s", agg_file)

        return all_summaries

    finally:
        if server_proc is not None:
            logger.info("Shutting down policy server …")
            server_proc.terminate()
            server_proc.wait(timeout=10)
            if server_proc.poll() is None:
                server_proc.kill()
            logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness")
    parser.add_argument("--config", type=str, default="configs/eval_libero.yaml",
                        help="Eval config YAML.")
    parser.add_argument("--serving-config", type=str, default="configs/serving.yaml",
                        help="Serving config YAML.")
    parser.add_argument("--suites", nargs="+", default=None,
                        help="Override task suites.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-server", action="store_true",
                        help="Skip server startup (assume already running).")
    args = parser.parse_args(argv)

    eval_cfg = _load_yaml(args.config)
    serving_cfg = _load_yaml(args.serving_config)

    run_eval_harness(
        eval_config=eval_cfg,
        serving_config=serving_cfg,
        task_suites=args.suites,
        results_dir=args.results_dir,
        start_server=not args.no_server,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
