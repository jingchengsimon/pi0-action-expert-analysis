"""Closed-loop rollout runner for LIBERO/robosuite/MuJoCo.

Runs a full closed-loop evaluation: connects to a running policy server,
iterates over all tasks in a LIBERO benchmark suite, executes rollouts
with action-chunking, and collects per-episode metrics.

Usage::

    python sim/libero_runner.py --config configs/eval_libero.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import time
from typing import Any

import numpy as np
import yaml

from openpi_client import websocket_client_policy as _ws_client
from sim.obs_adapter import adapt_libero_obs
from sim.rollout_video import save_rollout_video

logger = logging.getLogger(__name__)

# Dummy action used during the initial "wait" steps where the sim drops objects.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # rendering resolution used during training

# Per-suite max_steps (matches openpi examples/libero/main.py).
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_libero_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    """Initialise and return the LIBERO env + task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


# ---------------------------------------------------------------------------
# Core rollout loop
# ---------------------------------------------------------------------------

def run_eval(
    *,
    task_suite: str = "libero_spatial",
    num_episodes: int = 50,
    seeds: list[int] | None = None,
    max_steps_override: int | None = None,
    action_horizon: int = 10,
    replan_steps: int = 5,
    resize_size: int = 224,
    num_steps_wait: int = 10,
    render: bool = True,
    video_out: str = "results/rollouts",
    host: str = "0.0.0.0",
    port: int = 8000,
    seed: int = 7,
) -> dict[str, Any]:
    """Run closed-loop evaluation on a LIBERO task suite.

    Returns a structured results dict with per-task and aggregate metrics.
    """
    from libero.libero import benchmark

    seeds = seeds or [seed]
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {task_suite}")
    task_suite_obj = benchmark_dict[task_suite]()
    num_tasks = task_suite_obj.n_tasks

    max_steps = max_steps_override or SUITE_MAX_STEPS.get(task_suite, 520)

    video_dir = pathlib.Path(video_out)
    video_dir.mkdir(parents=True, exist_ok=True)

    client = _ws_client.WebsocketClientPolicy(host, port)

    all_results: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0

    for s in seeds:
        np.random.seed(s)
        for task_id in range(num_tasks):
            task = task_suite_obj.get_task(task_id)
            init_states = task_suite_obj.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, s)

            task_episodes = 0
            task_successes = 0

            for ep_idx in range(num_episodes):
                env.reset()
                action_plan: collections.deque = collections.deque()
                obs = env.set_init_state(init_states[ep_idx % len(init_states)])

                replay_frames: list[np.ndarray] = []
                t = 0
                done = False
                episode_actions: list[np.ndarray] = []
                inference_times: list[float] = []

                while t < max_steps + num_steps_wait:
                    try:
                        if t < num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        adapted = adapt_libero_obs(obs, task_description, resize_size=resize_size)
                        if render:
                            replay_frames.append(adapted["observation/image"])

                        if not action_plan:
                            t0 = time.monotonic()
                            result = client.infer(adapted)
                            inference_times.append(time.monotonic() - t0)
                            chunk = result["actions"]
                            action_plan.extend(chunk[:replan_steps])

                        action = action_plan.popleft()
                        episode_actions.append(np.asarray(action))

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as e:
                        logger.error("Exception during rollout: %s", e)
                        break

                task_episodes += 1
                total_episodes += 1

                # Save video
                if render and replay_frames:
                    suffix = "success" if done else "failure"
                    seg = task_description.replace(" ", "_")
                    save_rollout_video(
                        replay_frames,
                        video_dir / f"seed{s}_{seg}_{suffix}_ep{ep_idx}.mp4",
                        task_description=task_description,
                        success=done,
                        fps=10,
                    )

                all_results.append({
                    "seed": s,
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode": ep_idx,
                    "success": bool(done),
                    "steps": t,
                    "avg_inference_ms": (
                        np.mean(inference_times) * 1000 if inference_times else None
                    ),
                })

            logger.info(
                "Task %d/%d '%s' seed=%d  SR=%.1f%%",
                task_id + 1, num_tasks, task_description, s,
                task_successes / max(task_episodes, 1) * 100,
            )

    # Aggregate
    success_rate = total_successes / max(total_episodes, 1)
    logger.info("Overall SR: %.1f%%  (%d/%d)", success_rate * 100, total_successes, total_episodes)

    summary = {
        "task_suite": task_suite,
        "num_tasks": num_tasks,
        "seeds": seeds,
        "num_episodes_per_task": num_episodes,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "episodes": all_results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LIBERO closed-loop evaluation")
    parser.add_argument("--config", type=str, default="configs/eval_libero.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override all seeds with a single seed.")
    parser.add_argument("--results-out", type=str, default="results/eval_results.json")
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)

    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [7])
    host = args.host or cfg.get("host", "0.0.0.0")
    port = args.port or cfg.get("port", 8000)

    summary = run_eval(
        task_suite=cfg.get("task_suite", "libero_spatial"),
        num_episodes=cfg.get("num_episodes", 50),
        seeds=seeds,
        max_steps_override=cfg.get("max_steps"),
        action_horizon=cfg.get("action_horizon", 10),
        replan_steps=cfg.get("replan_steps", 5),
        resize_size=cfg.get("resize_size", 224),
        num_steps_wait=cfg.get("num_steps_wait", 10),
        render=cfg.get("render", True),
        video_out=cfg.get("video_out", "results/rollouts"),
        host=host,
        port=port,
        seed=seeds[0],
    )

    # Save results
    out_path = pathlib.Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
