"""Convert raw demos into LeRobot dataset format.

Downloads LIBERO RLDS data from HuggingFace and converts it to the
LeRobot format expected by the openpi training pipeline.

Usage::

    python finetune/convert_to_lerobot.py \\
        --suite libero_spatial \\
        --output-dir data/libero_lerobot \\
        --repo-id libero_spatial
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any

logger = logging.getLogger(__name__)


def convert_libero_rlds_to_lerobot(
    *,
    suite: str = "libero_spatial",
    output_dir: str = "data/libero_lerobot",
    repo_id: str | None = None,
    image_size: tuple[int, int] = (256, 256),
    max_episodes: int | None = None,
) -> pathlib.Path:
    """Convert LIBERO RLDS data to LeRobot format.

    Wraps the logic from ``third_party/openpi/examples/libero/convert_libero_data_to_lerobot.py``.

    Parameters
    ----------
    suite:
        LIBERO task suite name (libero_spatial, libero_object, libero_goal, libero_10).
    output_dir:
        Root directory for the LeRobot dataset.
    repo_id:
        LeRobot repo_id; defaults to suite name.
    image_size:
        Target image dimensions.
    max_episodes:
        Limit number of episodes (useful for quick tests).

    Returns
    -------
    pathlib.Path
        Path to the created LeRobot dataset directory.
    """
    repo_id = repo_id or suite
    out = pathlib.Path(output_dir) / repo_id
    out.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error("lerobot is not installed. Install it with: pip install lerobot")
        sys.exit(1)

    try:
        import tensorflow_datasets as tfds
    except ImportError:
        logger.error("tensorflow_datasets is not installed. Install it with: pip install tensorflow-datasets")
        sys.exit(1)

    logger.info("Downloading LIBERO RLDS data for suite: %s", suite)

    # Use the openpi conversion script if available.
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party" / "openpi" / "examples" / "libero"))
        import convert_libero_data_to_lerobot as _converter
        logger.info("Using openpi's built-in LIBERO converter.")
        # Delegate to the openpi converter.
        _converter.main(suite_name=suite, output_dir=str(out), repo_id=repo_id)
        return out
    except Exception as e:
        logger.warning("openpi converter failed (%s), falling back to manual conversion.", e)

    # Fallback: manual conversion.
    logger.info("Performing manual RLDS → LeRobot conversion …")
    ds_name = f"libero_{suite}" if not suite.startswith("libero") else suite

    dataset = tfds.load(
        f"{ds_name}/image",
        split="train",
        data_dir=str(pathlib.Path(output_dir) / "rlds_cache"),
    )

    episodes = list(dataset)
    if max_episodes:
        episodes = episodes[:max_episodes]
    logger.info("Processing %d episodes …", len(episodes))

    features = {
        "observation.image": {"dtype": "image", "shape": list(image_size) + [3], "names": ["height", "width", "channels"]},
        "observation.wrist_image": {"dtype": "image", "shape": list(image_size) + [3], "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [7], "names": ["action"]},
    }

    lerobot_ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(out),
        features=features,
        fps=20,
    )

    for ep_idx, episode in enumerate(episodes):
        steps = episode["steps"]
        task = episode.get("language_instruction", b"").decode("utf-8") or f"task_{ep_idx}"

        for step in steps:
            frame = {
                "observation.image": step["observation"]["image"],
                "observation.wrist_image": step["observation"].get("wrist_image", step["observation"]["image"]),
                "observation.state": step["observation"]["state"].numpy().astype("float32"),
                "action": step["action"].numpy().astype("float32")[:7],
                "task": task,
            }
            lerobot_ds.add_frame(frame)

        if (ep_idx + 1) % 10 == 0:
            logger.info("  Processed %d / %d episodes", ep_idx + 1, len(episodes))

    lerobot_ds.save()
    logger.info("LeRobot dataset saved to %s", out)
    return out


def verify_dataset(path: str | pathlib.Path) -> bool:
    """Verify a converted LeRobot dataset has the expected structure."""
    path = pathlib.Path(path)
    if not path.exists():
        logger.error("Dataset path does not exist: %s", path)
        return False

    data_dir = path / "data"
    if not data_dir.exists():
        logger.error("No data/ subdirectory found in %s", path)
        return False

    logger.info("Dataset at %s contains %d parquet files.", path, len(list(data_dir.glob("*.parquet"))))
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert LIBERO data to LeRobot format")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        help="LIBERO suite name.")
    parser.add_argument("--output-dir", type=str, default="data/libero_lerobot",
                        help="Output directory for LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="LeRobot repo_id (default: suite name).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit episodes (for quick tests).")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify an existing dataset.")
    args = parser.parse_args(argv)

    if args.verify:
        ok = verify_dataset(pathlib.Path(args.output_dir) / (args.repo_id or args.suite))
        sys.exit(0 if ok else 1)

    convert_libero_rlds_to_lerobot(
        suite=args.suite,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
