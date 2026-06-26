"""Compute normalisation statistics from a LeRobot dataset.

Produces ``norm_stats.json`` containing mean, std, q01, q99 for state
and action tensors, compatible with the openpi normalisation pipeline.

Usage::

    python finetune/compute_norm_stats.py \\
        --dataset data/libero_lerobot/libero_spatial \\
        --output assets/libero/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

logger = logging.getLogger(__name__)


def compute_norm_stats_from_lerobot(
    dataset_path: str | pathlib.Path,
    *,
    keys: tuple[str, ...] = ("observation.state", "action"),
) -> dict[str, dict]:
    """Compute NormStats from a LeRobot dataset directory.

    Parameters
    ----------
    dataset_path:
        Path to the LeRobot dataset (containing ``data/*.parquet``).
    keys:
        Which data keys to compute statistics for.

    Returns
    -------
    dict
        ``{key: {"mean": [...], "std": [...], "q01": [...], "q99": [...]}}``
    """
    try:
        from openpi.shared.normalize import RunningStats
    except ImportError:
        logger.warning("openpi.shared.normalize not available — using manual computation.")
        RunningStats = None

    dataset_path = pathlib.Path(dataset_path)
    parquet_files = sorted(dataset_path.glob("data/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}/data/")

    logger.info("Found %d parquet files in %s", len(parquet_files), dataset_path)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is not installed. Install it with: pip install pyarrow")
        sys.exit(1)

    # Collect all data per key.
    all_data: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for pf in parquet_files:
        table = pq.read_table(str(pf))
        for key in keys:
            col = table.column(key)
            arr = col.to_numpy(zero_copy_only=False)
            # Handle nested arrays (list of arrays per row).
            if arr.dtype == object:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr])
            all_data[key].append(arr)

    result: dict[str, dict] = {}
    for key in keys:
        data = np.concatenate(all_data[key], axis=0).astype(np.float64)
        logger.info("Key '%s': shape=%s", key, data.shape)

        if RunningStats is not None:
            rs = RunningStats()
            rs.update(data)
            stats = rs.get_statistics()
            result[key] = {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "q01": stats.q01.tolist() if stats.q01 is not None else None,
                "q99": stats.q99.tolist() if stats.q99 is not None else None,
            }
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            q01 = np.quantile(data, 0.01, axis=0)
            q99 = np.quantile(data, 0.99, axis=0)
            result[key] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
            }

        logger.info(
            "  mean range: [%.4f, %.4f], std range: [%.4f, %.4f]",
            float(mean.min()), float(mean.max()),
            float(std.min()), float(std.max()),
        )

    return result


def save_norm_stats(stats: dict[str, dict], output_path: str | pathlib.Path) -> None:
    """Save norm stats to a JSON file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Norm stats saved to %s", output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute normalisation statistics")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to LeRobot dataset directory.")
    parser.add_argument("--output", type=str, default="assets/norm_stats.json",
                        help="Output JSON path.")
    parser.add_argument("--keys", nargs="+", default=["observation.state", "action"],
                        help="Data keys to compute stats for.")
    args = parser.parse_args(argv)

    stats = compute_norm_stats_from_lerobot(args.dataset, keys=tuple(args.keys))
    save_norm_stats(stats, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
