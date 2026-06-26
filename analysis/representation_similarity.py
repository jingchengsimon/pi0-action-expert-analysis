"""Representation similarity analysis using CKA.

Computes Centered Kernel Alignment (CKA) between representations
from different layers of the PaliGemma VLM and Action Expert.

Usage::

    python analysis/representation_similarity.py \\
        --activations-dir results/activations \\
        --output results/analysis/cka_matrix.png
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Parameters
    ----------
    X: (n_samples, d1) array
    Y: (n_samples, d2) array

    Returns
    -------
    float in [0, 1]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Centered kernel matrices.
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Center the kernels.
    n = K_X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_X = H @ K_X @ H
    K_Y = H @ K_Y @ H

    hsic_xy = np.trace(K_X @ K_Y)
    hsic_xx = np.trace(K_X @ K_X)
    hsic_yy = np.trace(K_Y @ K_Y)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_cka_matrix(
    layer_activations: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise CKA between all layers.

    Parameters
    ----------
    layer_activations:
        Dict mapping layer name → (n_samples, d) activation matrix.

    Returns
    -------
    (layer_names, cka_matrix) where cka_matrix[i,j] = CKA(layer_i, layer_j).
    """
    names = sorted(layer_activations.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cka = linear_cka(layer_activations[names[i]], layer_activations[names[j]])
            matrix[i, j] = cka
            matrix[j, i] = cka

    return names, matrix


def plot_cka_matrix(
    names: list[str],
    matrix: np.ndarray,
    output_path: str | pathlib.Path,
    title: str = "CKA Similarity Matrix",
) -> pathlib.Path:
    """Plot CKA matrix as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("CKA matrix saved to %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CKA representation similarity")
    parser.add_argument("--activations-dir", type=str, default="results/activations")
    parser.add_argument("--output", type=str, default="results/analysis/cka_matrix.png")
    args = parser.parse_args(argv)

    act_dir = pathlib.Path(args.activations_dir)

    # Load activations.
    layer_acts: dict[str, np.ndarray] = {}
    for npz_file in act_dir.glob("*.npz"):
        data = np.load(npz_file, allow_pickle=True)
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 2:
                # Flatten to (samples, features).
                if arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], -1)
                elif arr.ndim > 3:
                    arr = arr.reshape(arr.shape[0], -1)
                layer_acts[f"{npz_file.stem}/{key}"] = arr

    if not layer_acts:
        logger.error("No activations found in %s. Run extract_activations.py first.", act_dir)
        return

    logger.info("Computing CKA for %d layers …", len(layer_acts))
    names, matrix = compute_cka_matrix(layer_acts)

    # Log top pairs.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j] > 0.8:
                logger.info("  High CKA: %s ↔ %s = %.3f", names[i], names[j], matrix[i, j])

    plot_cka_matrix(names, matrix, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
