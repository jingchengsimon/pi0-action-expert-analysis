"""Smoke test: pre-processing pipeline validation.

Verifies that:
  1. Quaternion → axis-angle round-trip is consistent.
  2. Image resize preserves dtype and expected shape.
  3. Normalize / Unnormalize are inverse operations (round-trip).
  4. State vector assembly has the correct shape.

Usage::

    python smoke_tests/test_preprocessing.py
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test: quaternion → axis-angle
# ---------------------------------------------------------------------------

def test_quat_to_axisangle() -> bool:
    """Verify quat→axisangle conversion is correct."""
    from sim.obs_adapter import quat_to_axisangle

    # Identity rotation → zero vector.
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])
    result = quat_to_axisangle(q_identity)
    ok = np.allclose(result, 0.0, atol=1e-6)
    logger.info("quat_to_axisangle(identity) = %s  ok=%s", result, ok)

    # 90° rotation around Z-axis: q = [0, 0, sin(45°), cos(45°)]
    angle = math.pi / 2
    q_z90 = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
    result = quat_to_axisangle(q_z90)
    expected = np.array([0.0, 0.0, angle])
    ok2 = np.allclose(result, expected, atol=1e-6)
    logger.info("quat_to_axisangle(Z_90°) = %s  expected=%s  ok=%s", result, expected, ok2)

    # 180° rotation around X-axis: q = [1, 0, 0, 0]
    q_x180 = np.array([1.0, 0.0, 0.0, 0.0])
    result = quat_to_axisangle(q_x180)
    expected = np.array([math.pi, 0.0, 0.0])
    ok3 = np.allclose(result, expected, atol=1e-6)
    logger.info("quat_to_axisangle(X_180°) = %s  expected=%s  ok=%s", result, expected, ok3)

    return ok and ok2 and ok3


# ---------------------------------------------------------------------------
# Test: image resize
# ---------------------------------------------------------------------------

def test_image_resize() -> bool:
    """Verify image resize produces correct shape and dtype."""
    from sim.obs_adapter import _resize_image

    # 256×256 → 224×224
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    resized = _resize_image(img, 224)
    ok1 = resized.shape == (224, 224, 3) and resized.dtype == np.uint8
    logger.info("resize(256→224): shape=%s dtype=%s  ok=%s", resized.shape, resized.dtype, ok1)

    # 128×256 → 224×224 (non-square, should pad)
    img2 = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    resized2 = _resize_image(img2, 224)
    ok2 = resized2.shape == (224, 224, 3) and resized2.dtype == np.uint8
    logger.info("resize(128×256→224²): shape=%s  ok=%s", resized2.shape, ok2)

    return ok1 and ok2


# ---------------------------------------------------------------------------
# Test: observation adapter end-to-end
# ---------------------------------------------------------------------------

def test_libero_obs_adapter() -> bool:
    """Verify the full LIBERO observation adapter pipeline."""
    from sim.obs_adapter import adapt_libero_obs

    fake_obs = {
        "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.random.randn(3).astype(np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.random.randn(2).astype(np.float32),
    }

    result = adapt_libero_obs(fake_obs, "pick up the red block", resize_size=224)

    ok_img = result["observation/image"].shape == (224, 224, 3)
    ok_wrist = result["observation/wrist_image"].shape == (224, 224, 3)
    ok_state = result["observation/state"].shape == (8,)
    ok_prompt = result["prompt"] == "pick up the red block"

    logger.info(
        "adapt_libero_obs: image=%s wrist=%s state=%s prompt=%s",
        result["observation/image"].shape,
        result["observation/wrist_image"].shape,
        result["observation/state"].shape,
        result["prompt"],
    )

    all_ok = ok_img and ok_wrist and ok_state and ok_prompt
    logger.info("  all_ok=%s", all_ok)
    return all_ok


# ---------------------------------------------------------------------------
# Test: normalise / unnormalise round-trip
# ---------------------------------------------------------------------------

def test_normalize_roundtrip() -> bool:
    """Verify that Normalize → Unnormalize is (approximately) identity."""
    try:
        from openpi.transforms import Normalize, Unnormalize
        from openpi.shared.normalize import NormStats
    except ImportError:
        logger.warning("openpi not importable — skipping normalize round-trip test.")
        return True

    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 8)).astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-6
    q01 = np.quantile(data, 0.01, axis=0)
    q99 = np.quantile(data, 0.99, axis=0)

    stats = {"observation/state": NormStats(mean=mean, std=std, q01=q01, q99=q99)}

    norm_t = Normalize(stats, use_quantiles=False)
    unnorm_t = Unnormalize(stats, use_quantiles=False)

    sample = {"observation/state": data[:1]}
    normed = norm_t(sample)
    restored = unnorm_t(normed)

    diff = np.abs(restored["observation/state"] - sample["observation/state"])
    ok = np.all(diff < 1e-4)
    logger.info("Normalize→Unnormalize max_diff=%.6e  ok=%s", float(diff.max()), ok)
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    """Run all preprocessing smoke tests and return overall pass/fail."""
    results: dict[str, bool] = {}

    results["quat_to_axisangle"] = test_quat_to_axisangle()
    results["image_resize"] = test_image_resize()
    results["libero_obs_adapter"] = test_libero_obs_adapter()
    results["normalize_roundtrip"] = test_normalize_roundtrip()

    logger.info("=" * 50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info("  %s: %s", name, status)
    logger.info("=" * 50)

    return all(results.values())


def main() -> None:
    ok = run_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
