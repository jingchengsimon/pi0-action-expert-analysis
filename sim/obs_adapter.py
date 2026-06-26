"""Adapt rendered simulator observations into pi0 policy inputs.

Converts raw LIBERO / robosuite / MuJoCo observations into the format
expected by the pi0 policy server via the WebSocket protocol.

Key transformations
-------------------
- ``agentview_image`` → ``observation/image``   (rotate 180°, resize)
- ``robot0_eye_in_hand_image`` → ``observation/wrist_image`` (rotate 180°, resize)
- ``robot0_eef_pos`` + ``robot0_eef_quat`` + ``robot0_gripper_qpos``
  → ``observation/state`` (8-d: pos(3) + quat→axisangle(3) + gripper(2))
- ``task.language`` → ``prompt``

Usage::

    from sim.obs_adapter import adapt_libero_obs, quat_to_axisangle

    obs_dict = adapt_libero_obs(env_obs, task_description, resize_size=224)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from openpi_client import image_tools
except ImportError:
    image_tools = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Quaternion → axis-angle conversion (from robosuite)
# ---------------------------------------------------------------------------

def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion ``[x, y, z, w]`` to an axis-angle vector (3-d).

    Copied from robosuite ``transform_utils.py`` with minor clean-up.
    """
    quat = np.asarray(quat, dtype=np.float64).copy()
    # Clamp w to [-1, 1] for numerical safety.
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float64)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an image to ``(size, size)`` keeping aspect ratio + padding.

    Uses :func:`openpi_client.image_tools.resize_with_pad` when available,
    otherwise falls back to a simple PIL-based implementation.
    """
    if image_tools is not None:
        img = image_tools.convert_to_uint8(img)
        return image_tools.resize_with_pad(img, size, size)

    # Fallback: PIL
    from PIL import Image
    pil = Image.fromarray(img)
    pil.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - pil.width) // 2, (size - pil.height) // 2)
    canvas.paste(pil, offset)
    return np.array(canvas, dtype=np.uint8)


def _rotate_180(img: np.ndarray) -> np.ndarray:
    """Rotate an image by 180° (flip both axes)."""
    return np.ascontiguousarray(img[::-1, ::-1])


# ---------------------------------------------------------------------------
# LIBERO observation adapter
# ---------------------------------------------------------------------------

def adapt_libero_obs(
    obs: dict[str, Any],
    task_description: str,
    *,
    resize_size: int = 224,
    rotate_images: bool = True,
) -> dict[str, Any]:
    """Convert a raw LIBERO env observation into the pi0 client format.

    Parameters
    ----------
    obs:
        Raw observation dict from ``OffScreenRenderEnv.step()``.
    task_description:
        Natural-language task instruction (e.g. ``"pick up the red block"``).
    resize_size:
        Target edge length for square resize (default 224 to match model
        input expectations).
    rotate_images:
        Whether to apply the 180° rotation that LIBERO training data uses.

    Returns
    -------
    dict
        Keys: ``observation/image``, ``observation/wrist_image``,
        ``observation/state`` (8-d ndarray), ``prompt`` (str).
    """
    # --- images ---
    agentview = obs["agentview_image"]
    wrist = obs["robot0_eye_in_hand_image"]

    if rotate_images:
        agentview = _rotate_180(agentview)
        wrist = _rotate_180(wrist)

    agentview = _resize_image(agentview, resize_size)
    wrist = _resize_image(wrist, resize_size)

    # --- proprioceptive state (8-d) ---
    eef_pos = obs["robot0_eef_pos"]                       # (3,)
    eef_quat = obs["robot0_eef_quat"]                     # (4,) → axis-angle (3,)
    eef_aa = quat_to_axisangle(eef_quat)
    gripper_qpos = obs["robot0_gripper_qpos"]             # (2,)
    state = np.concatenate([eef_pos, eef_aa, gripper_qpos]).astype(np.float32)

    return {
        "observation/image": agentview,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }


# ---------------------------------------------------------------------------
# Generic MuJoCo / robosuite adapter (extensible)
# ---------------------------------------------------------------------------

def adapt_mujoco_obs(
    obs: dict[str, Any],
    task_description: str,
    *,
    image_key: str = "agentview_image",
    wrist_image_key: str = "robot0_eye_in_hand_image",
    eef_pos_key: str = "robot0_eef_pos",
    eef_quat_key: str = "robot0_eef_quat",
    gripper_key: str = "robot0_gripper_qpos",
    resize_size: int = 224,
    rotate_images: bool = True,
) -> dict[str, Any]:
    """Generic adapter for MuJoCo-based environments.

    Same logic as :func:`adapt_libero_obs` but with configurable key names,
    making it usable with robosuite and other MuJoCo wrappers.
    """
    agentview = obs[image_key]
    wrist = obs[wrist_image_key]

    if rotate_images:
        agentview = _rotate_180(agentview)
        wrist = _rotate_180(wrist)

    agentview = _resize_image(agentview, resize_size)
    wrist = _resize_image(wrist, resize_size)

    eef_pos = obs[eef_pos_key]
    eef_aa = quat_to_axisangle(obs[eef_quat_key])
    gripper_qpos = obs[gripper_key]
    state = np.concatenate([eef_pos, eef_aa, gripper_qpos]).astype(np.float32)

    return {
        "observation/image": agentview,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }
