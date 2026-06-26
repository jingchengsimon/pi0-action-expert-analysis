"""DROID camera/state/action schema and I/O transforms.

Documents the mapping between DROID real-robot observations and the
π0 policy interface, and provides helper functions for runtime conversion.

Schema (DROID → π0)
--------------------
| π0 key                   | DROID source                       | shape  | notes                    |
|--------------------------|--------------------------------------|--------|--------------------------|
| observation/image        | exterior_image_1_left (or right)   | (H,W,3)| uint8, resized to 224²   |
| observation/wrist_image  | wrist_image_left (or right)        | (H,W,3)| uint8, resized to 224²   |
| observation/state        | joint_pos(7) + gripper_width(1)    | (8,)   | float32                  |
| prompt                   | language instruction               | str    | natural-language task    |
| actions                  | joint_velocity(7) + gripper(1)     | (8,)   | 8-d action space         |

Control frequency: 15 Hz (DROID default)
Normalisation: uses norm_stats from DROID checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DROID_STATE_DIM = 8     # joint_pos(7) + gripper_width(1)
DROID_ACTION_DIM = 8    # joint_velocity(7) + gripper(1)
DROID_MODEL_INPUT_RES = 224

# DROID camera names → π0 key mapping.
DROID_CAMERA_MAP: dict[str, str] = {
    "exterior_image_1_left": "observation/image",
    "exterior_image_1_right": "observation/image",
    "wrist_image_left": "observation/wrist_image",
    "wrist_image_right": "observation/wrist_image",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def droid_obs_to_pi0(
    obs: dict[str, Any],
    task_description: str = "",
    *,
    resize_size: int = DROID_MODEL_INPUT_RES,
    use_left_cameras: bool = True,
) -> dict[str, Any]:
    """Convert a DROID observation dict to the π0 client format.

    Parameters
    ----------
    obs:
        Raw DROID observation dict with keys like
        ``exterior_image_1_left``, ``wrist_image_left``,
        ``joint_positions``, ``gripper_position``.
    task_description:
        Language instruction for the current task.
    resize_size:
        Target image edge length.
    use_left_cameras:
        Whether to use the left camera images (DROID has stereo pairs).
    """
    suffix = "_left" if use_left_cameras else "_right"

    # --- images ---
    ext_key = f"exterior_image_1{suffix}"
    wrist_key = f"wrist_image{suffix}"

    try:
        from openpi_client import image_tools
        ext_img = image_tools.convert_to_uint8(obs[ext_key])
        ext_img = image_tools.resize_with_pad(ext_img, resize_size, resize_size)
        wrist_img = image_tools.convert_to_uint8(obs[wrist_key])
        wrist_img = image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    except ImportError:
        from PIL import Image
        ext_img = _pil_resize(obs[ext_key], resize_size)
        wrist_img = _pil_resize(obs[wrist_key], resize_size)

    # --- proprioceptive state (8-d) ---
    joint_pos = np.asarray(obs.get("joint_positions", obs.get("robot_state", {}).get("joint_positions", np.zeros(7))))
    gripper_pos = np.asarray([obs.get("gripper_position", 0.0)])
    state = np.concatenate([joint_pos[:7], gripper_pos]).astype(np.float32)

    return {
        "observation/image": ext_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def pi0_actions_to_droid(actions: np.ndarray) -> np.ndarray:
    """Post-process π0 action output for DROID robot.

    Trims to the first 8 dimensions (joint_velocity(7) + gripper(1)).
    """
    return np.asarray(actions)[..., :DROID_ACTION_DIM]


def _pil_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize using PIL as fallback."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(np.asarray(img, dtype=np.uint8))
    pil.thumbnail((size, size), PILImage.BILINEAR)
    canvas = PILImage.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - pil.width) // 2, (size - pil.height) // 2)
    canvas.paste(pil, offset)
    return np.array(canvas, dtype=np.uint8)


def describe_schema() -> dict[str, Any]:
    """Return a structured description of the DROID→π0 schema."""
    return {
        "source": "DROID (real-robot Franka Emika platform)",
        "cameras": {
            "observation/image": {
                "droid_keys": ["exterior_image_1_left", "exterior_image_1_right"],
                "shape_model": f"({DROID_MODEL_INPUT_RES}, {DROID_MODEL_INPUT_RES}, 3)",
                "preprocessing": "resize_with_pad",
                "dtype": "uint8",
            },
            "observation/wrist_image": {
                "droid_keys": ["wrist_image_left", "wrist_image_right"],
                "shape_model": f"({DROID_MODEL_INPUT_RES}, {DROID_MODEL_INPUT_RES}, 3)",
                "preprocessing": "resize_with_pad",
                "dtype": "uint8",
            },
        },
        "state": {
            "key": "observation/state",
            "dim": DROID_STATE_DIM,
            "components": [
                ("joint_positions", 7, "Franka joint angles (rad)"),
                ("gripper_position", 1, "gripper width (m)"),
            ],
        },
        "action": {
            "key": "actions",
            "dim": DROID_ACTION_DIM,
            "components": [
                ("joint_velocity", 7, "joint velocity commands (rad/s)"),
                ("gripper", 1, "gripper velocity command"),
            ],
        },
        "control_freq_hz": 15,
    }
