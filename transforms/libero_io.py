"""LIBERO camera/state/action schema and I/O transforms.

Documents the mapping between LIBERO simulation observations and the
π0 policy interface, and provides helper functions for runtime conversion.

Schema (LIBERO → π0)
---------------------
| π0 key                   | LIBERO source                 | shape  | notes                    |
|--------------------------|-------------------------------|--------|--------------------------|
| observation/image        | agentview_image (rot180)    | (H,W,3)| uint8, resized to 224²   |
| observation/wrist_image  | robot0_eye_in_hand_image    | (H,W,3)| uint8, resized to 224²   |
| observation/state        | eef_pos(3)+quat2aa(3)+grip(2)| (8,)  | float32                  |
| prompt                   | task.language                | str    | natural-language task    |
| actions                  | OSC delta + gripper          | (7,)   | xyz(3)+rot(3)+grip(1)   |

Control frequency: 20 Hz (LIBERO default)
Normalisation: uses norm_stats from checkpoint or computed from data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.obs_adapter import adapt_libero_obs, quat_to_axisangle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_STATE_DIM = 8   # eef_pos(3) + axisangle(3) + gripper_qpos(2)
LIBERO_ACTION_DIM = 7  # delta_xyz(3) + delta_rot(3) + gripper(1)
LIBERO_RENDER_RES = 256
LIBERO_MODEL_INPUT_RES = 224

# Per-suite recommended max_steps.
LIBERO_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def libero_obs_to_pi0(
    obs: dict[str, Any],
    task_description: str,
    *,
    resize_size: int = LIBERO_MODEL_INPUT_RES,
) -> dict[str, Any]:
    """Convert a raw LIBERO env observation to the π0 client dict.

    Thin wrapper around :func:`sim.obs_adapter.adapt_libero_obs` with
    LIBERO-specific defaults.
    """
    return adapt_libero_obs(obs, task_description, resize_size=resize_size, rotate_images=True)


def pi0_actions_to_libero(actions: np.ndarray) -> np.ndarray:
    """Post-process π0 action output for LIBERO env.

    The π0 model outputs actions of shape ``(T, 32)`` (padded).  This
    function trims to the first 7 dimensions used by LIBERO.
    """
    return np.asarray(actions)[..., :LIBERO_ACTION_DIM]


def describe_schema() -> dict[str, Any]:
    """Return a structured description of the LIBERO→π0 schema."""
    return {
        "source": "LIBERO (MuJoCo-based simulation)",
        "cameras": {
            "observation/image": {
                "libero_key": "agentview_image",
                "shape_raw": f"({LIBERO_RENDER_RES}, {LIBERO_RENDER_RES}, 3)",
                "shape_model": f"({LIBERO_MODEL_INPUT_RES}, {LIBERO_MODEL_INPUT_RES}, 3)",
                "preprocessing": "rotate_180 + resize_with_pad",
                "dtype": "uint8",
            },
            "observation/wrist_image": {
                "libero_key": "robot0_eye_in_hand_image",
                "shape_raw": f"({LIBERO_RENDER_RES}, {LIBERO_RENDER_RES}, 3)",
                "shape_model": f"({LIBERO_MODEL_INPUT_RES}, {LIBERO_MODEL_INPUT_RES}, 3)",
                "preprocessing": "rotate_180 + resize_with_pad",
                "dtype": "uint8",
            },
        },
        "state": {
            "key": "observation/state",
            "dim": LIBERO_STATE_DIM,
            "components": [
                ("robot0_eef_pos", 3, "end-effector position (xyz)"),
                ("robot0_eef_quat→axisangle", 3, "end-effector orientation (axis-angle)"),
                ("robot0_gripper_qpos", 2, "gripper joint positions"),
            ],
        },
        "action": {
            "key": "actions",
            "dim": LIBERO_ACTION_DIM,
            "components": [
                ("delta_xyz", 3, "position delta"),
                ("delta_rot", 3, "rotation delta (axis-angle)"),
                ("gripper", 1, "gripper command (−1=close, +1=open)"),
            ],
        },
        "control_freq_hz": 20,
    }
