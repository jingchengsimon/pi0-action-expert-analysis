"""Render rollout frames into a video for qualitative inspection.

Saves an MP4 video from a list of RGB frames, optionally overlaying task
description, step count, and success/failure labels.

Usage::

    from sim.rollout_video import save_rollout_video

    save_rollout_video(frames, "results/rollouts/task1_success.mp4",
                       task_description="pick up red block", success=True)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_imageio():
    """Import imageio with a graceful fallback."""
    try:
        import imageio
        return imageio
    except ImportError:
        return None


def _overlay_text(
    frame: np.ndarray,
    text: str,
    *,
    position: tuple[int, int] = (10, 20),
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.5,
) -> np.ndarray:
    """Overlay text on a frame using OpenCV if available, else PIL."""
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
        x, y = position
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg_color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)
        return frame
    except ImportError:
        pass

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text(position, text, fill=color, font=font)
        return np.array(pil)
    except ImportError:
        return frame


def save_rollout_video(
    frames: Sequence[np.ndarray],
    output_path: str | pathlib.Path,
    *,
    task_description: str = "",
    success: bool = False,
    fps: int = 10,
    overlay_info: bool = True,
) -> pathlib.Path:
    """Save a list of RGB frames as an MP4 video.

    Parameters
    ----------
    frames:
        Sequence of ``(H, W, 3)`` uint8 numpy arrays.
    output_path:
        Destination file path (parent dirs created automatically).
    task_description:
        Task label shown in the overlay.
    success:
        Whether the rollout succeeded (shown in overlay).
    fps:
        Frames per second.
    overlay_info:
        If *True*, overlay task description and success/failure on each frame.

    Returns
    -------
    pathlib.Path
        The path to the written video file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        logger.warning("No frames to save for %s", output_path)
        return output_path

    imageio = _try_import_imageio()
    if imageio is None:
        logger.error("imageio is not installed — cannot save video to %s", output_path)
        return output_path

    processed: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        f = frame.copy()
        if overlay_info:
            label = "SUCCESS" if success else "FAILURE"
            _overlay_text(f, f"{task_description}  [{label}]", position=(5, 18))
            _overlay_text(f, f"step {i}", position=(5, 38), color=(200, 200, 200))
        processed.append(f)

    imageio.mimwrite(str(output_path), processed, fps=fps)
    logger.info("Saved rollout video: %s (%d frames)", output_path, len(processed))
    return output_path


def save_rollout_gif(
    frames: Sequence[np.ndarray],
    output_path: str | pathlib.Path,
    *,
    fps: int = 10,
    max_frames: int = 100,
) -> pathlib.Path:
    """Save frames as an animated GIF (useful for quick inspection)."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio = _try_import_imageio()
    if imageio is None:
        logger.error("imageio is not installed — cannot save GIF.")
        return output_path

    selected = list(frames[:max_frames])
    imageio.mimwrite(str(output_path), selected, fps=fps, loop=0)
    logger.info("Saved GIF: %s (%d frames)", output_path, len(selected))
    return output_path
