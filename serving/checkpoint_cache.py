"""Checkpoint cache / warm-up for fast policy server startup.

Provides an LRU-style cache that holds already-loaded :class:`Policy`
objects keyed by ``(config_name, checkpoint_dir)``.  This avoids the
expensive model-load overhead when switching between checkpoints during
batch evaluation runs.

Usage::

    from serving.checkpoint_cache import PolicyCache

    cache = PolicyCache(max_size=3)
    policy = cache.get("pi0_libero", "gs://openpi-assets/checkpoints/pi0_base")
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from openpi.policies import policy as _policy
from serving.launch_policy_server import create_policy

logger = logging.getLogger(__name__)


class PolicyCache:
    """Thread-safe LRU cache for loaded :class:`Policy` instances.

    Parameters
    ----------
    max_size:
        Maximum number of policies to keep in memory.  When exceeded the
        least-recently-used policy is evicted.
    default_prompt:
        Default language prompt forwarded to :func:`create_policy`.
    """

    def __init__(self, max_size: int = 3, *, default_prompt: str | None = None) -> None:
        self._max_size = max_size
        self._default_prompt = default_prompt
        self._cache: OrderedDict[tuple[str, str], _policy.Policy] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        config_name: str,
        checkpoint_dir: str,
        *,
        model: str | None = None,
        pytorch_device: str | None = None,
    ) -> _policy.Policy:
        """Return a cached (or freshly loaded) policy.

        Parameters
        ----------
        config_name:
            Training config name (e.g. ``"pi0_libero"``).
        checkpoint_dir:
            Checkpoint directory or GCS URI.
        model:
            Optional model variant override.
        pytorch_device:
            PyTorch device override (e.g. ``"cuda"``).
        """
        key = (config_name, checkpoint_dir)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                logger.info("Cache HIT: %s", key)
                return self._cache[key]

        # Cache miss — load outside the lock (may be slow).
        logger.info("Cache MISS: loading %s from %s …", config_name, checkpoint_dir)
        policy = create_policy(
            model=model or config_name,
            checkpoint=checkpoint_dir,
            config_name=config_name,
            default_prompt=self._default_prompt,
            pytorch_device=pytorch_device,
        )

        with self._lock:
            self._cache[key] = policy
            self._cache.move_to_end(key)
            # Evict oldest if over capacity.
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.info("Cache EVICT: %s", evicted_key)

        return policy

    def preload(
        self,
        entries: list[tuple[str, str]],
        *,
        model: str | None = None,
        pytorch_device: str | None = None,
    ) -> None:
        """Pre-load a list of ``(config_name, checkpoint_dir)`` pairs.

        Useful at server start-up to warm the cache before any client
        connects.
        """
        for config_name, checkpoint_dir in entries:
            self.get(config_name, checkpoint_dir, model=model, pytorch_device=pytorch_device)

    def clear(self) -> None:
        """Drop all cached policies."""
        with self._lock:
            self._cache.clear()
        logger.info("Cache cleared.")

    @property
    def size(self) -> int:
        """Number of policies currently cached."""
        return len(self._cache)

    def __repr__(self) -> str:
        keys = list(self._cache.keys())
        return f"PolicyCache(max_size={self._max_size}, cached={keys})"


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_global_cache: PolicyCache | None = None


def get_global_cache(max_size: int = 3, default_prompt: str | None = None) -> PolicyCache:
    """Return (and lazily create) a process-wide :class:`PolicyCache`."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PolicyCache(max_size=max_size, default_prompt=default_prompt)
    return _global_cache
