"""Replay Buffer for SuperMimic Quadrotor Training.

Provides:
  - Circular GPU / JAX array storage for transitions: (s, s1, a, a_prev, xi, r, f, traj_id, ep_step, done).
  - Multi-step window sampling (k-step windows) for autoregressive World Model training (Eq. 1).
  - Imagination starting-state sampling with observation history for policy BPTT (Eq. 2).
  - Seedable from offline expert trajectory corpus.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

STATE_DIM = 22
ACT_DIM = 4
XI_DIM = 10


class SuperMimicReplayBuffer:
    """Fixed-capacity circular replay buffer stored in RAM / device memory."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        # Preallocate NumPy storage (fast slice and window extraction)
        self.s = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.s1 = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.a_prev = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.xi = np.zeros((capacity, XI_DIM), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.f = np.zeros(capacity, dtype=np.int32)
        self.traj_id = np.zeros(capacity, dtype=np.int32)
        self.ep_step = np.zeros(capacity, dtype=np.int32)
        self.done = np.zeros(capacity, dtype=bool)

    def add_batch(self,
                  s: np.ndarray,
                  s1: np.ndarray,
                  a: np.ndarray,
                  a_prev: np.ndarray,
                  xi: np.ndarray,
                  r: np.ndarray,
                  f: np.ndarray,
                  traj_id: np.ndarray,
                  ep_step: np.ndarray,
                  done: np.ndarray):
        """Add a batch of transitions to the ring buffer."""
        n = len(s)
        if n == 0:
            return

        indices = (self.ptr + np.arange(n)) % self.capacity
        self.s[indices] = s
        self.s1[indices] = s1
        self.a[indices] = a
        self.a_prev[indices] = a_prev
        self.xi[indices] = xi
        self.r[indices] = r
        self.f[indices] = f
        self.traj_id[indices] = traj_id
        self.ep_step[indices] = ep_step
        self.done[indices] = done

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample_windows(self, batch_size: int, k: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Sample k-step contiguous windows for autoregressive World Model training.
        
        Returns dict of arrays with shape (batch_size, k, ...), plus mask of shape (batch_size, k).
        A step is masked out if an episode boundary occurred earlier in the window.
        """
        if self.size < k:
            raise ValueError(f"Buffer has {self.size} transitions, cannot sample {k}-step windows.")

        # Candidate starting indices
        max_idx = self.size - k if self.size < self.capacity else self.capacity
        start_indices = rng.integers(0, max_idx, size=batch_size)

        # Offsets for window of length k
        offsets = np.arange(k)
        win_indices = (start_indices[:, None] + offsets[None, :]) % self.capacity

        s_win = self.s[win_indices]
        s1_win = self.s1[win_indices]
        a_win = self.a[win_indices]
        xi_win = self.xi[win_indices]
        done_win = self.done[win_indices]
        traj_win = self.traj_id[win_indices]

        # Valid mask: step j is valid if no 'done' occurred at 0..j-1 and all steps belong to same trajectory
        same_traj = (traj_win == traj_win[:, 0:1])
        done_float = done_win.astype(np.float32)
        no_prior_done = np.cumprod(np.concatenate([np.ones((batch_size, 1), dtype=np.float32), 1.0 - done_float[:, :-1]], axis=-1), axis=-1)
        mask = (same_traj.astype(np.float32) * no_prior_done)

        return {
            "s": s_win,
            "s1": s1_win,
            "a": a_win,
            "xi": xi_win,
            "mask": mask,  # (batch_size, k)
        }

    def sample_starts(self, batch_size: int, hist_len: int, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Sample starting states for imagination rollouts along with observation history."""
        if self.size == 0:
            raise ValueError("Buffer is empty.")

        indices = rng.integers(0, self.size, size=batch_size)
        start = {
            "s": self.s[indices],
            "a_prev": self.a_prev[indices],
            "xi": self.xi[indices],
            "f": self.f[indices],
            "traj_id": self.traj_id[indices],
        }

        # Backtrack within same trajectory up to hist_len steps
        ep_steps = self.ep_step[indices]
        back_offsets = np.arange(hist_len - 1, -1, -1)  # hist-1 down to 0
        # Clamp back offsets so we don't cross episode beginning
        clamped_offsets = np.minimum(back_offsets[None, :], ep_steps[:, None])
        hist_indices = (indices[:, None] - clamped_offsets) % self.capacity

        hist = {
            "s": self.s[hist_indices],
            "a_prev": self.a_prev[hist_indices],
            "xi": self.xi[hist_indices],
            "f": self.f[hist_indices],
        }
        return start, hist
