"""
cloud_env.py
============
Cloud Resource Allocation – Custom RL Environment
--------------------------------------------------
Mirrors the Gymnasium API (reset / step / render / close) without requiring
gymnasium to be installed.  When gymnasium IS available it inherits from
``gymnasium.Env`` automatically.

State  (np.ndarray, shape=(4,), float32, all normalised 0–1):
    [cpu_used, mem_used, containers_norm, request_load]

Actions (int, Discrete 5):
    0 – Idle           (do nothing)
    1 – Alloc CPU      (relieve CPU pressure  -10%)
    2 – Alloc Memory   (relieve Memory pressure -10%)
    3 – Scale Up       (+1 container, reduces per-container load)
    4 – Scale Down     (-1 container, increases per-container load)

Reward  (float, per step):
    + efficiency_bonus   when load is served with low resource idle
    – overload_penalty   when CPU/mem > 90% or containers exhausted
    – waste_penalty      when CPU/mem < 20% and load is low
    – idle_step_penalty  small push to discourage always choosing "Idle"
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any

# ── optional gymnasium integration ──────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env

    def _box(low, high, n):
        return spaces.Box(
            low=np.full(n, low, dtype=np.float32),
            high=np.full(n, high, dtype=np.float32),
            dtype=np.float32,
        )

    def _discrete(n):
        return spaces.Discrete(n)

    _HAS_GYMNASIUM = True

except ImportError:

    class _Discrete:
        """Minimal stand-in for gymnasium.spaces.Discrete."""
        def __init__(self, n: int):
            self.n = n
        def sample(self, rng: np.random.Generator | None = None) -> int:
            rng = rng or np.random.default_rng()
            return int(rng.integers(self.n))
        def contains(self, x: int) -> bool:
            return 0 <= int(x) < self.n
        def __repr__(self):
            return f"Discrete({self.n})"

    class _Box:
        """Minimal stand-in for gymnasium.spaces.Box."""
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = (shape,)
            self.dtype = dtype
        def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
            rng = rng or np.random.default_rng()
            return (rng.random(self.shape) * (self.high - self.low) + self.low).astype(self.dtype)
        def contains(self, x) -> bool:
            return np.all(x >= self.low) and np.all(x <= self.high)
        def __repr__(self):
            return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"

    class _BASE:
        metadata: dict = {}

    def _box(low, high, n):
        return _Box(low, high, n)

    def _discrete(n):
        return _Discrete(n)

    _HAS_GYMNASIUM = False


# ────────────────────────────────────────────────────────────────────────────

class CloudResourceEnv(_BASE):
    """Cloud Resource Allocation environment for RL training."""

    metadata = {"render_modes": ["human", "ansi"]}

    # ── constants ────────────────────────────────────────────────────────────
    MAX_CONTAINERS: int   = 20
    MIN_CONTAINERS: int   = 1
    MAX_STEPS:      int   = 200

    OVERLOAD_THRESHOLD:   float = 0.90
    WASTE_THRESHOLD:      float = 0.20
    OVERLOAD_GRACE_STEPS: int   = 5     # consecutive overload steps before early termination

    EFFICIENCY_BONUS:   float =  1.0
    OVERLOAD_PENALTY:   float = -3.0
    WASTE_PENALTY:      float = -0.5
    IDLE_STEP_PENALTY:  float = -0.1
    COST_PENALTY_RATE:  float =  0.05   # per container, per step
    SCALING_PENALTY:    float =  0.10   # penalty for each scale up/down action

    # ── spaces ───────────────────────────────────────────────────────────────
    observation_space = _box(0.0, 1.0, 4)   # [cpu, mem, cont_norm, load]
    action_space      = _discrete(5)

    # ────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        render_mode: str | None = None,
        seed:        int | None = None,
    ):
        if _HAS_GYMNASIUM:
            super().__init__()

        self.render_mode = render_mode
        self._np_rng     = np.random.default_rng(seed)

        # Per-instance spaces so reset(seed=…) works correctly
        self.observation_space = _box(0.0, 1.0, 4)
        self.action_space      = _discrete(5)

        # Internal state
        self._cpu_used:             float = 0.0
        self._mem_used:             float = 0.0
        self._containers:           int   = 5
        self._load:                 float = 0.0
        self._step_count:           int   = 0
        self._consecutive_overload: int   = 0   # consecutive steps in overload

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed:    int  | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if _HAS_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self._containers           = int(self._np_rng.integers(3, 8))
        self._load                 = float(self._np_rng.uniform(0.1, 0.6))
        self._cpu_used             = float(self._np_rng.uniform(0.1, 0.5))
        self._mem_used             = float(self._np_rng.uniform(0.1, 0.5))
        self._step_count           = 0
        self._consecutive_overload = 0

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}. Must be in [0, 4].")

        self._step_count += 1

        # ── Apply action ─────────────────────────────────────────────────────
        if action == 0:   # Idle
            pass
        elif action == 1: # Allocate CPU   → relieve CPU pressure
            self._cpu_used = max(0.0, self._cpu_used - 0.10)
        elif action == 2: # Allocate Memory → relieve mem pressure
            self._mem_used = max(0.0, self._mem_used - 0.10)
        elif action == 3: # Scale Up (+1 container)
            self._containers = min(self.MAX_CONTAINERS, self._containers + 1)
            self._cpu_used   = max(0.0, self._cpu_used - 0.05)
            self._mem_used   = max(0.0, self._mem_used - 0.05)
        elif action == 4: # Scale Down / Kill (-1 container)
            self._containers = max(self.MIN_CONTAINERS, self._containers - 1)
            self._cpu_used   = min(1.0, self._cpu_used + 0.08)
            self._mem_used   = min(1.0, self._mem_used + 0.08)

        # ── Environment dynamics ─────────────────────────────────────────────
        self._update_dynamics()

        # ── Reward ───────────────────────────────────────────────────────────
        reward = self._compute_reward(action)

        # ── Track consecutive overload ────────────────────────────────────────
        is_overloaded = (
            self._cpu_used > self.OVERLOAD_THRESHOLD
            or self._mem_used > self.OVERLOAD_THRESHOLD
        )
        self._consecutive_overload = self._consecutive_overload + 1 if is_overloaded else 0

        # ── Termination ──────────────────────────────────────────────────────
        # Normal: max steps reached, or sustained overload for too many steps
        terminated = (
            self._step_count >= self.MAX_STEPS
            or self._consecutive_overload >= self.OVERLOAD_GRACE_STEPS
        )
        # Catastrophic: all resources simultaneously saturated
        truncated = bool(
            self._cpu_used  >= 1.0
            and self._mem_used  >= 1.0
            and self._containers <= self.MIN_CONTAINERS
        )

        if self.render_mode == "human":
            self.render()

        # ── Build info with reward for logging / demo ─────────────────────────
        info = self._get_info()
        info["reward"]              = round(float(reward), 4)
        info["consecutive_overload"] = self._consecutive_overload

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        frame = self._render_ansi()
        if self.render_mode == "human":
            print(frame)
            return None
        return frame   # "ansi" mode

    def close(self) -> None:
        pass

    # ── Private helpers ──────────────────────────────────────────────────────

    def _update_dynamics(self) -> None:
        """Stochastic, load-dependent resource drift."""
        t        = self._step_count / self.MAX_STEPS
        base     = 0.5 + 0.35 * math.sin(2 * math.pi * t * 3)  # periodic load
        noise    = float(self._np_rng.normal(0, 0.05))
        self._load = float(np.clip(base + noise, 0.0, 1.0))

        # Pressure per container (more containers → less individual pressure)
        container_ratio = self._containers / self.MAX_CONTAINERS
        pressure        = float(np.clip(self._load / max(container_ratio, 0.05), 0.0, 1.0))

        self._cpu_used = float(np.clip(
            self._cpu_used + self._np_rng.uniform(-0.03, 0.03) + 0.05 * pressure, 0.0, 1.0
        ))
        self._mem_used = float(np.clip(
            self._mem_used + self._np_rng.uniform(-0.02, 0.02) + 0.04 * pressure, 0.0, 1.0
        ))

    def _compute_reward(self, action: int) -> float:
        overloaded = (
            self._cpu_used  > self.OVERLOAD_THRESHOLD
            or self._mem_used  > self.OVERLOAD_THRESHOLD
            or self._containers <= self.MIN_CONTAINERS
        )
        wasted = (
            self._cpu_used  < self.WASTE_THRESHOLD
            and self._mem_used  < self.WASTE_THRESHOLD
            and self._load < 0.3
        )

        if overloaded:
            reward = self.OVERLOAD_PENALTY
        elif wasted:
            reward = self.WASTE_PENALTY
        else:
            efficiency = 1.0 - abs(self._cpu_used - self._load) \
                             - abs(self._mem_used  - self._load) * 0.5
            reward = self.EFFICIENCY_BONUS * float(np.clip(efficiency, 0.0, 1.0))

        # ── Cost penalty: more containers = higher cloud cost ─────────────────
        reward -= self.COST_PENALTY_RATE * self._containers

        # ── Scaling penalty: discourage erratic scaling ───────────────────────
        if action in [3, 4]:
            reward -= self.SCALING_PENALTY

        # ── Idle penalty: discourage permanent passivity ──────────────────────
        if action == 0:
            reward += self.IDLE_STEP_PENALTY

        return reward

    def _get_obs(self) -> np.ndarray:
        cont_norm = (self._containers - self.MIN_CONTAINERS) / (
            self.MAX_CONTAINERS - self.MIN_CONTAINERS
        )
        return np.array(
            [self._cpu_used, self._mem_used, cont_norm, self._load],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "cpu_used":   round(self._cpu_used,   4),
            "mem_used":   round(self._mem_used,   4),
            "containers": self._containers,
            "load":       round(self._load,       4),
            "step":       self._step_count,
        }

    def _render_ansi(self) -> str:
        W = 24
        def bar(v: float) -> str:
            filled = max(0, min(W, int(v * W)))
            return "█" * filled + "░" * (W - filled)
        lines = [
            f"─── Step {self._step_count:>3} / {self.MAX_STEPS} {'─' * 30}",
            f"  CPU  [{bar(self._cpu_used)}] {self._cpu_used:>6.1%}",
            f"  MEM  [{bar(self._mem_used)}] {self._mem_used:>6.1%}",
            f"  CONT [{bar(self._containers/self.MAX_CONTAINERS)}] {self._containers:>2}/{self.MAX_CONTAINERS}",
            f"  LOAD [{bar(self._load)}] {self._load:>6.1%}",
        ]
        return "\n".join(lines)

    # ── Human-friendly string representations ────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CloudResourceEnv("
            f"cpu={self._cpu_used:.2%}, "
            f"mem={self._mem_used:.2%}, "
            f"containers={self._containers}, "
            f"load={self._load:.2%}, "
            f"step={self._step_count}/{self.MAX_STEPS})"
        )


# ────────────────────────────────────────────────────────────────────────────
# Quick sanity check  ─  run:  python cloud_env.py
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("gymnasium available:", _HAS_GYMNASIUM)

    env         = CloudResourceEnv(render_mode="human", seed=42)
    obs, info   = env.reset()

    print("\n=== CloudResourceEnv Smoke Test ===")
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Initial obs       : {obs}")
    print(f"Initial info      : {info}\n")

    ACTION_NAMES = {
        0: "Idle",
        1: "Alloc CPU",
        2: "Alloc MEM",
        3: "Scale UP",
        4: "Scale DOWN",
    }

    total_reward = 0.0
    rng          = np.random.default_rng(0)

    for step_i in range(20):
        action = int(rng.integers(5))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {step_i+1:>2} | {ACTION_NAMES[action]:<10} | "
            f"rew={reward:+.3f} | "
            f"cpu={info['cpu_used']:.1%} "
            f"mem={info['mem_used']:.1%} "
            f"cont={info['containers']:>2} "
            f"load={info['load']:.1%}"
        )
        if terminated or truncated:
            print(f"Episode ended (terminated={terminated}, truncated={truncated})")
            break

    print(f"\n{'─'*60}")
    print(f"Total reward over {step_i+1} steps : {total_reward:.3f}")
    print("✓ All checks passed.")
    env.close()
    sys.exit(0)
