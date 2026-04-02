"""
server/cloud_environment.py
===========================
OpenEnv-compliant environment for Cloud Resource Allocation.

Subclasses ``Environment`` from ``openenv.core.env_server`` and implements:

    reset(seed, episode_id)  → CloudObservation
    step(action, timeout_s)  → CloudObservation  (with .done and .reward set)
    state (property)         → CloudState

The Observation base class has built-in `done`, `reward`, `metadata` fields.
Physics / dynamics are ported from the original ``cloud_env.py``.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Any, Optional

import numpy as np
from openenv.core.env_server import Environment

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CloudAction, CloudObservation, CloudState
from tasks import TASK_CONFIGS, DEFAULT_TASK, generate_load, grade_episode


# ────────────────────────────────────────────────────────────────────────────
# Constants  (mirrored from cloud_env.py)
# ────────────────────────────────────────────────────────────────────────────

MAX_CONTAINERS   = 20
MIN_CONTAINERS   = 1

OVERLOAD_THRESHOLD   = 0.90
WASTE_THRESHOLD      = 0.20
OVERLOAD_GRACE_STEPS = 5

EFFICIENCY_BONUS   =  1.0
OVERLOAD_PENALTY   = -3.0
WASTE_PENALTY_VAL  = -0.5
IDLE_STEP_PENALTY  = -0.1
COST_PENALTY_RATE  =  0.05
SCALING_PENALTY    =  0.10

ACTION_NAMES = {
    0: "Idle",
    1: "Alloc CPU",
    2: "Alloc MEM",
    3: "Scale Up",
    4: "Scale Down",
}


class CloudResourceEnvironment(Environment[CloudAction, CloudObservation, CloudState]):
    """OpenEnv Cloud Resource Allocation environment."""

    def __init__(self):
        super().__init__()

        # Determine which task to run (env var or default)
        task_name = os.environ.get("TASK_NAME", DEFAULT_TASK)
        if task_name not in TASK_CONFIGS:
            task_name = DEFAULT_TASK

        self._task_name = task_name
        self._task_cfg  = TASK_CONFIGS[task_name]

        # RNG
        seed = int(os.environ.get("ENV_SEED", "42"))
        self._rng = np.random.default_rng(seed)

        # Internal mutable state
        self._cpu_used: float = 0.0
        self._mem_used: float = 0.0
        self._containers: int = 5
        self._load: float     = 0.0
        self._step_count: int = 0
        self._consecutive_overload: int = 0

        # Episode tracking
        self._episode_id: str = str(uuid.uuid4())
        self._total_reward: float = 0.0
        self._steps_overloaded: int = 0
        self._steps_wasted: int = 0
        self._done: bool = False

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CloudObservation:
        """Initialise a new episode and return the first observation."""
        # Re-read task config in case env var changed
        task_name = os.environ.get("TASK_NAME", self._task_name)
        if task_name not in TASK_CONFIGS:
            task_name = DEFAULT_TASK
        self._task_name = task_name
        self._task_cfg  = TASK_CONFIGS[task_name]

        # Set seed
        actual_seed = seed if seed is not None else int(os.environ.get("ENV_SEED", "42"))
        self._rng = np.random.default_rng(actual_seed)

        # Episode ID
        self._episode_id = episode_id or str(uuid.uuid4())

        # Randomise initial conditions
        self._containers = int(self._rng.integers(3, 8))
        self._load       = float(self._rng.uniform(0.1, 0.5))
        self._cpu_used   = float(self._rng.uniform(0.1, 0.4))
        self._mem_used   = float(self._rng.uniform(0.1, 0.4))
        self._step_count = 0
        self._consecutive_overload = 0

        # Reset tracking
        self._total_reward = 0.0
        self._steps_overloaded = 0
        self._steps_wasted = 0
        self._done = False

        return CloudObservation(
            cpu_used=round(self._cpu_used, 4),
            mem_used=round(self._mem_used, 4),
            containers=self._containers,
            load=round(self._load, 4),
            step_count=0,
            message=f"Episode reset. Task: {task_name} ({self._task_cfg.difficulty})",
            done=False,
            reward=None,
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        action: CloudAction,
        timeout_s: Optional[float] = None,
    ) -> CloudObservation:
        """Execute one action and return the resulting observation."""
        act = action.action_id
        if act not in ACTION_NAMES:
            act = 0  # default to Idle for invalid actions

        self._step_count += 1

        # ── Apply action ─────────────────────────────────────────────────
        if act == 0:    # Idle
            pass
        elif act == 1:  # Alloc CPU
            self._cpu_used = max(0.0, self._cpu_used - 0.10)
        elif act == 2:  # Alloc MEM
            self._mem_used = max(0.0, self._mem_used - 0.10)
        elif act == 3:  # Scale Up
            self._containers = min(MAX_CONTAINERS, self._containers + 1)
            self._cpu_used   = max(0.0, self._cpu_used - 0.05)
            self._mem_used   = max(0.0, self._mem_used - 0.05)
        elif act == 4:  # Scale Down
            self._containers = max(MIN_CONTAINERS, self._containers - 1)
            self._cpu_used   = min(1.0, self._cpu_used + 0.08)
            self._mem_used   = min(1.0, self._mem_used + 0.08)

        # ── Environment dynamics ─────────────────────────────────────────
        self._update_dynamics()

        # ── Reward ───────────────────────────────────────────────────────
        reward = self._compute_reward(act)
        self._total_reward += reward

        # ── Track overload / waste for grader ─────────────────────────────
        is_overloaded = (
            self._cpu_used > OVERLOAD_THRESHOLD
            or self._mem_used > OVERLOAD_THRESHOLD
        )
        is_wasted = (
            self._cpu_used < WASTE_THRESHOLD
            and self._mem_used < WASTE_THRESHOLD
            and self._load < 0.3
        )
        if is_overloaded:
            self._steps_overloaded += 1
            self._consecutive_overload += 1
        else:
            self._consecutive_overload = 0
        if is_wasted:
            self._steps_wasted += 1

        # ── Termination ──────────────────────────────────────────────────
        terminated = (
            self._step_count >= self._task_cfg.episode_length
            or self._consecutive_overload >= OVERLOAD_GRACE_STEPS
        )
        truncated = bool(
            self._cpu_used >= 1.0
            and self._mem_used >= 1.0
            and self._containers <= MIN_CONTAINERS
        )
        done = terminated or truncated
        self._done = done

        # Build status message
        status = "running"
        if done:
            score = grade_episode(self._task_name, {
                "total_steps":      self._step_count,
                "steps_overloaded": self._steps_overloaded,
                "steps_wasted":     self._steps_wasted,
            })
            status = f"done | score={score:.3f}"

        return CloudObservation(
            cpu_used=round(self._cpu_used, 4),
            mem_used=round(self._mem_used, 4),
            containers=self._containers,
            load=round(self._load, 4),
            step_count=self._step_count,
            message=f"Action: {ACTION_NAMES[act]} | Reward: {reward:+.3f} | {status}",
            done=done,
            reward=reward,
        )

    # ── state property ───────────────────────────────────────────────────────

    @property
    def state(self) -> CloudState:
        """Return current episode metadata."""
        return CloudState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_reward=round(self._total_reward, 4),
            steps_overloaded=self._steps_overloaded,
            steps_wasted=self._steps_wasted,
            episode_done=self._done,
        )

    # ── Private helpers (ported from cloud_env.py) ───────────────────────────

    def _update_dynamics(self) -> None:
        """Stochastic, load-dependent resource drift."""
        self._load = generate_load(
            self._task_name,
            self._step_count,
            self._task_cfg.episode_length,
            self._rng,
        )

        # Pressure per container
        container_ratio = self._containers / MAX_CONTAINERS
        pressure = float(np.clip(
            self._load / max(container_ratio, 0.05), 0.0, 1.0
        ))

        self._cpu_used = float(np.clip(
            self._cpu_used + self._rng.uniform(-0.03, 0.03) + 0.05 * pressure,
            0.0, 1.0,
        ))
        self._mem_used = float(np.clip(
            self._mem_used + self._rng.uniform(-0.02, 0.02) + 0.04 * pressure,
            0.0, 1.0,
        ))

    def _compute_reward(self, action: int) -> float:
        """Multi-component reward signal."""
        overloaded = (
            self._cpu_used > OVERLOAD_THRESHOLD
            or self._mem_used > OVERLOAD_THRESHOLD
            or self._containers <= MIN_CONTAINERS
        )
        wasted = (
            self._cpu_used < WASTE_THRESHOLD
            and self._mem_used < WASTE_THRESHOLD
            and self._load < 0.3
        )

        if overloaded:
            reward = OVERLOAD_PENALTY
        elif wasted:
            reward = WASTE_PENALTY_VAL
        else:
            efficiency = (
                1.0
                - abs(self._cpu_used - self._load)
                - abs(self._mem_used - self._load) * 0.5
            )
            reward = EFFICIENCY_BONUS * float(np.clip(efficiency, 0.0, 1.0))

        # Cost penalty: more containers = higher cloud cost
        reward -= COST_PENALTY_RATE * self._containers

        # Scaling penalty: discourage erratic scaling
        if action in (3, 4):
            reward -= SCALING_PENALTY

        # Idle penalty: discourage permanent passivity
        if action == 0:
            reward += IDLE_STEP_PENALTY

        return reward
