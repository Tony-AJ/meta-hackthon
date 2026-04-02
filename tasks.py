"""
tasks.py
========
Task definitions, load-pattern generators, and graders for the
Cloud Resource Allocation OpenEnv environment.

Three tasks (easy → medium → hard):
  1. steady-load       constant moderate load
  2. diurnal-cycle     sinusoidal daily traffic
  3. spike-resilience  random flash-crowd spikes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# 1.  Task configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single task."""
    name: str
    difficulty: str            # "easy", "medium", "hard"
    description: str
    episode_length: int        # max steps per episode
    load_fn_name: str          # key into _LOAD_GENERATORS


TASK_CONFIGS: dict[str, TaskConfig] = {
    "steady-load": TaskConfig(
        name="steady-load",
        difficulty="easy",
        description="Maintain stability under constant low-to-moderate load (0.3–0.5).",
        episode_length=100,
        load_fn_name="steady",
    ),
    "diurnal-cycle": TaskConfig(
        name="diurnal-cycle",
        difficulty="medium",
        description="Handle predictable daily traffic patterns (sinusoidal 0.2–0.8).",
        episode_length=150,
        load_fn_name="diurnal",
    ),
    "spike-resilience": TaskConfig(
        name="spike-resilience",
        difficulty="hard",
        description="Survive unpredictable flash-crowd spikes (random surges to 0.95).",
        episode_length=200,
        load_fn_name="spiky",
    ),
}

DEFAULT_TASK = "steady-load"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Load-pattern generators
# ────────────────────────────────────────────────────────────────────────────

def _steady_load(step: int, total_steps: int, rng: np.random.Generator) -> float:
    """Constant load in [0.3, 0.5] with small Gaussian noise."""
    base = 0.40
    noise = float(rng.normal(0, 0.03))
    return float(np.clip(base + noise, 0.1, 0.6))


def _diurnal_load(step: int, total_steps: int, rng: np.random.Generator) -> float:
    """Sinusoidal load simulating a daily traffic cycle."""
    t = step / max(total_steps, 1)
    base = 0.5 + 0.3 * math.sin(2 * math.pi * t)
    noise = float(rng.normal(0, 0.04))
    return float(np.clip(base + noise, 0.05, 0.95))


def _spiky_load(step: int, total_steps: int, rng: np.random.Generator) -> float:
    """Mostly moderate load with random flash-crowd spikes."""
    base = 0.35 + float(rng.normal(0, 0.05))
    # 15 % chance of a flash-crowd spike
    if rng.random() < 0.15:
        spike = float(rng.uniform(0.35, 0.60))
        base += spike
    return float(np.clip(base, 0.05, 0.98))


_LOAD_GENERATORS: dict[str, Callable] = {
    "steady":  _steady_load,
    "diurnal": _diurnal_load,
    "spiky":   _spiky_load,
}


def generate_load(
    task_name: str,
    step: int,
    total_steps: int,
    rng: np.random.Generator,
) -> float:
    """
    Return the request load for the current step given the active task.
    Falls back to steady-load if the task name is unknown.
    """
    cfg = TASK_CONFIGS.get(task_name, TASK_CONFIGS[DEFAULT_TASK])
    fn = _LOAD_GENERATORS[cfg.load_fn_name]
    return fn(step, total_steps, rng)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Graders  (deterministic, 0.0 – 1.0)
# ────────────────────────────────────────────────────────────────────────────

def grade_episode(task_name: str, trajectory: dict) -> float:
    """
    Score an episode trajectory on a 0.0 – 1.0 scale.

    Parameters
    ----------
    task_name : str
        One of the registered task names.
    trajectory : dict
        Must contain:
          - total_steps       (int)
          - steps_overloaded  (int)  steps where CPU or MEM > 90 %
          - steps_wasted      (int)  steps where CPU & MEM < 20 % and load < 30 %

    Returns
    -------
    float   in [0.0, 1.0]
    """
    total = max(trajectory.get("total_steps", 1), 1)
    overloaded = trajectory.get("steps_overloaded", 0)
    wasted = trajectory.get("steps_wasted", 0)

    overload_ratio = overloaded / total
    waste_ratio = wasted / total

    # Base score: fraction of time NOT in overload
    base_score = 1.0 - overload_ratio

    # Waste penalty: up to –0.3 for fully idle episodes
    waste_penalty = waste_ratio * 0.3

    score = base_score - waste_penalty
    return max(0.0, min(1.0, score))
