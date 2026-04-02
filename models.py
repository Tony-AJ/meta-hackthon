"""
models.py
=========
Typed Pydantic models for the Cloud Resource Allocation OpenEnv environment.

OpenEnv base classes (from openenv.core.env_server):
  - Action(BaseModel):      has 'metadata' field
  - Observation(BaseModel): has 'done', 'reward', 'metadata' fields
  - State(BaseModel):       has 'episode_id', 'step_count' fields
"""

from typing import Any, Dict, Optional
from openenv.core.env_server import Action, Observation, State


# ── Action ──────────────────────────────────────────────────────────────────

class CloudAction(Action):
    """
    Agent's resource-management decision.

    action_id mapping:
        0 = Idle           (do nothing)
        1 = Alloc CPU      (relieve CPU pressure −10%)
        2 = Alloc Memory   (relieve MEM pressure −10%)
        3 = Scale Up       (+1 container, distributes load)
        4 = Scale Down     (−1 container, saves cost)
    """
    action_id: int = 0


# ── Observation ─────────────────────────────────────────────────────────────

class CloudObservation(Observation):
    """
    Current cluster state visible to the agent.

    All resource values are normalised to [0.0, 1.0].
    Inherits 'done', 'reward', 'metadata' from Observation base.
    """
    cpu_used: float = 0.0        # fraction of CPU consumed
    mem_used: float = 0.0        # fraction of memory consumed
    containers: int = 5          # number of running containers (1–20)
    load: float = 0.0            # incoming request load
    step_count: int = 0          # current step in the episode
    message: str = ""            # human-readable status string


# ── State ───────────────────────────────────────────────────────────────────

class CloudState(State):
    """
    Internal episode metadata tracked by the environment.
    Exposed via the `state` property for monitoring / grading.

    Inherits 'episode_id', 'step_count' from State base.
    """
    task_name: str = "steady-load"
    total_reward: float = 0.0
    steps_overloaded: int = 0
    steps_wasted: int = 0
    episode_done: bool = False
