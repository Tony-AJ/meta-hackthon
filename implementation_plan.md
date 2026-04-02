# Implementation Plan: Migrating to OpenEnv Spec (CORRECTED)

The goal is to transform the existing `CloudResourceEnv` (Gymnasium-based) into a fully compliant **OpenEnv** environment that meets the Meta Hackathon criteria.

---

## User Review Required

> [!IMPORTANT]
> This plan follows the **corrected** OpenEnv spec based on actual `openenv-core` library requirements:
> - Uses `@dataclass` decorators (not Pydantic)
> - Sync methods (framework handles async internally)
> - `state()` as `@property` (not async method)
> - Reward is plain `float` (no reward model)
> - Requires `EnvClient` subclass for client-side interaction
> - Server uses `create_fastapi_app()` one-liner

> [!WARNING]
> We need 3 tasks with programmatic graders scoring 0.0–1.0.

---

## Task Definitions (Finalized)

| Task | Difficulty | Description | Success Criteria |
|------|------------|-------------|------------------|
| **steady-load** | Easy | Maintain stability under constant low-to-moderate load (0.3-0.5) | Score ≥ 0.7: Keep overload < 5% of steps, avoid resource waste |
| **diurnal-cycle** | Medium | Handle predictable daily traffic patterns (sinusoidal load 0.2-0.8) | Score ≥ 0.6: Anticipate peaks with proactive scaling, keep overload < 15% |
| **spike-resilience** | Hard | Survive unpredictable flash crowds (random spikes to 0.95 load) | Score ≥ 0.5: React quickly to spikes, recover within 10 steps, overload < 25% |

### Grader Formula (Per Task)

```python
def compute_score(episode_info: dict) -> float:
    overload_ratio = steps_overloaded / total_steps
    waste_penalty = avg_idle_resource_time  # CPU or MEM < 20% when load is low
    recovery_bonus = 1.0 / (1.0 + avg_recovery_steps)  # Faster recovery = higher score

    # Base score: survival rate
    base_score = 1.0 - overload_ratio

    # Adjustments
    final_score = base_score * (1.0 - waste_penalty * 0.3) * (1.0 + recovery_bonus * 0.2)

    return max(0.0, min(1.0, final_score))  # clamp to [0.0, 1.0]
```

---

## Proposed Changes

### Phase 1: Core Environment (OpenEnv Spec)

#### [NEW] `server/open_env.py`
```python
from dataclasses import dataclass
from openenv.core.env_server import OpenEnv

@dataclass
class CloudAction:
    action_id: int  # 0-4 (Discrete)

@dataclass
class CloudObservation:
    cpu_used: float
    mem_used: float
    containers: int
    load: float

# NO reward model — reward is plain float

class CloudResourceOpenEnv(OpenEnv):
    """OpenEnv-compliant cloud resource allocation environment."""

    def reset(self, task: str | None = None) -> CloudObservation:
        # Reset internal state, select task, return initial observation
        ...

    def step(self, action: CloudAction) -> tuple[CloudObservation, float, bool, dict]:
        # Apply action, return (next_obs, reward, done, info)
        ...

    @property
    def state(self) -> dict:
        # Return current state dict (used for debugging/inspection)
        ...
```

**Key corrections from previous plan:**
- `@dataclass` instead of Pydantic models
- Sync `reset()` and `step()` (not async)
- `state` as `@property` returning `dict`
- Reward is `float`, not a model class

---

### Phase 2: Task System & Graders

#### [NEW] `server/task_config.py`
```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class TaskConfig:
    name: str
    difficulty: str  # "easy" | "medium" | "hard"
    load_pattern: str  # "constant" | "sinusoidal" | "spiky"
    episode_length: int  # 50-200 steps
    grader_fn: Callable[[dict], float]  # returns 0.0-1.0
    thresholds: dict  # task-specific thresholds

# Registry
TASKS = {
    "steady-load": TaskConfig(...),
    "diurnal-cycle": TaskConfig(...),
    "spike-resilience": TaskConfig(...),
}
```

#### [NEW] `server/tasks.py`
- Load pattern generators (constant, sinusoidal, spiky)
- Grader implementations for each task
- Deterministic scoring based on episode trajectory

---

### Phase 3: Server & Client

#### [NEW] `server/server.py`
```python
from openenv.core.env_server import create_fastapi_app
from .open_env import CloudResourceOpenEnv, CloudAction, CloudObservation

env = CloudResourceOpenEnv()

# One-liner server creation (OpenEnv provides this)
app = create_fastapi_app(env, CloudAction, CloudObservation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

**Key corrections:**
- No manual FastAPI routes
- Uses `create_fastapi_app()` from openenv-core

#### [NEW] `server/client.py` (MANDATORY — was missing)
```python
from openenv.core.env_server import EnvClient
from .open_env import CloudAction, CloudObservation

class CloudEnvClient(EnvClient):
    """Client for interacting with Cloud Resource OpenEnv server."""

    def _step_payload(self, action: CloudAction) -> dict:
        return {"action_id": action.action_id}

    def _parse_result(self, data: dict) -> tuple[CloudObservation, float, bool, dict]:
        obs = CloudObservation(
            cpu_used=data["cpu_used"],
            mem_used=data["mem_used"],
            containers=data["containers"],
            load=data["load"],
        )
        reward = data["reward"]
        done = data["done"]
        info = data.get("info", {})
        return obs, reward, done, info

    def _parse_state(self, data: dict) -> dict:
        return data
```

#### [NEW] `server/__init__.py`
```python
from .open_env import CloudResourceOpenEnv, CloudAction, CloudObservation
from .client import CloudEnvClient
from .task_config import TASKS

__all__ = [
    "CloudResourceOpenEnv",
    "CloudAction",
    "CloudObservation",
    "CloudEnvClient",
    "TASKS",
]
```

---

### Phase 4: Deployment

#### [NEW] `openenv.yaml`
```yaml
name: cloud-resource-allocation
version: 1.0.0
description: AI agent learns to allocate cloud resources (CPU, Memory, Containers)
tasks:
  - name: steady-load
    difficulty: easy
    description: Maintain stability under constant low-to-moderate load
  - name: diurnal-cycle
    difficulty: medium
    description: Handle predictable daily traffic patterns
  - name: spike-resilience
    difficulty: hard
    description: Survive unpredictable flash crowds
entrypoint: server/server.py
image: <docker-hub-repo>/cloud-resource-env:latest
```

#### [MODIFY] `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install dependencies including openenv-core
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Environment variable to switch between server and demo modes
ENV MODE=server

# Run OpenEnv server by default, or Gradio demo with MODE=demo
CMD if [ "$MODE" = "server" ]; then python server/server.py; else python app.py; fi
```

---

### Phase 5: Baseline Inference

#### [NEW] `inference.py`
```python
import os
import asyncio
from openai import OpenAI
from server.client import CloudEnvClient

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME", "cloud-resource-env:latest")

TASKS = ["steady-load", "diurnal-cycle", "spike-resilience"]
MAX_STEPS = 200

# Logging functions (strict format per hackathon spec)
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

async def run_task(task_name: str) -> float:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env_client = CloudEnvClient.from_docker_image(IMAGE_NAME, task=task_name)

    log_start(task=task_name, env="cloud-resource", model=MODEL_NAME)

    obs = await env_client.reset()
    rewards = []

    for step in range(1, MAX_STEPS + 1):
        # Get action from LLM
        action = get_llm_action(client, obs, step)

        obs, reward, done, info = await env_client.step(CloudAction(action_id=action))
        rewards.append(reward)

        error = info.get("error")
        log_step(step=step, action=str(action), reward=reward, done=done, error=error)

        if done:
            break

    success = sum(rewards) > 0  # Simple success criterion
    log_end(success=success, steps=step, rewards=rewards)

    return sum(rewards) / len(rewards) if rewards else 0.0

async def main():
    for task in TASKS:
        score = await run_task(task)
        print(f"Task {task}: {score:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Phase 6: Documentation

#### [MODIFY] `README.md`
Contents:
- Environment description and motivation (cloud resource allocation = Kubernetes autoscaling)
- Observation space: 4-dim vector (CPU, MEM, containers, load)
- Action space: `Discrete(5)` with meanings
- Task descriptions with difficulty levels
- Setup and usage instructions
- Baseline scores table (filled after inference)
- Required environment variables

#### [NEW] `.env.example`
```bash
# OpenAI-compatible API (required for inference.py)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token_here

# Docker image name for local testing
IMAGE_NAME=cloud-resource-env:latest

# Server mode (server or demo)
MODE=server
```

---

## Resource Constraints Check

| Constraint | Limit | Our Implementation |
|------------|-------|-------------------|
| vCPU | 2 | Lightweight env, no heavy computation |
| Memory | 8GB | Minimal state, ~100MB footprint |
| Runtime | <20min | Episodes capped at 200 steps, ~2-3min per task |
| Total inference | ~10min for 3 tasks | Well under limit |

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Model Selection for baseline | Rule-based heuristic first, then LLM agent |
| Grader Thresholds | Easy: 0.7+, Medium: 0.5+, Hard: 0.3+ |
| Server vs Gradio | Single Dockerfile with `MODE` env var switch |
| Directory structure | `server/` subdirectory for OpenEnv files |

---

## Verification Plan

### Automated Tests (Pre-Submission Checklist)
- [ ] `openenv validate` passes
- [ ] `docker build .` completes successfully
- [ ] `python inference.py` runs and produces valid `[START]/[STEP]/[END]` logs
- [ ] HF Space responds to `/reset` with HTTP 200
- [ ] All 3 tasks return scores in 0.0–1.0 range
- [ ] Runtime < 20 minutes on vcpu=2, mem=8gb

### Manual Verification
- [ ] `MODE=demo` runs Gradio app
- [ ] `MODE=server` runs OpenEnv server
- [ ] Original `q_agent.py --train` still works with `cloud_env.py`
- [ ] Baseline scores are reproducible

---

## File Structure (Final)

```
meta-hackthon/
├── server/
│   ├── __init__.py           # NEW: exports
│   ├── open_env.py           # NEW: OpenEnv environment
│   ├── client.py             # NEW: EnvClient subclass (MANDATORY)
│   ├── task_config.py        # NEW: Task configurations
│   ├── tasks.py              # NEW: Graders and load patterns
│   └── server.py             # NEW: FastAPI server (one-liner)
├── cloud_env.py              # Original Gymnasium env (keep for DQN training)
├── app.py                    # Gradio demo (unchanged)
├── q_agent.py                # DQN training (unchanged)
├── inference.py              # NEW: Baseline LLM agent script
├── openenv.yaml              # NEW: OpenEnv manifest
├── Dockerfile                # MODIFIED: MODE switch
├── requirements.txt          # MODIFIED: Add openenv-core
├── README.md                 # MODIFIED: Full documentation
└── .env.example              # NEW: Env var template
```

---

## Timeline Estimate (Medium Effort)

| Phase | Files | Estimated Time |
|-------|-------|----------------|
| Phase 1: Core Env | `server/open_env.py` | 1 hour |
| Phase 2: Tasks | `server/task_config.py`, `server/tasks.py` | 1 hour |
| Phase 3: Server/Client | `server/server.py`, `server/client.py`, `server/__init__.py` | 1 hour |
| Phase 4: Deployment | `openenv.yaml`, `Dockerfile` | 30 min |
| Phase 5: Inference | `inference.py` | 1 hour |
| Phase 6: Docs | `README.md`, `.env.example` | 30 min |
| Testing & Validation | All verification checks | 1 hour |
| **Total** | | **~5-6 hours** |
