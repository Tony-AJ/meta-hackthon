"""
inference.py
============
Baseline inference script for the Cloud Resource Allocation OpenEnv environment.

MANDATORY:
  - Must be named ``inference.py`` and placed in the root directory.
  - Uses OpenAI Client for all LLM calls.
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
  - Emits structured [START], [STEP], [END] stdout logs.

Falls back to the trained DQN agent when no LLM API key is available.

Usage:
    export HF_TOKEN=hf_xxx
    export IMAGE_NAME=cloud-resource-env:latest
    python inference.py
"""

import asyncio
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import CloudEnv
from models import CloudAction, CloudObservation
from tasks import TASK_CONFIGS, grade_episode

# ── Logging setup ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── Configuration ────────────────────────────────────────────────────────────

IMAGE_NAME    = os.getenv("IMAGE_NAME")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK     = "cloud-resource-allocation"
MAX_STEPS     = 200    # safety cap (tasks have their own limits)
TEMPERATURE   = 0.3
MAX_TOKENS    = 100

ACTION_MAP = {
    "idle": 0,       "0": 0,
    "alloc_cpu": 1,  "1": 1,
    "alloc_mem": 2,  "2": 2,
    "scale_up": 3,   "3": 3,
    "scale_down": 4, "4": 4,
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an AI agent managing a cloud infrastructure cluster.
    Each step you observe:
      - cpu_used: fraction of CPU consumed (0.0-1.0)
      - mem_used: fraction of memory consumed (0.0-1.0)
      - containers: number of running containers (1-20)
      - load: incoming request load (0.0-1.0)

    You must choose ONE action by responding with ONLY the action number:
      0 = Idle (do nothing)
      1 = Alloc CPU (reduce CPU pressure by 10%)
      2 = Alloc MEM (reduce memory pressure by 10%)
      3 = Scale Up (+1 container, distributes load)
      4 = Scale Down (-1 container, saves cost)

    GOAL: Keep CPU and Memory below 90% (avoid overload) while
    minimizing waste (don't keep resources idle when load is low).
    Balance efficiency vs cost.

    Respond with ONLY a single digit (0, 1, 2, 3, or 4). Nothing else.
""").strip()


# ── DQN fallback agent ──────────────────────────────────────────────────────

_DQN_AGENT = None

def _load_dqn_fallback() -> bool:
    """Try to load the trained DQN agent as a fallback. Returns True on success."""
    global _DQN_AGENT
    try:
        from q_agent import DQNAgent
        model_path = Path("models/dqn_cloud.pth")
        if model_path.exists():
            _DQN_AGENT = DQNAgent(obs_dim=4, n_actions=5)
            _DQN_AGENT.load(str(model_path))
            logger.info("DQN fallback agent loaded from %s", model_path)
            return True
        else:
            logger.warning("No trained DQN model found at %s", model_path)
    except Exception as exc:
        logger.warning("Could not load DQN fallback: %s", exc)
    return False


def _dqn_action(obs: CloudObservation) -> int:
    """Get action from the DQN agent, or default to Idle."""
    if _DQN_AGENT is None:
        return 0
    import numpy as np
    state = np.array([obs.cpu_used, obs.mem_used, obs.containers / 20.0, obs.load],
                     dtype=np.float32)
    return _DQN_AGENT.select_action(state, greedy=True)


# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM decision-making ─────────────────────────────────────────────────────

def build_user_prompt(obs: CloudObservation, last_reward: float) -> str:
    return textwrap.dedent(f"""\
        Current cluster state (step {obs.step_count}):
          cpu_used:   {obs.cpu_used:.2f}
          mem_used:   {obs.mem_used:.2f}
          containers: {obs.containers}
          load:       {obs.load:.2f}
          last_reward: {last_reward:+.2f}

        Choose your action (0-4):
    """).strip()


def get_action_from_llm(
    client: "OpenAI",
    obs: CloudObservation,
    last_reward: float,
) -> int:
    """Call the LLM and parse its response into an action_id."""
    user_prompt = build_user_prompt(obs, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse: try to find a digit 0-4 in the response
        for char in text:
            if char in "01234":
                return int(char)

        # Fallback: try keyword matching
        text_lower = text.lower()
        for keyword, act_id in ACTION_MAP.items():
            if keyword in text_lower:
                return act_id

        return 0  # default to Idle if unparseable

    except Exception as exc:
        logger.error("LLM request failed: %s", exc)
        return 0  # safe fallback


# ── Action name for logging ──────────────────────────────────────────────────

ACTION_NAMES = {
    0: "idle()",
    1: "alloc_cpu()",
    2: "alloc_mem()",
    3: "scale_up()",
    4: "scale_down()",
}


# ── Main inference loop ─────────────────────────────────────────────────────

async def run_task(
    task_name: str,
    env: CloudEnv,
    llm_client: Optional["OpenAI"],
    use_dqn: bool = False,
) -> dict:
    """Run a single task and return results."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    # Decide agent label
    agent_label = "DQN" if use_dqn else MODEL_NAME

    # Set task via env var before reset
    os.environ["TASK_NAME"] = task_name

    log_start(task=task_name, env=BENCHMARK, model=agent_label)

    try:
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action
            if use_dqn:
                action_id = _dqn_action(obs)
            else:
                action_id = get_action_from_llm(llm_client, obs, last_reward)

            # Execute action
            result = await env.step(CloudAction(action_id=action_id))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=ACTION_NAMES.get(action_id, f"action({action_id})"),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        # Compute grader score
        state = await env.state()
        score = grade_episode(task_name, {
            "total_steps":      steps_taken,
            "steps_overloaded": state.steps_overloaded if hasattr(state, 'steps_overloaded') else 0,
            "steps_wasted":     state.steps_wasted if hasattr(state, 'steps_wasted') else 0,
        })
        success = score > 0.1

    except Exception as exc:
        logger.error("Task %s error: %s", task_name, exc)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "total_reward": sum(rewards),
    }


# ── Environment variable validation ─────────────────────────────────────────

def validate_config() -> tuple[bool, bool]:
    """
    Validate required environment variables.
    Returns (has_llm, has_dqn) indicating which agents are available.
    """
    missing = []

    if not IMAGE_NAME:
        missing.append("IMAGE_NAME")

    has_llm = bool(API_KEY)
    has_dqn = _load_dqn_fallback()

    if not has_llm and not has_dqn:
        logger.error(
            "No agent available! Set HF_TOKEN/API_KEY for LLM agent, "
            "or provide models/dqn_cloud.pth for DQN fallback."
        )
        sys.exit(1)

    if not has_llm:
        logger.warning(
            "HF_TOKEN/API_KEY not set — using trained DQN agent as fallback. "
            "Set HF_TOKEN for LLM-based inference."
        )

    if missing:
        logger.error(
            "Missing required environment variables: %s. "
            "See .env.example for details.",
            ", ".join(missing),
        )
        sys.exit(1)

    return has_llm, has_dqn


async def main() -> None:
    """Run all 3 tasks and report results."""
    has_llm, has_dqn = validate_config()

    # Only create LLM client if API key is available
    llm_client = None
    use_dqn = False
    if has_llm:
        from openai import OpenAI
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        use_dqn = True
        logger.info("Using DQN agent for inference (no LLM API key).")

    # Connect to environment via Docker image
    env = await CloudEnv.from_docker_image(IMAGE_NAME)

    results = []

    try:
        for task_name in TASK_CONFIGS:
            result = await run_task(task_name, env, llm_client, use_dqn=use_dqn)
            results.append(result)
            print("", flush=True)  # blank line between tasks
    finally:
        await env.close()

    # Summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['task']:<20s}  score={r['score']:.3f}  "
            f"steps={r['steps']}  reward={r['total_reward']:+.2f}  "
            f"{'✅' if r['success'] else '❌'}"
        )
    avg_score = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"\n  Average score: {avg_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
