"""
app.py — Hugging Face Spaces (Gradio) demo
==========================================
Cloud Resource Allocation – DevOps AI Agent

Three tabs:
  1. 🎮 Interactive  – step through one episode manually
  2. 🚀 Auto-Run     – run a full episode and plot results
  3. ℹ️  About       – problem description

Deploy to Hugging Face Spaces (SDK: Gradio, app_file: app.py)
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
# Use non-interactive Agg backend — required for headless server / HF Spaces
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloud_env import CloudResourceEnv

# ── Logging setup ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── try to load a trained agent (optional) ───────────────────────────────────
_AGENT = None
try:
    from q_agent import DQNAgent
    _MODEL_PATH = Path("models/dqn_cloud.pth")
    if _MODEL_PATH.exists():
        _AGENT = DQNAgent(obs_dim=4, n_actions=5)
        _AGENT.load(str(_MODEL_PATH))
        logger.info("Trained DQN agent loaded.")
    else:
        logger.warning("No trained model found – using random agent for demo.")
except Exception as e:
    logger.error("Could not load DQN agent: %s", e)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "⏸ Idle",
    1: "🖥 Alloc CPU",
    2: "💾 Alloc MEM",
    3: "⬆ Scale UP",
    4: "⬇ Scale DOWN",
}

def _pick_action(obs: np.ndarray) -> int:
    """Use trained agent if available, otherwise random."""
    if _AGENT is not None:
        return _AGENT.select_action(obs, greedy=True)
    return random.randrange(5)


def _bar(value: float, width: int = 20) -> str:
    filled = max(0, min(width, int(value * width)))
    return "█" * filled + "░" * (width - filled)


def _state_table(info: dict, reward: float | None = None) -> str:
    rows = [
        f"| Resource     | Value      | Bar                       |",
        f"|--------------|------------|---------------------------|",
        f"| 🖥 CPU Used   | {info['cpu_used']:.1%}    | {_bar(info['cpu_used'])}  |",
        f"| 💾 MEM Used   | {info['mem_used']:.1%}    | {_bar(info['mem_used'])}  |",
        f"| 📦 Containers | {info['containers']:>2} / 20  | {_bar(info['containers']/20)}  |",
        f"| 📈 Load       | {info['load']:.1%}    | {_bar(info['load'])}  |",
        f"| 🔄 Step       | {info['step']}         |                           |",
    ]
    if reward is not None:
        rows.append(f"| 🏅 Reward     | {reward:+.3f}   |                           |")
    if "consecutive_overload" in info:
        rows.append(f"| 🔥 Overload   | {info['consecutive_overload']} steps  |                           |")
    return "\n".join(rows)


# ────────────────────────────────────────────────────────────────────────────
# Tab 1 – Interactive simulation (uses gr.State for per-session isolation)
# ────────────────────────────────────────────────────────────────────────────

def interactive_reset(
    state: dict[str, Any],
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Reset the environment and return initial display + updated session state."""
    env = CloudResourceEnv(seed=random.randint(0, 9999))
    obs, info = env.reset()
    state = {"env": env, "obs": obs, "total_reward": 0.0, "log": []}
    table = _state_table(info)
    return table, "0.000", "—", "Episode reset ✅", state


def interactive_step(
    manual_action: str,
    state: dict[str, Any],
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Execute one step and return updated display + session state."""
    if not state or "env" not in state:
        return (
            "⚠️ Click **Reset Episode** first.",
            "—", "—", "Not started",
            state or {},
        )

    env = state["env"]
    obs = state["obs"]

    # Action selection
    action_map = {name: idx for idx, name in ACTION_NAMES.items()}
    if manual_action and manual_action in action_map:
        action = action_map[manual_action]
    else:
        action = _pick_action(obs)
    chosen = ACTION_NAMES[action]

    obs, reward, terminated, truncated, info = env.step(action)
    state["obs"] = obs
    state["total_reward"] += reward
    log_line = (
        f"Step {info['step']:>3} | {chosen:<14} | "
        f"rew={reward:+.3f} | "
        f"cpu={info['cpu_used']:.1%} mem={info['mem_used']:.1%} "
        f"cont={info['containers']:>2} load={info['load']:.1%}"
    )
    state["log"].append(log_line)

    table = _state_table(info, reward)
    total = f"{state['total_reward']:.3f}"
    status = "🏁 Done" if (terminated or truncated) else "▶ Running"
    log_text = "\n".join(state["log"][-20:])  # last 20 lines

    return table, total, chosen, log_text, state


# ────────────────────────────────────────────────────────────────────────────
# Tab 2 – Auto-run full episode + plots
# ────────────────────────────────────────────────────────────────────────────

def run_episode(seed_val: int, use_agent: bool) -> tuple[plt.Figure, str]:
    """Run a full episode and return (matplotlib figure, summary markdown)."""
    env = CloudResourceEnv(seed=int(seed_val))
    obs, _ = env.reset()

    rewards, cpu_trace, mem_trace, cont_trace, load_trace = [], [], [], [], []

    while True:
        action = _pick_action(obs) if use_agent else random.randrange(5)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        cpu_trace.append(info["cpu_used"])
        mem_trace.append(info["mem_used"])
        cont_trace.append(info["containers"] / 20)
        load_trace.append(info["load"])
        if terminated or truncated:
            break

    env.close()

    steps = list(range(1, len(rewards) + 1))
    cum_r = np.cumsum(rewards)

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor="#0d1117")
    fig.suptitle("Cloud Resource Allocation – Episode Summary",
                 color="white", fontsize=14, weight="bold", y=1.01)

    style = dict(linewidth=1.5)
    ax_style = dict(facecolor="#161b22", grid=True)

    # Panel 1 – Cumulative reward
    ax1 = axes[0]
    ax1.set_facecolor(ax_style["facecolor"])
    ax1.plot(steps, cum_r, color="#58a6ff", **style, label="Cumulative Reward")
    ax1.axhline(0, color="#ffffff22", linewidth=0.8)
    ax1.set_ylabel("Cum. Reward", color="white")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#30363d")
    ax1.legend(facecolor="#0d1117", labelcolor="white", fontsize=9)
    for spine in ax1.spines.values(): spine.set_color("#30363d")
    ax1.grid(color="#30363d", linewidth=0.5)

    # Panel 2 – Resource utilisation
    ax2 = axes[1]
    ax2.set_facecolor(ax_style["facecolor"])
    ax2.plot(steps, cpu_trace,  color="#f0883e", **style, label="CPU")
    ax2.plot(steps, mem_trace,  color="#3fb950", **style, label="MEM")
    ax2.plot(steps, load_trace, color="#bc8cff", **style, label="Load", linestyle="--")
    ax2.axhline(0.9, color="#ff4444", linewidth=0.8, linestyle=":", label="Overload (90%)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Utilisation", color="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#30363d")
    ax2.legend(facecolor="#0d1117", labelcolor="white", fontsize=9, ncol=2)
    for spine in ax2.spines.values(): spine.set_color("#30363d")
    ax2.grid(color="#30363d", linewidth=0.5)

    # Panel 3 – Container count (normalised)
    ax3 = axes[2]
    ax3.set_facecolor(ax_style["facecolor"])
    ax3.fill_between(steps, cont_trace, alpha=0.4, color="#58a6ff")
    ax3.plot(steps, cont_trace, color="#58a6ff", **style, label="Containers (norm)")
    ax3.set_ylim(0, 1.05); ax3.set_ylabel("Containers / 20", color="white")
    ax3.set_xlabel("Step", color="white")
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#30363d")
    ax3.legend(facecolor="#0d1117", labelcolor="white", fontsize=9)
    for spine in ax3.spines.values(): spine.set_color("#30363d")
    ax3.grid(color="#30363d", linewidth=0.5)

    fig.tight_layout()

    summary = (
        f"**Episode length:** {len(rewards)} steps\n\n"
        f"**Total reward:** {sum(rewards):.2f}\n\n"
        f"**Avg reward/step:** {np.mean(rewards):.3f}\n\n"
        f"**Peak CPU:** {max(cpu_trace):.1%}  |  **Peak MEM:** {max(mem_trace):.1%}\n\n"
        f"**Agent:** {'DQN (trained)' if (use_agent and _AGENT) else 'Random'}"
    )
    return fig, summary


# ────────────────────────────────────────────────────────────────────────────
# Tab 3 – About
# ────────────────────────────────────────────────────────────────────────────

ABOUT_TEXT = """
## ☁️ Cloud Resource Allocation — DevOps AI Agent

> **An OpenEnv-compliant reinforcement learning environment where AI agents learn to manage cloud infrastructure resources (CPU, Memory, Containers) under varying load patterns — simulating real-world Kubernetes autoscaling at scale.**

**Built for the Meta × Scaler AI Hackathon 2026**

---

### 🌍 Real-World Mapping

This simulation directly maps to production Kubernetes and cloud infrastructure:

| Simulation | Real-World Equivalent |
|---|---|
| CPU utilisation | VM / pod CPU usage |
| Memory utilisation | Container memory usage |
| Containers (1–20) | Kubernetes pods / VM Scale Set instances |
| Load (0.0–1.0) | Requests per second (normalised) |
| Scale Up / Down | Horizontal Pod Autoscaler (HPA) actions |
| Overload penalty (>90%) | SLA violation / latency spike |
| Waste penalty (<20%) | Over-provisioned cloud spend |

---

### 🧠 Observation Space  (`obs_dim = 4`, all normalised 0–1)

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | `cpu_used` | 0.0–1.0 | Fraction of total CPU consumed |
| 1 | `mem_used` | 0.0–1.0 | Fraction of total memory consumed |
| 2 | `containers` | 1–20 | Number of running containers/pods |
| 3 | `load` | 0.0–1.0 | Incoming request load (normalised RPS) |

---

### ⚙️ Action Space  (`Discrete(5)`)

| Action | Name | Effect | Real-World Analogy |
|---|---|---|---|
| 0 | **Idle** | Do nothing | No autoscaler intervention |
| 1 | **Alloc CPU** | Relieve CPU pressure (−10%) | Vertical scaling / CPU limit increase |
| 2 | **Alloc MEM** | Relieve MEM pressure (−10%) | Memory limit increase |
| 3 | **Scale UP** | +1 container; CPU/MEM relief (−5%) | HPA scale-up event |
| 4 | **Scale DOWN** | −1 container; CPU/MEM increase (+8%) | HPA scale-down to save cost |

---

### 🏅 Reward Function

| Component | Condition | Signal | Purpose |
|---|---|---|---|
| **Efficiency bonus** | CPU ≈ load, balanced utilisation | `+1.0 × efficiency` | Reward right-sizing |
| **Overload penalty** | CPU or MEM > 90% | `−3.0` | Punish SLA violations |
| **Waste penalty** | CPU & MEM < 20%, load < 30% | `−0.5` | Punish over-provisioning |
| **Container cost** | Per container, every step | `−0.05 × containers` | Incentivise frugality |
| **Scaling penalty** | Scale Up or Scale Down action | `−0.10` | Discourage erratic scaling |
| **Idle penalty** | Idle action chosen | `−0.10` | Discourage permanent passivity |

**Efficiency formula:** `efficiency = 1.0 − |cpu_used − load| − |mem_used − load| × 0.5`

---

### 🛑 Termination Conditions

| Type | Condition |
|---|---|
| **Normal** | `step_count >= episode_length` (task-specific) |
| **Overload** | 5 consecutive steps with CPU or MEM > 90% |
| **Catastrophic** | CPU ≥ 100% AND MEM ≥ 100% AND containers ≤ 1 |

---

### 📋 Task Difficulties (Easy → Hard)

| Task | Difficulty | Load Pattern | Steps |
|---|---|---|---|
| `steady-load` | 🟢 Easy | Constant 0.3–0.5, Gaussian noise (σ=0.03) | 100 |
| `diurnal-cycle` | 🟡 Medium | Sinusoidal 0.2–0.8 (daily traffic) | 150 |
| `spike-resilience` | 🔴 Hard | Random spikes to 0.95 (15% chance/step) | 200 |

**Grading:** `score = (1.0 − overload_ratio) − (waste_ratio × 0.3)` clamped to [0.0, 1.0]

---

### 🤖 Dual-Agent Architecture

#### 1. Deep Q-Network (DQN)
- 2-hidden-layer MLP (4 → 128 → 128 → 5)
- Experience replay buffer (20,000 transitions)
- Target network (hard update every 200 steps)
- ε-greedy exploration (1.0 → 0.05, linear decay over 5,000 steps)
- Gradient clipping (max norm 10.0)

#### 2. LLM Agent (Qwen 72B)
- Uses **Qwen/Qwen2.5-72B-Instruct** via HF Inference API
- Natural-language reasoning about cluster state each step
- Responds with a single action digit (0–4)
- Falls back to keyword matching, then DQN, on parse failure

**Automatic fallback:** If no `HF_TOKEN` is set, the system uses the trained DQN agent.

---

### 🐳 Deployment

- **Unified Server:** FastAPI (OpenEnv API) + Gradio UI on port 7860
- **Docker Ready:** Single Dockerfile, deploy as HF Space
- **OpenEnv Compliant:** Standard `POST /reset` · `POST /step` · `GET /state` API

---

> ⚡ Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · **Meta × Scaler AI Hackathon 2026**
"""


# ────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0d1117",
    body_text_color="#e6edf3",
    block_background_fill="#161b22",
    block_border_color="#30363d",
    button_primary_background_fill="#238636",
    button_primary_text_color="white",
)

with gr.Blocks(theme=THEME, title="Cloud Resource Allocation – DevOps AI Agent") as demo:

    gr.Markdown(
        """
        # ☁️ Cloud Resource Allocation – DevOps AI Agent
        > A Reinforcement Learning agent that schedules cloud resources to minimise cost, latency, and overload.
        """,
        elem_id="header",
    )

    # ── Per-session state (thread-safe, no global mutable dict) ──────────
    session_state = gr.State(value={})

    with gr.Tabs():

        # ── Tab 1: Interactive ───────────────────────────────────────────────
        with gr.Tab("🎮 Interactive Simulation"):
            gr.Markdown("Step through an episode manually or let the agent decide.")

            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary", scale=1)
                action_dd = gr.Dropdown(
                    choices=["Agent decides"] + list(ACTION_NAMES.values()),
                    value="Agent decides",
                    label="Action override",
                    scale=2,
                )
                step_btn = gr.Button("▶ Step", variant="secondary", scale=1)

            with gr.Row():
                state_md   = gr.Markdown("_Click Reset to start._")
                with gr.Column():
                    total_rew  = gr.Textbox(label="Total Reward", interactive=False)
                    last_act   = gr.Textbox(label="Last Action",  interactive=False)
                    ep_status  = gr.Textbox(label="Status",       interactive=False)

            log_box = gr.Textbox(
                label="Step Log (last 20)", lines=10, interactive=False,
                elem_id="log_box",
            )

            reset_btn.click(
                fn=interactive_reset,
                inputs=[session_state],
                outputs=[state_md, total_rew, last_act, log_box, session_state],
            )
            step_btn.click(
                fn=interactive_step,
                inputs=[action_dd, session_state],
                outputs=[state_md, total_rew, last_act, log_box, session_state],
            )

        # ── Tab 2: Auto-run ──────────────────────────────────────────────────
        with gr.Tab("🚀 Run Episode"):
            gr.Markdown("Automatically run a full episode and visualise results.")
            with gr.Row():
                seed_sl  = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                agent_cb = gr.Checkbox(
                    value=True,
                    label=f"Use DQN agent {'✅' if _AGENT else '(no model → random)'}",
                )
                run_btn  = gr.Button("▶ Run Episode", variant="primary")

            episode_plot    = gr.Plot(label="Episode Metrics")
            episode_summary = gr.Markdown()

            run_btn.click(
                fn=run_episode,
                inputs=[seed_sl, agent_cb],
                outputs=[episode_plot, episode_summary],
            )

        # ── Tab 3: About ─────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT_TEXT)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
