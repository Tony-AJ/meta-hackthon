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

import os
import random
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloud_env import CloudResourceEnv

# ── try to load a trained agent (optional) ───────────────────────────────────
_AGENT = None
try:
    from q_agent import DQNAgent
    _MODEL_PATH = Path("models/dqn_cloud.pth")
    if _MODEL_PATH.exists():
        _AGENT = DQNAgent(obs_dim=4, n_actions=5)
        _AGENT.load(str(_MODEL_PATH))
        print("[app] Trained DQN agent loaded.")
    else:
        print("[app] No trained model found – using random agent for demo.")
except Exception as e:
    print(f"[app] Could not load DQN agent: {e}")


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
# Tab 1 – Interactive simulation
# ────────────────────────────────────────────────────────────────────────────

_istate: dict = {}   # mutable session state (Gradio is stateless per call)


def interactive_reset():
    env = CloudResourceEnv(seed=random.randint(0, 9999))
    obs, info = env.reset()
    _istate.clear()
    _istate.update({"env": env, "obs": obs, "total_reward": 0.0, "log": []})
    table = _state_table(info)
    return table, "0.000", "—", "Episode reset ✅"


def interactive_step(manual_action: str):
    if "env" not in _istate:
        return "⚠️ Click **Reset Episode** first.", "—", "—", "Not started"

    env   = _istate["env"]
    obs   = _istate["obs"]

    # Action selection
    action_map = {name: idx for idx, name in ACTION_NAMES.items()}
    if manual_action and manual_action in action_map:
        action = action_map[manual_action]
    else:
        action = _pick_action(obs)
    chosen = ACTION_NAMES[action]

    obs, reward, terminated, truncated, info = env.step(action)
    _istate["obs"]          = obs
    _istate["total_reward"] += reward
    log_line = (
        f"Step {info['step']:>3} | {chosen:<14} | "
        f"rew={reward:+.3f} | "
        f"cpu={info['cpu_used']:.1%} mem={info['mem_used']:.1%} "
        f"cont={info['containers']:>2} load={info['load']:.1%}"
    )
    _istate["log"].append(log_line)

    table  = _state_table(info, reward)
    total  = f"{_istate['total_reward']:.3f}"
    status = "🏁 Done" if (terminated or truncated) else "▶ Running"
    log_text = "\n".join(_istate["log"][-20:])  # last 20 lines

    return table, total, chosen, log_text


# ────────────────────────────────────────────────────────────────────────────
# Tab 2 – Auto-run full episode + plots
# ────────────────────────────────────────────────────────────────────────────

def run_episode(seed_val: int, use_agent: bool):
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
## ☁️ Cloud Resource Allocation – DevOps AI Agent

An RL agent that learns to allocate cloud resources to minimize **cost**, **latency**, and **overload** — analogous to Kubernetes scheduling.

---

### 🧠 State Space  (`obs_dim = 4`, all normalised 0–1)
| Index | Feature           | Meaning                       |
|-------|-------------------|-------------------------------|
| 0     | `cpu_used`        | Fraction of CPU consumed      |
| 1     | `mem_used`        | Fraction of Memory consumed   |
| 2     | `containers_norm` | Number of running containers  |
| 3     | `request_load`    | Incoming request load         |

---

### ⚙️ Action Space  (`Discrete(5)`)
| Action | Name        | Effect                          |
|--------|-------------|---------------------------------|
| 0      | Idle        | Do nothing                      |
| 1      | Alloc CPU   | Relieve CPU pressure (−10%)     |
| 2      | Alloc MEM   | Relieve MEM pressure (−10%)     |
| 3      | Scale UP    | +1 container (distributes load) |
| 4      | Scale DOWN  | −1 container (saves cost)       |

---

### 🏅 Reward Function
| Condition                          | Signal                        |
|------------------------------------|-------------------------------|
| Balanced efficiency               | `+efficiency_bonus`            |
| CPU or MEM > 90%                  | `−3.0 overload_penalty`        |
| Resources idle & low load         | `−0.5 waste_penalty`           |
| Container cost (per container)    | `−0.05 × containers`           |
| Scaling action (erratic scaling)  | `−0.10 scaling_penalty`        |
| Idle action                       | `−0.10 idle_penalty`           |

---

### 🤖 Agent
**Deep Q-Network (DQN)** with:
- 2-hidden-layer MLP (4→128→128→5)
- Experience replay buffer (20k transitions)
- Target network (hard update every 200 steps)
- Epsilon-greedy exploration (ε: 1.0 → 0.05)

---

> Built for the **Meta Hackathon** · Industry-level DevOps + AI combination
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
                outputs=[state_md, total_rew, last_act, log_box],
            )
            step_btn.click(
                fn=interactive_step,
                inputs=[action_dd],
                outputs=[state_md, total_rew, last_act, log_box],
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
    demo.launch(show_error=True)
