"""
q_agent.py
==========
Deep Q-Network (DQN) agent for Cloud Resource Allocation.

Architecture
------------
- 2-hidden-layer MLP: 4 → 128 → 128 → 5
- Experience replay buffer (deque)
- Separate target network (hard-update every C steps)
- Epsilon-greedy exploration with linear decay

Usage
-----
    # Training
    python q_agent.py --train --episodes 500 --save models/dqn_cloud.pth

    # Evaluation (requires saved model)
    python q_agent.py --eval --load models/dqn_cloud.pth --episodes 10
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path
from typing import Deque, NamedTuple

import numpy as np

# ── optional torch import ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("[WARNING] PyTorch not found. Install with:  pip install torch")

from cloud_env import CloudResourceEnv


# ────────────────────────────────────────────────────────────────────────────
# 1. Replay Buffer
# ────────────────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int = 10_000, seed: int | None = None):
        self._buf: Deque[Transition] = collections.deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, *args) -> None:
        self._buf.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ────────────────────────────────────────────────────────────────────────────
# 2. Q-Network  (only defined when torch is available)
# ────────────────────────────────────────────────────────────────────────────

if _HAS_TORCH:
    class QNetwork(nn.Module):
        """
        MLP approximating Q(s, a) for all actions simultaneously.
        Input : state vector (obs_dim,)
        Output: Q-values for each action (n_actions,)
        """

        def __init__(self, obs_dim: int = 4, n_actions: int = 5, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ────────────────────────────────────────────────────────────────────────────
# 3. DQN Agent
# ────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN Agent with:
    - Epsilon-greedy exploration (linear decay)
    - Experience replay
    - Target network (hard update every `target_update` steps)
    """

    # ── hyper-parameters (easy to override) ─────────────────────────────────
    GAMMA:         float = 0.99       # discount factor
    LR:            float = 1e-3       # learning rate
    BATCH_SIZE:    int   = 64         # mini-batch size
    BUFFER_SIZE:   int   = 20_000     # replay buffer capacity
    EPS_START:     float = 1.0        # initial exploration rate
    EPS_END:       float = 0.05       # final exploration rate
    EPS_DECAY:     int   = 5_000      # linear decay over this many steps
    TARGET_UPDATE:  int  = 200        # hard-copy target network every N steps

    def __init__(
        self,
        obs_dim:   int = 4,
        n_actions: int = 5,
        hidden:    int = 128,
        seed:      int | None = None,
        device:    str = "cpu",
    ):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required. Run: pip install torch")

        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.device    = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Networks
        self.q_net     = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.LR)
        self.buffer    = ReplayBuffer(capacity=self.BUFFER_SIZE, seed=seed)

        self._total_steps: int   = 0
        self._epsilon:     float = self.EPS_START

    # ── Epsilon ──────────────────────────────────────────────────────────────

    def _update_epsilon(self) -> None:
        progress = min(1.0, self._total_steps / self.EPS_DECAY)
        self._epsilon = self.EPS_START + progress * (self.EPS_END - self.EPS_START)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Set greedy=True for evaluation (no exploration).
        """
        if not greedy and random.random() < self._epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Learning step ────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """Pull a mini-batch from the buffer and do one gradient step."""
        if len(self.buffer) < self.BATCH_SIZE:
            return None

        batch       = self.buffer.sample(self.BATCH_SIZE)
        states      = torch.tensor(np.array([t.state      for t in batch]), dtype=torch.float32, device=self.device)
        actions     = torch.tensor([t.action     for t in batch], dtype=torch.long,  device=self.device)
        rewards     = torch.tensor([t.reward     for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones       = torch.tensor([t.done       for t in batch], dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q   = rewards + self.GAMMA * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # ── Full step (store + learn + update target) ────────────────────────────

    def step(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> float | None:
        self.buffer.push(state, action, reward, next_state, done)
        self._total_steps += 1
        self._update_epsilon()

        loss = self.learn()

        if self._total_steps % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "q_net_state":     self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps":     self._total_steps,
            "epsilon":         self._epsilon,
            "hyperparams": {
                "obs_dim":   self.obs_dim,
                "n_actions": self.n_actions,
            },
        }
        torch.save(checkpoint, path)
        print(f"[DQNAgent] Saved → {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._total_steps = checkpoint.get("total_steps", 0)
        self._epsilon     = checkpoint.get("epsilon", self.EPS_END)
        print(f"[DQNAgent] Loaded ← {path}")


# ────────────────────────────────────────────────────────────────────────────
# 4. Training loop
# ────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "Idle",
    1: "Alloc CPU",
    2: "Alloc MEM",
    3: "Scale UP",
    4: "Scale DOWN",
}


def train(
    episodes:    int  = 300,
    max_steps:   int  = 200,
    seed:        int  = 42,
    save_path:   str  = "models/dqn_cloud.pth",
    log_every:   int  = 20,
    verbose:     bool = True,
) -> dict:
    """
    Train the DQN agent on CloudResourceEnv.

    Returns a dict with training history (episode rewards, losses).
    """
    env   = CloudResourceEnv(seed=seed)
    agent = DQNAgent(obs_dim=4, n_actions=5, seed=seed)

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses":          [],
        "epsilons":        [],
    }

    for ep in range(1, episodes + 1):
        obs, _    = env.reset(seed=seed + ep)
        total_rew = 0.0
        ep_losses = []

        for t in range(max_steps):
            action     = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done       = terminated or truncated

            loss = agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                ep_losses.append(loss)

            obs        = next_obs
            total_rew += reward

            if done:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["episode_rewards"].append(total_rew)
        history["episode_lengths"].append(t + 1)
        history["losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)

        if verbose and ep % log_every == 0:
            avg_rew = np.mean(history["episode_rewards"][-log_every:])
            print(
                f"Episode {ep:>4}/{episodes} | "
                f"AvgReward={avg_rew:>8.2f} | "
                f"Loss={avg_loss:.4f} | "
                f"ε={agent.epsilon:.3f} | "
                f"Steps={t+1}"
            )

    agent.save(save_path)
    return history


# ────────────────────────────────────────────────────────────────────────────
# 5. Evaluation loop
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    load_path: str  = "models/dqn_cloud.pth",
    episodes:  int  = 5,
    seed:      int  = 99,
    render:    bool = True,
) -> dict:
    """Run the trained agent (greedy) and return per-episode stats."""
    env   = CloudResourceEnv(render_mode="human" if render else None, seed=seed)
    agent = DQNAgent(obs_dim=4, n_actions=5)
    agent.load(load_path)

    results = []
    for ep in range(1, episodes + 1):
        obs, _    = env.reset(seed=seed + ep)
        total_rew = 0.0
        steps     = 0

        print(f"\n─── Episode {ep} ───────────────────────────────────────────")
        while True:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_rew += reward
            steps     += 1
            if render:
                print(f"  {ACTION_NAMES[action]:<10} | rew={reward:+.3f} | "
                      f"cpu={info['cpu_used']:.1%} mem={info['mem_used']:.1%} "
                      f"cont={info['containers']:>2} load={info['load']:.1%}")
            if terminated or truncated:
                break

        results.append({"episode": ep, "total_reward": total_rew, "steps": steps})
        print(f"  Total reward: {total_rew:.2f}  |  Steps: {steps}")

    env.close()
    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Cloud Resource Allocation")
    parser.add_argument("--train",    action="store_true", help="Run training")
    parser.add_argument("--eval",     action="store_true", help="Run evaluation")
    parser.add_argument("--episodes", type=int,  default=300,                    help="Number of episodes")
    parser.add_argument("--save",     type=str,  default="models/dqn_cloud.pth", help="Path to save model")
    parser.add_argument("--load",     type=str,  default="models/dqn_cloud.pth", help="Path to load model")
    parser.add_argument("--seed",     type=int,  default=42,                     help="Random seed")
    parser.add_argument("--no-render", action="store_true",                       help="Disable render during eval")
    args = parser.parse_args()

    if args.train:
        history = train(episodes=args.episodes, save_path=args.save, seed=args.seed)
        rewards = history["episode_rewards"]
        print(f"\nTraining complete. Final avg reward (last 50): {np.mean(rewards[-50:]):.2f}")

    if args.eval:
        evaluate(load_path=args.load, episodes=args.episodes, seed=args.seed, render=not args.no_render)

    if not args.train and not args.eval:
        print("Specify --train or --eval. Run with --help for usage.")
