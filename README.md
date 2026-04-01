---
title: Cloud Resource Allocation DevOps AI Agent
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: mit
---

# ☁️ Cloud Resource Allocation – DevOps AI Agent

A Reinforcement Learning agent that learns to **allocate CPU, Memory, and Containers** in a simulated cloud cluster — minimising cost, latency, and overload. Built for a Hackathon. Inspired by Kubernetes scheduling.

## 🧠 What it does

The **DQN agent** observes the current cluster state and decides the optimal resource action every step:

| Signal | Detail |
|---|---|
| **State** | CPU used · MEM used · Container count · Request load |
| **Actions** | Idle · Alloc CPU · Alloc MEM · Scale UP · Scale DOWN |
| **Reward** | +efficiency, −overload (×3), −waste, −cost per container |

## 📁 Project Structure

```
cloud_env.py   # Custom Gymnasium environment
q_agent.py     # DQN agent (replay buffer, target network, CLI)
app.py         # This Gradio app
requirements.txt
```

## 🚀 Train Locally

```bash
pip install -r requirements.txt
python q_agent.py --train --episodes 500 --save models/dqn_cloud.pth
python q_agent.py --eval  --load models/dqn_cloud.pth
```
