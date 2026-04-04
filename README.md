---
title: Cloud Resource Allocation – OpenEnv
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
tags:
  - openenv
  - reinforcement-learning
  - cloud-computing
  - devops
pinned: false
license: mit
---

# ☁️ Cloud Resource Allocation — DevOps AI Agent

> **An OpenEnv-compliant reinforcement learning environment where AI agents learn to manage cloud infrastructure resources (CPU, Memory, Containers) under varying load patterns — simulating real-world Kubernetes autoscaling at scale.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

**Built for the Meta × Scaler AI Hackathon 2026**

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Real-World Mapping](#-real-world-mapping)
- [Architecture](#-architecture)
- [Environment Design](#-environment-design)
  - [Observation Space](#-observation-space)
  - [Action Space](#%EF%B8%8F-action-space)
  - [Reward Function](#-reward-function)
  - [Termination Conditions](#-termination-conditions)
- [Task System](#-task-system-easy--hard)
- [Agent Implementations](#-agent-implementations)
  - [DQN Agent](#1-deep-q-network-dqn)
  - [LLM Agent](#2-llm-agent-qwen-72b)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [Gradio Interactive Demo](#-gradio-interactive-demo)
- [Environment Variables](#-environment-variables)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Baseline Scores](#-baseline-scores)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

This project implements a **Cloud Resource Allocation** environment as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL problem. The agent operates a simulated Kubernetes cluster, making real-time decisions about CPU allocation, memory management, and horizontal pod autoscaling under dynamically varying traffic loads.

The environment features **three progressively harder tasks** — from steady-state traffic to unpredictable flash-crowd spikes — and supports two agent backends:

1. **Deep Q-Network (DQN)** — a trained neural network agent for fast, local inference.
2. **LLM Agent (Qwen 72B)** — a large language model that reasons about cluster state in natural language.

The system is fully containerised with Docker and can be deployed as a **Hugging Face Space** (OpenEnv server or Gradio interactive demo).

---

## ✨ Key Features

| Category | Details |
|----------|---------|
| 🧠 **Dual-Agent Architecture** | DQN (offline-trained) + LLM (Qwen 72B via HF Inference API) with automatic fallback |
| 🎮 **Interactive Gradio UI** | Step-by-step simulation, auto-run episodes with real-time plotting |
| 🌐 **OpenEnv Compliant** | Standard `reset` / `step` / `state` API with typed Pydantic models |
| 📈 **Three Task Difficulties** | Steady load (easy), diurnal cycle (medium), spike resilience (hard) |
| 🐳 **Docker Ready** | Single Dockerfile with `MODE` switch (server / demo) |
| 🏗️ **Gymnasium Compatible** | Core environment inherits from `gymnasium.Env` when available |
| 📊 **Built-in Grading** | Deterministic 0.0–1.0 scoring based on overload and waste ratios |
| ✅ **Tested** | Pytest test suite for environment physics and edge cases |

---

## 🌍 Real-World Mapping

The simulation directly maps to production Kubernetes and cloud infrastructure concepts:

| Simulation Concept | Real-World Equivalent | Monitoring Tool |
|--------------------|-----------------------|-----------------|
| CPU utilisation | VM / pod CPU usage | Azure Monitor, Prometheus |
| Memory utilisation | Container memory usage | cAdvisor, Datadog |
| Containers (1–20) | Kubernetes pods / VM Scale Set instances | `kubectl get pods` |
| Load (0.0–1.0) | Requests per second (normalised) | Ingress controller metrics |
| Scale Up / Down | Horizontal Pod Autoscaler (HPA) actions | HPA events |
| Overload penalty (>90%) | SLA violation / customer-facing latency spike | Alertmanager |
| Waste penalty (<20%) | Over-provisioned cloud spend | Cost Explorer |
| Container cost | Per-pod cloud billing | Cloud billing API |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER                         │
│                                                                 │
│   ┌────────────────────┐    ┌──────────────────────────────┐   │
│   │   DQN Agent        │    │   LLM Agent (Qwen 72B)      │   │
│   │   q_agent.py       │    │   inference.py               │   │
│   │   MLP: 4→128→128→5 │    │   OpenAI-compatible client   │   │
│   └────────┬───────────┘    └──────────────┬───────────────┘   │
│            │            fallback           │                    │
│            └──────────────┬────────────────┘                    │
│                           ▼                                     │
├─────────────────────────────────────────────────────────────────┤
│                        CLIENT LAYER                             │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │   client.py — CloudEnv (EnvClient subclass)             │  │
│   │   Typed actions, observations, and state management     │  │
│   └──────────────────────────┬───────────────────────────────┘  │
│                              │  HTTP (Docker / URL)             │
├──────────────────────────────┼──────────────────────────────────┤
│                        SERVER LAYER                             │
│                                                                 │
│   ┌──────────────────────────┴───────────────────────────────┐  │
│   │   server/app.py — FastAPI (create_fastapi_app)          │  │
│   │   Endpoints: POST /reset, POST /step, GET /state        │  │
│   └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│   ┌──────────────────────────┴───────────────────────────────┐  │
│   │   server/cloud_environment.py                            │  │
│   │   OpenEnv Environment subclass (core physics engine)     │  │
│   └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│   ┌──────────────────────────┴───────────────────────────────┐  │
│   │   tasks.py — Load generators & graders                  │  │
│   │   cloud_env.py — Gymnasium-compatible env (DQN training)│  │
│   │   models.py — Pydantic data models                      │  │
│   └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        DEMO LAYER                               │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │   app.py — Gradio Interactive UI                        │  │
│   │   🎮 Interactive Sim │ 🚀 Auto-Run │ ℹ️ About           │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Environment Design

### 👁 Observation Space

The agent observes a 4-dimensional normalised state vector at each step:

| Index | Field | Type | Range | Description |
|-------|-------|------|-------|-------------|
| 0 | `cpu_used` | float | 0.0–1.0 | Fraction of total CPU consumed |
| 1 | `mem_used` | float | 0.0–1.0 | Fraction of total memory consumed |
| 2 | `containers` | int | 1–20 | Number of running containers/pods |
| 3 | `load` | float | 0.0–1.0 | Incoming request load (normalised RPS) |

Additional fields in the `CloudObservation` model:

| Field | Type | Description |
|-------|------|-------------|
| `step_count` | int | Current step in the episode (0–200) |
| `message` | str | Human-readable status string |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Reward received for the last action |

### ⚙️ Action Space

`Discrete(5)` — The agent selects exactly one action per step:

| `action_id` | Name | Effect | Real-World Analogy |
|-------------|------|--------|--------------------|
| 0 | **Idle** | Do nothing | No autoscaler intervention |
| 1 | **Alloc CPU** | Relieve CPU pressure (−10%) | Vertical scaling / CPU limit increase |
| 2 | **Alloc MEM** | Relieve memory pressure (−10%) | Memory limit increase |
| 3 | **Scale Up** | +1 container; slight CPU/MEM relief (−5% each) | HPA scale-up event |
| 4 | **Scale Down** | −1 container; CPU/MEM increase (+8% each) | HPA scale-down to save cost |

### 🏅 Reward Function

A multi-component reward signal balances efficiency, reliability, and cost:

| Component | Condition | Signal | Purpose |
|-----------|-----------|--------|---------|
| **Efficiency bonus** | CPU ≈ load, balanced utilisation | `+1.0 × efficiency` | Reward right-sizing |
| **Overload penalty** | CPU or MEM > 90%, or min containers | `−3.0` | Punish SLA violations |
| **Waste penalty** | CPU & MEM < 20% and load < 30% | `−0.5` | Punish over-provisioning |
| **Container cost** | Per container, every step | `−0.05 × containers` | Incentivise frugality |
| **Scaling penalty** | Scale Up or Scale Down action | `−0.10` | Discourage erratic scaling |
| **Idle penalty** | Idle action chosen | `−0.10` | Discourage permanent passivity |

**Efficiency formula:**
```
efficiency = 1.0 − |cpu_used − load| − |mem_used − load| × 0.5
reward_efficiency = clip(efficiency, 0.0, 1.0)
```

### 🛑 Termination Conditions

| Type | Condition | Description |
|------|-----------|-------------|
| **Normal termination** | `step_count >= episode_length` | Task-specific max steps reached |
| **Overload termination** | `consecutive_overload >= 5` | 5 consecutive steps with CPU or MEM > 90% |
| **Catastrophic truncation** | `cpu >= 100% AND mem >= 100% AND containers <= 1` | Total system collapse |

---

## 📋 Task System (Easy → Hard)

Each task defines a unique load pattern, episode length, and grading criteria:

| Task | Difficulty | Load Pattern | Episode Length | Description |
|------|------------|-------------|----------------|-------------|
| `steady-load` | 🟢 Easy | Constant 0.3–0.5 with Gaussian noise (σ=0.03) | 100 steps | Maintain stability under predictable moderate load |
| `diurnal-cycle` | 🟡 Medium | Sinusoidal 0.2–0.8 (daily traffic pattern) | 150 steps | Handle predictable but varying daily traffic |
| `spike-resilience` | 🔴 Hard | Random spikes to 0.95 (15% chance per step) | 200 steps | Survive unpredictable flash-crowd events |

### Grading Function

Each task is scored on a **0.0–1.0** scale using a deterministic grader:

```
score = (1.0 − overload_ratio) − (waste_ratio × 0.3)
score = clip(score, 0.0, 1.0)
```

Where:
- **overload_ratio** = fraction of steps where CPU or MEM > 90%
- **waste_ratio** = fraction of steps where CPU & MEM < 20% AND load < 30%

---

## 🤖 Agent Implementations

### 1. Deep Q-Network (DQN)

A fully trained DQN agent located in `q_agent.py` with pre-trained weights in `models/dqn_cloud.pth`.

**Architecture:**

```
Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(5) → Q-values
```

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Discount factor (γ)** | 0.99 | Future reward discount |
| **Learning rate** | 1e-3 | Adam optimiser LR |
| **Batch size** | 64 | Mini-batch from replay buffer |
| **Replay buffer** | 20,000 transitions | Experience replay capacity |
| **Target network update** | Every 200 steps | Hard copy to target network |
| **ε-greedy start** | 1.0 | Initial exploration rate |
| **ε-greedy end** | 0.05 | Final exploration rate |
| **ε decay** | 5,000 steps | Linear decay schedule |
| **Gradient clipping** | 10.0 | Max gradient norm |

**Train locally:**

```bash
# Train for 500 episodes and save the model
python q_agent.py --train --episodes 500 --save models/dqn_cloud.pth

# Evaluate the trained model for 10 episodes
python q_agent.py --eval --load models/dqn_cloud.pth --episodes 10

# Train with a custom seed
python q_agent.py --train --episodes 300 --seed 123 --save models/dqn_cloud.pth
```

### 2. LLM Agent (Qwen 72B)

The `inference.py` script uses an OpenAI-compatible chat API to call **Qwen/Qwen2.5-72B-Instruct** via the Hugging Face Inference Router.

**How it works:**

1. The system prompt instructs the LLM about the cluster state format and available actions.
2. Each step, the current observation is formatted into a user prompt.
3. The LLM responds with a single digit (0–4) representing the chosen action.
4. Parsing falls back to keyword matching, then to Idle (0) on failure.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Temperature | 0.3 |
| Max tokens | 100 |
| Default model | `Qwen/Qwen2.5-72B-Instruct` |
| API base | `https://router.huggingface.co/v1` |

**Automatic fallback:** If no `HF_TOKEN` is set, `inference.py` automatically falls back to the trained DQN agent.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerised deployment)
- A Hugging Face API token (optional, for LLM agent)

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/cloud-resource-allocation.git
cd cloud-resource-allocation

pip install -r requirements.txt
```

### 2. Run the OpenEnv server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The server exposes three endpoints:
- `POST /reset` — Start a new episode
- `POST /step` — Execute one action
- `GET /state` — Get current episode metadata

### 3. Run the Gradio interactive demo

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 4. Run the baseline inference

```bash
# Using LLM agent
export HF_TOKEN=your_token_here
export IMAGE_NAME=cloud-resource-env:latest
python inference.py

# Using DQN fallback (no API key needed)
export IMAGE_NAME=cloud-resource-env:latest
python inference.py
```

This runs all three tasks (`steady-load`, `diurnal-cycle`, `spike-resilience`) and prints a scored summary.

### 5. Train the DQN agent from scratch

```bash
python q_agent.py --train --episodes 500 --save models/dqn_cloud.pth
```

### 6. Smoke test the environment

```bash
python cloud_env.py
```

Runs 20 random steps and validates the environment works correctly.

---

## 🐳 Docker Deployment

A single Dockerfile supports two modes via the `MODE` environment variable:

### Build the image

```bash
docker build -t cloud-resource-env:latest .
```

### Run as OpenEnv server (default)

```bash
docker run -p 7860:7860 cloud-resource-env:latest
```

### Run as Gradio interactive demo

```bash
docker run -p 7860:7860 -e MODE=demo cloud-resource-env:latest
```

### Run with a specific task

```bash
docker run -p 7860:7860 -e TASK_NAME=spike-resilience cloud-resource-env:latest
```

### Deploy to Hugging Face Spaces

The project is ready for one-click deployment to HF Spaces:
1. Push the repo to a Hugging Face Space (SDK: Docker).
2. The Dockerfile automatically starts the OpenEnv server.
3. Set `MODE=demo` in Space settings for the Gradio UI instead.

---

## 🎮 Gradio Interactive Demo

The Gradio app (`app.py`) provides three tabs:

### Tab 1: 🎮 Interactive Simulation
- Step through an episode **manually** or let the agent decide.
- Real-time resource dashboard with bar charts.
- Action override dropdown (Idle, Alloc CPU, Alloc MEM, Scale Up, Scale Down).
- Running step log (last 20 steps).

### Tab 2: 🚀 Run Episode
- Auto-run a full episode with a configurable seed.
- Three-panel matplotlib plot:
  - **Cumulative reward** over time
  - **Resource utilisation** (CPU, MEM, Load) with 90% overload line
  - **Container count** (normalised, filled area chart)
- Episode summary with peak stats.

### Tab 3: ℹ️ About
- Full environment documentation (state space, action space, reward function, agent details).

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | For LLM inference | — | Hugging Face API token |
| `API_BASE_URL` | For LLM inference | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | For LLM inference | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `IMAGE_NAME` | For `inference.py` | — | Docker image name for OpenEnv server |
| `MODE` | For Docker | `server` | `server` (OpenEnv API) or `demo` (Gradio UI) |
| `TASK_NAME` | Optional | `steady-load` | Task to run: `steady-load`, `diurnal-cycle`, `spike-resilience` |
| `ENV_SEED` | Optional | `42` | Random seed for reproducibility |

A `.env.example` file is included in the repository for reference.

---

## 📁 Project Structure

```
meta-hackthon/
│
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest (task registry)
├── Dockerfile                  # Multi-mode Docker build (server / demo)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
│
├── models.py                   # Pydantic data models (Action, Observation, State)
├── client.py                   # EnvClient subclass for HTTP communication
├── tasks.py                    # Task configs, load generators, and grading functions
├── cloud_env.py                # Gymnasium-compatible RL env (for DQN training)
│
├── server/                     # OpenEnv FastAPI server
│   ├── __init__.py
│   ├── app.py                  # FastAPI app factory (create_fastapi_app)
│   └── cloud_environment.py    # Environment subclass (core physics engine)
│
├── inference.py                # Baseline inference script (LLM + DQN fallback)
├── q_agent.py                  # DQN agent: training, evaluation, and CLI
├── app.py                      # Gradio interactive demo (3-tab UI)
│
├── __init__.py                 # Package init
│
└── models/
    └── dqn_cloud.pth           # Pre-trained DQN weights
```

### File Responsibilities

| File | Purpose |
|------|---------|
| `models.py` | Typed Pydantic models (`CloudAction`, `CloudObservation`, `CloudState`) inheriting from OpenEnv base classes |
| `client.py` | `CloudEnv` — `EnvClient` subclass for async/sync HTTP communication with the server |
| `tasks.py` | Task definitions (`steady-load`, `diurnal-cycle`, `spike-resilience`), load-pattern generators, and the `grade_episode` grading function |
| `cloud_env.py` | Standalone `CloudResourceEnv` with Gymnasium API (`reset`/`step`/`render`/`close`) — used for DQN training |
| `server/cloud_environment.py` | `CloudResourceEnvironment` — OpenEnv `Environment` subclass with identical physics, used by the FastAPI server |
| `server/app.py` | FastAPI application factory using `create_fastapi_app` from `openenv-core` |
| `inference.py` | Mandatory OpenEnv inference script — connects to Docker, runs all 3 tasks, emits `[START]`/`[STEP]`/`[END]` logs |
| `q_agent.py` | Full DQN pipeline: `QNetwork`, `ReplayBuffer`, `DQNAgent`, `train()`, `evaluate()` with CLI |
| `app.py` | Gradio UI with dark theme, interactive simulation, auto-run plotting, and about page |

---

## 🧪 Testing

The project includes a pytest test suite in the `tests/` directory:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cloud_env.py -v
```

### Test Coverage

The test suite validates:
- **Environment physics** — reset initialisation, action effects, dynamics
- **Reward computation** — overload penalties, waste penalties, efficiency bonuses
- **Termination logic** — max steps, consecutive overload, catastrophic collapse
- **Edge cases** — boundary container counts, extreme load values
- **Reproducibility** — deterministic behaviour with fixed seeds

---

## 📊 Baseline Scores

| Method | steady-load | diurnal-cycle | spike-resilience | Avg |
|--------|-------------|---------------|------------------|-----|
| Random | TBD | TBD | TBD | TBD |
| Rule-based | TBD | TBD | TBD | TBD |
| DQN (trained) | TBD | TBD | TBD | TBD |
| LLM (Qwen 72B) | TBD | TBD | TBD | TBD |

*(Run `inference.py` to populate these scores)*

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/my-improvement`)
3. **Commit** your changes (`git commit -m "Add my improvement"`)
4. **Push** to the branch (`git push origin feature/my-improvement`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Smoke test the environment
python cloud_env.py

# Train a fresh DQN agent
python q_agent.py --train --episodes 300
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv</a> · <strong>Meta × Scaler AI Hackathon 2026</strong>
</p>
