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

# ☁️ Cloud Resource Allocation – OpenEnv Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent learns to manage cloud infrastructure resources (CPU, Memory, Containers) under varying load patterns. Simulates real-world **Kubernetes autoscaling** scenarios.

> **Built for the Meta × Scaler AI Hackathon 2026**

---

## 🌍 Real-World Mapping

| Simulation Concept | Real-World Equivalent |
|--------------------|-----------------------|
| CPU utilisation    | VM / pod CPU usage (Azure Monitor, Prometheus) |
| Memory utilisation | Container memory usage (cAdvisor, Datadog) |
| Containers         | Kubernetes pods / VM Scale Set instances |
| Load               | Requests per second (Ingress controller metrics) |
| Scale Up / Down    | Horizontal Pod Autoscaler (HPA) actions |
| Overload penalty   | SLA violation / customer-facing latency spike |
| Waste penalty      | Over-provisioned cloud spend |

---

## 🎯 Tasks (Easy → Hard)

| Task | Difficulty | Load Pattern | Episode Length | Description |
|------|------------|-------------|----------------|-------------|
| `steady-load` | 🟢 Easy | Constant 0.3–0.5 | 100 steps | Maintain stability under predictable moderate load |
| `diurnal-cycle` | 🟡 Medium | Sinusoidal 0.2–0.8 | 150 steps | Handle predictable daily traffic patterns |
| `spike-resilience` | 🔴 Hard | Random spikes to 0.95 | 200 steps | Survive unpredictable flash-crowd events |

### Grader

Each task is scored on a **0.0–1.0** scale:

```
score = (1.0 - overload_ratio) - (waste_ratio × 0.3)
```

- **Overload ratio**: fraction of steps where CPU or MEM > 90%
- **Waste ratio**: fraction of steps where CPU and MEM < 20% and load < 30%

---

## ⚙️ Action Space

| `action_id` | Name | Effect |
|-------------|------|--------|
| 0 | Idle | Do nothing |
| 1 | Alloc CPU | Relieve CPU pressure (−10%) |
| 2 | Alloc MEM | Relieve memory pressure (−10%) |
| 3 | Scale Up | +1 container (distributes load) |
| 4 | Scale Down | −1 container (saves cost) |

---

## 👁️ Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `cpu_used` | float | 0.0–1.0 | Fraction of CPU consumed |
| `mem_used` | float | 0.0–1.0 | Fraction of memory consumed |
| `containers` | int | 1–20 | Number of running containers |
| `load` | float | 0.0–1.0 | Incoming request load |
| `step_count` | int | 0–200 | Current step in the episode |
| `message` | str | — | Human-readable status |

---

## 🏅 Reward Function

| Condition | Signal |
|-----------|--------|
| Balanced efficiency (CPU ≈ load) | `+1.0 × efficiency` |
| CPU or MEM > 90% (overload) | `−3.0` |
| Resources idle & low load (waste) | `−0.5` |
| Per-container cost | `−0.05 × containers` |
| Scaling action (erratic changes) | `−0.10` |
| Idle action (passivity) | `−0.10` |

---

## 📊 Baseline Scores

| Method | steady-load | diurnal-cycle | spike-resilience | Avg |
|--------|-------------|---------------|------------------|-----|
| Random | TBD | TBD | TBD | TBD |
| Rule-based | TBD | TBD | TBD | TBD |
| LLM (Qwen 72B) | TBD | TBD | TBD | TBD |

*(Scores will be filled after running `inference.py`)*

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the OpenEnv server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run the baseline inference

```bash
export HF_TOKEN=your_token_here
export IMAGE_NAME=cloud-resource-env:latest
python inference.py
```

### 4. Docker

```bash
# Build
docker build -t cloud-resource-env:latest .

# Run OpenEnv server
docker run -p 7860:7860 cloud-resource-env:latest

# Run Gradio demo
docker run -p 7860:7860 -e MODE=demo cloud-resource-env:latest
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | For inference | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | For inference | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | For inference | — | Hugging Face API token |
| `IMAGE_NAME` | For inference | — | Docker image name |
| `MODE` | For Docker | `server` | `server` (OpenEnv) or `demo` (Gradio) |
| `TASK_NAME` | Optional | `steady-load` | Which task to run |
| `ENV_SEED` | Optional | `42` | Random seed for reproducibility |

---

## 📁 Project Structure

```
meta-hackthon/
├── models.py                # @dataclass Action, Observation, State
├── client.py                # EnvClient subclass
├── tasks.py                 # Task configs + grader functions
├── inference.py             # Baseline LLM agent (mandatory)
├── openenv.yaml             # OpenEnv manifest
├── server/
│   ├── cloud_environment.py # Environment subclass (core physics)
│   └── app.py               # create_fastapi_app server
├── cloud_env.py             # Original Gymnasium env (for DQN training)
├── q_agent.py               # DQN training/evaluation
├── app.py                   # Gradio interactive demo
├── Dockerfile               # Docker build (MODE switch)
├── requirements.txt         # Python dependencies
└── models/
    └── dqn_cloud.pth        # Trained DQN weights
```

---

## 🤖 Agent Details

**Deep Q-Network (DQN)** with:
- 2-hidden-layer MLP: 4 → 128 → 128 → 5
- Experience replay buffer (20k transitions)
- Target network (hard update every 200 steps)
- Epsilon-greedy exploration (ε: 1.0 → 0.05)

### Train locally

```bash
python q_agent.py --train --episodes 500 --save models/dqn_cloud.pth
python q_agent.py --eval  --load models/dqn_cloud.pth
```

---

## 📜 License

MIT

---

> Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · Meta × Scaler AI Hackathon 2026
