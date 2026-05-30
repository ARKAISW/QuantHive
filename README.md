# 🏛️ QuantHive V1 — Autonomous AI Agent Trading & LLM Governance

[![OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![Framework](https://img.shields.io/badge/Framework-Gymnasium-green.svg?style=for-the-badge)](https://gymnasium.farama.org/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen%202.5--1.5B-purple.svg?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Theme](https://img.shields.io/badge/Theme-Scalable%20Oversight-orange.svg?style=for-the-badge)](https://hackathon.openenv.org)

QuantHive V1 is a state-of-the-art **autonomous financial agent trading system** that leverages local fine-tuned causal language models and reinforcement learning (RL) alignment to execute trades under dynamic risk constraints. The core architecture bridges high-performance quantitative trading simulations with structured Chain-of-Thought (CoT) semantic reasoning, audited by an adversarial **LLM Judge**.

Developed as an elite project for the **OpenEnv April '26 Hackathon**, QuantHive V1 demonstrates how autonomous agents can learn to read technical market indicators, respect heuristic risk constraints, and explain their decisions in a verifiable, auditable format.

---

## 🔬 Core Architecture Overview

QuantHive V1 coordinates a single learning agent (the **Trader**) guided by heuristic technical and fundamental advisors, audited in real-time by an **LLM Judge**:

```text
                  ┌────────────────────────────────────────┐
                  │      Market Technicals (RSI/EMA/BB)    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
┌──────────────────┐    ┌───────────────────────────┐    ┌──────────────────┐
│Heuristic Risk    │    │Local Causal LLM (Qwen)    │    │Adversarial Judge │
│Constraints Model ├───>│Chain-of-Thought reasoning │<───┤LLM Auditing      │
│(Drawdown Caps)   │    │  <thought> & <action>     │    │(Reward Scoring)  │
└──────────────────┘    └─────────────┬─────────────┘    └──────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │OpenEnv Gymnasium Simulator│
                        │(Slippage & Funding Costs) │
                        └───────────────────────────┘
```

---

## ⚡ Key Features

* **🧠 Causal CoT Reasoning:** The trading agent processes market signals and explains its strategic intent inside structured `<thought>` tags before outputting executable JSON inside `<action>` tags.
* **🛡️ Calibrated Risk Management:** Trailing volatility and portfolio drawdown feed into a dynamic heuristic constraints engine, setting hard sizing caps that the agent must learn to respect.
* **⚖️ Scalable LLM Oversight:** An adversarial **LLM Judge** monitors trajectories in real-time, providing semantic alignment rewards to fine-tune and align the trading policy.
* **⚡ Parameter-Efficient Fine-Tuning:** Full training pipeline using **Unsloth (LoRA)** to train Qwen 2.5-1.5B locally on custom SFT trajectories, followed by **GRPO Alignment**.
* **💻 Interactive UI Dashboard:** A premium, state-of-the-art **Vite React** frontend dashboard connected to a high-speed **FastAPI** simulation backend.

---

## 📂 Detailed Directory Structure

The repository is modularly structured, separating quantitative environments, reasoning agents, and interactive server layers:

| Component | Path | Description |
| :--- | :--- | :--- |
| **Environment** | [`env/trading_env.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/env/trading_env.py) | gymnasium-compliant quantitative market simulator featuring 8 market regimes. |
| **Causal Brain** | [`policy/local_model.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/policy/local_model.py) | Engine managing inference for local causal models (Qwen) with custom tags. |
| **Trader Agent** | [`agents/trader.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/agents/trader.py) | Coordinates researchers, sentiment analysis, and risk limits for the LLM prompt. |
| **Risk Modeler** | [`agents/risk_model.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/agents/risk_model.py) | Calculates dynamic position limits and trailing drawdown constraints. |
| **Technicals** | [`agents/researcher.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/agents/researcher.py) | Rules-based researcher generating market indicator bounds (RSI, MACD, Bollinger Bands). |
| **Fundamental** | [`agents/fa_agent.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/agents/fa_agent.py) | Analyzes macro/fundamental trend sentiment bias. |
| **FastAPI Backend** | [`api/server.py`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/api/server.py) | High-speed API server orchestrating simulator resets, steps, and real-time streaming. |
| **Vite Frontend** | [`ui/`](file:///run/media/arka/New%20Volume/DEVELOPMENT/Quanthive%20v1/ui) | Sleek, modern dashboard built with TailwindCSS, Zustand, and React. |

---

## 🚀 The Training & Alignment Pipeline

QuantHive V1 features a robust two-stage training methodology to align the agent from raw quantitative rules to highly-compliant semantic reasoning:

```text
  [Stage 1: Trajectory Collection] ──> SFT Data Generation (Rule-based Trajectories)
                                                    │
                                                    ▼
  [Stage 2: Supervised Fine-Tuning] ──> Unsloth LoRA Fine-Tuning on Qwen 2.5
                                                    │
                                                    ▼
  [Stage 3: RL Alignment (GRPO)] ──> 5-Verifier Rule-Based & Semantic Reward Loop
                                        - Structured format compliance
                                        - Technical indicator alignment
                                        - Position sizing constraints
                                        - Trade direction profitability
                                        - Adaptive governance bounds
```

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisite Packages
Install standard dependencies for the simulation backend and server:
```bash
pip install -r requirements-space.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
USE_LOCAL_POLICY=true
LOCAL_MODEL_PATH="models/local_policy"
ALLOW_CPU_LOCAL_POLICY=true
OPENAI_API_KEY="your-api-key"
```

### 3. Launching the Local Interactive Dashboard
Launch the FastAPI backend server on port `7860`:
```bash
python app.py --demo
```

To run the Vite React dashboard locally:
```bash
cd ui
npm install
npm run dev
```

---

## 🧠 SFT & GRPO Training Run Commands

### 1. Collect Trajectories & Run standard Training
To run the SFT trajectory collector and fine-tune your model:
```bash
python -m training.train
```

### 2. Run CPU-Only Lightweight Training
Designed for low-resource environments:
```bash
python -m training.train_cpu
```

### 3. Run GRPO Alignment using Unsloth (CUDA Enabled)
To run a Group Relative Policy Optimization alignment session using Unsloth 4-bit quantization:
```bash
python -m training.train_grpo --model-name unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

---

## 🏛️ Verifiable Reasoning Output Example

When fully trained, the QuantHive Trader successfully coordinates market signals and risk constraints inside its auditable Chain-of-Thought reasoning:

```markdown
<thought>
The market technicals indicate an oversold condition (RSI is at 28.4, below the 35 threshold). 
However, current portfolio drawdown stands at 4.2%, which is close to our maximum limit. 
The Risk Modeler has set a dynamic position sizing limit of 0.35. 
To optimize profit while respecting governance bounds, I will propose a conservative long position of 0.25 size, with a strict stop-loss set at 98.50.
</thought>
<action>
{"direction": 1, "size": 0.25, "sl": 98.50, "tp": 105.20}
</action>
```

---

**Built for the OpenEnv April '26 Hackathon | Autonomous Trading & Scalable Oversight**  
**Author:** Arka Sarkar (arkasarkar1507@gmail.com)
