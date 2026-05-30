# QuantHive V1 — Single-Agent Hackathon Legacy Codebase

This directory houses the original V1 codebase of **QuantHive**, developed during the hackathon phase. It implements a single-agent financial simulation using a custom Gymnasium environment supervised by an LLM Judge and controlled by a local fine-tuned LLM policy.

You can package this folder directly as its own self-contained Git repository if you wish to maintain or publish it separately.

---

## Architecture Overview

In V1, the system is designed around a single learning agent (the **Trader**) guided by external heuristic advice and audited by an LLM Judge:

```
                  ┌────────────────────────────────────────┐
                  │          Market Technicals (RSI/EMA)   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
┌──────────────────┐    ┌───────────────────────────┐    ┌──────────────────┐
│Risk Model        ├───>│Local Policy Model (Qwen)  │<───┤LLM Judge         │
│(Dynamic Sizing)  │    │(Reasoning & Action Head)  │    │(Reward Auditing) │
└──────────────────┘    └─────────────┬─────────────┘    └──────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │Gymnasium Environment      │
                        │(Slippage & Funding Costs) │
                        └───────────────────────────┘
```

---

## Directory Layout

*   **`env/`**
    *   `trading_env.py` — The core Gymnasium single-agent trading environment, featuring synthetic regime generation (8 regimes), slippage modeling, and trade execution.
*   **`agents/`**
    *   `trader.py` — Coordinates TA signals, FA sentiment, and risk constraints, feeding them to the policy model to output the final trade details.
    *   `risk_model.py` — Sets dynamic position limits based on trailing volatility and portfolio drawdown.
    *   `researcher.py` — A rules-based technical analyst generating buy/sell signals from RSI, MACD, and Bollinger Bands.
    *   `fa_agent.py` — A rules-based fundamental analyst generating trend-following sentiment bias.
    *   `portfolio_manager.py` — An optional LLM-driven strategic veto/override meta-agent.
*   **`policy/`**
    *   `local_model.py` — Coordinates inference of local causal language models (e.g., Qwen2.5-1.5B) using `<thought>` and `<action>` tags for structured reasoning.
*   **`training/`**
    *   `train.py` — Main SFT data collection and RL training loop using the LLM Judge.
    *   `train_cpu.py` — A lightweight training script designed for CPU-only execution.
    *   `train_grpo.py` — GRPO reinforcement learning trainer using Unsloth for local parameter-efficient training.
    *   `grpo_verifiers_multiagent.py` & `train_grpo_multiagent.py` — Early experimental GRPO implementations for multi-agent negotiation.
    *   `prompt_utils.py` — Helpers for preparing formatting prompts and system contexts for the LLM.
    *   `benchmark.py` — Runs simple comparative evaluations of baselines.
    *   `evaluate_live.py` — Connects the trained agent to live simulation feeds.

---

## Getting Started (V1 Run Guide)

### 1. Requirements & Setup
Make sure you have your environment variables set up in a `.env` file at the root:
```env
USE_LOCAL_POLICY=true
LOCAL_MODEL_PATH="models/local_policy"
ALLOW_CPU_LOCAL_POLICY=true
OPENAI_API_KEY="your-key-here"  # Required if using OpenAI-based Portfolio Manager
```

### 2. Running Training
To collect trajectories and run the main single-agent loop:
```bash
python -m training.train
```

To run a GRPO training session using Unsloth (requires a CUDA GPU):
```bash
python -m training.train_grpo --model-name unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```
