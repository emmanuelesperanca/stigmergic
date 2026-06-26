# 🐜 StigmergicAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

> **A zero-dependency, mathematically secure multi-agent framework based on Stigmergy, Latent State Transfer, and Byzantine Fault Tolerance.**

Forget API chaining, central orchestrators, and fragile `Agent-A -> Agent-B` string-passing. **StigmergicAI** reimagines Multi-Agent Systems (MAS) not as corporate org charts, but as distributed physical ecosystems. 

If LangGraph is a rigid assembly line, StigmergicAI is an ant colony: decentralized, highly resilient, and mathematically protected against LLM hallucinations.

---

## 🧠 The Three Architectural Horizons

Current AI orchestration frameworks suffer from the **String Tax** (serializing rich latent thoughts into lossy text) and **Centralized Fragility** (if the orchestrator fails, the system halts). StigmergicAI solves this through three pillars:

### 1. Stigmergic Swarms (The Environment as the Orchestrator)
Agents in this framework **do not talk to each other**. Instead, they interact exclusively by reading and modifying a shared semantic field (a database like SQLite, DynamoDB, or OpenSearch). 
* **Zero Coupling:** An agent wakes up, senses high entropy (unresolved tasks), acts, lowers the entropy, and dies. 
* **Absolute Resilience:** If an LLM API crashes, the task remains in the database. When the API recovers, the swarm naturally resumes.

### 2. Byzantine Cognitive Consensus (Semantic Raft)
You wouldn't trust a single microservice with your database; why trust a single LLM? 
Before any state mutation is committed to the environment, it must pass a **Byzantine Quorum**. Lightweight NLI (Natural Language Inference) models acting as validation nodes independently verify the logical derivation of the proposed action. If a node detects a hallucination or prompt injection, the transaction is slashed.

### 3. Latent State Transfer (LSC)
Instead of forcing SLMs/LLMs to compress their reasoning into text strings (Remote Procedure Calls), StigmergicAI provides primitives for **Latent State Calls**. Agents can inject their hidden activation tensors directly into the residual stream of downstream agents, passing pure mathematical context with zero token-generation latency.

---

## ⚡ Quick Start: The Chaos Monkey Test

You don't need a heavy vector database to understand Stigmergy. Let's run a swarm using a local SQLite file as our physical environment.

```bash
pip install -r requirements.txt
python examples/01_hello_stigmergy.py
