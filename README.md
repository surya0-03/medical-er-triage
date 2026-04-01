---
title: Medical ER Triage OpenEnv
emoji: 👁
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - medical-triage
  - emergency-medicine
---

# 🏥 Medical ER Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4CAF50)](https://github.com/openenv)
[![Hugging Face](https://img.shields.io/badge/🤗-Space-FFD21E)](https://huggingface.co/spaces/maldini03/medical-er-triage-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM-FF6600)](https://console.groq.com)

**Production-grade, deterministic OpenEnv-compatible reinforcement learning environment for emergency room triage.**

> 🚑 **Real-world simulation** grounded in ESI algorithm, NHAMCS patient distributions, and CMS resource benchmarks.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Environment Architecture](#environment-architecture)
- [Quick Start](#quick-start)
- [Action & Observation Spaces](#action--observation-spaces)
- [Tasks & Difficulty Modes](#tasks--difficulty-modes)
- [Reward Structure](#reward-structure)
- [Baseline Performance](#baseline-performance)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Citation & Credits](#citation--credits)

---

## Overview

This environment simulates the complex decision-making process in an Emergency Department (ED) where an agent must:
- **Triage patients** using the Emergency Severity Index (ESI) algorithm (levels 1-5)
- **Allocate scarce resources** (ICU beds, general beds, hallway spaces)
- **Activate time-critical protocols** (stroke, STEMI, sepsis, trauma)
- **Manage patient flow** from arrival through treatment to discharge
- **Handle dynamic stressors** (crowding, staff load, patient deterioration)

### Key Features

| Feature | Description |
|---------|-------------|
| **Deterministic Seeding** | Fully reproducible simulations with fixed seeds |
| **Typed Data Contracts** | Strict Pydantic models for actions, observations, and rewards |
| **Difficulty Modes** | Easy (5 steps), Medium (12 steps), Hard (20 steps + ESI-1 death limits) |
| **Realistic Patient Mix** | 50-patient dataset with balanced ESI distributions |
| **Clinical Protocols** | Stroke, STEMI, Sepsis, and Trauma alerts with cooldown tracking |
| **Staff Load Modeling** | Nurse-to-patient ratios affect outcomes |
| **Delayed Outcomes** | Patient outcomes computed after simulation (mortality, length of stay) |
| **Docker Deployment** | Ready-to-run containerized application |

---

## Why This Matters

Emergency Departments face critical challenges globally:
- **Overcrowding** leads to increased mortality and poor outcomes
- **Triage errors** cause delayed care for critical patients
- **Resource misallocation** creates bottlenecks
- **Staff burnout** from unsustainable workloads

This environment provides a safe, reproducible sandbox to develop and test AI policies that could:
- Reduce wait times by 20-30%
- Improve patient outcomes through better prioritization
- Optimize bed allocation to prevent hallway boarding
- Support clinical decision-making during surge events

---

## Environment Architecture

### Simulation Flow (per 10-minute step)
```
┌─────────────────┐
│ Agent Action │
└────────┬────────┘
▼
┌─────────────────┐
│ Bed Allocation │ ◄── ICU/General/Hallway capacity checks
└────────┬────────┘
▼
┌─────────────────┐
│ Wait Time │ ◄── Queue progression, staffing impact
└────────┬────────┘
▼
┌─────────────────┐
│ Deterioration │ ◄── ESI-dependent clinical decline
└────────┬────────┘
▼
┌─────────────────┐
│ New Arrivals │ ◄── Poisson-distributed patient arrivals
└────────┬────────┘
▼
┌─────────────────┐
│ Reward │ ◄── Outcomes + Efficiency + Utilization
└─────────────────┘
```
### State Machine
```
Waiting ──allocate_bed──► Assigned Bed ──► In Treatment ──► Completed
│ │ │
│ │ │
└───────────────────────────┴────────────────┘
│
▼
Outcome Pending
│
▼
Resolved
```
---

## Quick Start

### Prerequisites


#### Clone 
```
git clone https://github.com/surya0-03/medical-er-triage.git
cd medical-er-triage
```
#### Install with uv (recommended)
```
pip install uv
uv sync
```
#### Or with pip directly
` pip install -r my_env/requirements.txt `

### Run Baseline Evaluation
#### Option 1 — Groq (free tier, recommended):
```
Get free API key at https://console.groq.com
export GROQ_API_KEY="your_key_here"
python inference.py
```
#### Option 2 — Deterministic fallback (no API key):

` python inference.py '  Falls back automatically if no key

#### Option 3 — OpenAI (if you have credits):

```
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
python inference.py
```
### Expected Output

Running Groq (llama-3.3-70b-versatile) baseline.
```
=== Groq (llama-3.3-70b-versatile) Baseline Scores ===
task=easy steps=5 total_reward=2.470 avg_reward=0.494 terminated=True

task=medium steps=12 total_reward=5.604 avg_reward=0.467 terminated=True

task=hard steps=20 total_reward=9.531 avg_reward=0.477 terminated=True

macro_avg_reward=0.479
macro_total_reward=17.606
```
### Run API Server Locally

```
 uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860 --reload
```
Then visit:

API: http://localhost:7860

Swagger UI: http://localhost:7860/docs

ReDoc: http://localhost:7860/redoc

## Action & Observation Spaces

### Action Space

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `assign_esi` | Assign ESI level (1-5) to waiting patient | `patient_id`, `esi_level` |
| `allocate_bed` | Assign bed to triaged patient | `patient_id`, `bed_type` (icu/general/hallway) |
| `trigger_protocol` | Activate clinical protocol | `patient_id`, `protocol_type` |
| `discharge` | Discharge treated patient | `patient_id` |
| `divert` | Divert incoming ambulances | None |

**Valid protocol_type values:**

+ ``` stroke_code ``` - Acute stroke protocol

+ ``` stemi_alert ``` - Heart attack protocol

+ ``` sepsis_alert ``` - Severe infection protocol

+ ``` trauma_alert ``` - Major trauma protocol

### Observation Space

```
{
    "time_step": int,                    # Current step (0-500)
    "patients": List[Patient],           # All patients in system
    "waiting_queue": List[str],          # Patient IDs awaiting triage
    "icu_beds_available": int,           # 0-8 ICU beds free
    "general_beds_available": int,       # 0-20 general beds free
    "hallway_used": int,                 # 0-3 hallway spaces occupied
    "crowding_score": float,             # 0-1 congestion metric
    "staff_load": {
        "icu_nurses": int,               # ICU nursing staff available
        "ed_nurses": int,                # ED nursing staff available
        "icu_patients": int,             # Current ICU patient count
        "ed_patients": int,              # Current ED patient count
        "icu_nurse_load": float,         # ICU nurse-to-patient ratio
        "ed_nurse_load": float,          # ED nurse-to-patient ratio
        "total_nurse_load": float        # Aggregate workload metric
    }
}
```
### Patient Object
```
{
    "patient_id": str,
    "age": int,                          #0-130 years
    "symptoms": List[str],               #Clinical presentation
    "vitals": {
        "heart_rate": int,               #bpm
        "blood_pressure": str,           #e.g., "120/80"
        "oxygen_level": float            #0-100%
    },
    "esi_level": int,                    #1-5 (1=most critical)
    "wait_time": int,                    #Minutes waiting
    "state": str,                        #See state machine above
    "deterioration_count": int           #Number of deteriorations
}
```
## Tasks & Difficulty Modes

| Mode | Max Steps | Termination Condition | Starting Resources | Description |
|------|-----------|----------------------|-------------------|-------------|
| **Easy** | 5 | Step limit reached | ICU: 8 beds, General: 20 beds | Simple patient flow, low acuity, ample resources |
| **Medium** | 12 | Step limit reached | ICU: 6 beds, General: 15 beds | Mixed acuity, moderate crowding, constrained resources |
| **Hard** | 20 | Step limit OR 3 ESI-1 deaths | ICU: 2 beds, General: 4 beds | High acuity, severe crowding, resource shortage |

**Hard mode constraints:**

+ Limited ICU capacity (2 beds) - critical for ESI-1 patients

+ High patient acuity with frequent deteriorations

+ Mortality risk if critical patients aren't prioritized

## Reward Structure

The reward function combines multiple clinical and operational metrics, normalized to [0.0, 1.0]:

### Components
Component	Weight	Description :
+ Patient Outcomes	0-0.5	Survival, mortality prevention, quality-adjusted life years
+ Wait Time Reduction	-0.01/min	Penalty for prolonged waiting
+ Resource Utilization	0-0.3	Efficient bed allocation, avoiding hallway boarding
+ Protocol Adherence	0-0.2	Following best practices (stroke, STEMI, sepsis alerts)
### Reward Formula
```
score = clamp01(
    (raw_step_reward + bounded_outcome_component + (1 + outcome_cap)) 
    / (2 * (1 + outcome_cap))
)
```
### Outcome Categories

Outcome	Score	Description
+ Excellent	0.8-1.0	Full recovery, timely care
+ Good	0.6-0.8	Minor complications, adequate care
+ Fair	0.4-0.6	Moderate complications, delays
+ Poor	0.2-0.4	Severe complications, major delays
+ Critical	0.0-0.2	Death or permanent disability

## Baseline Performance
### Groq — llama-3.3-70b-versatile (Free Tier)
| Task	| Steps |	Total Reward |	Avg Reward |
|------|-------|--------------|------------|
| **Easy**	| 5	| 2.470 |	0.494 |
| **Medium**	| 12	| 5.604	| 0.467 |
| **Hard**	| 20	| 9.531	| 0.477 |
| **Macro** | — |	17.606	| 0.479 |

### Deterministic Fallback (No API Key)
| Task | Steps | Total Reward | Avg Reward |
|------|-------|--------------|------------|
| Easy | 5 | 2.339 | 0.468 |
| Medium | 12 | 5.598 | 0.467 |
| Hard | 20 | 8.961 | 0.448 |
| **Macro** | — | **16.899** | **0.461** |
**LLM outperforms deterministic by ~4% on macro average reward, demonstrating the value of adaptive reasoning in complex medical triage decisions.**

## API Reference
### Base URL
Production: https://maldini03-medical-er-triage-env.hf.space

Local: http://localhost:7860

### Endpoints
Endpoint	Method	Description
+ ` /health `	- **GET**	Liveness check
+ ` /reset `	- **POST**	Reset environment with difficulty and seed
+ ` /step `	- **POST**	 Execute an action
+ ` /state `	- **GET**	Get current state snapshot
+ ` /grade	` - **GET**	Get episode score (0-1) and summary
+ ` /score	` - **GET**	Detailed clinical scoring
+ ` /tasks	` - **GET**	Task definitions and configurations
+ ` /grader `	- **GET**	Reward function details
+ ` /baseline `	- **GET**	Baseline policy metadata
+ ` /docs `	- **GET**	Swagger UI documentation

### Example API Calls

#### Reset environment
```
curl -X POST https://maldini03-medical-er-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'
```
#### Take action
```
curl -X POST https://maldini03-medical-er-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign_esi", "patient_id": "p_00000", "esi_level": 2}}'
```
#### Get current state
` curl "https://maldini03-medical-er-triage-env.hf.space/state?session_id=default" `

#### Get grade
` curl "https://maldini03-medical-er-triage-env.hf.space/grade?session_id=default" `

#### Get detailed score with clinical insights
` curl "https://maldini03-medical-er-triage-env.hf.space/score?session_id=default" `

## Deployment

### Docker Build

#### Build image
` docker build -t medical-er-triage -f Dockerfile . `

#### Run container
```
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_key" \
  medical-er-triage
```
### Hugging Face Space
The environment is live at:

+ Space: https://huggingface.co/spaces/maldini03/medical-er-triage-env

+ API: https://maldini03-medical-er-triage-env.hf.space

+ Swagger UI: https://maldini03-medical-er-triage-env.hf.space/docs

The Space uses:

+ Docker-based deployment

+ FastAPI backend

+ Uvicorn server

+ Automatic GPU detection (if available)

## Project Structure
```
medical-er-triage/
├── inference.py              #🎯 Hackathon evaluation entry point
├── Dockerfile                #🐳 Docker deployment config
├── pyproject.toml            #📦 Project metadata and dependencies
├── uv.lock                   #🔒 Locked dependencies
├── README.md                 #📖 This file
├── demo.py                   #🎬 Demo script
│
└── my_env/                   # 🧠 Core environment
    ├── __init__.py
    ├── models.py             # Pydantic models (action, observation, reward)
    ├── inference.py          # Baseline agent implementation
    ├── client.py             # Environment client wrapper
    ├── openenv.yaml          # OpenEnv specification
    ├── requirements.txt      # Python dependencies
    │
    └── server/               # 🚀 API server
        ├── __init__.py
        ├── app.py            # FastAPI application
        ├── environment.py    # MedicalEmergencyRoomEnv core class, 50-patient dataset (NHAMCS-based), Fixed numeric rules and grading and Protocol definitions and cooldown
        ├── Dockerfile        # Server-specific Docker config
```

## Citation & Credits
### Dataset Sources
+ NHAMCS (National Hospital Ambulatory Medical Care Survey): Patient arrival distributions

+ ESI (Emergency Severity Index): Triage algorithm 

+ CMS (Centers for Medicare & Medicaid Services): Resource benchmarks

### Academic References
+ Gilboy, N., et al. (2012). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. Agency for Healthcare Research and Quality.

+ McCarthy, M. L., et al. (2018). "Crowding in Emergency Departments." Health Affairs.

+ McHugh, M., & Van Dyke, K. (2017). "Nurse Staffing and Patient Outcomes." Medical Care Research and Review.

### Technologies Used
+ FastAPI - Modern web framework

+ Pydantic - Type validation

+ Groq - Free LLM inference

+ Docker - Containerization

+ Hugging Face Spaces - Hosting

### License
+ MIT License - See LICENSE file for details.

## Contact & Support
+ GitHub Issues: Report bugs or request features

+ Hugging Face Space: View live demo

+ Author: @surya0-03

## Acknowledgments
Special thanks to:

+ The OpenEnv team for the environment specification

+ Groq for providing free LLM inference credits

+ Hugging Face for hosting infrastructure

+ The emergency medicine community for domain expertise

Made with ❤️ for better emergency care through AI
