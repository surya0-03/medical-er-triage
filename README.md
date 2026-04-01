---
title: Medical ER Triage OpenEnv
emoji: 👁
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---


# Medical ER Triage OpenEnv
 
Production-grade, deterministic OpenEnv-compatible reinforcement learning environment for emergency room triage.
 
> **Note on API Credits:** We are not using OpenAI for inference due to lack of API credits. This project uses [Groq](https://console.groq.com) (free tier) as the LLM backend for the baseline agent, with a deterministic rule-based fallback requiring no API key at all.
 
---
 
## Description
 
This project implements a strict triage simulator with:
 
- Deterministic seeded dynamics for arrivals, deterioration, and delayed outcomes.
- Strongly typed data contracts via Pydantic models.
- Explicit task modes (`easy`, `medium`, `hard`) with fixed termination rules.
- Action/observation schemas exposed through `openenv.yaml`.
- FastAPI runtime API for reset/step/state/task metadata/grader/baseline metadata.
 
Core modules:
 
- `my_env/server/data/constants.py`: all fixed numeric rules and grading constants.
- `my_env/models.py`: strict observation/action/reward and validation.
- `my_env/server/data/patient_dataset.py`: 50-patient balanced ESI dataset.
- `my_env/server/data/protocols.py`: protocol definitions and cooldown tracking.
- `my_env/server/environment.py`: environment engine (`reset`, `step`, `state`).
- `my_env/server/app.py`: FastAPI endpoints.
- `my_env/baseline.py`: baseline inference runner (internal module).
- `my_env/openenv.yaml`: OpenEnv metadata and schemas.
- `inference.py`: **entry point for hackathon evaluation** — runs baseline agent across all tasks.
 
---
 
## Repository Structure
 
```
medical-er-triage/
├── inference.py          # Hackathon evaluation entry point
├── Dockerfile            # Docker deployment config
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependencies (required)
├── demo.py               # Demo script
└── my_env/
    ├── models.py
    ├── baseline.py
    ├── client.py
    ├── openenv.yaml
    ├── requirements.txt
    └── server/
        ├── app.py
        ├── environment.py
        └── Dockerfile
```
 
---
 
## Action and Observation Spaces
 
**Action space:**
- `action_type`: `assign_esi | allocate_bed | discharge | trigger_protocol | divert`
- `patient_id`: string or null
- `esi_level`: integer in `[1, 5]` or null
- `bed_type`: `icu | general | hallway | none | null`
- `protocol_type`: `stroke_code | stemi_alert | sepsis_alert | trauma_alert | null`
 
**Observation space:**
- `time_step`: integer `>= 0`
- `patients`: list of typed patient objects
- `waiting_queue`: list of patient ids
- `icu_beds_available`: integer in `[0, 8]`
- `general_beds_available`: integer in `[0, 20]`
- `hallway_used`: integer in `[0, 3]`
- `crowding_score`: float `>= 0`
- `staff_load`: typed load dictionary
 
---
 
## Tasks
 
| Task | Max Steps | Termination |
|------|-----------|-------------|
| `easy` | 5 | Step limit reached |
| `medium` | 12 | Step limit reached |
| `hard` | 20 | Step limit OR 3 ESI-1 deaths |
 
**Hard mode constraints:** ICU starts with 2 free beds, General starts with 4 free beds.
 
**Time and flow per step (10 min each):**
1. action → 2. beds → 3. wait time → 4. deterioration → 5. arrivals → 6. reward
 
---
 
## Baseline Scores
 
### Groq — llama-3.3-70b-versatile (primary, free tier)
 
> OpenAI is **not used** due to lack of API credits. Groq provides a free tier at [console.groq.com](https://console.groq.com).
 
| Task | Steps | Total Reward | Avg Reward |
|------|-------|-------------|------------|
| Easy | 5 | 2.470296 | 0.494059 |
| Medium | 12 | 5.604389 | 0.467032 |
| Hard | 20 | 9.531375 | 0.476569 |
| **Macro** | — | **17.606060** | **0.479220** |
 
### Deterministic Fallback (no API key required)
 
| Task | Steps | Total Reward | Avg Reward |
|------|-------|-------------|------------|
| Easy | 5 | 2.339261 | 0.467852 |
| Medium | 12 | 5.598458 | 0.466538 |
| Hard | 20 | 8.961030 | 0.448052 |
| **Macro** | — | **16.898749** | **0.460814** |
 
The LLM agent (Groq) outperforms the deterministic fallback by ~4% on macro average reward.
 
---
 
## Setup and Running
 
### Prerequisites
 
This project uses `uv` for dependency management. `uv.lock` is required for reproducibility.
 
```bash
pip install uv
uv sync
```
 
Or using pip directly:
 
```bash
pip install -r my_env/requirements.txt
```
 
### Run inference.py (hackathon evaluation)
 
**Option 1 — Groq (recommended, free tier):**
```bash
# Get a free API key at https://console.groq.com
export GROQ_API_KEY=your_groq_key_here
python inference.py
```
 
**Option 2 — OpenAI (if you have credits):**
```bash
export OPENAI_API_KEY=your_openai_key_here
export OPENAI_MODEL=gpt-4o-mini          # optional, default is gpt-4o-mini
export OPENAI_INFERENCE_SEED=12345       # optional, for reproducibility
python inference.py
```
 
**Option 3 — Deterministic fallback (no API key needed):**
```bash
export OPENAI_BASELINE_FALLBACK=deterministic
python inference.py
```
 
On Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="your_groq_key_here"
python inference.py
```
 
### Run the API server locally
 
```bash
uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860
```
 
### Run with Docker
 
```bash
docker build -t medical-er-triage -f Dockerfile .
docker run -p 7860:7860 medical-er-triage
```
 
---
 
## API Reference
 
FastAPI service: `my_env/server/app.py` — live at [https://maldini03-medical-er-triage-env.hf.space](https://maldini03-medical-er-triage-env.hf.space)
 
### POST /reset
 
```json
{
  "difficulty": "easy | medium | hard",
  "seed": 42,
  "session_id": "optional_existing_session"
}
```
 
### POST /step
 
```json
{
  "session_id": "optional_default_session",
  "action": {
    "action_type": "assign_esi",
    "patient_id": "p_00001",
    "esi_level": 2,
    "bed_type": null,
    "protocol_type": null
  }
}
```
 
### GET /state
Returns current session state snapshot. Query param: `session_id` (default: `default`).
 
### GET /grade
Returns per-episode score in `[0.0, 1.0]`, pass/fail flag, difficulty, steps, and summary metadata.
 
### GET /score
Returns detailed scoring with clinical insights: `priority_queue`, `time_of_day`, `is_peak_hours`, `acuity_score`, `mortality_rate`, `avg_wait_time`, `outcomes_pending`.
 
### GET /health
Returns liveness metadata.
 
### GET /tasks
Returns task definitions, hard-start settings, and resource capacities.
 
### GET /grader
Returns reward formula, grading constants, deterioration rules, staffing penalty settings, and delayed outcome mapping.
 
Normalization: `score = clamp01((raw_step_reward + bounded_outcome_component + (1 + outcome_cap)) / (2 * (1 + outcome_cap)))`
 
### GET /baseline
Returns baseline policy metadata.
 
### Test endpoints
 
```bash
# Health check
curl https://maldini03-medical-er-triage-env.hf.space/health
 
# Reset
curl -X POST https://maldini03-medical-er-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'
 
# Step
curl -X POST https://maldini03-medical-er-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign_esi", "patient_id": "p_00000", "esi_level": 2}}'
 
# Grade
curl "https://maldini03-medical-er-triage-env.hf.space/grade?session_id=default"
```
 
---
 
## OpenEnv Interface
 
Environment class: `my_env.server.environment.MedicalEmergencyRoomEnv`
 
- `reset(seed=...) -> Observation`
- `step(action) -> (Observation, Reward, done, info)`
- `state() -> dict`
- `step_gym(action) -> (Observation, Reward, terminated, truncated, info)`
 
OpenEnv manifest: `my_env/openenv.yaml` — includes environment name, version, entrypoint, task definitions, action schema, and observation schema.
 
---
 
## Hugging Face Spaces
 
Live deployment: [https://huggingface.co/spaces/maldini03/medical-er-triage-env](https://huggingface.co/spaces/maldini03/medical-er-triage-env)
 
- Docker-based Space running FastAPI via `uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860`
- Space tag: `openenv`
- Swagger UI: [https://maldini03-medical-er-triage-env.hf.space/docs](https://maldini03-medical-er-triage-env.hf.space/docs)
- Configure the Space metadata with tag `openenv`.
- Image entrypoint uses `uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860`.
