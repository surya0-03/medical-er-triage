# Medical ER Triage OpenEnv

Production-grade, deterministic OpenEnv-compatible reinforcement learning environment for emergency room triage.

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
- `my_env/baseline.py`: OpenAI API baseline inference runner.
- `my_env/openenv.yaml`: OpenEnv metadata and schemas.

## Action and Observation Spaces

- Action space:
  - `action_type`: `assign_esi | allocate_bed | discharge | trigger_protocol | divert`
  - `patient_id`: string or null
  - `esi_level`: integer in `[1, 5]` or null
  - `bed_type`: `icu | general | hallway | none | null`
  - `protocol_type`: `stroke_code | stemi_alert | sepsis_alert | trauma_alert | null`

- Observation space:
  - `time_step`: integer `>= 0`
  - `patients`: list of typed patient objects
  - `waiting_queue`: list of patient ids
  - `icu_beds_available`: integer in `[0, 8]`
  - `general_beds_available`: integer in `[0, 20]`
  - `hallway_used`: integer in `[0, 3]`
  - `crowding_score`: float `>= 0`
  - `staff_load`: typed load dictionary

## Tasks

Defined task modes:

- `easy`
  - Maximum steps: `5`
  - Termination: step limit reached
- `medium`
  - Maximum steps: `12`
  - Termination: step limit reached
- `hard`
  - Maximum steps: `20`
  - Additional termination: `3` ESI-1 deaths
  - Hard-start constraints: ICU starts with `2` free beds, General starts with `4` free beds

Time and flow:

- Time step: `10` minutes
- Update order in each step:
  1. action
  2. beds
  3. wait time
  4. deterioration
  5. arrivals
  6. reward

## API

FastAPI service file: `my_env/server/app.py`.

### POST /reset

Request JSON:

```json
{
  "difficulty": "easy | medium | hard",
  "seed": 42,
  "session_id": "optional_existing_session"
}
```

Response JSON:

```json
{
  "observation": {"...": "..."},
  "difficulty": "medium",
  "seed": 42,
  "session_id": "generated_or_reused_session"
}
```

### POST /step

Request JSON:

```json
{
  "session_id": "optional_default_session",
  "action": {
    "action_type": "assign_esi | allocate_bed | discharge | trigger_protocol | divert",
    "patient_id": "p_00001",
    "esi_level": 2,
    "bed_type": "icu | general | hallway | none",
    "protocol_type": "stroke_code | stemi_alert | sepsis_alert | trauma_alert"
  }
}
```

Response JSON:

```json
{
  "observation": {"...": "..."},
  "reward": {
    "value": 0.12,
    "breakdown": {"...": 0.0}
  },
  "terminated": false,
  "truncated": false,
  "info": {"...": "..."}
}
```

### GET /state

Returns current session state snapshot.

Query parameter:

- `session_id` (optional, defaults to `default`)

### GET /grade

Returns per-episode grading output:

- `score` in `[0.0, 1.0]`
- pass/fail flag
- difficulty, steps, and summary metadata

### GET /health

Returns liveness metadata for orchestrators and OpenEnv runners.

### GET /tasks

Returns task definitions, hard-start settings, and resource capacities.

### GET /grader

Returns reward formula, grading constants, deterioration rules, staffing penalty settings, and delayed outcome mapping.

- Normalization: `score = clamp01((raw_step_reward + bounded_outcome_component + (1 + outcome_cap)) / (2 * (1 + outcome_cap)))`

### GET /baseline

Returns baseline policy metadata.

## Setup

### Run locally

```bash
pip install -r my_env/requirements.txt
uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t medical-er-triage -f my_env/server/Dockerfile .
docker run -p 7860:7860 medical-er-triage
```

### Local setup

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r my_env/requirements.txt
```

2. Run API server:

```bash
uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860
```

3. Run OpenAI baseline agent:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_MODEL=gpt-4o-mini
set OPENAI_INFERENCE_SEED=12345
python -m my_env.baseline
```

If you hit `429 insufficient_quota`, run with local fallback:

```bash
set OPENAI_BASELINE_FALLBACK=deterministic
python -m my_env.baseline
```

### Docker setup

Build and run:

```bash
docker build -t medical-er-triage -f my_env/server/Dockerfile .
docker run -p 7860:7860 medical-er-triage
```

### Test endpoints

```bash
# Reset (returns session_id)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign_esi", "patient_id": "p_00000", "esi_level": 2}}'

# Grade
curl "http://localhost:7860/grade?session_id=default"

# State
curl "http://localhost:7860/state?session_id=default"
```

### Run baseline

```bash
export OPENAI_API_KEY=your_key_here
cd my_env
python -m baseline
```

## Baseline Scores (GPT-4o-mini, seed=42)

| Task   | Avg reward | Score  | Pass |
|--------|------------|--------|------|
| Easy   | ~0.72      | ~0.74  | Yes  |
| Medium | ~0.65      | ~0.70  | Yes  |
| Hard   | ~0.52      | ~0.59  | No   |

## Baseline Results

OpenAI baseline script evaluates all three tasks (`easy`, `medium`, `hard`) and prints:

- per-task steps, total reward, average reward
- macro average reward across all tasks
- macro total reward across all tasks

The script is configured for reproducibility with fixed environment seeds, temperature `0.0`, and `OPENAI_INFERENCE_SEED`.

## OpenEnv Metadata

OpenEnv manifest is provided in `my_env/openenv.yaml` and includes:

- environment name and version
- entrypoint
- task definitions
- action schema
- observation schema

## OpenEnv Interface

Environment class: `my_env.server.environment.MedicalEmergencyRoomEnv`

- `reset(seed=...) -> Observation`
- `step(action) -> (Observation, Reward, done, info)`
- `state() -> dict`

Compatibility helper:

- `step_gym(action) -> (Observation, Reward, terminated, truncated, info)`

## Hugging Face Spaces

- Docker deployment notes are in `my_env/HF_SPACES.md`.
- Configure the Space metadata with tag `openenv`.
- Image entrypoint uses `uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860`.
