# Antibiotic Resistance Stewardship Environment

## Overview
Simulates clinical decision-making for antibiotic selection under resistance constraints. An AI agent observes patient data and must prescribe the correct antibiotic and duration, adapting to resistance levels and culture results.

## Action Space
| Field | Values | Description |
|-------|--------|-------------|
| `antibiotic` | A, B, C | Antibiotic to prescribe |
| `duration` | 3, 5, 7 | Treatment duration in days |

## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | int | Unique patient identifier |
| `infection_type` | str | respiratory / urinary / skin |
| `severity` | int (1–10) | Infection severity score |
| `resistance_level` | str | low / medium / high |
| `culture_result` | str or None | Confirmed pathogen (hard task only) |
| `step_count` | int | Current step in episode |

## Tasks
| Task | Description | Challenge |
|------|-------------|-----------|
| `easy` | No resistance complications | Match antibiotic to infection type |
| `medium` | Resistance present | Avoid antibiotic A for high-resistance cases |
| `hard` | Culture results revealed at step 2 | Adapt treatment mid-episode |

## Reward Function
- `+0.5` correct antibiotic for infection type
- `+0.2` appropriate duration (5 or 7 days)
- `-0.3` antibiotic A prescribed with high resistance
- `-0.5` wrong antibiotic

Final score is normalized to `[0.0, 1.0]` via `grade_episode()`.

## Setup

### Environment Variables
```
API_BASE_URL   The API endpoint for the LLM
MODEL_NAME     The model identifier to use for inference
HF_TOKEN       Your Hugging Face / API key
```

### Docker
```bash
docker build -t antibiotic-env .
docker run \
  -e API_BASE_URL=<your_url> \
  -e MODEL_NAME=<your_model> \
  -e HF_TOKEN=<your_token> \
  antibiotic-env
```

### Local
```bash
pip install -r requirements.txt
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...
python inference.py
```

## Baseline Scores
Baseline scores using a rule-following LLM agent:
- easy: ~0.85
- medium: ~0.70
- hard: ~0.65
