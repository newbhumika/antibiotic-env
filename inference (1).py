import os
import json
from openai import OpenAI
from environment import AntibioticEnv, Action
from graders import grade_episode

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an antibiotic stewardship agent. Given a patient observation, choose the best antibiotic and treatment duration.

Rules:
- Antibiotic A treats respiratory infections
- Antibiotic B treats urinary infections  
- Antibiotic C treats skin infections
- Prefer duration 5 or 7 days over 3 days
- Avoid Antibiotic A when resistance_level is high

You must respond with ONLY a valid JSON object in this exact format:
{"antibiotic": "A", "duration": 5}

antibiotic must be one of: A, B, C
duration must be one of: 3, 5, 7"""


def get_action(obs) -> Action:
    obs_text = (
        f"Patient ID: {obs.patient_id}\n"
        f"Infection type: {obs.infection_type}\n"
        f"Severity: {obs.severity}/10\n"
        f"Resistance level: {obs.resistance_level}\n"
        f"Culture result: {obs.culture_result}\n"
        f"Step: {obs.step_count}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient observation:\n{obs_text}\n\nChoose antibiotic and duration (JSON only):"}
        ],
        max_tokens=50,
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    parsed = json.loads(content)
    return Action(
        antibiotic=parsed["antibiotic"],
        duration=int(parsed["duration"])
    )


def run_task(task):
    env = AntibioticEnv(task=task)
    obs = env.reset()

    rewards = []

    print(f"[START] {task}")

    done = False
    while not done:
        try:
            action = get_action(obs)
        except Exception as e:
            # Fallback to safe default if LLM fails
            print(f"[WARN] LLM call failed: {e}. Using fallback.")
            action = Action(antibiotic="A", duration=5)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)

        print(f"[STEP] obs={obs.model_dump_json()} reward={reward.value} done={done}")

    score = grade_episode(rewards)
    print(f"[END] score={score}")

    return score


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
