from pydantic import BaseModel
import random

# ---------- Models ----------
class Observation(BaseModel):
    patient_id: int
    infection_type: str
    severity: int
    resistance_level: str
    culture_result: str | None
    step_count: int

class Action(BaseModel):
    antibiotic: str   # A, B, C
    duration: int     # 3, 5, 7 days

class Reward(BaseModel):
    value: float


# ---------- Environment ----------
class AntibioticEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.reset()

    def reset(self):
        self.step_count = 0
        self.done = False

        self.state_data = {
            "patient_id": random.randint(1, 100),
            "infection_type": random.choice(["respiratory", "urinary", "skin"]),
            "severity": random.randint(1, 10),
            "resistance_level": random.choice(["low", "medium", "high"]),
            "culture_result": None
        }

        return Observation(**self.state_data, step_count=self.step_count)

    def step(self, action: Action):
        self.step_count += 1

        reward = 0.0

        correct_antibiotic = {
            "respiratory": "A",
            "urinary": "B",
            "skin": "C"
        }

        # ---------- Reward logic ----------
        if action.antibiotic == correct_antibiotic[self.state_data["infection_type"]]:
            reward += 0.5
        else:
            reward -= 0.5

        if action.duration in [5, 7]:
            reward += 0.2

        if self.state_data["resistance_level"] == "high" and action.antibiotic == "A":
            reward -= 0.3  # misuse

        # ---------- Hard task dynamic ----------
        if self.task == "hard" and self.step_count == 2:
            self.state_data["culture_result"] = correct_antibiotic[self.state_data["infection_type"]]

        if self.step_count >= 3:
            self.done = True

        obs = Observation(**self.state_data, step_count=self.step_count)
        return obs, Reward(value=reward), self.done, {}

    def state(self):
        return self.state_data
