from fastapi import FastAPI
from pydantic import BaseModel
from environment import AntibioticEnv, Action

app = FastAPI()

# One global env instance per worker (fine for validation)
env = AntibioticEnv(task="easy")

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok"}
