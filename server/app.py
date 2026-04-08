from fastapi import FastAPI
from pydantic import BaseModel
from environment import AntibioticEnv, Action
import uvicorn

app = FastAPI()

env = AntibioticEnv(task="hard")

class StepRequest(BaseModel):
    antibiotic: str
    duration: int

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(req: StepRequest):
    action = Action(**req.dict())
    obs, reward, done, _ = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done
    }

@app.get("/state")
def state():
    return env.state()

# ✅ REQUIRED FOR OPENENV
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
