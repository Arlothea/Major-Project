from fastapi import FastAPI
from datetime import datetime

# py -m uvicorn server:app --host 0.0.0.0 --port 8000

app = FastAPI()

current_alert = None

@app.post("/escalation")
async def escalation(data: dict):
    global current_alert
    current_alert = {
        "name": data["name"],
        "level": data["level"],
        "camera": data["camera"],
        "time": datetime.now().isoformat()
    }
    return {"ok": True}

@app.get("/current")
def get_current():
    return current_alert
