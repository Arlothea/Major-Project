from fastapi import FastAPI
from datetime import datetime

# py -m uvicorn server:app --host 0.0.0.0 --port 8000

app = FastAPI()

current_alert = None

@app.post("/escalation")
async def escalation(data: dict):
    global current_alert

    current_alert = {
        "name": data.get("name", "|Unknown"),
        "level": int(data.get("level", 0)),
        "camera": data.get("camera", "Unknown"),
        "time": datetime.now().isoformat()
    }
    return {"ok": True}

@app.get("/current")
def get_current():
    return current_alert

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

@app.get("/ping")
def ping():
    return {"status": "ok"}
