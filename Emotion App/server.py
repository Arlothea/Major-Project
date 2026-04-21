from fastapi import FastAPI
from datetime import datetime

# Creates a FastAPI instance
app = FastAPI()

# Global variable to store the current alert.
current_alert = None

# This is the endpoint that recieves the escalation data from main.py.
@app.post("/escalation")
async def escalation(data: dict):
    global current_alert

    # Store the recieved name, level and camera and include a timestamp of when the alert was received.
    current_alert = {
        "name": data.get("name", "Unknown"),
        "level": int(data.get("level", 0)),
        "camera": data.get("camera", "Unknown"),
        "time": datetime.now().isoformat()
    }
    return {"ok": True}

# This endpoint is for the UI to retrive the current alert data. Then returns name, level, camera and time.
@app.get("/current")
def get_current():
    return current_alert

# This allows the rserver.py file to be run directly, and starts the FastAPI server.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

# endpoint to check if the server is running.
@app.get("/ping")
def ping():
    return {"status": "ok"}
