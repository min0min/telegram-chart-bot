from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "Telegram Chart Bot Starter"}

@app.get("/health")
def health():
    return {"health": "ok"}
