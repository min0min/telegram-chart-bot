import os
import requests
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default-secret")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Telegram Chart Bot Webhook Ready"
    }


@app.get("/health")
def health():
    return {"health": "ok"}


def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    requests.post(url, json=payload, timeout=10)


def normalize_symbol(command: str) -> str:
    coin = command.replace("/", "").upper()

    aliases = {
        "BTC": "BTCUSDT",
        "BITCOIN": "BTCUSDT",
        "ETH": "ETHUSDT",
        "ETHEREUM": "ETHUSDT",
        "SOL": "SOLUSDT",
        "DOGE": "DOGEUSDT",
        "XRP": "XRPUSDT",
        "BNB": "BNBUSDT",
    }

    if coin in aliases:
        return aliases[coin]

    if coin.endswith("USDT"):
        return coin

    return f"{coin}USDT"


def parse_command(text: str):
    parts = text.strip().split()
    command = parts[0]
    interval = parts[1] if len(parts) >= 2 else "15m"

    allowed_intervals = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "12h",
        "1d"
    }

    if interval not in allowed_intervals:
        interval = "15m"

    symbol = normalize_symbol(command)
    return symbol, interval


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    update = await request.json()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "")

    if not chat_id or not text.startswith("/"):
        return {"ok": True}

    symbol, interval = parse_command(text)

    reply = (
        f"📊 <b>{symbol} {interval}</b>\\n\\n"
        f"명령어 수신 완료 ✅\\n"
        f"다음 단계에서 바이낸스 선물 캔들 + 차트 이미지 + 자동 작도 붙일 거임.\\n\\n"
        f"예시 명령어:\\n"
        f"/btc 15m\\n"
        f"/eth 1h\\n"
        f"/sol 15m"
    )

    send_message(chat_id, reply)

    return {"ok": True}
