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
        "message": "Telegram Chart Bot Bitget Ready"
    }


@app.get("/health")
def health():
    return {"health": "ok"}


def send_message(chat_id: int, text: str):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        },
        timeout=10
    )


def normalize_symbol(command: str) -> str:
    coin = command.replace("/", "").split("@")[0].upper()

    aliases = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "DOGE": "DOGEUSDT",
        "XRP": "XRPUSDT",
        "BNB": "BNBUSDT",
    }

    return aliases.get(coin, f"{coin}USDT")


def parse_command(text: str):
    parts = text.strip().split()
    command = parts[0]

    interval = parts[1].lower() if len(parts) >= 2 else "15m"

    symbol = normalize_symbol(command)

    return symbol, interval


def get_bitget_price(symbol: str):
    try:
        url = "https://api.bitget.com/api/v2/mix/market/ticker"

        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES"
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("code") != "00000":
            return None

        ticker = data["data"][0]

        return {
            "price": float(ticker["lastPr"]),
            "change": float(ticker["change24h"])
        }

    except Exception:
        return None


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):

    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403)

    update = await request.json()

    message = update.get("message") or update.get("edited_message")

    if not message:
        return {"ok": True}

    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if not text.startswith("/"):
        return {"ok": True}

    symbol, interval = parse_command(text)

    ticker = get_bitget_price(symbol)

    if not ticker:
        send_message(
            chat_id,
            f"⚠️ {symbol} 정보를 가져오지 못했어."
        )
        return {"ok": True}

    icon = "🟢" if ticker["change"] >= 0 else "🔴"

    send_message(
        chat_id,
        f"📊 <b>{symbol} {interval}</b>\n\n"
        f"현재가: <b>{ticker['price']}</b>\n"
        f"24시간 변동: {icon} {ticker['change']}%\n\n"
        f"다음 단계: 캔들 데이터 연결"
    )

    return {"ok": True}
