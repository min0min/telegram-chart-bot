import os
import requests
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default-secret")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
BINANCE_FUTURES_API = "https://fapi.binance.com"


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Telegram Chart Bot Price Ready"
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
    coin = command.replace("/", "").split("@")[0].upper()

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
    interval = parts[1].lower() if len(parts) >= 2 else "15m"

    allowed_intervals = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "12h",
        "1d"
    }

    if interval not in allowed_intervals:
        interval = "15m"

    symbol = normalize_symbol(command)
    return symbol, interval


def get_futures_ticker(symbol: str):
    url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
    response = requests.get(url, params={"symbol": symbol}, timeout=10)

    if response.status_code != 200:
        return None

    data = response.json()

    return {
        "symbol": data.get("symbol", symbol),
        "price": float(data.get("lastPrice", 0)),
        "change_percent": float(data.get("priceChangePercent", 0)),
        "high": float(data.get("highPrice", 0)),
        "low": float(data.get("lowPrice", 0)),
        "volume": float(data.get("volume", 0)),
    }


def format_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:,.8f}"


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
    ticker = get_futures_ticker(symbol)

    if not ticker:
        send_message(
            chat_id,
            f"⚠️ <b>{symbol}</b> 정보를 가져오지 못했어.\n"
            f"바이낸스 선물 상장 심볼인지 확인해줘.\n\n"
            f"예시: /btc, /eth 1h, /sol 15m"
        )
        return {"ok": True}

    change = ticker["change_percent"]
    change_icon = "🟢" if change >= 0 else "🔴"

    reply = (
        f"📊 <b>{symbol} {interval}</b>\n\n"
        f"현재가: <b>{format_price(ticker['price'])}</b> USDT\n"
        f"24h 변동: {change_icon} <b>{change:.2f}%</b>\n"
        f"24h 고가: {format_price(ticker['high'])}\n"
        f"24h 저가: {format_price(ticker['low'])}\n\n"
        f"다음 단계: {interval} 캔들 차트 이미지 생성"
    )

    send_message(chat_id, reply)

    return {"ok": True}
