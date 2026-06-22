import os
import time
from pathlib import Path

import requests
import pandas as pd
import mplfinance as mpf
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default-secret")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
BITGET_API = "https://api.bitget.com"


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Telegram Chart Bot Chart V1 Ready"
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
        timeout=15
    )


def send_photo(chat_id: int, image_path: str, caption: str):
    with open(image_path, "rb") as image:
        requests.post(
            f"{TELEGRAM_API}/sendPhoto",
            data={
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            },
            files={"photo": image},
            timeout=30
        )


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


def normalize_interval(interval: str) -> str:
    interval = interval.lower()

    allowed = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
    }

    return allowed.get(interval, "15m")


def display_interval(bitget_interval: str) -> str:
    return bitget_interval.replace("H", "h").replace("D", "d").replace("W", "w")


def parse_command(text: str):
    parts = text.strip().split()

    command = parts[0]
    interval = parts[1] if len(parts) >= 2 else "15m"

    symbol = normalize_symbol(command)
    bitget_interval = normalize_interval(interval)

    return symbol, bitget_interval


def get_bitget_ticker(symbol: str):
    url = f"{BITGET_API}/api/v2/mix/market/ticker"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES"
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(f"ticker HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()

    if data.get("code") != "00000":
        raise RuntimeError(f"ticker Bitget error: {data}")

    ticker = data["data"][0]

    price = float(ticker["lastPr"])
    change_raw = float(ticker.get("change24h", 0))

    # Bitget change24h is usually decimal form. 0.0123 = 1.23%
    change_percent = change_raw * 100

    return {
        "price": price,
        "change_percent": change_percent
    }


def get_bitget_candles(symbol: str, granularity: str, limit: int = 200) -> pd.DataFrame:
    url = f"{BITGET_API}/api/v2/mix/market/history-candles"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": granularity,
        "limit": str(limit)
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(f"candles HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()

    if data.get("code") != "00000":
        raise RuntimeError(f"candles Bitget error: {data}")

    rows = data.get("data", [])

    if not rows:
        raise RuntimeError("No candle data returned")

    parsed = []
    for row in rows:
        # Bitget candle row: timestamp, open, high, low, close, volume, quote volume...
        parsed.append({
            "Date": pd.to_datetime(int(row[0]), unit="ms"),
            "Open": float(row[1]),
            "High": float(row[2]),
            "Low": float(row[3]),
            "Close": float(row[4]),
            "Volume": float(row[5]),
        })

    df = pd.DataFrame(parsed)
    df = df.sort_values("Date")
    df = df.set_index("Date")

    return df


def format_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:,.8f}"


def create_chart_image(symbol: str, interval: str, df: pd.DataFrame) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"

    title = f"{symbol} {display_interval(interval)} | Bitget USDT Futures"

    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="inherit"
    )

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle="--",
        y_on_right=True
    )

    mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        title=title,
        mav=(20, 60),
        figsize=(12, 7),
        tight_layout=True,
        savefig=dict(fname=image_path, dpi=140, bbox_inches="tight")
    )

    return image_path


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    update = await request.json()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    text = message.get("text", "")
    chat_id = message.get("chat", {}).get("id")

    if not chat_id or not text.startswith("/"):
        return {"ok": True}

    symbol, interval = parse_command(text)
    shown_interval = display_interval(interval)

    try:
        ticker = get_bitget_ticker(symbol)
        df = get_bitget_candles(symbol, interval, limit=200)
        image_path = create_chart_image(symbol, interval, df)

        change = ticker["change_percent"]
        icon = "🟢" if change >= 0 else "🔴"

        caption = (
            f"📊 <b>{symbol} {shown_interval}</b>\n\n"
            f"현재가: <b>{format_price(ticker['price'])}</b>\n"
            f"24h 변동: {icon} <b>{change:.2f}%</b>\n"
            f"캔들: <b>{len(df)}개 수신 완료</b>\n\n"
            f"다음 단계: 지지/저항 자동 작도"
        )

        send_photo(chat_id, image_path, caption)

        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        send_message(
            chat_id,
            f"⚠️ <b>{symbol} {shown_interval}</b> 차트 생성 실패\n\n"
            f"원인: <code>{str(e)[:300]}</code>\n\n"
            f"예시: /btc, /btc 15m, /eth 1h, /sol 4h"
        )

    return {"ok": True}
