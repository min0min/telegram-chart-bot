import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import mplfinance as mpf
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default-secret")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
BITGET_API = "https://api.bitget.com"


@app.get("/")
def home():
    return {"status": "ok", "message": "Telegram Chart Bot Drawing V2 Ready"}


@app.get("/health")
def health():
    return {"health": "ok"}


def send_message(chat_id: int, text: str):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=15
    )


def send_photo(chat_id: int, image_path: str, caption: str):
    with open(image_path, "rb") as image:
        requests.post(
            f"{TELEGRAM_API}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": image},
            timeout=40
        )


def normalize_symbol(command: str) -> str:
    coin = command.replace("/", "").split("@")[0].upper()
    aliases = {
        "BTC": "BTCUSDT", "BITCOIN": "BTCUSDT",
        "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT",
        "SOL": "SOLUSDT", "DOGE": "DOGEUSDT",
        "XRP": "XRPUSDT", "BNB": "BNBUSDT",
    }
    if coin in aliases:
        return aliases[coin]
    if coin.endswith("USDT"):
        return coin
    return f"{coin}USDT"


def normalize_interval(interval: str) -> str:
    interval = interval.lower()
    allowed = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "3d": "3D", "1w": "1W",
    }
    return allowed.get(interval, "15m")


def display_interval(interval: str) -> str:
    return interval.replace("H", "h").replace("D", "d").replace("W", "w")


def parse_command(text: str):
    parts = text.strip().split()
    command = parts[0]
    interval = parts[1] if len(parts) >= 2 else "15m"
    return normalize_symbol(command), normalize_interval(interval)


def candle_limit_by_interval(interval: str) -> int:
    return 200


def pivot_window_by_interval(interval: str) -> int:
    if interval in {"1m", "3m", "5m"}:
        return 3
    if interval in {"15m", "30m", "1H"}:
        return 4
    if interval in {"2H", "4H", "6H"}:
        return 5
    return 6


def get_bitget_ticker(symbol: str):
    url = f"{BITGET_API}/api/v2/mix/market/ticker"
    params = {"symbol": symbol, "productType": "USDT-FUTURES"}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"ticker HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"ticker Bitget error: {data}")
    t = data["data"][0]
    price = float(t["lastPr"])
    change_raw = float(t.get("change24h", 0))
    return {"price": price, "change_percent": change_raw * 100}


def get_bitget_candles(symbol: str, granularity: str, limit: int) -> pd.DataFrame:
    url = f"{BITGET_API}/api/v2/mix/market/history-candles"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": granularity,
        "limit": str(limit)
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"candles HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"candles Bitget error: {data}")
    rows = data.get("data", [])
    if not rows:
        raise RuntimeError("No candle data returned")

    parsed = []
    for row in rows:
        parsed.append({
            "Date": pd.to_datetime(int(row[0]), unit="ms"),
            "Open": float(row[1]),
            "High": float(row[2]),
            "Low": float(row[3]),
            "Close": float(row[4]),
            "Volume": float(row[5]),
        })
    df = pd.DataFrame(parsed).sort_values("Date").set_index("Date")
    return df


def find_pivots(df: pd.DataFrame, window: int):
    highs, lows = [], []
    h = df["High"].values
    l = df["Low"].values

    for i in range(window, len(df) - window):
        if h[i] == max(h[i-window:i+window+1]):
            highs.append((i, df.index[i], float(h[i])))
        if l[i] == min(l[i-window:i+window+1]):
            lows.append((i, df.index[i], float(l[i])))

    return highs, lows


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def cluster_levels(pivots, current_price: float, atr: float, kind: str, max_levels: int = 3):
    if not pivots:
        return []

    tolerance = max(atr * 0.45, current_price * 0.0015)
    clusters = []

    for idx, dt, price in pivots:
        placed = False
        for c in clusters:
            if abs(price - c["price"]) <= tolerance:
                c["items"].append((idx, dt, price))
                c["price"] = float(np.mean([x[2] for x in c["items"]]))
                placed = True
                break
        if not placed:
            clusters.append({"price": price, "items": [(idx, dt, price)]})

    scored = []
    last_index = len(pivots)
    for c in clusters:
        touches = len(c["items"])
        most_recent_idx = max(x[0] for x in c["items"])
        recency = most_recent_idx / max(1, len(pivots))
        distance_score = 1 / (1 + abs(c["price"] - current_price) / max(atr, 1e-9))
        volume_score = touches
        score = touches * 2.5 + recency * 1.5 + distance_score * 2 + volume_score * 0.5
        scored.append({
            "price": c["price"],
            "touches": touches,
            "score": score,
            "kind": kind
        })

    if kind == "resistance":
        filtered = [x for x in scored if x["price"] >= current_price * 0.998]
    else:
        filtered = [x for x in scored if x["price"] <= current_price * 1.002]

    if len(filtered) < max_levels:
        filtered = scored

    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
    return filtered[:max_levels]


def trendline_from_pivots(pivots, current_price: float, kind: str):
    if len(pivots) < 2:
        return None

    # 최근 pivot 위주. 너무 오래된 선보다 현재 구조에 맞게.
    recent = pivots[-8:]

    best = None
    for a in range(len(recent)):
        for b in range(a + 1, len(recent)):
            p1 = recent[a]
            p2 = recent[b]

            x1, dt1, y1 = p1
            x2, dt2, y2 = p2

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 저항선은 대체로 하락/완만, 지지선은 상승/완만을 선호하되 강제하지 않음
            projected = y2 + slope * ((999999) - x2)
            score = abs(x2 - x1)

            if kind == "support" and slope < 0:
                score *= 0.75
            if kind == "resistance" and slope > 0:
                score *= 0.75

            # 현재가와 너무 먼 선은 감점
            end_est = y2 + slope * (len(pivots) - x2)
            score *= 1 / (1 + abs(end_est - current_price) / max(current_price * 0.02, 1e-9))

            if best is None or score > best["score"]:
                best = {
                    "points": [(dt1, y1), (dt2, y2)],
                    "score": score,
                    "slope": slope,
                    "kind": kind
                }

    return best


def detect_range(df: pd.DataFrame, current_price: float, atr: float):
    recent = df.tail(80)
    top = float(recent["High"].quantile(0.92))
    bottom = float(recent["Low"].quantile(0.08))
    width = top - bottom
    mid = (top + bottom) / 2

    # 박스폭이 너무 넓으면 박스권 아님
    is_range = width <= max(current_price * 0.06, atr * 10)
    position = (current_price - bottom) / max(width, 1e-9)

    if not is_range:
        return None

    return {
        "top": top,
        "bottom": bottom,
        "mid": mid,
        "position": position
    }


def analyze_structure(df: pd.DataFrame, interval: str):
    current = float(df["Close"].iloc[-1])
    atr = atr_value(df)
    window = pivot_window_by_interval(interval)

    pivot_highs, pivot_lows = find_pivots(df, window)

    resistances = cluster_levels(pivot_highs, current, atr, "resistance", 3)
    supports = cluster_levels(pivot_lows, current, atr, "support", 3)

    resistance_trend = trendline_from_pivots(pivot_highs, current, "resistance")
    support_trend = trendline_from_pivots(pivot_lows, current, "support")

    box = detect_range(df, current, atr)

    return {
        "current": current,
        "atr": atr,
        "pivot_highs": pivot_highs,
        "pivot_lows": pivot_lows,
        "resistances": resistances,
        "supports": supports,
        "resistance_trend": resistance_trend,
        "support_trend": support_trend,
        "box": box
    }


def format_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:,.8f}"


def make_analysis_text(structure) -> str:
    current = structure["current"]
    supports = structure["supports"]
    resistances = structure["resistances"]
    box = structure["box"]

    nearest_support = min(supports, key=lambda x: abs(x["price"] - current)) if supports else None
    nearest_resistance = min(resistances, key=lambda x: abs(x["price"] - current)) if resistances else None

    lines = []

    if box:
        pos = box["position"]
        if pos >= 0.72:
            lines.append("박스권 상단부 접근")
        elif pos <= 0.28:
            lines.append("박스권 하단부 접근")
        else:
            lines.append("박스권 중단부")

    if nearest_resistance:
        dist = (nearest_resistance["price"] - current) / current * 100
        lines.append(f"가까운 저항: {format_price(nearest_resistance['price'])} ({dist:+.2f}%)")

    if nearest_support:
        dist = (nearest_support["price"] - current) / current * 100
        lines.append(f"가까운 지지: {format_price(nearest_support['price'])} ({dist:+.2f}%)")

    if not lines:
        lines.append("뚜렷한 핵심 레벨 부족")

    return "\n".join(lines[:4])


def create_chart_image(symbol: str, interval: str, df: pd.DataFrame, structure) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"

    title = f"{symbol} {display_interval(interval)} | Auto Drawing V2"

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

    hlines = []
    hcolors = []
    hstyles = []

    # 지지/저항 수평선
    for r in structure["resistances"]:
        hlines.append(r["price"])
        hcolors.append("#ff5252")
        hstyles.append("-.")

    for s in structure["supports"]:
        hlines.append(s["price"])
        hcolors.append("#00e676")
        hstyles.append("-.")

    # 박스권 상하단
    box = structure["box"]
    if box:
        hlines.extend([box["top"], box["bottom"]])
        hcolors.extend(["#ffa726", "#ffa726"])
        hstyles.extend(["--", "--"])

    alines = []
    acolors = []

    if structure["resistance_trend"]:
        alines.append(structure["resistance_trend"]["points"])
        acolors.append("#ff1744")

    if structure["support_trend"]:
        alines.append(structure["support_trend"]["points"])
        acolors.append("#00c853")

    kwargs = {}

    if hlines:
        kwargs["hlines"] = dict(
            hlines=hlines,
            colors=hcolors,
            linestyle=hstyles,
            linewidths=1.1,
            alpha=0.85
        )

    if alines:
        kwargs["alines"] = dict(
            alines=alines,
            colors=acolors,
            linewidths=1.3,
            alpha=0.95
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
        savefig=dict(fname=image_path, dpi=140, bbox_inches="tight"),
        **kwargs
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
        limit = candle_limit_by_interval(interval)
        ticker = get_bitget_ticker(symbol)
        df = get_bitget_candles(symbol, interval, limit=limit)
        structure = analyze_structure(df, interval)
        image_path = create_chart_image(symbol, interval, df, structure)

        change = ticker["change_percent"]
        icon = "🟢" if change >= 0 else "🔴"

        analysis_text = make_analysis_text(structure)

        caption = (
            f"📊 <b>{symbol} {shown_interval}</b>\n\n"
            f"현재가: <b>{format_price(ticker['price'])}</b>\n"
            f"24h 변동: {icon} <b>{change:.2f}%</b>\n"
            f"캔들: <b>{len(df)}개</b>\n\n"
            f"🟥 저항 / 🟩 지지 / 🟧 박스권\n"
            f"📌 <b>자동 분석</b>\n"
            f"{analysis_text}"
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
