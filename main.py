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
    return {"status": "ok", "message": "Telegram Chart Bot Drawing V4 Ready"}


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


def get_bitget_candles(symbol: str, granularity: str, limit: int = 200) -> pd.DataFrame:
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
    return pd.DataFrame(parsed).sort_values("Date").set_index("Date")


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = (df["High"] - df["Low"]).tail(period).mean()
    return float(atr)


def find_pivots(df: pd.DataFrame, window: int):
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values

    for i in range(window, len(df) - window):
        if h[i] > max(h[i-window:i]) and h[i] >= max(h[i+1:i+window+1]):
            highs.append((i, df.index[i], float(h[i])))
        if l[i] < min(l[i-window:i]) and l[i] <= min(l[i+1:i+window+1]):
            lows.append((i, df.index[i], float(l[i])))

    return highs, lows


def market_regime(df: pd.DataFrame, current: float):
    recent = df.tail(55)
    start = float(recent["Close"].iloc[0])
    end = float(recent["Close"].iloc[-1])
    move_pct = (end - start) / current * 100
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    range_pct = (high - low) / current * 100

    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["Close"].rolling(60).mean().iloc[-1])

    if abs(move_pct) < max(1.1, range_pct * 0.22):
        regime = "range"
    elif move_pct > 0 and ma20 >= ma60:
        regime = "uptrend"
    elif move_pct < 0 and ma20 <= ma60:
        regime = "downtrend"
    else:
        regime = "mixed"

    return {
        "regime": regime,
        "move_pct": move_pct,
        "range_pct": range_pct,
        "ma20": ma20,
        "ma60": ma60
    }


def reaction_score_around_level(df: pd.DataFrame, level: float, tolerance: float):
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values
    avg_volume = max(float(np.mean(volumes)), 1e-9)

    touches = 0
    rejections = 0
    volume_score = 0.0

    for i in range(len(df)):
        touched = lows[i] - tolerance <= level <= highs[i] + tolerance
        if not touched:
            continue

        touches += 1
        volume_score += min(volumes[i] / avg_volume, 3.0)

        if abs(highs[i] - level) <= tolerance and closes[i] < level:
            rejections += 1
        elif abs(lows[i] - level) <= tolerance and closes[i] > level:
            rejections += 1

    return touches, rejections, volume_score


def build_major_levels(df: pd.DataFrame, current: float, atr: float, pivot_highs, pivot_lows, regime: str):
    tolerance = max(atr * 0.45, current * 0.0013)
    candidates = []

    for idx, dt, price in pivot_highs:
        candidates.append({"price": price, "source": "pivot_high", "idx": idx})
    for idx, dt, price in pivot_lows:
        candidates.append({"price": price, "source": "pivot_low", "idx": idx})

    vol_threshold = df["Volume"].quantile(0.90)
    for i, row in enumerate(df.itertuples()):
        if row.Volume >= vol_threshold:
            candidates.append({"price": float(row.High), "source": "volume_high", "idx": i})
            candidates.append({"price": float(row.Low), "source": "volume_low", "idx": i})
            candidates.append({"price": float(row.Close), "source": "volume_close", "idx": i})

    clusters = []
    for c in candidates:
        placed = False
        for cl in clusters:
            if abs(c["price"] - cl["price"]) <= tolerance:
                cl["items"].append(c)
                weights = [2.0 if "pivot" in x["source"] else 1.15 for x in cl["items"]]
                prices = [x["price"] for x in cl["items"]]
                cl["price"] = float(np.average(prices, weights=weights))
                placed = True
                break
        if not placed:
            clusters.append({"price": c["price"], "items": [c]})

    levels = []
    for cl in clusters:
        price = cl["price"]
        touches, rejections, vol_score = reaction_score_around_level(df, price, tolerance)

        if touches < 2:
            continue

        recent_idx = max(x["idx"] for x in cl["items"])
        recency = recent_idx / max(len(df) - 1, 1)
        distance_pct = (price - current) / current * 100
        distance_abs = abs(distance_pct)

        if distance_abs > 7.5:
            continue

        pivot_count = sum(1 for x in cl["items"] if "pivot" in x["source"])
        volume_count = sum(1 for x in cl["items"] if "volume" in x["source"])

        score = (
            touches * 1.7
            + rejections * 2.4
            + min(vol_score, 13) * 0.45
            + pivot_count * 1.25
            + volume_count * 0.35
            + recency * 2.0
            + (1 / (1 + distance_abs / 2.2)) * 2.2
        )

        levels.append({
            "price": price,
            "kind": "resistance" if price >= current else "support",
            "touches": touches,
            "rejections": rejections,
            "score": score,
            "distance_pct": distance_pct
        })

    levels = sorted(levels, key=lambda x: x["score"], reverse=True)

    selected = []
    min_gap = max(atr * 1.15, current * 0.0035)

    for lv in levels:
        if all(abs(lv["price"] - s["price"]) > min_gap for s in selected):
            selected.append(lv)

    supports = [x for x in selected if x["price"] < current]
    resistances = [x for x in selected if x["price"] > current]

    # 추세장에서는 현재가 가까운 레벨을 더 적게, 박스권에서는 상하단을 선명하게
    max_each = 2 if regime in {"uptrend", "downtrend"} else 3

    supports = sorted(supports, key=lambda x: abs(x["distance_pct"]))[:max_each]
    resistances = sorted(resistances, key=lambda x: abs(x["distance_pct"]))[:max_each]

    supports = sorted(supports, key=lambda x: x["price"], reverse=True)
    resistances = sorted(resistances, key=lambda x: x["price"])

    return supports, resistances


def line_quality(df: pd.DataFrame, x1, y1, x2, y2, kind: str, atr: float):
    if x2 == x1:
        return -999, 0, 0

    current = float(df["Close"].iloc[-1])
    slope = (y2 - y1) / (x2 - x1)

    # 급경사 제거
    slope_pct_per_bar = abs(slope) / current * 100
    if slope_pct_per_bar > 0.055:
        return -999, 0, 0

    intercept = y1 - slope * x1
    tolerance = max(atr * 0.5, current * 0.0012)

    touches = 0
    violations = 0

    start = max(0, min(x1, x2))
    for i in range(start, len(df)):
        expected = slope * i + intercept
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])

        if kind == "support":
            if abs(low - expected) <= tolerance:
                touches += 1
            if low < expected - tolerance * 1.45:
                violations += 1
        else:
            if abs(high - expected) <= tolerance:
                touches += 1
            if high > expected + tolerance * 1.45:
                violations += 1

    projected = slope * (len(df) - 1) + intercept
    dist_pct = abs(projected - current) / current * 100

    if dist_pct > 4.0:
        return -999, touches, violations

    score = touches * 2.0 - violations * 2.5 - dist_pct * 0.5

    return score, touches, violations


def best_trendline(df: pd.DataFrame, pivots, current: float, atr: float, kind: str, regime: str):
    if len(pivots) < 3:
        return None

    # 박스권에서는 대각선 과감히 줄임
    if regime == "range":
        return None

    recent = pivots[-10:]
    best = None

    for a in range(len(recent)):
        for b in range(a + 1, len(recent)):
            x1, dt1, y1 = recent[a]
            x2, dt2, y2 = recent[b]

            if x2 - x1 < max(14, int(len(df) * 0.10)):
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 방향성에 안 맞는 추세선 제거
            if regime == "uptrend" and kind == "resistance":
                continue
            if regime == "downtrend" and kind == "support":
                continue
            if kind == "support" and slope <= 0:
                continue
            if kind == "resistance" and slope >= 0:
                continue

            score, touches, violations = line_quality(df, x1, y1, x2, y2, kind, atr)

            if score < 1.0 or touches < 2:
                continue

            projected = y1 + slope * ((len(df) - 1) - x1)

            item = {
                "points": [(df.index[x1], y1), (df.index[-1], projected)],
                "score": score,
                "touches": touches,
                "violations": violations,
                "slope": slope,
                "projected": projected,
                "kind": kind
            }

            if best is None or item["score"] > best["score"]:
                best = item

    return best


def detect_range(df: pd.DataFrame, current: float, regime: str):
    if regime != "range":
        return None

    recent = df.tail(90)
    top = float(recent["High"].quantile(0.91))
    bottom = float(recent["Low"].quantile(0.09))
    width = top - bottom

    if width <= 0:
        return None

    width_pct = width / current * 100

    if width_pct > 6.5:
        return None

    return {
        "top": top,
        "bottom": bottom,
        "mid": (top + bottom) / 2,
        "position": (current - bottom) / width,
        "width_pct": width_pct
    }


def analyze_structure(df: pd.DataFrame, interval: str):
    current = float(df["Close"].iloc[-1])
    atr = atr_value(df)
    window = pivot_window_by_interval(interval)

    regime_info = market_regime(df, current)
    regime = regime_info["regime"]

    pivot_highs, pivot_lows = find_pivots(df, window)
    supports, resistances = build_major_levels(df, current, atr, pivot_highs, pivot_lows, regime)

    support_trend = best_trendline(df, pivot_lows, current, atr, "support", regime)
    resistance_trend = best_trendline(df, pivot_highs, current, atr, "resistance", regime)

    box = detect_range(df, current, regime)

    return {
        "current": current,
        "atr": atr,
        "regime": regime,
        "regime_info": regime_info,
        "supports": supports,
        "resistances": resistances,
        "support_trend": support_trend,
        "resistance_trend": resistance_trend,
        "box": box,
    }


def format_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:,.8f}"


def make_analysis_text(structure) -> str:
    lines = []

    regime = structure["regime"]
    if regime == "uptrend":
        lines.append("상승 추세장: 지지선/눌림 구간 우선")
    elif regime == "downtrend":
        lines.append("하락 추세장: 저항선/반등 매도 구간 우선")
    elif regime == "range":
        lines.append("박스권: 상단·하단 반응 확인")
    else:
        lines.append("혼조 구간: 수평 지지/저항 우선")

    supports = structure["supports"]
    resistances = structure["resistances"]

    if resistances:
        r = resistances[0]
        lines.append(f"핵심 저항: {format_price(r['price'])} ({r['distance_pct']:+.2f}%)")

    if supports:
        s = supports[0]
        lines.append(f"핵심 지지: {format_price(s['price'])} ({s['distance_pct']:+.2f}%)")

    box = structure["box"]
    if box:
        pos = box["position"]
        if pos >= 0.72:
            lines.append("박스 상단부: 추격 진입 주의")
        elif pos <= 0.28:
            lines.append("박스 하단부: 이탈/반등 확인")
        else:
            lines.append("박스 중단부: 방향성 애매")

    if structure["support_trend"]:
        lines.append("상승 추세 지지선 유효")
    if structure["resistance_trend"]:
        lines.append("하락 추세 저항선 유효")

    return "\n".join(lines[:5])


def create_chart_image(symbol: str, interval: str, df: pd.DataFrame, structure) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"
    title = f"{symbol} {display_interval(interval)} | Clean Drawing V4"

    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit", volume="inherit"
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle="--",
        y_on_right=True
    )

    hlines, hcolors, hstyles, hwidths = [], [], [], []

    current = structure["current"]

    # 현재가 라인
    hlines.append(current)
    hcolors.append("#ffffff")
    hstyles.append(":")
    hwidths.append(0.9)

    # 수평 레벨: 너무 많이 긋지 않기
    for r in structure["resistances"][:2]:
        hlines.append(r["price"])
        hcolors.append("#ff5252")
        hstyles.append("-")
        hwidths.append(1.25)

    for s in structure["supports"][:2]:
        hlines.append(s["price"])
        hcolors.append("#00e676")
        hstyles.append("-")
        hwidths.append(1.25)

    box = structure["box"]
    if box:
        hlines.extend([box["top"], box["bottom"]])
        hcolors.extend(["#ffa726", "#ffa726"])
        hstyles.extend(["--", "--"])
        hwidths.extend([1.0, 1.0])

    alines, acolors, awidths = [], [], []

    # 추세선은 시장 상태에 맞는 것만 1개 정도
    if structure["support_trend"]:
        alines.append(structure["support_trend"]["points"])
        acolors.append("#00c853")
        awidths.append(1.45)

    if structure["resistance_trend"]:
        alines.append(structure["resistance_trend"]["points"])
        acolors.append("#ff1744")
        awidths.append(1.45)

    kwargs = {}

    if hlines:
        kwargs["hlines"] = dict(
            hlines=hlines,
            colors=hcolors,
            linestyle=hstyles,
            linewidths=hwidths,
            alpha=0.82
        )

    if alines:
        kwargs["alines"] = dict(
            alines=alines,
            colors=acolors,
            linewidths=awidths,
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
        savefig=dict(fname=image_path, dpi=145, bbox_inches="tight"),
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
        ticker = get_bitget_ticker(symbol)
        df = get_bitget_candles(symbol, interval, limit=200)
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
            f"⚪ 현재가 / 🟥 저항 / 🟩 지지 / 🟧 박스권\n"
            f"📌 <b>Clean 분석 V4</b>\n"
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
