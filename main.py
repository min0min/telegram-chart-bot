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
    return {"status": "ok", "message": "Telegram Chart Bot Drawing V3 Ready"}


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
        left_h = h[i-window:i]
        right_h = h[i+1:i+window+1]
        left_l = l[i-window:i]
        right_l = l[i+1:i+window+1]

        if h[i] > max(left_h) and h[i] >= max(right_h):
            highs.append((i, df.index[i], float(h[i])))
        if l[i] < min(left_l) and l[i] <= min(right_l):
            lows.append((i, df.index[i], float(l[i])))

    return highs, lows


def reaction_score_around_level(df: pd.DataFrame, level: float, tolerance: float):
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values

    touches = 0
    rejections = 0
    volume_score = 0.0
    avg_volume = max(float(np.mean(volumes)), 1e-9)

    for i in range(len(df)):
        touched = lows[i] - tolerance <= level <= highs[i] + tolerance
        if not touched:
            continue

        touches += 1
        volume_score += min(volumes[i] / avg_volume, 3.0)

        candle_range = max(highs[i] - lows[i], 1e-9)
        body_mid = (df["Open"].iloc[i] + closes[i]) / 2

        # 위에서 맞고 밀림 = 저항 반응, 아래에서 맞고 반등 = 지지 반응
        if abs(highs[i] - level) <= tolerance and closes[i] < level:
            rejections += 1
        elif abs(lows[i] - level) <= tolerance and closes[i] > level:
            rejections += 1
        elif abs(body_mid - level) <= tolerance * 0.7:
            rejections += 0.25

    return touches, rejections, volume_score


def build_major_levels(df: pd.DataFrame, current: float, atr: float, pivot_highs, pivot_lows):
    tolerance = max(atr * 0.42, current * 0.0012)
    candidates = []

    # 1) 피벗 기반 후보
    for idx, dt, price in pivot_highs:
        candidates.append({"price": price, "source": "pivot_high", "idx": idx})
    for idx, dt, price in pivot_lows:
        candidates.append({"price": price, "source": "pivot_low", "idx": idx})

    # 2) 거래량 터진 캔들의 고가/저가/종가 후보 추가
    vol_threshold = df["Volume"].quantile(0.88)
    for i, row in enumerate(df.itertuples()):
        if row.Volume >= vol_threshold:
            candidates.append({"price": float(row.High), "source": "volume_high", "idx": i})
            candidates.append({"price": float(row.Low), "source": "volume_low", "idx": i})
            candidates.append({"price": float(row.Close), "source": "volume_close", "idx": i})

    if not candidates:
        return [], []

    # 3) 후보 가격대 클러스터링
    clusters = []
    for c in candidates:
        placed = False
        for cl in clusters:
            if abs(c["price"] - cl["price"]) <= tolerance:
                cl["items"].append(c)
                weights = [2.0 if "pivot" in x["source"] else 1.2 for x in cl["items"]]
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

        recent_idx = max(x["idx"] for x in cl["items"])
        recency = recent_idx / max(len(df) - 1, 1)
        distance = abs(price - current) / current * 100
        distance_score = 1 / (1 + distance / 2.5)

        pivot_count = sum(1 for x in cl["items"] if "pivot" in x["source"])
        volume_count = sum(1 for x in cl["items"] if "volume" in x["source"])

        # 실제 반응 + 거래량 + 최근성 + 현재가 근처를 종합
        score = (
            touches * 1.8
            + rejections * 2.2
            + min(vol_score, 14) * 0.45
            + pivot_count * 1.3
            + volume_count * 0.45
            + recency * 2.0
            + distance_score * 2.2
        )

        if touches < 2:
            continue

        kind = "resistance" if price >= current else "support"
        levels.append({
            "price": price,
            "kind": kind,
            "touches": touches,
            "rejections": rejections,
            "score": score,
            "distance_pct": (price - current) / current * 100
        })

    # 4) 너무 가까운 선 제거: 같은 구역 선은 최고 점수 1개만
    levels = sorted(levels, key=lambda x: x["score"], reverse=True)
    selected = []

    min_gap = max(atr * 0.75, current * 0.0025)

    for lv in levels:
        if all(abs(lv["price"] - s["price"]) > min_gap for s in selected):
            selected.append(lv)

    supports = sorted(
        [x for x in selected if x["price"] < current],
        key=lambda x: (abs(x["distance_pct"]), -x["score"])
    )[:3]

    resistances = sorted(
        [x for x in selected if x["price"] > current],
        key=lambda x: (abs(x["distance_pct"]), -x["score"])
    )[:3]

    # 가까운 순으로 표시하되, 너무 쓰레기 점수는 제외
    supports = sorted(supports, key=lambda x: x["price"], reverse=True)
    resistances = sorted(resistances, key=lambda x: x["price"])

    return supports, resistances


def line_error(df: pd.DataFrame, line, kind: str, atr: float):
    x1, y1, x2, y2 = line
    if x2 == x1:
        return 1e9, 0
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    touches = 0
    violations = 0
    total_error = 0.0
    tolerance = max(atr * 0.45, np.mean(df["Close"]) * 0.001)

    start = max(0, min(x1, x2))
    end = len(df) - 1

    for i in range(start, end + 1):
        expected = slope * i + intercept
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])

        if kind == "support":
            dist = abs(low - expected)
            if dist <= tolerance:
                touches += 1
            if low < expected - tolerance * 1.35:
                violations += 1
            total_error += min(dist / tolerance, 3)
        else:
            dist = abs(high - expected)
            if dist <= tolerance:
                touches += 1
            if high > expected + tolerance * 1.35:
                violations += 1
            total_error += min(dist / tolerance, 3)

    avg_error = total_error / max(end - start + 1, 1)
    return avg_error + violations * 1.5 - touches * 0.35, touches


def best_trendline(df: pd.DataFrame, pivots, current: float, atr: float, kind: str):
    if len(pivots) < 3:
        return None

    recent = pivots[-10:]
    best = None

    for a in range(len(recent)):
        for b in range(a + 1, len(recent)):
            x1, dt1, y1 = recent[a]
            x2, dt2, y2 = recent[b]

            if x2 - x1 < max(12, len(df) * 0.08):
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 15m 같은 단기봉에서 터무니없는 대각선 제거
            slope_pct_per_bar = abs(slope) / current * 100
            if slope_pct_per_bar > 0.08:
                continue

            err, touches = line_error(df, (x1, y1, x2, y2), kind, atr)
            projected = y1 + slope * ((len(df) - 1) - x1)
            dist_pct = abs(projected - current) / current * 100

            # 현재가와 너무 먼 추세선 제거
            if dist_pct > 4.5:
                continue

            score = touches * 2.0 - err - dist_pct * 0.35

            # 방향성 가중치
            if kind == "support" and slope > 0:
                score += 1.0
            if kind == "resistance" and slope < 0:
                score += 1.0

            if touches < 2:
                continue

            if best is None or score > best["score"]:
                best = {
                    "points": [(df.index[x1], y1), (df.index[-1], projected)],
                    "score": score,
                    "touches": touches,
                    "slope": slope,
                    "projected": projected,
                    "kind": kind
                }

    return best


def detect_range(df: pd.DataFrame, current: float, atr: float):
    recent = df.tail(90)
    highs = recent["High"]
    lows = recent["Low"]
    top = float(highs.quantile(0.90))
    bottom = float(lows.quantile(0.10))
    width = top - bottom

    if width <= 0:
        return None

    width_pct = width / current * 100
    trend_move_pct = abs(float(recent["Close"].iloc[-1] - recent["Close"].iloc[0])) / current * 100

    # 박스권은 폭은 적당하고, 기간 시작~끝 이동이 과하지 않아야 함
    if width_pct > 7.0 or trend_move_pct > width_pct * 0.85:
        return None

    pos = (current - bottom) / width
    return {
        "top": top,
        "bottom": bottom,
        "mid": (top + bottom) / 2,
        "position": pos,
        "width_pct": width_pct
    }


def analyze_structure(df: pd.DataFrame, interval: str):
    current = float(df["Close"].iloc[-1])
    atr = atr_value(df)
    window = pivot_window_by_interval(interval)

    pivot_highs, pivot_lows = find_pivots(df, window)

    supports, resistances = build_major_levels(df, current, atr, pivot_highs, pivot_lows)

    support_trend = best_trendline(df, pivot_lows, current, atr, "support")
    resistance_trend = best_trendline(df, pivot_highs, current, atr, "resistance")

    box = detect_range(df, current, atr)

    return {
        "current": current,
        "atr": atr,
        "supports": supports,
        "resistances": resistances,
        "support_trend": support_trend,
        "resistance_trend": resistance_trend,
        "box": box,
        "pivot_highs": pivot_highs,
        "pivot_lows": pivot_lows
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

    lines = []

    if box:
        pos = box["position"]
        if pos >= 0.72:
            lines.append("박스 상단 접근: 추격 진입 주의")
        elif pos <= 0.28:
            lines.append("박스 하단 접근: 반등/이탈 확인 구간")
        else:
            lines.append("박스 중단부: 방향성 애매")

    if resistances:
        r = resistances[0]
        lines.append(f"주요 저항: {format_price(r['price'])} ({r['distance_pct']:+.2f}%)")

    if supports:
        s = supports[0]
        lines.append(f"주요 지지: {format_price(s['price'])} ({s['distance_pct']:+.2f}%)")

    if structure["support_trend"]:
        lines.append("상승 추세 지지선 감지")
    if structure["resistance_trend"]:
        lines.append("하락/상단 추세 저항선 감지")

    if not lines:
        lines.append("현재 구간은 명확한 핵심 레벨 부족")

    return "\n".join(lines[:5])


def create_chart_image(symbol: str, interval: str, df: pd.DataFrame, structure) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"

    title = f"{symbol} {display_interval(interval)} | Smart Drawing V3"

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

    hlines, hcolors, hstyles, hwidths = [], [], [], []

    # 가까운 주요 레벨만 표시
    for r in structure["resistances"]:
        hlines.append(r["price"])
        hcolors.append("#ff5252")
        hstyles.append("-")
        hwidths.append(1.35 if r["score"] > 10 else 1.0)

    for s in structure["supports"]:
        hlines.append(s["price"])
        hcolors.append("#00e676")
        hstyles.append("-")
        hwidths.append(1.35 if s["score"] > 10 else 1.0)

    box = structure["box"]
    if box:
        hlines.extend([box["top"], box["bottom"]])
        hcolors.extend(["#ffa726", "#ffa726"])
        hstyles.extend(["--", "--"])
        hwidths.extend([1.0, 1.0])

    alines, acolors, awidths = [], [], []

    if structure["resistance_trend"]:
        alines.append(structure["resistance_trend"]["points"])
        acolors.append("#ff1744")
        awidths.append(1.4)

    if structure["support_trend"]:
        alines.append(structure["support_trend"]["points"])
        acolors.append("#00c853")
        awidths.append(1.4)

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
            f"🟥 주요 저항 / 🟩 주요 지지 / 🟧 박스권\n"
            f"📌 <b>Smart 분석 V3</b>\n"
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
