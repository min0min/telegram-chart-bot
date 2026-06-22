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

SYMBOL_CACHE = {"symbols": set(), "updated_at": 0}


@app.get("/")
def home():
    return {"status": "ok", "message": "Telegram Chart Bot Strategy V7 Ready"}


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
            timeout=45
        )


def get_bitget_symbols(force: bool = False) -> set:
    now = time.time()
    if not force and SYMBOL_CACHE["symbols"] and now - SYMBOL_CACHE["updated_at"] < 3600:
        return SYMBOL_CACHE["symbols"]

    r = requests.get(
        f"{BITGET_API}/api/v2/mix/market/contracts",
        params={"productType": "USDT-FUTURES"},
        timeout=15
    )
    if r.status_code != 200:
        raise RuntimeError(f"contracts HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"contracts Bitget error: {data}")

    symbols = {item.get("symbol", "").upper() for item in data.get("data", []) if item.get("symbol")}
    SYMBOL_CACHE["symbols"] = symbols
    SYMBOL_CACHE["updated_at"] = now
    return symbols


def normalize_symbol(raw: str) -> str:
    coin = raw.replace("/", "").split("@")[0].upper().strip()
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


def is_valid_symbol(symbol: str) -> bool:
    try:
        return symbol.upper() in get_bitget_symbols()
    except Exception:
        return True


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


def parse_chart_command(text: str):
    parts = text.strip().split()
    symbol = normalize_symbol(parts[0])
    interval = normalize_interval(parts[1] if len(parts) >= 2 else "15m")
    return symbol, interval


def timeframe_profile(interval: str):
    profiles = {
        "1m":  {"name": "초단타", "entry_buffer": 0.18, "sl_atr": 0.70, "tp1_atr": 0.85, "tp2_atr": 1.35, "max_wait": "5~20분"},
        "3m":  {"name": "초단타", "entry_buffer": 0.20, "sl_atr": 0.75, "tp1_atr": 0.95, "tp2_atr": 1.55, "max_wait": "10~40분"},
        "5m":  {"name": "단타", "entry_buffer": 0.22, "sl_atr": 0.85, "tp1_atr": 1.05, "tp2_atr": 1.75, "max_wait": "20분~1시간"},
        "15m": {"name": "단타/스캘핑", "entry_buffer": 0.25, "sl_atr": 1.00, "tp1_atr": 1.25, "tp2_atr": 2.00, "max_wait": "1~4시간"},
        "30m": {"name": "단기", "entry_buffer": 0.28, "sl_atr": 1.10, "tp1_atr": 1.40, "tp2_atr": 2.25, "max_wait": "2~8시간"},
        "1H":  {"name": "단기 스윙", "entry_buffer": 0.32, "sl_atr": 1.25, "tp1_atr": 1.60, "tp2_atr": 2.60, "max_wait": "반나절~1일"},
        "2H":  {"name": "스윙", "entry_buffer": 0.35, "sl_atr": 1.35, "tp1_atr": 1.80, "tp2_atr": 3.00, "max_wait": "1~2일"},
        "4H":  {"name": "스윙", "entry_buffer": 0.40, "sl_atr": 1.55, "tp1_atr": 2.10, "tp2_atr": 3.50, "max_wait": "2~5일"},
        "6H":  {"name": "스윙", "entry_buffer": 0.42, "sl_atr": 1.70, "tp1_atr": 2.30, "tp2_atr": 3.80, "max_wait": "3~7일"},
        "12H": {"name": "중기 스윙", "entry_buffer": 0.45, "sl_atr": 1.90, "tp1_atr": 2.60, "tp2_atr": 4.20, "max_wait": "1~2주"},
        "1D":  {"name": "일봉 스윙", "entry_buffer": 0.50, "sl_atr": 2.20, "tp1_atr": 3.00, "tp2_atr": 5.00, "max_wait": "1~4주"},
        "3D":  {"name": "중장기", "entry_buffer": 0.60, "sl_atr": 2.60, "tp1_atr": 3.80, "tp2_atr": 6.50, "max_wait": "수 주"},
        "1W":  {"name": "장기", "entry_buffer": 0.70, "sl_atr": 3.00, "tp1_atr": 5.00, "tp2_atr": 8.00, "max_wait": "수 주~수 개월"},
    }
    return profiles.get(interval, profiles["15m"])


def get_bitget_ticker(symbol: str):
    r = requests.get(
        f"{BITGET_API}/api/v2/mix/market/ticker",
        params={"symbol": symbol, "productType": "USDT-FUTURES"},
        timeout=15
    )
    if r.status_code != 200:
        raise RuntimeError(f"ticker HTTP {r.status_code}: {r.text[:200]}")

    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"ticker Bitget error: {data}")

    t = data["data"][0]
    return {
        "price": float(t["lastPr"]),
        "change_percent": float(t.get("change24h", 0)) * 100,
        "high24h": float(t.get("high24h", 0) or 0),
        "low24h": float(t.get("low24h", 0) or 0)
    }


def get_bitget_candles(symbol: str, granularity: str, limit: int = 200) -> pd.DataFrame:
    r = requests.get(
        f"{BITGET_API}/api/v2/mix/market/history-candles",
        params={"symbol": symbol, "productType": "USDT-FUTURES", "granularity": granularity, "limit": str(limit)},
        timeout=15
    )
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
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = (df["High"] - df["Low"]).tail(period).mean()
    return float(atr)


def pivot_window_by_interval(interval: str) -> int:
    if interval in {"1m", "3m", "5m"}:
        return 3
    if interval in {"15m", "30m", "1H"}:
        return 4
    if interval in {"2H", "4H", "6H"}:
        return 5
    return 6


def find_pivots(df: pd.DataFrame, window: int):
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(window, len(df)-window):
        if h[i] > max(h[i-window:i]) and h[i] >= max(h[i+1:i+window+1]):
            highs.append((i, df.index[i], float(h[i])))
        if l[i] < min(l[i-window:i]) and l[i] <= min(l[i+1:i+window+1]):
            lows.append((i, df.index[i], float(l[i])))
    return highs, lows


def market_regime(df: pd.DataFrame, current: float):
    recent = df.tail(55)
    start, end = float(recent["Close"].iloc[0]), float(recent["Close"].iloc[-1])
    move_pct = (end - start) / current * 100
    range_pct = (float(recent["High"].max()) - float(recent["Low"].min())) / current * 100
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
    ma120 = float(df["Close"].rolling(120).mean().iloc[-1]) if len(df) >= 120 else ma60

    if abs(move_pct) < max(1.1, range_pct * 0.22):
        regime = "range"
    elif move_pct > 0 and ma20 >= ma60:
        regime = "uptrend"
    elif move_pct < 0 and ma20 <= ma60:
        regime = "downtrend"
    else:
        regime = "mixed"

    if ma20 > ma60 > ma120:
        strength = "상승 정배열"
    elif ma20 < ma60 < ma120:
        strength = "하락 정배열"
    else:
        strength = "혼조 배열"

    return {"regime": regime, "move_pct": move_pct, "range_pct": range_pct, "ma20": ma20, "ma60": ma60, "ma120": ma120, "strength": strength}


def reaction_score_around_level(df: pd.DataFrame, level: float, tolerance: float):
    highs, lows, closes, volumes = df["High"].values, df["Low"].values, df["Close"].values, df["Volume"].values
    avg_volume = max(float(np.mean(volumes)), 1e-9)
    touches, rejections, volume_score = 0, 0, 0.0

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
        if distance_abs > 10:
            continue

        pivot_count = sum(1 for x in cl["items"] if "pivot" in x["source"])
        volume_count = sum(1 for x in cl["items"] if "volume" in x["source"])

        score = touches*1.7 + rejections*2.4 + min(vol_score, 13)*0.45 + pivot_count*1.25 + volume_count*0.35 + recency*2.0 + (1/(1+distance_abs/2.2))*2.2
        levels.append({"price": price, "kind": "resistance" if price >= current else "support", "touches": touches, "rejections": rejections, "score": score, "distance_pct": distance_pct})

    levels = sorted(levels, key=lambda x: x["score"], reverse=True)
    selected, min_gap = [], max(atr * 1.15, current * 0.0035)

    for lv in levels:
        if all(abs(lv["price"] - s["price"]) > min_gap for s in selected):
            selected.append(lv)

    supports = [x for x in selected if x["price"] < current]
    resistances = [x for x in selected if x["price"] > current]
    max_each = 3 if regime == "range" else 2

    supports = sorted(supports, key=lambda x: abs(x["distance_pct"]))[:max_each]
    resistances = sorted(resistances, key=lambda x: abs(x["distance_pct"]))[:max_each]

    return sorted(supports, key=lambda x: x["price"], reverse=True), sorted(resistances, key=lambda x: x["price"])


def line_quality(df: pd.DataFrame, x1, y1, x2, y2, kind: str, atr: float):
    if x2 == x1:
        return -999, 0, 0
    current = float(df["Close"].iloc[-1])
    slope = (y2-y1)/(x2-x1)
    if abs(slope)/current*100 > 0.055:
        return -999, 0, 0
    intercept = y1 - slope*x1
    tolerance = max(atr*0.5, current*0.0012)
    touches, violations = 0, 0
    start = max(0, min(x1, x2))
    for i in range(start, len(df)):
        expected = slope*i + intercept
        high, low = float(df["High"].iloc[i]), float(df["Low"].iloc[i])
        if kind == "support":
            if abs(low-expected) <= tolerance: touches += 1
            if low < expected - tolerance*1.45: violations += 1
        else:
            if abs(high-expected) <= tolerance: touches += 1
            if high > expected + tolerance*1.45: violations += 1
    projected = slope*(len(df)-1) + intercept
    dist_pct = abs(projected-current)/current*100
    if dist_pct > 4.0:
        return -999, touches, violations
    return touches*2.0 - violations*2.5 - dist_pct*0.5, touches, violations


def best_trendline(df: pd.DataFrame, pivots, current: float, atr: float, kind: str, regime: str):
    if len(pivots) < 3 or regime == "range":
        return None
    recent, best = pivots[-10:], None
    for a in range(len(recent)):
        for b in range(a+1, len(recent)):
            x1, dt1, y1 = recent[a]
            x2, dt2, y2 = recent[b]
            if x2-x1 < max(14, int(len(df)*0.10)):
                continue
            slope = (y2-y1)/(x2-x1)
            if regime == "uptrend" and kind == "resistance": continue
            if regime == "downtrend" and kind == "support": continue
            if kind == "support" and slope <= 0: continue
            if kind == "resistance" and slope >= 0: continue
            score, touches, violations = line_quality(df, x1, y1, x2, y2, kind, atr)
            if score < 1.0 or touches < 2:
                continue
            projected = y1 + slope*((len(df)-1)-x1)
            item = {"points": [(df.index[x1], y1), (df.index[-1], projected)], "score": score, "touches": touches, "violations": violations, "slope": slope, "projected": projected, "kind": kind}
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
    width_pct = width/current*100
    if width_pct > 6.5:
        return None
    return {"top": top, "bottom": bottom, "mid": (top+bottom)/2, "position": (current-bottom)/width, "width_pct": width_pct}


def analyze_structure(df: pd.DataFrame, interval: str):
    current = float(df["Close"].iloc[-1])
    atr = atr_value(df)
    atr_pct = atr/current*100
    window = pivot_window_by_interval(interval)
    regime_info = market_regime(df, current)
    regime = regime_info["regime"]
    pivot_highs, pivot_lows = find_pivots(df, window)
    supports, resistances = build_major_levels(df, current, atr, pivot_highs, pivot_lows, regime)
    support_trend = best_trendline(df, pivot_lows, current, atr, "support", regime)
    resistance_trend = best_trendline(df, pivot_highs, current, atr, "resistance", regime)
    box = detect_range(df, current, regime)

    return {"current": current, "atr": atr, "atr_pct": atr_pct, "regime": regime, "regime_info": regime_info, "supports": supports, "resistances": resistances, "support_trend": support_trend, "resistance_trend": resistance_trend, "box": box}


def format_price(price: float) -> str:
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    return f"{price:,.8f}"


def volatility_label(atr_pct: float):
    if atr_pct < 0.35: return "낮음"
    if atr_pct < 0.8: return "보통"
    if atr_pct < 1.6: return "높음"
    return "매우 높음"


def clamp_positive(price: float):
    return max(price, 0.00000001)


def nearest_level_above(current: float, levels, fallback: float):
    above = [x["price"] for x in levels if x["price"] > current]
    return min(above) if above else fallback


def nearest_level_below(current: float, levels, fallback: float):
    below = [x["price"] for x in levels if x["price"] < current]
    return max(below) if below else fallback


def build_trade_plan(structure, interval: str):
    current = structure["current"]
    atr = structure["atr"]
    atr_pct = structure["atr_pct"]
    regime = structure["regime"]
    profile = timeframe_profile(interval)

    supports = structure["supports"]
    resistances = structure["resistances"]

    main_support = supports[0]["price"] if supports else current - atr * 1.5
    main_resistance = resistances[0]["price"] if resistances else current + atr * 1.5

    buffer = atr * profile["entry_buffer"]
    sl_atr = atr * profile["sl_atr"]
    tp1_atr = atr * profile["tp1_atr"]
    tp2_atr = atr * profile["tp2_atr"]

    long_entry = main_resistance + buffer
    long_sl = min(main_support - buffer, long_entry - sl_atr)
    long_tp1 = max(long_entry + tp1_atr, main_resistance + atr * 0.8)
    long_tp2 = max(long_entry + tp2_atr, long_tp1 + atr * 0.8)

    short_entry = main_support - buffer
    short_sl = max(main_resistance + buffer, short_entry + sl_atr)
    short_tp1 = min(short_entry - tp1_atr, main_support - atr * 0.8)
    short_tp2 = min(short_entry - tp2_atr, short_tp1 - atr * 0.8)

    long_rr1 = abs(long_tp1 - long_entry) / max(abs(long_entry - long_sl), 1e-9)
    long_rr2 = abs(long_tp2 - long_entry) / max(abs(long_entry - long_sl), 1e-9)
    short_rr1 = abs(short_entry - short_tp1) / max(abs(short_sl - short_entry), 1e-9)
    short_rr2 = abs(short_entry - short_tp2) / max(abs(short_sl - short_entry), 1e-9)

    # 구간 중간값: 여기서는 신규 진입 비추
    mid_low = main_support + atr * 0.35
    mid_high = main_resistance - atr * 0.35
    in_mid_zone = mid_low < current < mid_high

    if regime == "uptrend":
        bias = "롱 우선"
        note = "눌림 지지 확인 또는 저항 돌파 안착 시나리오 우선."
    elif regime == "downtrend":
        bias = "숏 우선"
        note = "반등 저항 확인 또는 지지 이탈 시나리오 우선."
    elif regime == "range":
        bias = "박스 매매"
        note = "상단 추격 롱·하단 추격 숏 주의. 상단/하단 반응 확인."
    else:
        bias = "중립"
        note = "방향성 확인 전까지 돌파/이탈 확인 위주."

    if atr_pct >= 1.6:
        risk_note = "변동성 매우 높음: 손절폭 과소 설정 주의, 레버리지 축소 권장."
    elif atr_pct >= 0.8:
        risk_note = "변동성 높음: 진입 후 되돌림 폭 감안 필요."
    elif atr_pct < 0.35:
        risk_note = "변동성 낮음: 돌파 실패/휩쏘 가능성 주의."
    else:
        risk_note = "변동성 보통: 일반적인 ATR 기준 대응 가능."

    return {
        "profile": profile,
        "bias": bias,
        "note": note,
        "risk_note": risk_note,
        "main_support": main_support,
        "main_resistance": main_resistance,
        "long_entry": long_entry,
        "long_sl": clamp_positive(long_sl),
        "long_tp1": long_tp1,
        "long_tp2": long_tp2,
        "long_rr1": long_rr1,
        "long_rr2": long_rr2,
        "short_entry": clamp_positive(short_entry),
        "short_sl": short_sl,
        "short_tp1": clamp_positive(short_tp1),
        "short_tp2": clamp_positive(short_tp2),
        "short_rr1": short_rr1,
        "short_rr2": short_rr2,
        "in_mid_zone": in_mid_zone,
        "mid_low": mid_low,
        "mid_high": mid_high
    }


def split_entry_plan(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    current = structure["current"]
    atr = structure["atr"]
    atr_pct = structure["atr_pct"]

    if atr_pct >= 1.6:
        return {
            "mode": "avoid",
            "text": (
                "🚨 <b>변동성 주의</b>\\n"
                "현재 변동성이 매우 높아서 분할진입보다 돌파/이탈 확인 후 1회 진입이 더 안전해 보임.\\n"
                "무리한 물타기식 분할진입은 비추천."
            )
        }

    if atr_pct >= 0.8:
        ratio = "30% / 30% / 40%"
        gap = atr * 0.45
        note = "변동성 높음: 마지막 진입 비중을 가장 크게 두고, 확인 후 진입 권장."
    elif atr_pct < 0.35:
        ratio = "50% / 30% / 20%"
        gap = atr * 0.25
        note = "변동성 낮음: 휩쏘 가능성이 있어 첫 진입을 과하게 크게 잡지 말 것."
    else:
        ratio = "40% / 30% / 30%"
        gap = atr * 0.35
        note = "변동성 보통: 일반적인 3분할 접근 가능."

    long_entries = [
        plan["long_entry"],
        plan["long_entry"] - gap,
        plan["main_support"] + gap
    ]

    short_entries = [
        plan["short_entry"],
        plan["short_entry"] + gap,
        plan["main_resistance"] - gap
    ]

    # 가격 순서 보정
    long_entries = sorted(long_entries, reverse=True)
    short_entries = sorted(short_entries)

    return {
        "mode": "split",
        "text": (
            f"🧱 <b>분할진입 계산</b>\\n"
            f"추천 비중: <b>{ratio}</b>\\n"
            f"롱 분할 후보: {format_price(long_entries[0])} / {format_price(long_entries[1])} / {format_price(long_entries[2])}\\n"
            f"숏 분할 후보: {format_price(short_entries[0])} / {format_price(short_entries[1])} / {format_price(short_entries[2])}\\n"
            f"{note}"
        )
    }


def signal_points(df: pd.DataFrame, structure, interval: str):
    current = structure["current"]
    atr = structure["atr"]
    regime = structure["regime"]
    supports = structure["supports"]
    resistances = structure["resistances"]

    if not supports and not resistances:
        return [], []

    support_price = supports[0]["price"] if supports else current - atr * 1.5
    resistance_price = resistances[0]["price"] if resistances else current + atr * 1.5

    tolerance = max(atr * 0.55, current * 0.0015)

    long_points = []
    short_points = []

    # 최근 80개 안에서만 표기. 너무 많이 찍히면 지저분하므로 최대 3개씩.
    start = max(1, len(df) - 80)

    for i in range(start, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        close = float(row["Close"])
        open_ = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        prev_close = float(prev["Close"])

        bullish = close > open_
        bearish = close < open_

        near_support = abs(low - support_price) <= tolerance or low <= support_price <= high
        near_resistance = abs(high - resistance_price) <= tolerance or low <= resistance_price <= high

        breakout_long = prev_close <= resistance_price and close > resistance_price and bullish
        breakdown_short = prev_close >= support_price and close < support_price and bearish

        bounce_long = near_support and bullish and close > support_price
        reject_short = near_resistance and bearish and close < resistance_price

        if regime in {"uptrend", "mixed", "range"} and (breakout_long or bounce_long):
            long_points.append((i, low - atr * 0.45))

        if regime in {"downtrend", "mixed", "range"} and (breakdown_short or reject_short):
            short_points.append((i, high + atr * 0.45))

    # 최근성 우선으로 최대 3개
    return long_points[-3:], short_points[-3:]


def strategy_text(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    regime_info = structure["regime_info"]
    atr_pct = structure["atr_pct"]
    split = split_entry_plan(structure, interval)

    if structure["regime"] == "uptrend":
        trend = "📈 상승 우세"
    elif structure["regime"] == "downtrend":
        trend = "📉 하락 우세"
    elif structure["regime"] == "range":
        trend = "📦 박스권"
    else:
        trend = "⚖️ 혼조"

    vol_warning = ""
    if atr_pct >= 1.6:
        vol_warning = "🚨 변동성 매우 높음: 손절폭 과소 설정 금지, 레버리지 축소 권장"
    elif atr_pct >= 0.8:
        vol_warning = "⚠️ 변동성 주의: 진입 후 흔들림 폭이 커질 수 있음"
    elif atr_pct < 0.35:
        vol_warning = "⚠️ 저변동성 주의: 돌파 실패/휩쏘 가능성 있음"
    else:
        vol_warning = "✅ 변동성 보통: ATR 기준 대응 가능"

    lines = [
        f"🧩 프레임: {plan['profile']['name']} / 예상 관리 시간 {plan['profile']['max_wait']}",
        f"추세: {trend} · {regime_info['strength']}",
        f"변동성: {volatility_label(atr_pct)} / ATR {format_price(structure['atr'])} ({atr_pct:.2f}%)",
        f"{vol_warning}",
        f"방향성: <b>{plan['bias']}</b>",
        "",
        f"🟥 핵심 저항: <b>{format_price(plan['main_resistance'])}</b>",
        f"🟩 핵심 지지: <b>{format_price(plan['main_support'])}</b>",
        "",
        "🟢 <b>롱 시나리오</b>",
        f"진입 조건: {format_price(plan['long_entry'])} 돌파 후 안착",
        f"TP1: {format_price(plan['long_tp1'])} / R:R {plan['long_rr1']:.2f}",
        f"TP2: {format_price(plan['long_tp2'])} / R:R {plan['long_rr2']:.2f}",
        f"SL 후보: {format_price(plan['long_sl'])}",
        "",
        "🔴 <b>숏 시나리오</b>",
        f"진입 조건: {format_price(plan['short_entry'])} 이탈 후 저항화",
        f"TP1: {format_price(plan['short_tp1'])} / R:R {plan['short_rr1']:.2f}",
        f"TP2: {format_price(plan['short_tp2'])} / R:R {plan['short_rr2']:.2f}",
        f"SL 후보: {format_price(plan['short_sl'])}",
        "",
        split["text"],
        "",
        f"📌 {plan['note']}",
        f"⚠️ {plan['risk_note']}",
    ]

    if plan["in_mid_zone"]:
        lines.append(f"관망 구간: {format_price(plan['mid_low'])} ~ {format_price(plan['mid_high'])}")

    return "\\n".join(lines)


def create_chart_image(symbol: str, interval: str, df: pd.DataFrame, structure) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"
    title = f"{symbol} {display_interval(interval)} | Strategy V7"

    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle="--", y_on_right=True)

    hlines, hcolors, hstyles, hwidths = [], [], [], []
    current = structure["current"]

    hlines.append(current); hcolors.append("#ffffff"); hstyles.append(":"); hwidths.append(0.9)

    for r in structure["resistances"][:2]:
        hlines.append(r["price"]); hcolors.append("#ff5252"); hstyles.append("-"); hwidths.append(1.25)
    for s in structure["supports"][:2]:
        hlines.append(s["price"]); hcolors.append("#00e676"); hstyles.append("-"); hwidths.append(1.25)

    plan = build_trade_plan(structure, interval)

    # 전략 트리거
    hlines.extend([plan["long_entry"], plan["short_entry"]])
    hcolors.extend(["#64b5f6", "#ffb74d"])
    hstyles.extend(["--", "--"])
    hwidths.extend([0.8, 0.8])

    box = structure["box"]
    if box:
        hlines.extend([box["top"], box["bottom"]])
        hcolors.extend(["#ffa726", "#ffa726"])
        hstyles.extend(["--", "--"])
        hwidths.extend([1.0, 1.0])

    alines, acolors, awidths = [], [], []
    if structure["support_trend"]:
        alines.append(structure["support_trend"]["points"]); acolors.append("#00c853"); awidths.append(1.45)
    if structure["resistance_trend"]:
        alines.append(structure["resistance_trend"]["points"]); acolors.append("#ff1744"); awidths.append(1.45)

    kwargs = {}
    if hlines:
        kwargs["hlines"] = dict(hlines=hlines, colors=hcolors, linestyle=hstyles, linewidths=hwidths, alpha=0.82)
    if alines:
        kwargs["alines"] = dict(alines=alines, colors=acolors, linewidths=awidths, alpha=0.95)

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        title=title,
        mav=(20, 60),
        figsize=(12, 7),
        tight_layout=True,
        returnfig=True,
        **kwargs
    )

    ax = axes[0]

    long_pts, short_pts = signal_points(df, structure, interval)

    # 이모지 폰트가 서버에서 안 잡힐 수 있어도 텔레그램 이미지에는 보통 문자로 표시됨.
    # 실패해도 차트 생성은 계속되도록 안전 처리.
    try:
        for x, y in long_pts:
            ax.text(
                x, y, "🚀",
                fontsize=16,
                ha="center",
                va="top",
                zorder=10
            )

        for x, y in short_pts:
            ax.text(
                x, y, "🔻",
                fontsize=16,
                ha="center",
                va="bottom",
                zorder=10
            )

        # 우측 상단 상태 박스
        atr_pct = structure["atr_pct"]
        if atr_pct >= 1.6:
            risk = "HIGH VOLATILITY"
        elif atr_pct >= 0.8:
            risk = "VOLATILITY CAUTION"
        else:
            risk = "NORMAL VOL"

        ax.text(
            0.015, 0.97,
            f"🚀 Long zone / 🔻 Short zone\\n⚠ {risk}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#111111", ec="#888888", alpha=0.72),
            zorder=10
        )

    except Exception as e:
        print(f"emoji annotation skipped: {e}")

    fig.savefig(image_path, dpi=145, bbox_inches="tight")
    return image_path


@app.on_event("startup")
def startup_event():
    try:
        get_bitget_symbols(force=True)
        print(f"Loaded {len(SYMBOL_CACHE['symbols'])} Bitget USDT futures symbols")
    except Exception as e:
        print(f"symbol preload failed: {e}")


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    text = message.get("text", "").strip()
    chat_id = message.get("chat", {}).get("id")
    if not chat_id or not text.startswith("/"):
        return {"ok": True}

    try:
        symbol, interval = parse_chart_command(text)
        shown_interval = display_interval(interval)

        if not is_valid_symbol(symbol):
            send_message(chat_id, f"⚠️ <b>{symbol}</b> 는 Bitget USDT 선물 상장 심볼이 아닌 것 같아.")
            return {"ok": True}

        ticker = get_bitget_ticker(symbol)
        df = get_bitget_candles(symbol, interval, limit=200)
        structure = analyze_structure(df, interval)
        image_path = create_chart_image(symbol, interval, df, structure)

        change = ticker["change_percent"]
        icon = "🟢" if change >= 0 else "🔴"
        analysis = strategy_text(structure, interval)

        caption = (
            f"📊 <b>{symbol} {shown_interval}</b>\n"
            f"현재가: <b>{format_price(ticker['price'])}</b> / 24h {icon} <b>{change:.2f}%</b>\n"
            f"⚪ 현재가 · 🟥저항 · 🟩지지 · 🔵롱트리거 · 🟧숏트리거 · 🚀롱우세 · 🔻숏우세\n\n"
            f"{analysis}\n\n"
            f"※ 자동 분석이며 확정 신호가 아니라 리스크 관리용 시나리오야. 변동성 구간에서는 포지션 크기를 줄이는 게 우선."
        )

        send_photo(chat_id, image_path, caption)

        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        send_message(
            chat_id,
            f"⚠️ 분석 실패\n\n"
            f"원인: <code>{str(e)[:300]}</code>\n\n"
            f"예시: /btc, /eth 1h, /1000pepe 15m, /wif 4h"
        )

    return {"ok": True}
