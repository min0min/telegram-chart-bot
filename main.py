
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
    return {"status": "ok", "message": "Telegram Chart Bot Strategy V15 Ready"}


@app.get("/health")
def health():
    return {"health": "ok"}


def send_message(chat_id: int, text: str):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
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
        risk_note = "변동성 매우 높음: 손절폭 과소 설정 금지, 레버리지 축소 권장."
    elif atr_pct >= 0.8:
        risk_note = "변동성 높음: 진입 후 되돌림 폭 감안 필요."
    elif atr_pct < 0.35:
        risk_note = "변동성 낮음: 돌파 실패/휩쏘 가능성 주의."
    else:
        risk_note = "변동성 보통: 일반적인 ATR 기준 대응 가능."

    return {
        "profile": profile, "bias": bias, "note": note, "risk_note": risk_note,
        "main_support": main_support, "main_resistance": main_resistance,
        "long_entry": long_entry, "long_sl": clamp_positive(long_sl), "long_tp1": long_tp1, "long_tp2": long_tp2, "long_rr1": long_rr1, "long_rr2": long_rr2,
        "short_entry": clamp_positive(short_entry), "short_sl": short_sl, "short_tp1": clamp_positive(short_tp1), "short_tp2": clamp_positive(short_tp2), "short_rr1": short_rr1, "short_rr2": short_rr2,
        "in_mid_zone": in_mid_zone, "mid_low": mid_low, "mid_high": mid_high
    }


def split_entry_plan(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    current, atr, atr_pct = structure["current"], structure["atr"], structure["atr_pct"]
    regime = structure["regime"]
    support, resistance = plan["main_support"], plan["main_resistance"]
    price_range = max(resistance - support, atr * 3)
    position = (current - support) / max(price_range, 1e-9)

    if atr_pct >= 1.6:
        return {"mode": "avoid", "lines": ["🚨 <b>분할진입 판단</b>", "현재 변동성이 매우 높아서 분할진입 비추천.", "핵심 지지/저항 반응 확인 후 1회 진입 또는 포지션 크기 축소가 우선."]}

    spacing_by_tf = {"1m":0.35,"3m":0.45,"5m":0.55,"15m":0.80,"30m":1.05,"1H":1.35,"2H":1.65,"4H":2.10,"6H":2.35,"12H":2.80,"1D":3.50,"3D":4.50,"1W":5.50}
    spacing = atr * spacing_by_tf.get(interval, 0.80)
    zone_step = price_range / 3
    step = max(spacing, zone_step * 0.75)

    if atr_pct >= 0.8:
        ratio, note = "30% / 30% / 40%", "변동성 높음: 첫 진입 작게, 마지막 진입은 핵심 구간 반응 확인 후."
    elif atr_pct < 0.35:
        ratio, note = "40% / 30% / 30%", "저변동성: 너무 촘촘한 분할은 휩쏘에 말릴 수 있음."
    else:
        ratio, note = "40% / 30% / 30%", "보통 변동성: 시간봉 기준 넓은 3분할 접근."

    long_entries = [current - step * 0.65, current - step * 1.35, max(support + atr * 0.20, current - step * 2.10)]
    short_entries = [current + step * 0.65, current + step * 1.35, min(resistance - atr * 0.20, current + step * 2.10)]

    long_floor = support + atr * 0.15
    short_ceiling = resistance - atr * 0.15
    long_entries = [max(x, long_floor) for x in long_entries]
    short_entries = [min(x, short_ceiling) for x in short_entries]

    if position <= 0.25:
        long_entries = [current - atr * 0.45, support + atr * 0.55, support + atr * 0.20]
    if position >= 0.75:
        short_entries = [current + atr * 0.45, resistance - atr * 0.55, resistance - atr * 0.20]

    long_entries = sorted(set([round(x, 12) for x in long_entries]), reverse=True)
    short_entries = sorted(set([round(x, 12) for x in short_entries]))

    min_gap = atr * spacing_by_tf.get(interval, 0.80) * 0.55

    def clean_levels(levels, direction):
        cleaned = []
        for lv in levels:
            if all(abs(lv - c) >= min_gap for c in cleaned):
                cleaned.append(lv)
        while len(cleaned) < 3:
            if not cleaned:
                cleaned.append(current)
            else:
                cleaned.append(cleaned[-1] - min_gap if direction == "long" else cleaned[-1] + min_gap)
        return cleaned[:3]

    long_entries = clean_levels(long_entries, "long")
    short_entries = clean_levels(short_entries, "short")

    if position >= 0.78:
        zone_note = "현재가가 상단부라 롱 분할은 추격 주의. 숏은 저항 반응 확인 중심."
    elif position <= 0.22:
        zone_note = "현재가가 하단부라 숏 분할은 추격 주의. 롱은 지지 반응 확인 중심."
    else:
        zone_note = "현재가는 중간 구간. 분할은 핵심 지지/저항 쪽으로 넓게 대기."

    tf_note = f"{display_interval(interval)} 프레임: 촘촘한 분할보다 넓은 구간 분할 우선." if interval in {"1H","2H","4H","6H","12H","1D","3D","1W"} else f"{display_interval(interval)} 프레임: 단기 분할 가능하나 추격 진입 주의."

    if regime == "uptrend":
        preferred = "우선순위: 롱 눌림 분할 > 숏 추격"
    elif regime == "downtrend":
        preferred = "우선순위: 숏 반등 분할 > 롱 추격"
    elif regime == "range":
        preferred = "우선순위: 박스 하단 롱 / 박스 상단 숏"
    else:
        preferred = "우선순위: 돌파·이탈 확인 전까지 보수적"

    return {"mode": "split", "lines": ["🧱 <b>분할진입 계산</b>", f"추천 비중: <b>{ratio}</b>", f"롱 넓은 분할 후보: {format_price(long_entries[0])} / {format_price(long_entries[1])} / {format_price(long_entries[2])}", f"숏 넓은 분할 후보: {format_price(short_entries[0])} / {format_price(short_entries[1])} / {format_price(short_entries[2])}", zone_note, tf_note, preferred, note]}


def signal_points(df: pd.DataFrame, structure, interval: str):
    current, atr, regime = structure["current"], structure["atr"], structure["regime"]
    supports, resistances = structure["supports"], structure["resistances"]
    support_price = supports[0]["price"] if supports else current - atr * 1.5
    resistance_price = resistances[0]["price"] if resistances else current + atr * 1.5
    tolerance = max(atr * 0.55, current * 0.0015)
    long_points, short_points = [], []
    start = max(1, len(df) - 80)
    for i in range(start, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        close, open_, high, low, prev_close = float(row["Close"]), float(row["Open"]), float(row["High"]), float(row["Low"]), float(prev["Close"])
        bullish, bearish = close > open_, close < open_
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
    return long_points[-3:], short_points[-3:]


def score_grade(score: float):
    if score >= 85: return "A+ 강력 우세"
    if score >= 75: return "A 우세"
    if score >= 65: return "B+ 양호"
    if score >= 55: return "B 약우세"
    if score >= 45: return "C 관망"
    return "D 비추천"


def rr_grade(rr: float):
    if rr >= 2.0: return "우수"
    if rr >= 1.5: return "양호"
    if rr >= 1.1: return "보통"
    return "나쁨"


def entry_decision(score: float, rr: float, atr_pct: float):
    if atr_pct >= 1.6 and score < 80:
        return "진입 비추천", "변동성이 매우 높고 확률 우위가 충분하지 않음."
    if rr < 1.0:
        return "진입 비추천", "손익비가 1 미만이라 기대값이 불리함."
    if score >= 75 and rr >= 1.3:
        return "진입 가능", "확률과 손익비가 모두 양호함."
    if score >= 65 and rr >= 1.1:
        return "소액/축소 진입", "조건은 있으나 강한 확정 구간은 아님."
    if 45 <= score < 65:
        return "관망 우선", "방향성 확인이 더 필요함."
    return "진입 비추천", "현재 조건에서 우위가 약함."


def probability_score(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    regime, regime_info, atr_pct = structure["regime"], structure["regime_info"], structure["atr_pct"]
    supports, resistances, current = structure["supports"], structure["resistances"], structure["current"]
    long_score, short_score = 50.0, 50.0
    long_reasons, short_reasons, risk_reasons = [], [], []
    ma20, ma60, ma120 = regime_info["ma20"], regime_info["ma60"], regime_info["ma120"]

    if ma20 > ma60:
        long_score += 8; long_reasons.append("EMA20이 EMA60 위")
    else:
        short_score += 8; short_reasons.append("EMA20이 EMA60 아래")

    if ma60 > ma120:
        long_score += 5; long_reasons.append("중기 평균선 상승 우위")
    elif ma60 < ma120:
        short_score += 5; short_reasons.append("중기 평균선 하락 우위")

    if regime == "uptrend":
        long_score += 14; short_score -= 6; long_reasons.append("최근 구조가 상승 추세")
    elif regime == "downtrend":
        short_score += 14; long_score -= 6; short_reasons.append("최근 구조가 하락 추세")
    elif regime == "range":
        long_score -= 2; short_score -= 2; risk_reasons.append("박스권이라 상하단 반응 확인 필요")
    else:
        long_score -= 3; short_score -= 3; risk_reasons.append("혼조 구간")

    if supports:
        s = supports[0]
        dist_s = abs((current - s["price"]) / current * 100)
        if dist_s <= max(0.8, atr_pct * 0.9):
            long_score += 9; long_reasons.append("현재가가 핵심 지지권과 가까움")
        if s.get("touches", 0) >= 3:
            long_score += 5; long_reasons.append("지지선 반응 횟수 양호")

    if resistances:
        r = resistances[0]
        dist_r = abs((r["price"] - current) / current * 100)
        if dist_r <= max(0.8, atr_pct * 0.9):
            short_score += 9; short_reasons.append("현재가가 핵심 저항권과 가까움")
        if r.get("touches", 0) >= 3:
            short_score += 5; short_reasons.append("저항선 반응 횟수 양호")

    if resistances and current > resistances[0]["price"]:
        long_score += 10; long_reasons.append("저항 돌파 상태")
    if supports and current < supports[0]["price"]:
        short_score += 10; short_reasons.append("지지 이탈 상태")
    if structure["support_trend"]:
        long_score += 6; long_reasons.append("상승 추세 지지선 유효")
    if structure["resistance_trend"]:
        short_score += 6; short_reasons.append("하락 추세 저항선 유효")

    if atr_pct >= 1.6:
        long_score -= 8; short_score -= 8; risk_reasons.append("변동성 매우 높음")
    elif atr_pct >= 0.8:
        long_score -= 3; short_score -= 3; risk_reasons.append("변동성 높음")
    elif atr_pct < 0.35:
        long_score -= 4; short_score -= 4; risk_reasons.append("저변동성 휩쏘 가능성")

    if plan["long_rr1"] < 1.0:
        long_score -= 12; risk_reasons.append("롱 TP1 손익비 1 미만")
    elif plan["long_rr1"] >= 1.5:
        long_score += 6; long_reasons.append("롱 손익비 양호")
    if plan["short_rr1"] < 1.0:
        short_score -= 12; risk_reasons.append("숏 TP1 손익비 1 미만")
    elif plan["short_rr1"] >= 1.5:
        short_score += 6; short_reasons.append("숏 손익비 양호")

    long_score, short_score = max(0, min(100, long_score)), max(0, min(100, short_score))
    total = max(long_score + short_score, 1)
    long_prob = round(long_score / total * 100)
    short_prob = 100 - long_prob
    long_decision, long_decision_reason = entry_decision(long_score, plan["long_rr1"], atr_pct)
    short_decision, short_decision_reason = entry_decision(short_score, plan["short_rr1"], atr_pct)

    if long_decision in {"진입 가능", "소액/축소 진입"} and long_score > short_score + 8:
        final = "롱 우세"
    elif short_decision in {"진입 가능", "소액/축소 진입"} and short_score > long_score + 8:
        final = "숏 우세"
    else:
        final = "관망 우선"

    return {"long_score": long_score, "short_score": short_score, "long_prob": long_prob, "short_prob": short_prob, "long_grade": score_grade(long_score), "short_grade": score_grade(short_score), "long_decision": long_decision, "short_decision": short_decision, "long_decision_reason": long_decision_reason, "short_decision_reason": short_decision_reason, "final": final, "long_reasons": long_reasons[:4], "short_reasons": short_reasons[:4], "risk_reasons": list(dict.fromkeys(risk_reasons))[:4]}


def trade_quality(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    score = probability_score(structure, interval)

    atr_pct = structure["atr_pct"]
    regime = structure["regime"]

    best_score = max(score["long_score"], score["short_score"])
    best_rr = max(plan["long_rr1"], plan["short_rr1"])

    quality = 50.0

    quality += (best_score - 50) * 0.65

    if best_rr >= 2.0:
        quality += 18
    elif best_rr >= 1.5:
        quality += 12
    elif best_rr >= 1.1:
        quality += 5
    else:
        quality -= 18

    if atr_pct >= 1.6:
        quality -= 18
    elif atr_pct >= 0.8:
        quality -= 7
    elif atr_pct < 0.35:
        quality -= 5

    if regime == "range":
        quality -= 4
    elif regime in {"uptrend", "downtrend"}:
        quality += 6

    if plan["in_mid_zone"]:
        quality -= 8

    quality = max(0, min(100, quality))

    if quality >= 85:
        grade = "A+"
    elif quality >= 75:
        grade = "A"
    elif quality >= 65:
        grade = "B+"
    elif quality >= 55:
        grade = "B"
    elif quality >= 45:
        grade = "C"
    else:
        grade = "D"

    return quality, grade


def final_action(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    score = probability_score(structure, interval)
    quality, grade = trade_quality(structure, interval)

    long_ok = score["long_decision"] in {"진입 가능", "소액/축소 진입"}
    short_ok = score["short_decision"] in {"진입 가능", "소액/축소 진입"}

    # 방향 우위와 진입 품질을 분리해서 판단
    direction_gap = abs(score["long_score"] - score["short_score"])

    if score["long_score"] > score["short_score"] + 8:
        direction = "롱 방향 우위"
    elif score["short_score"] > score["long_score"] + 8:
        direction = "숏 방향 우위"
    else:
        direction = "방향성 애매"

    if quality < 45:
        if direction == "롱 방향 우위":
            return "🟡 LONG 관점 / NO TRADE", "롱 근거는 있지만 진입 품질이 낮아 신규 진입은 비추천."
        if direction == "숏 방향 우위":
            return "🟡 SHORT 관점 / NO TRADE", "숏 근거는 있지만 진입 품질이 낮아 신규 진입은 비추천."
        return "🚫 NO TRADE", "시장 품질이 낮아서 신규 진입보다 관망이 유리."

    if structure["atr_pct"] >= 1.6 and quality < 75:
        if direction == "롱 방향 우위":
            return "🟡 LONG 관점 / 변동성 주의", "롱 방향성은 있으나 변동성이 매우 높아 확인 후 접근 필요."
        if direction == "숏 방향 우위":
            return "🟡 SHORT 관점 / 변동성 주의", "숏 방향성은 있으나 변동성이 매우 높아 확인 후 접근 필요."
        return "🚫 NO TRADE", "변동성이 매우 높아 손절 흔들림이 커질 가능성이 높음."

    if plan["long_rr1"] < 1.0 and plan["short_rr1"] < 1.0:
        if direction == "롱 방향 우위":
            return "🟡 LONG 관점 / RR 부족", "롱 방향성은 있으나 TP1 손익비가 부족해 추격 진입은 비추천."
        if direction == "숏 방향 우위":
            return "🟡 SHORT 관점 / RR 부족", "숏 방향성은 있으나 TP1 손익비가 부족해 추격 진입은 비추천."
        return "🚫 NO TRADE", "롱/숏 모두 TP1 기준 손익비가 불리함."

    if score["final"] == "롱 우세" and long_ok:
        if plan["in_mid_zone"]:
            return "🟡 LONG 대기", "롱 우세지만 현재가는 중간값이라 눌림 확인 후 접근."
        return "🟢 LONG 우세", "롱 확률과 손익비가 상대적으로 우위."

    if score["final"] == "숏 우세" and short_ok:
        if plan["in_mid_zone"]:
            return "🟡 SHORT 대기", "숏 우세지만 현재가는 중간값이라 반등 확인 후 접근."
        return "🔴 SHORT 우세", "숏 확률과 손익비가 상대적으로 우위."

    if direction == "롱 방향 우위":
        return "🟡 LONG 관점 / 관망", "롱 근거가 더 많지만 진입 조건이 아직 부족함."
    if direction == "숏 방향 우위":
        return "🟡 SHORT 관점 / 관망", "숏 근거가 더 많지만 진입 조건이 아직 부족함."

    return "🟡 관망 우선", "확률 우위 또는 손익비가 충분하지 않아 확인이 더 필요."



def ai_commentary(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    score = probability_score(structure, interval)
    action, reason = final_action(structure, interval)

    comments = []

    if "LONG 관점" in action:
        comments.append("방향성은 롱 쪽 근거가 더 많지만, 현재 구간은 진입 품질이 낮아서 바로 추격하기보다 눌림이나 안착 확인이 필요해.")
    elif "SHORT 관점" in action:
        comments.append("방향성은 숏 쪽 근거가 더 많지만, 현재 구간은 진입 품질이 낮아서 바로 추격하기보다 반등 저항이나 이탈 확인이 필요해.")
    elif action.startswith("🚫"):
        comments.append("현재 구간은 방향성보다 리스크 관리가 우선이야.")
    elif "LONG" in action:
        comments.append("롱 관점은 유효하지만, 추격보다는 안착 확인이 중요해.")
    elif "SHORT" in action:
        comments.append("숏 관점은 유효하지만, 반등 저항 확인 없이 추격하면 위험해.")
    else:
        comments.append("현재는 방향성보다 확인 구간이 더 중요해.")

    if structure["atr_pct"] >= 1.6:
        comments.append("ATR이 매우 높아서 평소보다 손절폭과 레버리지를 보수적으로 잡는 게 좋아.")
    elif structure["atr_pct"] >= 0.8:
        comments.append("변동성이 큰 편이라 분할 진입은 넓게 잡는 게 유리해.")

    if plan["in_mid_zone"]:
        comments.append("현재가는 지지와 저항 사이 중간값에 가까워서 신규 진입 기대값이 애매할 수 있어.")

    if score["risk_reasons"]:
        comments.append("리스크 요인: " + " / ".join(score["risk_reasons"][:2]))

    return " ".join(comments)



def strategy_text(structure, interval: str):
    plan = build_trade_plan(structure, interval)
    split = split_entry_plan(structure, interval)
    score = probability_score(structure, interval)
    quality, grade = trade_quality(structure, interval)
    action, action_reason = final_action(structure, interval)

    regime_info = structure["regime_info"]
    atr_pct = structure["atr_pct"]

    if structure["regime"] == "uptrend":
        trend = "📈 상승 우세"
    elif structure["regime"] == "downtrend":
        trend = "📉 하락 우세"
    elif structure["regime"] == "range":
        trend = "📦 박스권"
    else:
        trend = "⚖️ 혼조"

    if atr_pct >= 1.6:
        vol_warning = "🚨 매우 높음 — 레버리지 축소 / 손절폭 과소 설정 금지"
    elif atr_pct >= 0.8:
        vol_warning = "⚠️ 높음 — 진입 후 흔들림 폭 감안"
    elif atr_pct < 0.35:
        vol_warning = "⚠️ 낮음 — 돌파 실패·휩쏘 주의"
    else:
        vol_warning = "✅ 보통 — ATR 기준 대응 가능"

    lines = [
        "━━━━━━━━━━━━━━",
        "📊 <b>전략 분석 V15</b>",
        "━━━━━━━━━━━━━━",
        "",
        f"🎯 <b>최종 행동</b>: {action}",
        f"└ {action_reason}",
        "",
        f"🏆 <b>Trade Quality</b>: {quality:.0f}/100 ({grade})",
        f"🟢 롱 방향 우위: <b>{score['long_prob']}%</b> · {score['long_score']:.0f}점 ({score['long_grade']})",
        f"🔴 숏 방향 우위: <b>{score['short_prob']}%</b> · {score['short_score']:.0f}점 ({score['short_grade']})",
        "",
        "━━━━━━━━━━━━━━",
        "🧭 <b>시장 상태</b>",
        "━━━━━━━━━━━━━━",
        f"프레임: <b>{plan['profile']['name']}</b>",
        f"관리 시간: {plan['profile']['max_wait']}",
        f"추세: {trend} · {regime_info['strength']}",
        f"변동성: {volatility_label(atr_pct)} / ATR {format_price(structure['atr'])} ({atr_pct:.2f}%)",
        f"{vol_warning}",
        "",
        f"🟥 핵심 저항: <b>{format_price(plan['main_resistance'])}</b>",
        f"🟩 핵심 지지: <b>{format_price(plan['main_support'])}</b>",
        "",
        "━━━━━━━━━━━━━━",
        "🟢 <b>롱 시나리오</b>",
        "━━━━━━━━━━━━━━",
        f"판단: <b>{score['long_decision']}</b>",
        f"진입: {format_price(plan['long_entry'])} 돌파 후 안착",
        f"TP1: {format_price(plan['long_tp1'])} / R:R {plan['long_rr1']:.2f} ({rr_grade(plan['long_rr1'])})",
        f"TP2: {format_price(plan['long_tp2'])} / R:R {plan['long_rr2']:.2f} ({rr_grade(plan['long_rr2'])})",
        f"SL: {format_price(plan['long_sl'])}",
        f"이유: {score['long_decision_reason']}",
        "",
        "━━━━━━━━━━━━━━",
        "🔴 <b>숏 시나리오</b>",
        "━━━━━━━━━━━━━━",
        f"판단: <b>{score['short_decision']}</b>",
        f"진입: {format_price(plan['short_entry'])} 이탈 후 저항화",
        f"TP1: {format_price(plan['short_tp1'])} / R:R {plan['short_rr1']:.2f} ({rr_grade(plan['short_rr1'])})",
        f"TP2: {format_price(plan['short_tp2'])} / R:R {plan['short_rr2']:.2f} ({rr_grade(plan['short_rr2'])})",
        f"SL: {format_price(plan['short_sl'])}",
        f"이유: {score['short_decision_reason']}",
        "",
        "━━━━━━━━━━━━━━",
        "📌 <b>근거</b>",
        "━━━━━━━━━━━━━━",
    ]

    if score["long_reasons"]:
        lines.append("롱 근거: " + " / ".join(score["long_reasons"]))
    if score["short_reasons"]:
        lines.append("숏 근거: " + " / ".join(score["short_reasons"]))
    if score["risk_reasons"]:
        lines.append("리스크: " + " / ".join(score["risk_reasons"]))

    lines.extend([
        "",
        "━━━━━━━━━━━━━━",
        "🧱 <b>분할진입</b>",
        "━━━━━━━━━━━━━━",
        *split["lines"][1:],
        "",
        "━━━━━━━━━━━━━━",
        "🧠 <b>AI 코멘트</b>",
        "━━━━━━━━━━━━━━",
        ai_commentary(structure, interval),
    ])

    if plan["in_mid_zone"]:
        lines.extend(["", f"관망 구간: {format_price(plan['mid_low'])} ~ {format_price(plan['mid_high'])}"])

    return "\n".join(lines)



def create_chart_image(symbol: str, interval: str, df: pd.DataFrame, structure) -> str:
    image_path = f"/tmp/{symbol}_{interval}_{int(time.time())}.png"
    title = f"{symbol} {display_interval(interval)} | Strategy V15"

    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle="--", y_on_right=True)

    plan, score = build_trade_plan(structure, interval), probability_score(structure, interval)
    hlines, hcolors, hstyles, hwidths = [], [], [], []
    current = structure["current"]

    hlines.append(current); hcolors.append("#ffffff"); hstyles.append(":"); hwidths.append(0.9)
    for r in structure["resistances"][:2]:
        hlines.append(r["price"]); hcolors.append("#ff5252"); hstyles.append("-"); hwidths.append(1.25)
    for s in structure["supports"][:2]:
        hlines.append(s["price"]); hcolors.append("#00e676"); hstyles.append("-"); hwidths.append(1.25)

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
        alines.append(structure["support_trend"]["points"]); acolors.append("#00c853"); awidths.append(1.35)
    if structure["resistance_trend"]:
        alines.append(structure["resistance_trend"]["points"]); acolors.append("#ff1744"); awidths.append(1.35)

    kwargs = {}
    if hlines:
        kwargs["hlines"] = dict(hlines=hlines, colors=hcolors, linestyle=hstyles, linewidths=hwidths, alpha=0.78)
    if alines:
        kwargs["alines"] = dict(alines=alines, colors=acolors, linewidths=awidths, alpha=0.9)

    fig, axes = mpf.plot(df, type="candle", style=style, volume=True, title=title, mav=(20, 60), figsize=(12, 7), tight_layout=True, returnfig=True, **kwargs)
    ax = axes[0]

    long_pts, short_pts = signal_points(df, structure, interval)
    if long_pts:
        x, y = long_pts[-1]
        ax.scatter(x, y, marker="^", s=180, color="#00e676", edgecolors="white", linewidths=0.9, zorder=10)
        ax.text(x, y, f"LONG {score['long_prob']}%", fontsize=8, ha="center", va="bottom", color="white", bbox=dict(boxstyle="round,pad=0.20", fc="#006b3c", ec="white", alpha=0.88), zorder=11)
    if short_pts:
        x, y = short_pts[-1]
        ax.scatter(x, y, marker="v", s=180, color="#ff5252", edgecolors="white", linewidths=0.9, zorder=10)
        ax.text(x, y, f"SHORT {score['short_prob']}%", fontsize=8, ha="center", va="top", color="white", bbox=dict(boxstyle="round,pad=0.20", fc="#8b0000", ec="white", alpha=0.88), zorder=11)

    atr_pct = structure["atr_pct"]
    risk = "HIGH VOLATILITY" if atr_pct >= 1.6 else ("VOLATILITY CAUTION" if atr_pct >= 0.8 else "NORMAL VOL")
    ax.text(0.015, 0.97, f"{final_action(structure, interval)[0]}\nQ {trade_quality(structure, interval)[0]:.0f}/100\nL {score['long_prob']}% / S {score['short_prob']}%\n{risk}", transform=ax.transAxes, fontsize=9, ha="left", va="top", color="white", bbox=dict(boxstyle="round,pad=0.35", fc="#111111", ec="#888888", alpha=0.72), zorder=12)

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

        caption = (
            f"📊 <b>{symbol} {shown_interval}</b>\n"
            f"현재가: <b>{format_price(ticker['price'])}</b> / 24h {icon} <b>{change:.2f}%</b>\n"
            f"⚪ 현재가 · 🟥 저항 · 🟩 지지 · 🔵 롱트리거 · 🟧 숏트리거"
        )

        send_photo(chat_id, image_path, caption)
        send_message(chat_id, strategy_text(structure, interval) + "\n\n※ 자동 분석이며 확정 신호가 아니라 리스크 관리용 시나리오야.")

        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        send_message(chat_id, f"⚠️ 분석 실패\n\n원인: <code>{str(e)[:300]}</code>\n\n예시: /btc, /eth 1h, /1000pepe 15m, /wif 4h")

    return {"ok": True}
