"""
Confluence Trade Alert Bot -> Telegram  ("Sniper" edition)
------------------------------------------------------------
Sends Entry / SL / TP alerts for crypto + forex pairs, gated by:
  1. Price at a support/resistance zone
  2. An actual rejection candle confirming it (not just proximity)
  3. Trend alignment (no counter-trend entries)
  4. At least 2 of: RSI extreme, liquidity grab, unfilled FVG
  5. Deduplication - won't re-alert the same setup every cycle

Runs on 15min and 1h.

SETUP (local test):
1. pip install -r requirements.txt
2. Fill in env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY
3. python sr_trade_alerts.py

For GitHub Actions automation, see SETUP.md
"""

import os
import json
import time
from datetime import datetime, timezone

import requests
import numpy as np

# ============ CONFIG ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

# Symbols in Twelve Data format. Crypto: "BTC/USD". Forex: "EUR/USD"
PAIRS = [
    "BTC/USD",
    "XAU/USD",
    "USD/JPY",
]

TIMEFRAMES = ["15min", "1h"]
LOOKBACK_CANDLES = 150
PIVOT_WINDOW = 5          # candles each side to confirm a swing high/low
ZONE_MERGE_PCT = 0.15     # % distance to merge nearby levels into one zone
ATR_PERIOD = 14
SL_ATR_BUFFER = 2.2
DEFAULT_RR = 2.0
REACTION_LOOKBACK = 3     # candles checked for a "reaction" at a zone

RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

LIQUIDITY_WINDOW = 20     # candles searched for swept swing high/low
FVG_LOOKBACK = 20         # candles searched for unfilled fair value gaps

MIN_OPTIONAL_CONFLUENCE = 2   # of {RSI, liquidity grab, FVG}, how many required on top of the hard gates

# --- Sniper filters ---
EMA_TREND_PERIOD = 50
REJECTION_WICK_RATIO = 0.4     # wick must be >= 40% of the candle's range
REJECTION_CLOSE_POSITION = 0.6 # close must sit in the outer 40% of the candle, favoring the reaction direction

# --- Deduplication ---
STATE_FILE = "state.json"
COOLDOWN_HOURS = {"15min": 4, "1h": 10}   # don't re-alert same symbol+timeframe+direction within this window
# =================================


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Missing token/chat id, skipping send.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code != 200:
        print(f"[Telegram error] {r.text}")


def fetch_candles(symbol: str, interval: str):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": LOOKBACK_CANDLES,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "values" not in data:
        print(f"[Data error] {symbol} {interval}: {data}")
        return None
    opens = np.array([float(c["open"]) for c in data["values"]])
    closes = np.array([float(c["close"]) for c in data["values"]])
    highs = np.array([float(c["high"]) for c in data["values"]])
    lows = np.array([float(c["low"]) for c in data["values"]])
    return {"open": opens, "high": highs, "low": lows, "close": closes}


def atr(highs, lows, closes, period=ATR_PERIOD):
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return np.mean(trs) if trs else 0
    return np.mean(trs[-period:])


def rsi(closes, period=RSI_PERIOD):
    deltas = np.diff(closes)
    if len(deltas) < period:
        return 50
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ema_series(closes, period):
    ema = np.zeros_like(closes)
    ema[0] = closes[0]
    k = 2 / (period + 1)
    for i in range(1, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
    return ema


def trend_direction(closes, period=EMA_TREND_PERIOD):
    """Returns 'up', 'down', or 'flat' based on price vs EMA and EMA slope."""
    if len(closes) < period + 5:
        return "flat"
    ema = ema_series(closes, period)
    price_above = closes[-1] > ema[-1]
    price_below = closes[-1] < ema[-1]
    slope_up = ema[-1] > ema[-5]
    slope_down = ema[-1] < ema[-5]
    if price_above and slope_up:
        return "up"
    if price_below and slope_down:
        return "down"
    return "flat"


def find_swing_points(highs, lows, window=PIVOT_WINDOW):
    swing_highs, swing_lows = [], []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])
    return swing_highs, swing_lows


def merge_zones(levels, pct=ZONE_MERGE_PCT):
    if not levels:
        return []
    levels = sorted(levels)
    zones = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - zones[-1][-1]) / zones[-1][-1] * 100 <= pct:
            zones[-1].append(lv)
        else:
            zones.append([lv])
    return [np.mean(z) for z in zones]


def check_reaction(price_now, closes, zone, direction, tolerance_pct=0.25):
    recent = closes[-REACTION_LOOKBACK:]
    touched = any(abs(c - zone) / zone * 100 <= tolerance_pct for c in recent)
    if not touched:
        return False
    return price_now > zone if direction == "support" else price_now < zone


def has_rejection_candle(opens, highs, lows, closes, direction):
    """Checks the most recent candles for a genuine rejection wick + strong close,
    rather than just price sitting near a zone."""
    for i in range(-REACTION_LOOKBACK, 0):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        rng = h - l
        if rng <= 0:
            continue
        close_pos = (c - l) / rng  # 0 = closed at low, 1 = closed at high
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if direction == "bullish":
            if lower_wick / rng >= REJECTION_WICK_RATIO and close_pos >= REJECTION_CLOSE_POSITION:
                return True
        else:
            if upper_wick / rng >= REJECTION_WICK_RATIO and (1 - close_pos) >= REJECTION_CLOSE_POSITION:
                return True
    return False


def detect_liquidity_grab(highs, lows, closes, window=LIQUIDITY_WINDOW):
    if len(highs) < window + 1:
        return False, False
    recent_high = max(highs[-window:-1])
    recent_low = min(lows[-window:-1])
    last_high, last_low, last_close = highs[-1], lows[-1], closes[-1]
    buy_side_grab = last_low < recent_low and last_close > recent_low   # swept lows, closed back above -> bullish
    sell_side_grab = last_high > recent_high and last_close < recent_high  # swept highs, closed back below -> bearish
    return buy_side_grab, sell_side_grab


def find_unfilled_fvgs(highs, lows, closes, lookback=FVG_LOOKBACK):
    n = len(highs)
    start = max(2, n - lookback)
    bullish_zones, bearish_zones = [], []
    for i in range(start, n):
        if highs[i - 2] < lows[i]:
            gap_low, gap_high = highs[i - 2], lows[i]
            bullish_zones.append((gap_low, gap_high))
        if lows[i - 2] > highs[i]:
            gap_low, gap_high = highs[i], lows[i - 2]
            bearish_zones.append((gap_low, gap_high))
    return bullish_zones, bearish_zones


def price_in_zone(price, zones, tolerance_pct=0.2):
    for lo, hi in zones:
        lo_t = lo * (1 - tolerance_pct / 100)
        hi_t = hi * (1 + tolerance_pct / 100)
        if lo_t <= price <= hi_t:
            return True
    return False


# ---------- Deduplication state ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def already_alerted_recently(state, key, timeframe):
    last = state.get(key)
    if not last:
        return False
    last_time = datetime.fromisoformat(last)
    elapsed_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
    return elapsed_hours < COOLDOWN_HOURS.get(timeframe, 4)


def mark_alerted(state, key):
    state[key] = datetime.now(timezone.utc).isoformat()


# ---------- Core analysis ----------
def analyze_pair(symbol: str, timeframe: str, state: dict):
    data = fetch_candles(symbol, timeframe)
    if data is None:
        return
    opens, closes, highs, lows = data["open"], data["close"], data["high"], data["low"]
    if len(closes) < max(30, EMA_TREND_PERIOD + 5):
        print(f"[Skip] {symbol} {timeframe}: not enough candles ({len(closes)})")
        return

    price_now = closes[-1]
    current_atr = atr(highs, lows, closes)
    current_rsi = rsi(closes)
    buy_grab, sell_grab = detect_liquidity_grab(highs, lows, closes)
    bull_fvgs, bear_fvgs = find_unfilled_fvgs(highs, lows, closes)
    trend = trend_direction(closes)

    swing_highs, swing_lows = find_swing_points(highs, lows)
    resistance_zones = merge_zones(swing_highs)
    support_zones = merge_zones(swing_lows)
    supports_below = sorted([z for z in support_zones if z < price_now], reverse=True)
    resistances_above = sorted([z for z in resistance_zones if z > price_now])

    signal_fired = False

    # --- LONG setup: needs uptrend or flat (never counter-trend against 'down') ---
    if supports_below and trend != "down":
        zone = supports_below[0]
        if check_reaction(price_now, closes, zone, "support") and has_rejection_candle(
            opens, highs, lows, closes, "bullish"
        ):
            score = 0
            tags = ["S/R reaction", "Rejection candle confirmed", f"Trend: {trend}"]
            if current_rsi <= RSI_OVERSOLD:
                score += 1
                tags.append(f"RSI oversold ({current_rsi:.0f})")
            if buy_grab:
                score += 1
                tags.append("Liquidity grab (buy-side)")
            if price_in_zone(price_now, bull_fvgs):
                score += 1
                tags.append("Bullish FVG")
            if score >= MIN_OPTIONAL_CONFLUENCE:
                key = f"{symbol}_{timeframe}_LONG"
                if already_alerted_recently(state, key, timeframe):
                    print(f"[Deduped] {symbol} {timeframe} LONG - already alerted within cooldown")
                else:
                    entry = price_now
                    structural_low = min(lows[-REACTION_LOOKBACK:])
                    anchor = min(zone, structural_low)
                    sl = anchor - (current_atr * SL_ATR_BUFFER)
                    risk = entry - sl
                    tp = resistances_above[0] if resistances_above else entry + risk * DEFAULT_RR
                    rr = round((tp - entry) / risk, 2) if risk > 0 else 0
                    alert_signal(symbol, timeframe, "LONG", entry, sl, tp, rr, tags)
                    mark_alerted(state, key)
                    signal_fired = True

    # --- SHORT setup: needs downtrend or flat (never counter-trend against 'up') ---
    if resistances_above and trend != "up":
        zone = resistances_above[0]
        if check_reaction(price_now, closes, zone, "resistance") and has_rejection_candle(
            opens, highs, lows, closes, "bearish"
        ):
            score = 0
            tags = ["S/R reaction", "Rejection candle confirmed", f"Trend: {trend}"]
            if current_rsi >= RSI_OVERBOUGHT:
                score += 1
                tags.append(f"RSI overbought ({current_rsi:.0f})")
            if sell_grab:
                score += 1
                tags.append("Liquidity grab (sell-side)")
            if price_in_zone(price_now, bear_fvgs):
                score += 1
                tags.append("Bearish FVG")
            if score >= MIN_OPTIONAL_CONFLUENCE:
                key = f"{symbol}_{timeframe}_SHORT"
                if already_alerted_recently(state, key, timeframe):
                    print(f"[Deduped] {symbol} {timeframe} SHORT - already alerted within cooldown")
                else:
                    entry = price_now
                    structural_high = max(highs[-REACTION_LOOKBACK:])
                    anchor = max(zone, structural_high)
                    sl = anchor + (current_atr * SL_ATR_BUFFER)
                    risk = sl - entry
                    tp = supports_below[0] if supports_below else entry - risk * DEFAULT_RR
                    rr = round((entry - tp) / risk, 2) if risk > 0 else 0
                    alert_signal(symbol, timeframe, "SHORT", entry, sl, tp, rr, tags)
                    mark_alerted(state, key)
                    signal_fired = True

    if not signal_fired:
        near_support = f"{supports_below[0]:.5f}" if supports_below else "none"
        near_resistance = f"{resistances_above[0]:.5f}" if resistances_above else "none"
        print(
            f"[No signal] {symbol} {timeframe}: price={price_now:.5f} "
            f"RSI={current_rsi:.0f} trend={trend} nearest_support={near_support} "
            f"nearest_resistance={near_resistance}"
        )


def alert_signal(symbol, timeframe, direction, entry, sl, tp, rr, tags):
    emoji = "🎯"
    tag_str = "\n".join(f"  • {t}" for t in tags)
    msg = (
        f"{emoji} *{symbol} - {direction}* ({timeframe})\n\n"
        f"Entry: `{entry:.5f}`\n"
        f"SL: `{sl:.5f}`\n"
        f"TP: `{tp:.5f}`\n"
        f"R:R  ~1:{rr}\n\n"
        f"Confluences:\n{tag_str}\n\n"
        f"_Verify before entering. Not financial advice._"
    )
    print(msg)
    send_telegram(msg)


def run_once():
    state = load_state()
    for pair in PAIRS:
        for tf in TIMEFRAMES:
            try:
                analyze_pair(pair, tf, state)
            except Exception as e:
                print(f"[Error] {pair} {tf}: {e}")
            time.sleep(1)  # avoid hammering free-tier rate limits
    save_state(state)


if __name__ == "__main__":
    run_once()
