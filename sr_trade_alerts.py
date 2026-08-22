"""
Confluence Trade Alert Bot -> Telegram
("Sniper" edition + outcome tracking + news blackout + multi-timeframe confluence)
-------------------------------------------------------------------------------------------
Sends Entry / SL / TP alerts for crypto + forex pairs, gated by:
  1. Price at a support/resistance zone
  2. An actual rejection candle confirming it (not just proximity)
  3. Trend alignment on its own timeframe (no counter-trend entries)
  4. Multi-timeframe confluence - a 15min signal must agree with the 1h trend;
     bonus confluence if the 15min zone lines up with a 1h zone too
  5. At least 2 of: RSI extreme, liquidity grab, unfilled FVG
  6. Deduplication - won't re-alert the same setup every cycle
  7. News blackout - pauses new signals around NFP / FOMC releases

Also tracks whether each alert actually hit TP or SL, and sends a daily
Telegram summary with win rate so you can judge if the system is any good.

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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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

TIMEFRAMES = ["15min", "1h"]  # default, used for any pair not listed in PAIR_TIMEFRAMES below

# Per-pair timeframe overrides. BTC/USD and XAU/USD get an extra 5min layer;
# USD/JPY stays on 15min + 1h only (keeps total requests/day under the free API cap).
PAIR_TIMEFRAMES = {
    "BTC/USD": ["5min", "15min", "1h"],
    "XAU/USD": ["5min", "15min", "1h"],
    "USD/JPY": ["15min", "1h"],
}

HTF_FOR = {"5min": "15min", "15min": "1h"}   # maps a timeframe to the higher timeframe that must confirm it

LOOKBACK_CANDLES = 400    # generous history so the 200 EMA has room to stabilize
PIVOT_WINDOW = 5          # candles each side to confirm a swing high/low
ZONE_MERGE_PCT = 0.15     # % distance to merge nearby levels into one zone
ATR_PERIOD = 14
SL_ATR_BUFFER = 2.2
DEFAULT_RR = 2.0
REACTION_LOOKBACK = 3     # candles checked for a "reaction" at a zone

# --- Trade quality floor ---
MIN_RR_RATIO = 2.0        # reject any setup whose real R:R (based on actual zone distance) falls below this

# Per-instrument profit floor, based on what a 0.01 lot position actually earns -
# more meaningful than a blanket % since it reflects your real position sizing.
# contract_size_per_lot uses common broker defaults (BTC/USD: 1 lot = 1 BTC,
# XAU/USD: 1 lot = 100 oz) - confirm these against your broker's contract specs.
INSTRUMENT_PROFIT_TARGETS = {
    "BTC/USD": {"lot_size": 0.01, "contract_size_per_lot": 1.0, "min_profit_usd": 2.0},
    "XAU/USD": {"lot_size": 0.01, "contract_size_per_lot": 100.0, "min_profit_usd": 4.0},
}
MIN_REWARD_PCT = 0.3      # fallback % floor for any pair not listed above (currently: USD/JPY)

RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

LIQUIDITY_WINDOW = 20     # candles searched for swept swing high/low
FVG_LOOKBACK = 20         # candles searched for unfilled fair value gaps

# --- Volume confirmation ---
# Only meaningful where real traded volume exists (mainly crypto - BTC/USD).
# Forex (USD/JPY) and spot gold (XAU/USD) trade OTC with no central volume figure,
# so this automatically no-ops for them rather than using a placeholder number.
VOLUME_SPIKE_WINDOW = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

MIN_OPTIONAL_CONFLUENCE = 2   # of {RSI, liquidity grab, FVG, HTF zone alignment}, how many required

# --- Sniper filters ---
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
REJECTION_WICK_RATIO = 0.4     # wick must be >= 40% of the candle's range
REJECTION_CLOSE_POSITION = 0.6 # close must sit in the outer 40% of the candle, favoring the reaction direction

# --- Multi-timeframe confluence ---
HTF_ZONE_TOLERANCE_PCT = 0.35  # how close a lower-TF zone must be to an HTF zone to count as "aligned"

# --- Deduplication ---
STATE_FILE = "state.json"
COOLDOWN_HOURS = {"5min": 2, "15min": 4, "1h": 10}   # don't re-alert same symbol+timeframe+direction within this window

# --- Outcome tracking ---
TRADE_LOG_FILE = "trade_log.json"

# --- News blackout ---
FOMC_DATES_2026 = ["2026-09-16", "2026-10-28", "2026-12-09"]
FOMC_TIME_ET = "14:00"
NFP_MONTHS_AHEAD = 3
BLACKOUT_BEFORE_MIN = 30
BLACKOUT_AFTER_MIN = 60
MANUAL_BLACKOUT_EVENTS = []
# =================================

NY_TZ = ZoneInfo("America/New_York")


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
    times = [c["datetime"] for c in data["values"]]
    opens = np.array([float(c["open"]) for c in data["values"]])
    closes = np.array([float(c["close"]) for c in data["values"]])
    highs = np.array([float(c["high"]) for c in data["values"]])
    lows = np.array([float(c["low"]) for c in data["values"]])
    volumes = []
    for c in data["values"]:
        try:
            volumes.append(float(c.get("volume") or 0.0))
        except (TypeError, ValueError):
            volumes.append(0.0)
    volumes = np.array(volumes)
    return {
        "datetime": times, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }


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


def trend_direction(closes, fast_period=EMA_FAST_PERIOD, slow_period=EMA_SLOW_PERIOD):
    """
    'up'   -> fast EMA above slow EMA, fast EMA rising, price above both
    'down' -> fast EMA below slow EMA, fast EMA falling, price below both
    'flat' -> anything else (EMAs crossing, price caught between them, etc.)
    """
    if len(closes) < slow_period + 10:
        return "flat"
    ema_fast = ema_series(closes, fast_period)
    ema_slow = ema_series(closes, slow_period)
    price = closes[-1]

    fast_above_slow = ema_fast[-1] > ema_slow[-1]
    fast_below_slow = ema_fast[-1] < ema_slow[-1]
    fast_slope_up = ema_fast[-1] > ema_fast[-5]
    fast_slope_down = ema_fast[-1] < ema_fast[-5]
    price_above_both = price > ema_fast[-1] and price > ema_slow[-1]
    price_below_both = price < ema_fast[-1] and price < ema_slow[-1]

    if fast_above_slow and fast_slope_up and price_above_both:
        return "up"
    if fast_below_slow and fast_slope_down and price_below_both:
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
    for i in range(-REACTION_LOOKBACK, 0):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        rng = h - l
        if rng <= 0:
            continue
        close_pos = (c - l) / rng
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
    buy_side_grab = last_low < recent_low and last_close > recent_low
    sell_side_grab = last_high > recent_high and last_close < recent_high
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


def has_usable_volume(volumes, min_nonzero_ratio=0.5):
    """Guards against instruments (forex, spot gold) that report no real volume -
    without this, a run of zeros would falsely look like a 'spike' of 0 >= 0."""
    if volumes is None or len(volumes) == 0:
        return False
    nonzero = np.count_nonzero(volumes)
    return (nonzero / len(volumes)) >= min_nonzero_ratio


def volume_spike(volumes, window=VOLUME_SPIKE_WINDOW, multiplier=VOLUME_SPIKE_MULTIPLIER):
    """True if the latest candle's volume meaningfully exceeds its recent average -
    i.e. the rejection/breakout candle was backed by real participation, not thin noise."""
    if not has_usable_volume(volumes) or len(volumes) < window + 1:
        return False
    avg_vol = np.mean(volumes[-window - 1:-1])
    if avg_vol <= 0:
        return False
    return volumes[-1] >= avg_vol * multiplier


def min_reward_distance(symbol, entry_price):
    """Returns (min_price_distance, target_profit_usd_or_None) for the reward floor.
    Uses the instrument's configured lot/contract size if available, otherwise
    falls back to a flat % of entry price."""
    cfg = INSTRUMENT_PROFIT_TARGETS.get(symbol)
    if cfg:
        units = cfg["lot_size"] * cfg["contract_size_per_lot"]
        if units > 0:
            return cfg["min_profit_usd"] / units, cfg["min_profit_usd"]
    return entry_price * (MIN_REWARD_PCT / 100), None


# ---------- Multi-timeframe confluence ----------
def get_htf_bias(htf_data):
    """Computes the higher-timeframe trend and nearest zones, used to gate/confirm lower-TF signals."""
    closes, highs, lows = htf_data["close"], htf_data["high"], htf_data["low"]
    trend = trend_direction(closes)
    swing_highs, swing_lows = find_swing_points(highs, lows)
    resistance_zones = merge_zones(swing_highs)
    support_zones = merge_zones(swing_lows)
    price_now = closes[-1]
    supports_below = sorted([z for z in support_zones if z < price_now], reverse=True)
    resistances_above = sorted([z for z in resistance_zones if z > price_now])
    return {
        "trend": trend,
        "nearest_support": supports_below[0] if supports_below else None,
        "nearest_resistance": resistances_above[0] if resistances_above else None,
    }


def zone_aligned_with_htf(zone, htf_zone, tolerance_pct=HTF_ZONE_TOLERANCE_PCT):
    if htf_zone is None:
        return False
    return abs(zone - htf_zone) / htf_zone * 100 <= tolerance_pct


# ---------- News blackout ----------
def fomc_blackout_windows():
    windows = []
    for d in FOMC_DATES_2026:
        dt_et = datetime.strptime(f"{d} {FOMC_TIME_ET}", "%Y-%m-%d %H:%M").replace(tzinfo=NY_TZ)
        windows.append((dt_et.astimezone(timezone.utc), "FOMC Rate Decision"))
    return windows


def nfp_blackout_windows(months_ahead=NFP_MONTHS_AHEAD):
    windows = []
    today = datetime.now(timezone.utc)
    for i in range(months_ahead):
        month = today.month + i
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        d = datetime(year, month, 1)
        while d.weekday() != 4:  # Friday
            d += timedelta(days=1)
        dt_et = d.replace(hour=8, minute=30, tzinfo=NY_TZ)
        windows.append((dt_et.astimezone(timezone.utc), "NFP (Non-Farm Payrolls)"))
    return windows


def manual_blackout_windows():
    windows = []
    for time_str, name in MANUAL_BLACKOUT_EVENTS:
        dt_utc = datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        windows.append((dt_utc, name))
    return windows


def is_blackout(now_utc=None):
    now = now_utc or datetime.now(timezone.utc)
    events = fomc_blackout_windows() + nfp_blackout_windows() + manual_blackout_windows()
    for event_time, name in events:
        delta_min = (now - event_time).total_seconds() / 60
        if -BLACKOUT_BEFORE_MIN <= delta_min <= BLACKOUT_AFTER_MIN:
            return True, name
    return False, None


# ---------- Deduplication state ----------
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_state():
    return load_json(STATE_FILE) or {}


def load_trade_log():
    return load_json(TRADE_LOG_FILE) or []


def already_alerted_recently(state, key, timeframe):
    last = state.get(key)
    if not last:
        return False
    last_time = datetime.fromisoformat(last)
    elapsed_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
    return elapsed_hours < COOLDOWN_HOURS.get(timeframe, 4)


def mark_alerted(state, key):
    state[key] = datetime.now(timezone.utc).isoformat()


# ---------- Outcome tracking ----------
def log_trade(trade_log, symbol, timeframe, direction, entry, sl, tp, entry_time):
    trade_log.append({
        "symbol": symbol, "timeframe": timeframe, "direction": direction,
        "entry": entry, "sl": sl, "tp": tp,
        "entry_time": entry_time, "status": "open", "closed_time": None,
    })


def check_open_trades(symbol, timeframe, data, trade_log):
    times, highs, lows = data["datetime"], data["high"], data["low"]
    for trade in trade_log:
        if trade["status"] != "open":
            continue
        if trade["symbol"] != symbol or trade["timeframe"] != timeframe:
            continue
        for i, t in enumerate(times):
            if t <= trade["entry_time"]:
                continue
            if trade["direction"] == "LONG":
                if lows[i] <= trade["sl"]:
                    trade["status"], trade["closed_time"] = "LOSS", t
                    break
                if highs[i] >= trade["tp"]:
                    trade["status"], trade["closed_time"] = "WIN", t
                    break
            else:
                if highs[i] >= trade["sl"]:
                    trade["status"], trade["closed_time"] = "LOSS", t
                    break
                if lows[i] <= trade["tp"]:
                    trade["status"], trade["closed_time"] = "WIN", t
                    break
        if trade["status"] != "open":
            emoji = "✅" if trade["status"] == "WIN" else "❌"
            send_telegram(
                f"{emoji} *{trade['symbol']} {trade['direction']}* ({trade['timeframe']}) "
                f"closed: *{trade['status']}*\nEntry: `{trade['entry']:.5f}` -> "
                f"{'TP' if trade['status']=='WIN' else 'SL'}: "
                f"`{trade['tp'] if trade['status']=='WIN' else trade['sl']:.5f}`"
            )


def compute_stats(trade_log):
    wins = sum(1 for t in trade_log if t["status"] == "WIN")
    losses = sum(1 for t in trade_log if t["status"] == "LOSS")
    open_trades = sum(1 for t in trade_log if t["status"] == "open")
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved else 0
    return wins, losses, open_trades, win_rate


def maybe_send_daily_summary(state, trade_log):
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_summary_date") == today_str:
        return
    wins, losses, open_trades, win_rate = compute_stats(trade_log)
    resolved = wins + losses
    if resolved == 0 and open_trades == 0:
        state["last_summary_date"] = today_str
        return
    msg = (
        f"📊 *Daily Summary*\n"
        f"Resolved trades: {resolved}\n"
        f"Wins: {wins}   Losses: {losses}\n"
        f"Win rate: {win_rate:.1f}%\n"
        f"Open trades: {open_trades}"
    )
    send_telegram(msg)
    state["last_summary_date"] = today_str


# ---------- Core analysis ----------
def analyze_pair(symbol: str, timeframe: str, data: dict, state: dict, trade_log: list,
                  blackout_active: bool, htf_bias: dict = None):
    times, opens, closes, highs, lows, volumes = (
        data["datetime"], data["open"], data["close"], data["high"], data["low"], data.get("volume")
    )
    if len(closes) < max(30, EMA_SLOW_PERIOD + 10):
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

    if blackout_active:
        print(f"[Blackout] {symbol} {timeframe}: new signal generation paused")
        return

    # --- LONG setup ---
    if supports_below and trend != "down":
        # Multi-timeframe gate: don't take a long if the higher timeframe is bearish
        htf_blocks_long = htf_bias is not None and htf_bias["trend"] == "down"
        if not htf_blocks_long:
            zone = supports_below[0]
            if check_reaction(price_now, closes, zone, "support") and has_rejection_candle(
                opens, highs, lows, closes, "bullish"
            ):
                score = 0
                tags = ["S/R reaction", "Rejection candle confirmed", f"Trend: {trend}"]
                if htf_bias is not None:
                    tags.append(f"1H trend: {htf_bias['trend']}")
                    if zone_aligned_with_htf(zone, htf_bias["nearest_support"]):
                        score += 1
                        tags.append("1H support zone alignment")
                if current_rsi <= RSI_OVERSOLD:
                    score += 1
                    tags.append(f"RSI oversold ({current_rsi:.0f})")
                if buy_grab:
                    score += 1
                    tags.append("Liquidity grab (buy-side)")
                if price_in_zone(price_now, bull_fvgs):
                    score += 1
                    tags.append("Bullish FVG")
                if volume_spike(volumes):
                    score += 1
                    tags.append("Volume spike")
                if score >= MIN_OPTIONAL_CONFLUENCE:
                    entry = price_now
                    structural_low = min(lows[-REACTION_LOOKBACK:])
                    anchor = min(zone, structural_low)
                    sl = anchor - (current_atr * SL_ATR_BUFFER)
                    risk = entry - sl
                    tp = resistances_above[0] if resistances_above else entry + risk * DEFAULT_RR
                    rr = round((tp - entry) / risk, 2) if risk > 0 else 0
                    reward_abs = tp - entry
                    min_dist, target_profit = min_reward_distance(symbol, entry)
                    if rr < MIN_RR_RATIO:
                        print(f"[Filtered] {symbol} {timeframe} LONG: R:R {rr} below minimum {MIN_RR_RATIO}")
                    elif reward_abs < min_dist:
                        note = f"(~${target_profit} target on 0.01 lot)" if target_profit else f"({MIN_REWARD_PCT}% floor)"
                        print(f"[Filtered] {symbol} {timeframe} LONG: reward {reward_abs:.5f} below required {min_dist:.5f} {note}")
                    else:
                        key = f"{symbol}_{timeframe}_LONG"
                        if already_alerted_recently(state, key, timeframe):
                            print(f"[Deduped] {symbol} {timeframe} LONG - already alerted within cooldown")
                        else:
                            profit_note = None
                            cfg = INSTRUMENT_PROFIT_TARGETS.get(symbol)
                            if cfg:
                                units = cfg["lot_size"] * cfg["contract_size_per_lot"]
                                profit_note = reward_abs * units
                            alert_signal(symbol, timeframe, "LONG", entry, sl, tp, rr, tags, profit_note)
                            mark_alerted(state, key)
                            log_trade(trade_log, symbol, timeframe, "LONG", entry, sl, tp, times[-1])
                            signal_fired = True

    # --- SHORT setup ---
    if resistances_above and trend != "up":
        htf_blocks_short = htf_bias is not None and htf_bias["trend"] == "up"
        if not htf_blocks_short:
            zone = resistances_above[0]
            if check_reaction(price_now, closes, zone, "resistance") and has_rejection_candle(
                opens, highs, lows, closes, "bearish"
            ):
                score = 0
                tags = ["S/R reaction", "Rejection candle confirmed", f"Trend: {trend}"]
                if htf_bias is not None:
                    tags.append(f"1H trend: {htf_bias['trend']}")
                    if zone_aligned_with_htf(zone, htf_bias["nearest_resistance"]):
                        score += 1
                        tags.append("1H resistance zone alignment")
                if current_rsi >= RSI_OVERBOUGHT:
                    score += 1
                    tags.append(f"RSI overbought ({current_rsi:.0f})")
                if sell_grab:
                    score += 1
                    tags.append("Liquidity grab (sell-side)")
                if price_in_zone(price_now, bear_fvgs):
                    score += 1
                    tags.append("Bearish FVG")
                if volume_spike(volumes):
                    score += 1
                    tags.append("Volume spike")
                if score >= MIN_OPTIONAL_CONFLUENCE:
                    entry = price_now
                    structural_high = max(highs[-REACTION_LOOKBACK:])
                    anchor = max(zone, structural_high)
                    sl = anchor + (current_atr * SL_ATR_BUFFER)
                    risk = sl - entry
                    tp = supports_below[0] if supports_below else entry - risk * DEFAULT_RR
                    rr = round((entry - tp) / risk, 2) if risk > 0 else 0
                    reward_abs = entry - tp
                    min_dist, target_profit = min_reward_distance(symbol, entry)
                    if rr < MIN_RR_RATIO:
                        print(f"[Filtered] {symbol} {timeframe} SHORT: R:R {rr} below minimum {MIN_RR_RATIO}")
                    elif reward_abs < min_dist:
                        note = f"(~${target_profit} target on 0.01 lot)" if target_profit else f"({MIN_REWARD_PCT}% floor)"
                        print(f"[Filtered] {symbol} {timeframe} SHORT: reward {reward_abs:.5f} below required {min_dist:.5f} {note}")
                    else:
                        key = f"{symbol}_{timeframe}_SHORT"
                        if already_alerted_recently(state, key, timeframe):
                            print(f"[Deduped] {symbol} {timeframe} SHORT - already alerted within cooldown")
                        else:
                            profit_note = None
                            cfg = INSTRUMENT_PROFIT_TARGETS.get(symbol)
                            if cfg:
                                units = cfg["lot_size"] * cfg["contract_size_per_lot"]
                                profit_note = reward_abs * units
                            alert_signal(symbol, timeframe, "SHORT", entry, sl, tp, rr, tags, profit_note)
                            mark_alerted(state, key)
                            log_trade(trade_log, symbol, timeframe, "SHORT", entry, sl, tp, times[-1])
                            signal_fired = True

    if not signal_fired:
        near_support = f"{supports_below[0]:.5f}" if supports_below else "none"
        near_resistance = f"{resistances_above[0]:.5f}" if resistances_above else "none"
        htf_note = f" htf_trend={htf_bias['trend']}" if htf_bias is not None else ""
        vol_note = " volume_data=yes" if has_usable_volume(volumes) else " volume_data=no"
        print(
            f"[No signal] {symbol} {timeframe}: price={price_now:.5f} "
            f"RSI={current_rsi:.0f} trend={trend}{htf_note} nearest_support={near_support} "
            f"nearest_resistance={near_resistance}{vol_note}"
        )


def alert_signal(symbol, timeframe, direction, entry, sl, tp, rr, tags, profit_note=None):
    emoji = "🎯"
    tag_str = "\n".join(f"  • {t}" for t in tags)
    profit_line = f"\nEst. profit (0.01 lot): ${profit_note:.2f}" if profit_note is not None else ""
    msg = (
        f"{emoji} *{symbol} - {direction}* ({timeframe})\n\n"
        f"Entry: `{entry:.5f}`\n"
        f"SL: `{sl:.5f}`\n"
        f"TP: `{tp:.5f}`\n"
        f"R:R  ~1:{rr}{profit_line}\n\n"
        f"Confluences:\n{tag_str}\n\n"
        f"_Verify before entering. Not financial advice._"
    )
    print(msg)
    send_telegram(msg)


def run_once():
    state = load_state()
    trade_log = load_trade_log()

    blackout_active, blackout_name = is_blackout()
    if blackout_active:
        print(f"[Blackout] Active: {blackout_name} - pausing new signal generation this run")

    for pair in PAIRS:
        pair_tfs = PAIR_TIMEFRAMES.get(pair, TIMEFRAMES)
        candle_data = {}
        for tf in pair_tfs:
            try:
                data = fetch_candles(pair, tf)
                if data is not None:
                    candle_data[tf] = data
            except Exception as e:
                print(f"[Error] fetching {pair} {tf}: {e}")
            time.sleep(1)  # avoid hammering free-tier rate limits

        # Compute higher-timeframe bias once per pair, reused as a gate for the lower timeframe
        htf_biases = {}
        for tf, htf in HTF_FOR.items():
            if tf in pair_tfs and htf in candle_data:
                try:
                    htf_biases[tf] = get_htf_bias(candle_data[htf])
                except Exception as e:
                    print(f"[Error] computing HTF bias for {pair} ({htf}): {e}")

        for tf in pair_tfs:
            if tf not in candle_data:
                continue
            data = candle_data[tf]
            try:
                check_open_trades(pair, tf, data, trade_log)
                analyze_pair(pair, tf, data, state, trade_log, blackout_active, htf_bias=htf_biases.get(tf))
            except Exception as e:
                print(f"[Error] analyzing {pair} {tf}: {e}")

    maybe_send_daily_summary(state, trade_log)
    save_json(STATE_FILE, state)
    save_json(TRADE_LOG_FILE, trade_log)

    wins, losses, open_trades, win_rate = compute_stats(trade_log)
    print(f"[Stats] Resolved={wins + losses} Wins={wins} Losses={losses} WinRate={win_rate:.1f}% Open={open_trades}")


if __name__ == "__main__":
    run_once()
