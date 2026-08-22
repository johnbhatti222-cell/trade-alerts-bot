# Trade Alert Bot — Setup

## What this does
Checks BTC/USD, ETH/USD, EUR/USD, GBP/USD on both the 15min and 1h
timeframe, every 15 minutes, and sends a Telegram alert when it finds
an Entry/SL/TP setup backed by at least 2 of: support/resistance
reaction, RSI extreme, liquidity grab (stop hunt), or a fair value gap.

## 1. Create your Telegram bot
1. In Telegram, message **@BotFather** → `/newbot` → follow prompts → copy the **bot token**.
2. Message your new bot anything (e.g. "hi").
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   → find `"chat":{"id":123456789` → that number is your **chat ID**.

## 2. Get a free Twelve Data API key
Sign up at https://twelvedata.com (free tier: 800 requests/day, 8/min).
Copy your API key.

> Rate limit note: 4 pairs × 2 timeframes = 8 requests per run.
> Every 15 min = ~768 requests/day, close to the 800/day free cap.
> If you hit limits, either drop a pair from `PAIRS` in
> `sr_trade_alerts.py`, or reduce frequency in the workflow file.

## 3. Push this folder to a GitHub repo
Create a **new repo** (public repos get unlimited free Actions minutes;
private repos get 2,000 free min/month, which this fits easily).
Push all these files, keeping the `.github/workflows/` folder structure intact.

## 4. Add your secrets
In your repo: **Settings → Secrets and variables → Actions → New repository secret**.
Add all three:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TWELVEDATA_API_KEY`

## 5. Done
The workflow (`.github/workflows/trade_alerts.yml`) runs automatically
every 15 minutes once pushed. To test immediately: go to the **Actions**
tab → "Trade Alerts" → **Run workflow** (manual trigger).

## Tuning
All thresholds live at the top of `sr_trade_alerts.py`:
- `PAIRS` — add/remove symbols (Twelve Data format, e.g. `"BTC/USD"`)
- `MIN_CONFLUENCE` — raise to 3 for stricter/rarer alerts
- `RSI_OVERSOLD` / `RSI_OVERBOUGHT` — RSI thresholds
- `SL_ATR_BUFFER` / `DEFAULT_RR` — stop distance and fallback risk:reward
- `ZONE_MERGE_PCT` — how close swing points must be to count as one S/R zone

## Disclaimer
This is a rules-based pattern detector, not financial advice. Always
verify setups yourself before entering a trade.
