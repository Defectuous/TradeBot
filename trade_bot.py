#!/usr/bin/env python3
"""
Simple RSI->GPT->Alpaca trading bot.

Run in paper mode first. Uses TAAPI, OpenAI (gpt-3.5-turbo-instruct), and Alpaca Python SDK.

Configure via environment variables or a .env file. See README.md and .env.example.
"""
import os
import time
import logging
from datetime import datetime, time as dtime, timedelta
import requests
import pytz
from decimal import Decimal

try:
    import openai
except Exception:
    openai = None

try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Required env vars
TAAPI_KEY = os.environ.get("TAAPI_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Bot settings
SYMBOL = os.environ.get("SYMBOL", "AAPL")
# Optional comma-separated list of symbols (overrides SYMBOL when set)
SYMBOLS_ENV = os.environ.get("SYMBOLS")
if SYMBOLS_ENV:
    SYMBOLS = [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]
else:
    SYMBOLS = [SYMBOL]

QTY = int(os.environ.get("QTY", "1"))
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo-instruct")

EAST = pytz.timezone("US/Eastern")


def in_market_hours(now=None):
    now = now or datetime.now(EAST)
    start = datetime.combine(now.date(), dtime(hour=9, minute=30), tzinfo=EAST)
    end = datetime.combine(now.date(), dtime(hour=16, minute=0), tzinfo=EAST)
    return start <= now <= end


def fetch_rsi(symbol: str) -> Decimal:
    if not TAAPI_KEY:
        raise RuntimeError("TAAPI_KEY not set")
    url = "https://api.taapi.io/rsi"
    params = {"secret": TAAPI_KEY, "symbol": symbol, "interval": "1m", "type": "stocks"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Example response: {'value': 34.12, 'time': ...}
    if "value" not in data:
        raise RuntimeError(f"Unexpected TAAPI response: {data}")
    return Decimal(str(data["value"]))


def ask_gpt_for_decision(rsi_value: Decimal) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    if openai is None:
        raise RuntimeError("openai package not installed")
    openai.api_key = OPENAI_API_KEY
    prompt = (
        f"You are an expert stock trader and you are tasked with looking at the Relative Strength "
        f"Index (RSI) of a stock and from this information decide to BUY, SELL, or do NOTHING. "
        f"The stock RSI value is {rsi_value}. Please reply with exactly one of: BUY, SELL, NOTHING."
    )

    # Use ChatCompletion with model that the user specified.
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Normalize to first token
    token = text.split()[0].upper()
    if token not in ("BUY", "SELL", "NOTHING"):
        logger.warning("GPT returned unexpected reply: %s", text)
        return "NOTHING"
    return token


def connect_alpaca():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Alpaca credentials not set")
    if tradeapi is None:
        raise RuntimeError("alpaca_trade_api package not installed")
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
    return api


def can_buy(api, price: Decimal, qty: int) -> bool:
    account = api.get_account()
    buying_power = Decimal(account.buying_power)
    needed = price * qty
    return buying_power >= needed


def owns_at_least(api, symbol: str, qty: int) -> bool:
    try:
        pos = api.get_position(symbol)
        return int(Decimal(pos.qty)) >= qty
    except Exception:
        return False


def place_order(api, symbol: str, qty: int, side: str):
    if DRY_RUN:
        logger.info("DRY RUN: would %s %d %s", side, qty, symbol)
        return None
    logger.info("Submitting %s order: %s %d", side, symbol, qty)
    order = api.submit_order(symbol=symbol, qty=qty, side=side.lower(), type="market", time_in_force="day")
    logger.info("Order submitted: id=%s status=%s", getattr(order, "id", None), getattr(order, "status", None))
    return order


def get_last_trade_price(api, symbol: str) -> Decimal:
    barset = api.get_barset(symbol, "1Min", limit=1)
    bars = barset.get(symbol)
    if not bars:
        # fallback to last trade
        t = api.get_last_trade(symbol)
        return Decimal(str(t.price))
    return Decimal(str(bars[-1].c))


def run_once(api, symbol: str):
    try:
        rsi = fetch_rsi(symbol)
        logger.info("RSI for %s = %s", symbol, rsi)
    except Exception as e:
        logger.exception("Failed to fetch RSI for %s: %s", symbol, e)
        return

    try:
        decision = ask_gpt_for_decision(rsi)
        logger.info("GPT decision for %s: %s", symbol, decision)
    except Exception as e:
        logger.exception("GPT error for %s: %s", symbol, e)
        return

    if decision == "BUY":
        try:
            price = get_last_trade_price(api, symbol)
            if not can_buy(api, price, QTY):
                logger.info("Insufficient buying power to buy %d %s at %s", QTY, symbol, price)
                return
            place_order(api, symbol, QTY, "buy")
        except Exception as e:
            logger.exception("Error handling BUY for %s: %s", symbol, e)
    elif decision == "SELL":
        try:
            if not owns_at_least(api, symbol, QTY):
                logger.info("Do not own %d %s to sell", QTY, symbol)
                return
            place_order(api, symbol, QTY, "sell")
        except Exception as e:
            logger.exception("Error handling SELL for %s: %s", symbol, e)
    else:
        logger.info("Decision NOTHING for %s — no action taken", symbol)


def main():
    logger.info("Starting RSI->GPT->Alpaca bot for %s (DRY_RUN=%s)", SYMBOL, DRY_RUN)
    api = connect_alpaca()

    while True:
        now = datetime.now(EAST)
        if not in_market_hours(now):
            logger.info("Outside market hours (%s). Sleeping till market open.", now.isoformat())
            # Sleep until next market open (tomorrow 9:30 ET)
            tomorrow = now.date() + timedelta(days=1)
            next_open = datetime.combine(tomorrow, dtime(hour=9, minute=30), tzinfo=EAST)
            secs = (next_open - now).total_seconds()
            time.sleep(max(60, int(secs)))
            continue

        # Iterate over configured symbols for this minute
        for sym in SYMBOLS:
            run_once(api, sym)

        # Sleep until the next full minute
        now_local = datetime.now()
        sleep_seconds = 60 - now_local.second
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
