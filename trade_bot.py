#!/usr/bin/env python3
"""
Simple RSI->GPT->Alpaca trading bot.

Run in paper mode first. Uses TAAPI, OpenAI (gpt-3.5-turbo), and Alpaca Python SDK.

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

import threading

# Optional file logging (toggle via .env: LOG_TO_FILE=true)
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "false").lower() in ("1", "true", "yes")
if LOG_TO_FILE:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "Logs")
        os.makedirs(logs_dir, exist_ok=True)

        tz = pytz.timezone("US/Eastern")

        def make_handler():
            ts = datetime.now(tz).strftime("%m%d%Y.%H%M")
            log_filename = os.path.join(logs_dir, f"TradeBot.{ts}.log")
            fh = logging.FileHandler(log_filename)
            fh.setLevel(LOG_LEVEL)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            return fh

        # current file handler stored in a mutable container for rotation
        file_handler = [make_handler()]
        logger.addHandler(file_handler[0])
        logger.info("File logging enabled: %s", file_handler[0].baseFilename)

        def rotator():
            while True:
                try:
                    # compute seconds until next top of hour in Eastern time
                    now_e = datetime.now(tz)
                    next_hour = (now_e + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    sleep_secs = (next_hour - now_e).total_seconds()
                    # sleep until just after the hour boundary
                    time.sleep(max(1, sleep_secs + 1))

                    new_h = make_handler()
                    logger.addHandler(new_h)
                    # remove and close old handler
                    old = file_handler[0]
                    logger.removeHandler(old)
                    try:
                        old.close()
                    except Exception:
                        pass
                    file_handler[0] = new_h
                    logger.info("Rotated log file, new file: %s", file_handler[0].baseFilename)
                except Exception:
                    logger.exception("Log rotation failed")
                    # on failure, wait a minute before retrying
                    time.sleep(60)

        t = threading.Thread(target=rotator, daemon=True)
        t.start()
    except Exception:
        logger.exception("Failed to enable file logging")

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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

EAST = pytz.timezone("US/Eastern")


def in_market_hours(now=None):
    now = now or datetime.now(EAST)
    # Build aware datetimes in US/Eastern correctly using localize
    start_naive = datetime.combine(now.date(), dtime(hour=9, minute=30))
    # Use early close at 13:00 on certain days
    if is_early_close(now.date()):
        end_naive = datetime.combine(now.date(), dtime(hour=13, minute=0))
    else:
        end_naive = datetime.combine(now.date(), dtime(hour=16, minute=0))
    start = EAST.localize(start_naive)
    end = EAST.localize(end_naive)
    return start <= now <= end


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    # weekday: Monday=0 .. Sunday=6
    d = datetime(year, month, 1).date()
    first_weekday = d.weekday()
    delta_days = (weekday - first_weekday) % 7
    day = 1 + delta_days + (n - 1) * 7
    return datetime(year, month, day).date()


def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    # find last weekday in month
    from calendar import monthrange
    last_day = monthrange(year, month)[1]
    d = datetime(year, month, last_day).date()
    delta = (d.weekday() - weekday) % 7
    return d - timedelta(days=delta)


def _easter_date(year: int) -> datetime.date:
    # Anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day).date()


def is_market_holiday(d: datetime.date) -> bool:
    year = d.year
    holidays = set()

    # New Year's Day (observed)
    ny = datetime(year, 1, 1).date()
    if ny.weekday() == 5:  # Saturday -> observed Friday
        holidays.add(ny - timedelta(days=1))
    elif ny.weekday() == 6:  # Sunday -> observed Monday
        holidays.add(ny + timedelta(days=1))
    else:
        holidays.add(ny)

    # Martin Luther King Jr. Day: third Monday in January
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # Presidents' Day (Washington): third Monday in February
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday: Friday before Easter
    eas = _easter_date(year)
    holidays.add(eas - timedelta(days=2))

    # Memorial Day: last Monday in May
    holidays.add(_last_weekday(year, 5, 0))

    # Juneteenth: June 19 (observed)
    j = datetime(year, 6, 19).date()
    if j.weekday() == 5:
        holidays.add(j - timedelta(days=1))
    elif j.weekday() == 6:
        holidays.add(j + timedelta(days=1))
    else:
        holidays.add(j)

    # Independence Day: July 4 (observed)
    ind = datetime(year, 7, 4).date()
    if ind.weekday() == 5:
        holidays.add(ind - timedelta(days=1))
    elif ind.weekday() == 6:
        holidays.add(ind + timedelta(days=1))
    else:
        holidays.add(ind)

    # Labor Day: first Monday in September
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving: fourth Thursday in November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    # Christmas Day: Dec 25 (observed)
    x = datetime(year, 12, 25).date()
    if x.weekday() == 5:
        holidays.add(x - timedelta(days=1))
    elif x.weekday() == 6:
        holidays.add(x + timedelta(days=1))
    else:
        holidays.add(x)

    return d in holidays


def next_trading_day_start(now: datetime) -> datetime:
    # now is tz-aware in EAST
    cur_date = now.date()
    candidate = cur_date
    # start searching from today if before open, else next day
    start_naive = datetime.combine(candidate, dtime(hour=9, minute=30))
    start_dt = EAST.localize(start_naive)
    if now >= start_dt:
        candidate = candidate + timedelta(days=1)

    # find next day that is Mon-Fri and not a market holiday
    while True:
        if candidate.weekday() >= 5 or is_market_holiday(candidate):
            candidate = candidate + timedelta(days=1)
            continue
        # found trading day
        next_open_naive = datetime.combine(candidate, dtime(hour=9, minute=30))
        return EAST.localize(next_open_naive)


def is_early_close(d: datetime.date) -> bool:
    # Early close at 13:00 ET on:
    # - The day before Independence Day (if July 4 is a weekday)
    # - The day after Thanksgiving (Friday)
    # - Christmas Eve (Dec 24)
    year = d.year
    # Day before Independence Day
    ind = datetime(year, 7, 4).date()
    day_before_ind = ind - timedelta(days=1)

    # Day after Thanksgiving: Thanksgiving is fourth Thursday in November
    thanks = _nth_weekday(year, 11, 3, 4)
    day_after_thanks = thanks + timedelta(days=1)

    # Christmas Eve
    xmas_eve = datetime(year, 12, 24).date()

    return d in (day_before_ind, day_after_thanks, xmas_eve)


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

    # Require the new OpenAI client available in openai>=1.0.0
    Client = getattr(openai, "OpenAI", None)
    if Client is None:
        raise RuntimeError("openai package does not expose OpenAI client; please install openai>=1.0.0")

    prompt = (
        f"You are an expert stock trader and you are tasked with looking at the Relative Strength "
        f"Index (RSI) of a stock and from this information decide to BUY, SELL, or do NOTHING. "
        f"The stock RSI value is {rsi_value}. Please reply with exactly one of: BUY, SELL, NOTHING."
    )

    client = Client(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
    except Exception:
        # If the model is not a chat model, the server returns a 404 with a helpful message.
        err = None
        try:
            raise
        except Exception as e:
            err = e
        msg = str(err)
        # detect the specific error about non-chat model and fallback to completions endpoint
        if "not a chat model" in msg or "v1/chat/completions" in msg:
            try:
                resp2 = client.completions.create(
                    model=OPENAI_MODEL,
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0,
                )
                # parse legacy completions response
                try:
                    text = resp2.choices[0].text.strip()
                except Exception:
                    text = str(resp2).strip()
            except Exception:
                logger.exception("Fallback completions API call failed")
                raise
        else:
            logger.exception("GPT API call failed")
            raise

    # Parse response using expected v1 response attributes
    # If text not set by a prior fallback branch, parse chat response
    if 'text' not in locals() or not locals().get('text'):
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            try:
                text = resp["choices"][0]["message"]["content"].strip()
            except Exception:
                text = str(resp).strip()
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
    # Sanitize the base URL to avoid inline comments or trailing spaces
    base_url_raw = os.environ.get("ALPACA_BASE_URL", ALPACA_BASE_URL) or ALPACA_BASE_URL
    # If someone added an inline comment like "https://paper-api.alpaca.markets // For Paper Trading",
    # dotenv will include the comment. Split on whitespace and take the first token.
    base_url = base_url_raw.split()[0].strip()
    # Remove any trailing slashes
    base_url = base_url.rstrip('/')
    logger.info("Connecting to Alpaca base URL: %s", base_url)
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=base_url)
    return api


def get_wallet_amount(api, field: str = "cash") -> Decimal:
    """
    Return a Decimal representing the requested wallet/account field from Alpaca.

    field: one of 'cash', 'buying_power', 'portfolio_value', 'equity'.
    Raises RuntimeError if the account or field cannot be read.
    """
    if api is None:
        raise RuntimeError("Alpaca API client is not provided")

    try:
        account = api.get_account()
    except Exception as e:
        logger.exception("Failed to fetch Alpaca account: %s", e)
        raise RuntimeError("Failed to fetch Alpaca account") from e

    # Normalize requested field and try a few common attribute names
    normalized = field.strip().lower()
    candidates = [normalized, normalized.replace('-', '_')]
    # common alternate names
    alt_map = {
        "cash": ["cash", "cash_balance"],
        "buying_power": ["buying_power", "buyingpower"],
        "portfolio_value": ["portfolio_value", "portfoliovalue", "equity"],
        "equity": ["equity", "portfolio_value"],
    }
    if normalized in alt_map:
        candidates = alt_map[normalized] + candidates

    val = None
    # Try attribute access first
    for c in candidates:
        if hasattr(account, c):
            try:
                val = getattr(account, c)
                break
            except Exception:
                continue

    # If not found as attribute, try mapping/dict access
    if val is None:
        try:
            # account could be a mapping-like object
            if isinstance(account, dict):
                for c in candidates:
                    if c in account:
                        val = account[c]
                        break
        except Exception:
            pass

    if val is None:
        # Last resort: try common attribute names directly
        for c in ("cash", "buying_power", "portfolio_value", "equity"):
            if hasattr(account, c):
                try:
                    val = getattr(account, c)
                    break
                except Exception:
                    continue

    if val is None:
        logger.error("Unable to find requested account field '%s' on Alpaca account object", field)
        raise RuntimeError(f"Account field '{field}' not found on Alpaca account")

    # Return Decimal converted value; account fields are typically strings
    try:
        return Decimal(str(val))
    except Exception as e:
        logger.exception("Failed to convert account field '%s' value to Decimal: %s", field, e)
        raise RuntimeError(f"Invalid numeric value for account field '{field}'") from e


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


def place_order(api, symbol: str, qty, side: str):
    if DRY_RUN:
        logger.info("DRY RUN: would %s %s %s", side, qty, symbol)
        return None
    logger.info("Submitting %s order: %s %s", side, symbol, qty)
    order = api.submit_order(symbol=symbol, qty=qty, side=side.lower(), type="market", time_in_force="day")
    logger.info("Order submitted: id=%s status=%s", getattr(order, "id", None), getattr(order, "status", None))
    return order


def get_last_trade_price(api, symbol: str) -> Decimal:
    # Try old alpaca-trade-api method first
    try:
        if hasattr(api, "get_barset"):
            barset = api.get_barset(symbol, "1Min", limit=1)
            bars = barset.get(symbol)
            if bars:
                return Decimal(str(bars[-1].c))
    except Exception:
        # continue to fallbacks
        pass

    # Common fallback: latest trade endpoint
    for fn in ("get_last_trade", "get_latest_trade", "get_last_trades", "get_latest_trades"):
        if hasattr(api, fn):
            try:
                t = getattr(api, fn)(symbol)
                # t may be an object with attribute price, or have .price, or be a mapping
                price = None
                if hasattr(t, "price"):
                    price = getattr(t, "price")
                elif isinstance(t, dict) and "price" in t:
                    price = t["price"]
                else:
                    # try common attribute names
                    for attr in ("p", "price_raw"):
                        if hasattr(t, attr):
                            price = getattr(t, attr)
                            break
                if price is None:
                    # as a last resort, stringify
                    return Decimal(str(t))
                return Decimal(str(price))
            except Exception:
                continue

    # Another fallback: data API that may provide bars
    if hasattr(api, "get_bars"):
        try:
            bars = api.get_bars(symbol, "1Min", limit=1)
            # bars may be a dict or sequence
            try:
                last = bars[symbol][-1]
                return Decimal(str(last.c))
            except Exception:
                try:
                    # bars may be a sequence
                    last = list(bars)[-1]
                    # try attrs
                    if hasattr(last, "c"):
                        return Decimal(str(last.c))
                    if hasattr(last, "close"):
                        return Decimal(str(last.close))
                    return Decimal(str(last))
                except Exception:
                    pass
        except Exception:
            pass

    raise RuntimeError("Unable to get last trade price for %s with available Alpaca client" % symbol)


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
            # Log wallet amounts before attempting to buy
            try:
                cash_before = get_wallet_amount(api, "cash")
            except Exception:
                cash_before = None
            try:
                bp_before = get_wallet_amount(api, "buying_power")
            except Exception:
                bp_before = None
            logger.info("Wallet before BUY for %s: cash=%s buying_power=%s", symbol, str(cash_before) if cash_before is not None else "<unknown>", str(bp_before) if bp_before is not None else "<unknown>")

            price = get_last_trade_price(api, symbol)
            if not can_buy(api, price, QTY):
                logger.info("Insufficient buying power to buy %d %s at %s", QTY, symbol, price)
                return
            place_order(api, symbol, QTY, "buy")
        except Exception as e:
            logger.exception("Error handling BUY for %s: %s", symbol, e)
    elif decision == "SELL":
        try:
            # Sell entire position for this symbol
            try:
                pos = api.get_position(symbol)
            except Exception:
                logger.info("Do not own any %s to sell", symbol)
                return

            try:
                qty_owned = Decimal(str(pos.qty))
            except Exception:
                # fallback if pos.qty isn't present
                try:
                    qty_owned = Decimal(str(getattr(pos, "qty", 0)))
                except Exception:
                    logger.info("Could not determine position quantity for %s", symbol)
                    return

            if qty_owned <= 0:
                logger.info("Do not own any %s to sell", symbol)
                return

            # Prepare qty for order: use integer shares when whole number, otherwise use float for fractional
            if qty_owned == qty_owned.to_integral_value():
                qty_for_order = int(qty_owned)
            else:
                qty_for_order = float(qty_owned)

            place_order(api, symbol, qty_for_order, "sell")
            # Log wallet amounts after SELL
            try:
                cash_after = get_wallet_amount(api, "cash")
            except Exception:
                cash_after = None
            try:
                pv_after = get_wallet_amount(api, "portfolio_value")
            except Exception:
                pv_after = None
            logger.info("Wallet after SELL for %s: cash=%s portfolio_value=%s", symbol, str(cash_after) if cash_after is not None else "<unknown>", str(pv_after) if pv_after is not None else "<unknown>")
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
            next_open_naive = datetime.combine(tomorrow, dtime(hour=9, minute=30))
            next_open = EAST.localize(next_open_naive)
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
