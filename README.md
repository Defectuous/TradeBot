RSI -> GPT -> Alpaca Trading Bot

Overview

This small scaffold implements a minute-by-minute loop that:
- Fetches RSI from TAAPI.io for a configured symbol (1m interval)
- Sends the RSI value to OpenAI (`gpt-3.5-turbo-instruct`) asking for a single decision: BUY/SELL/NOTHING
- Uses the Alpaca Python SDK to place market orders (paper account recommended)

Files

- `trade_bot.py` - main script
- `requirements.txt` - Python dependencies
- `.env.example` - example environment variables

Quickstart

1. Create a Python 3.9+ virtual environment and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill your keys.

3. Run in dry-run first:

```powershell
$env:DRY_RUN = "true"; python .\trade_bot.py
```

Notes

- The script uses US/Eastern hours (market open 9:30 to 16:00 ET).
- Default model is `gpt-3.5-turbo-instruct` but can be overridden with `OPENAI_MODEL`.
- Use `ALPACA_BASE_URL` to switch to paper/live endpoints. Default is paper.
- Always test in `DRY_RUN=true` and with Alpaca paper account before enabling live trading.
