RSI -> GPT -> Alpaca Trading Bot

Overview

This small scaffold implements a minute-by-minute loop that:
- Fetches RSI from TAAPI.io for a configured symbol (1m interval)
- Sends the RSI value to OpenAI (`gpt-3.5-turbo`) asking for a single decision: BUY/SELL/NOTHING
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
- Default model is `gpt-3.5-turbo` but can be overridden with `OPENAI_MODEL`.
- Use `ALPACA_BASE_URL` to switch to paper/live endpoints. Default is paper.
- Always test in `DRY_RUN=true` and with Alpaca paper account before enabling live trading.

- You can monitor multiple symbols by setting a comma-separated `SYMBOLS` environment variable. This
	overrides `SYMBOL`. Example: `SYMBOLS=SPY,AAPL,TSLA`.

Raspberry Pi
-----------

If you want to run the bot on a Raspberry Pi (Raspberry Pi OS / Debian), follow these concise steps.

1. Install system packages and Python tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

2. Create and activate a virtual environment (run inside the project directory):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Configure secrets in `./.env` (keep `DRY_RUN=true` while testing). Example values:

```
TAAPI_KEY=your_taapi_key
OPENAI_API_KEY=your_openai_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DRY_RUN=true
SYMBOLS=AAPL,TSLA,SPY
QTY=1
LOG_LEVEL=INFO
```

5. Run interactively for testing:

```bash
# ensure venv is active
source .venv/bin/activate
python3 ./trade_bot.py
```

6. Run in the background (quick): use `tmux` or `screen`:

```bash
sudo apt install -y tmux
tmux new -s tradebot
# inside tmux:
source .venv/bin/activate
python3 ./trade_bot.py
# detach: Ctrl-B then D
```

7. Recommended: run as a systemd service so it starts at boot. Create `/etc/systemd/system/trade_bot.service` with:

```
[Unit]
Description=RSI->GPT->Alpaca trade bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Trade_Bot
EnvironmentFile=/home/pi/Trade_Bot/.env
ExecStart=/home/pi/Trade_Bot/.venv/bin/python /home/pi/Trade_Bot/trade_bot.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

Then enable and start/stop:

```bash
sudo systemctl daemon-reload
sudo systemctl enable trade_bot.service
sudo systemctl start trade_bot.service
sudo systemctl stop trade_bot.service
sudo journalctl -u trade_bot.service -f
```

Notes
- Keep secrets in `./.env` and secure the Pi filesystem.
- Test thoroughly with `DRY_RUN=true` and Alpaca paper account before using real funds.
- Watch for API rate limits when using many symbols; ask if you want rate-limiting or per-symbol quantities.
