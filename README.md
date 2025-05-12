# Reddit Telegram Bot

A bot that analyzes Reddit sentiment for specified stock symbols and sends daily reports to Telegram.

## Features

- Fetches and analyzes Reddit posts for stock sentiment
- Generates technical analysis for stocks
- Creates visual reports with charts
- Integrates with Grok AI for insightful summaries
- Scheduled to run automatically on weekdays at 9:00 PM

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your configuration
4. Run the bot: `python telegram_reporter.py`

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Chat ID to send reports to
- `SYMBOLS_TO_ANALYZE`: Comma-separated list of stock symbols (default: "TSLA, PLTR, RKLB")
- `SENTIMENT_HISTORY_DAYS`: Number of days to analyze for sentiment (default: 7)
- `ENABLE_GROK_SUMMARY`: Enable/disable Grok AI summaries (default: True)
- `XAI_API_KEY`: API key for Grok AI integration

## License

This project is licensed under the terms of the license included in the repository. 