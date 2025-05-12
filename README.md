# Technical Naps 💊 Telegram Bot

**Automate your market analysis with actionable insights delivered directly to Telegram.**

Technical Naps 💊 combines Reddit sentiment analysis with comprehensive technical indicators for your chosen stocks and cryptocurrencies. Get daily, automated reports featuring key metrics, sentiment trends, visual charts, and optional AI-powered summaries to help you stay informed and make data-driven decisions.

Built for traders and investors who value timely, consolidated market intelligence.

## ✨ Key Features ✨

*   **Multi-Asset Analysis:** Track both traditional stocks (via Polygon.io) and major cryptocurrencies (via CoinAPI).
*   **Reddit Sentiment Analysis:** Scans relevant subreddits (configurable for stocks vs. crypto) to gauge market sentiment using NLTK's VADER analyzer. Tracks historical sentiment trends.
*   **Technical Indicator Suite:** Calculates and analyzes key TA indicators using `pandas-ta`, including:
    *   Moving Averages (SMA 20, 50, 200)
    *   Bollinger Bands (BBands)
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Stochastic Oscillator (%K, %D)
    *   Volume SMA
*   **DOM Analysis:** Analyzes historical mean daily log returns based on the day of the month to identify potential contrarian signals.
*   **Chart Visualization:** Generates a composite image visualizing:
    *   Price action with SMAs and Bollinger Bands
    *   Trading Volume with Volume SMA
    *   RSI, MACD, and Stochastic Oscillator indicators
    *   Recent Reddit sentiment distribution
    *   Historical sentiment trend
    *   Day-of-Month log return analysis
*   **AI-Powered Summaries (Optional):** Integrates with Grok (via X.ai API) to provide concise, AI-generated summaries synthesizing the sentiment and technical data with a unique "degen" persona (configurable prompt).
*   **Scheduled Reporting:** Uses APScheduler to deliver reports automatically to your specified Telegram chat on a configurable schedule (defaults to weekdays at 9:00 PM UTC).
*   **Robust Error Handling & Retries:** Includes mechanisms for handling API rate limits and network issues for both data fetching and Telegram messaging.

## ⚙️ Getting Started ⚙️

Follow these steps to get your Technical Naps 💊 bot running:

**1. Prerequisites:**

*   Python 3.8+
*   Git
*   Telegram Account & Bot Token
*   API Keys (see Configuration)

**2. Clone the Repository:**

```bash
# Choose a directory name, e.g., TechnicalNapsBot
git clone https://github.com/your-username/TechnicalNaps.git TechnicalNapsBot # Replace with your actual repo URL & desired local folder name
cd TechnicalNapsBot
```
*(Note: You'll likely want to rename the repository itself on GitHub/GitLab if you haven't already)*

**3. Install Dependencies:**

It's highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**4. Download NLTK Data:**

The bot uses NLTK for sentiment analysis. The script attempts to download the necessary `vader_lexicon` data automatically if it's not found in the standard NLTK paths or the included `functions/nltk_data` directory. If you encounter issues, you might need to run this manually in a Python interpreter within your virtual environment:

```python
import nltk
nltk.download('vader_lexicon')
```

**5. Configuration (Crucial!):**

The bot uses environment variables for configuration.

*   **Copy the Example:** Create a `.env` file by copying the example:
    ```bash
    cp .env.example .env
    ```
*   **Edit `.env`:** Open the `.env` file and fill in your credentials and preferences.

    ```dotenv
    # --- Telegram Settings ---
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN" # Get from BotFather on Telegram
    TELEGRAM_CHAT_ID="YOUR_TARGET_CHAT_ID"     # Can be a user ID or group/channel ID

    # --- Asset & Analysis Settings ---
    SYMBOLS_TO_ANALYZE="TSLA,AAPL,BTC,ETH" # Comma-separated list of stock/crypto symbols
    SENTIMENT_HISTORY_DAYS="30"          # How many days back to look for historical sentiment
    HISTORICAL_POST_FETCH_LIMIT="500"    # Max Reddit posts to fetch for historical sentiment (balance data vs API limits)
    STOCK_DATA_LOOKBACK_DAYS="90"        # Days of historical price data for TA (or "YTD")

    # --- API Keys ---
    REDDIT_CLIENT_ID="YOUR_REDDIT_APP_CLIENT_ID"     # From Reddit App settings (script type)
    REDDIT_CLIENT_SECRET="YOUR_REDDIT_APP_SECRET"    # From Reddit App settings
    REDDIT_USER_AGENT="TechnicalNapsBot/1.0 by YourUsername" # Replace with descriptive user agent (Updated Name)

    POLYGON_API_KEY="YOUR_POLYGON_IO_API_KEY"        # For stock data (required for stocks)
    COIN_API_KEY="YOUR_COINAPI_IO_API_KEY"           # For crypto data (required for crypto)

    # --- Grok AI Summary (Optional) ---
    ENABLE_GROK_SUMMARY="True"                     # Set to "False" to disable AI summaries
    XAI_API_KEY="YOUR_X_AI_API_KEY"                # Required if ENABLE_GROK_SUMMARY is True

    # --- Scheduler Settings (Optional - Defaults are in the script) ---
    # REPORT_SCHEDULE_HOUR="21" # Default: 21 (9 PM)
    # REPORT_SCHEDULE_MINUTE="0"  # Default: 0
    # REPORT_SCHEDULE_DAY_OF_WEEK="0-4" # Default: '0-4' (Mon-Fri)
    ```

**6. Running the Bot:**

*   **For a One-Time Test Run:**
    Modify the *last two lines* of `telegram_reporter.py` to comment out the scheduler (`# asyncio.run(main())`) and uncomment the single run (`asyncio.run(run_once())`). Then execute:
    ```bash
    python telegram_reporter.py
    ```
    This will generate and send reports immediately for the configured symbols. Check your Telegram chat.

*   **To Run with the Scheduler:**
    Ensure the *last two lines* of `telegram_reporter.py` have the single run commented out (`# asyncio.run(run_once())`) and the scheduler uncommented (`asyncio.run(main())`). Then run:
    ```bash
    python telegram_reporter.py
    ```
    The script will now run continuously, and the scheduler will trigger the report generation at the specified time (default: 9:00 PM UTC, Mon-Fri). Keep the process running (e.g., using `screen`, `tmux`, or a process manager like `systemd` on a server).

## 🔧 Configuration Details

*   `TELEGRAM_BOT_TOKEN`: Your unique bot token from Telegram's @BotFather.
*   `TELEGRAM_CHAT_ID`: The ID of the user, group, or channel where the bot should send reports. You can get this ID from bots like @userinfobot or @RawDataBot.
*   `SYMBOLS_TO_ANALYZE`: A comma-separated list of tickers (e.g., `AAPL`, `GOOGL`) and crypto symbols (e.g., `BTC`, `ETH`). The bot infers the type based on common crypto symbols (see `COMMON_CRYPTO_SYMBOLS` in `stock_data.py`).
*   `SENTIMENT_HISTORY_DAYS`: Number of past days to analyze for historical sentiment plotting.
*   `HISTORICAL_POST_FETCH_LIMIT`: Controls how many recent Reddit posts are fetched to build the historical sentiment data. Higher values give better history but increase fetch time and API usage.
*   `STOCK_DATA_LOOKBACK_DAYS`: How many days of OHLCV data to fetch for technical analysis calculations. Can be set to `"YTD"` to fetch data since the start of the current year.
*   `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`: Credentials for accessing the Reddit API. Create a 'script' type application on Reddit's app preferences page. **Remember to update the `REDDIT_USER_AGENT` to reflect the new bot name.**
*   `POLYGON_API_KEY`: Your API key from [Polygon.io](https://polygon.io/) for fetching stock market data.
*   `COIN_API_KEY`: Your API key from [CoinAPI.io](https://coinapi.io/) for fetching cryptocurrency market data.
*   `ENABLE_GROK_SUMMARY`: Set to `True` or `False` to enable/disable the AI summary feature.
*   `XAI_API_KEY`: Your API key for the X.ai (Grok) API, required if Grok summaries are enabled.

## 🤖 Usage

Once configured and running with the scheduler, the Technical Naps 💊 bot will automatically send a message containing the analysis report and the generated chart image to the specified `TELEGRAM_CHAT_ID` according to the schedule (default: weekdays at 9:00 PM UTC). If Grok summaries are enabled, the AI-generated text will be sent as a separate message before the chart.

## 🛠️ Technology Stack

*   **Data Fetching:**
    *   Polygon.io (Stocks) via `polygon-api-client`
    *   CoinAPI.io (Crypto) via `httpx`
    *   Reddit API via `asyncpraw`
*   **Analysis:**
    *   `pandas` for data manipulation
    *   `numpy` for numerical operations
    *   `nltk` (VADER) for sentiment analysis
    *   `pandas-ta` for technical indicators
*   **Visualization:** `matplotlib`, `seaborn`
*   **Telegram Integration:** `python-telegram-bot`
*   **Scheduling:** `APScheduler`
*   **Configuration:** `python-dotenv`
*   **AI Summaries (Optional):** X.ai API (Grok) via `httpx`

## ⚠️ Disclaimer

This bot is intended for informational and educational purposes only. The analysis and data provided should **not** be considered financial advice. Trading and investing involve risk, and you should conduct your own research or consult with a qualified financial advisor before making any investment decisions. The accuracy and timeliness of the data depend on the external APIs used.

## 📄 License

This project is licensed under the [MIT License](LICENSE). (You should add a LICENSE file, typically MIT for open-source projects like this).