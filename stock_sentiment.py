import asyncpraw as praw
import asyncio
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import nltk
import matplotlib
matplotlib.use('Agg') # <-- IMPORTANT: Add this BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas_ta as pta # Replaced ta with pandas_ta
import numpy as np
import uuid # For unique filenames
from stock_data import StockDataFetcher
import time # Added for timing
import traceback # Added for better error reporting

import os # Make sure os is imported
# Determine the absolute path to the project's 'functions/nltk_data' directory
# This assumes stock_sentiment.py is in the project root directory.
script_root_dir = os.path.dirname(os.path.abspath(__file__))
custom_nltk_data_path = os.path.join(script_root_dir, 'functions', 'nltk_data')

# Add the custom path to NLTK's data path list if not already present, prioritizing it
if custom_nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_data_path)
# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

sns.set_theme(style="darkgrid")

# --- Configuration for Asset Type Inference ---
COMMON_CRYPTO_SYMBOLS = {"BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "DOGE", "SOL", "SHIB"}
CRYPTO_FULL_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "LTC": "Litecoin",
    "BCH": "Bitcoin Cash", "ADA": "Cardano", "DOT": "Polkadot", "DOGE": "Dogecoin",
    "SOL": "Solana", "SHIB": "Shiba Inu"
}
# ---

class StockSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.sia = SentimentIntensityAnalyzer()
        self.stock_fetcher = StockDataFetcher()

    def _is_crypto(self, symbol):
        return symbol.upper() in COMMON_CRYPTO_SYMBOLS

    def _get_polygon_ticker(self, symbol, is_crypto):
        if is_crypto:
            return f"X:{symbol.upper()}USD"
        return symbol.upper()

    def _get_reddit_search_params(self, symbol, is_crypto):
        if is_crypto:
            subreddit_list = 'bitcoin+CryptoCurrency+CryptoMoonShots+altstreetbets'
            full_name = CRYPTO_FULL_NAMES.get(symbol.upper(), symbol.upper())
            search_query = f'"{symbol.upper()}" OR "{full_name}"'
        else:
            subreddit_list = 'stocks+investing+wallstreetbets+StockMarket'
            search_query = f'"{symbol.upper()}" OR "${symbol.upper()}" stock OR ticker {symbol.upper()}'
        return subreddit_list, search_query

    async def get_reddit_posts(self, symbol, limit=100, is_crypto_asset=None, sort_type='new', time_filter='all'):
        if is_crypto_asset is None:
            is_crypto_asset = self._is_crypto(symbol)

        subreddit_list, search_query = self._get_reddit_search_params(symbol, is_crypto_asset)
        posts_data = []
        try:
            subreddit = await self.reddit.subreddit(subreddit_list)
            async for post in subreddit.search(search_query, limit=limit, sort=sort_type, time_filter=time_filter):
                posts_data.append({
                    'title': post.title, 'text': post.selftext, 'score': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url
                })
        except Exception as e:
            print(f"Error fetching Reddit posts for {symbol} (is_crypto={is_crypto_asset}): {e}")
            return pd.DataFrame()
        return pd.DataFrame(posts_data)

    async def get_historical_reddit_sentiment(self, symbol, history_days=90,
                                        post_fetch_limit=3000,
                                        is_crypto_asset=None):
        if is_crypto_asset is None:
            is_crypto_asset = self._is_crypto(symbol)

        print(f"\nAttempting to fetch up to {post_fetch_limit} RECENT posts for '{symbol}' to analyze historical sentiment.")
        print(f"Targeting sentiment for the last {history_days} days (is_crypto={is_crypto_asset}).")

        all_posts_df = await self.get_reddit_posts(symbol, limit=post_fetch_limit,
                                             is_crypto_asset=is_crypto_asset,
                                             sort_type='new', time_filter='all')

        empty_cutoff_date = datetime.now() - timedelta(days=history_days)
        empty_full_date_range = pd.date_range(start=empty_cutoff_date.date(), end=datetime.now().date(), freq='D')
        empty_historical_df = pd.DataFrame(index=empty_full_date_range, columns=['Avg_Sentiment'])

        if all_posts_df.empty:
            print(f"No posts found for {symbol} with fetch limit {post_fetch_limit}.")
            return empty_historical_df

        print(f"Fetched {len(all_posts_df)} posts for {symbol}. Date range of fetched posts: {all_posts_df['created_utc'].min().strftime('%Y-%m-%d')} to {all_posts_df['created_utc'].max().strftime('%Y-%m-%d')}")

        if not pd.api.types.is_datetime64_any_dtype(all_posts_df['created_utc']):
            all_posts_df['created_utc'] = pd.to_datetime(all_posts_df['created_utc'])

        cutoff_date = datetime.now() - timedelta(days=history_days)
        historical_posts_df_filtered = all_posts_df[all_posts_df['created_utc'] >= cutoff_date].copy()

        if historical_posts_df_filtered.empty:
            print(f"No posts found for {symbol} within the target last {history_days} days (after filtering {len(all_posts_df)} fetched posts).")
            print(f"The oldest post fetched was from {all_posts_df['created_utc'].min().strftime('%Y-%m-%d')}.")
            return empty_historical_df

        actual_min_date = historical_posts_df_filtered['created_utc'].min()
        actual_max_date = historical_posts_df_filtered['created_utc'].max()
        actual_days_covered = (actual_max_date - actual_min_date).days + 1
        print(f"Analyzing sentiment for {len(historical_posts_df_filtered)} posts. Actual data coverage: {actual_min_date.strftime('%Y-%m-%d')} to {actual_max_date.strftime('%Y-%m-%d')} ({actual_days_covered} days).")

        historical_posts_df_filtered['sentiment'] = historical_posts_df_filtered['text'].apply(self.analyze_sentiment)
        historical_posts_df_filtered['date_only'] = historical_posts_df_filtered['created_utc'].dt.date
        daily_sentiment_df = historical_posts_df_filtered.groupby('date_only')['sentiment'].mean().reset_index()
        daily_sentiment_df.rename(columns={'date_only': 'Date', 'sentiment': 'Avg_Sentiment'}, inplace=True)

        if daily_sentiment_df.empty: return empty_historical_df

        daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date'])
        daily_sentiment_df.set_index('Date', inplace=True)

        full_date_range = pd.date_range(start=cutoff_date.date(), end=datetime.now().date(), freq='D')
        daily_sentiment_df = daily_sentiment_df.reindex(full_date_range, fill_value=np.nan)

        return daily_sentiment_df

    def analyze_sentiment(self, text):
        if not text or not isinstance(text, str): return 0.0
        return self.sia.polarity_scores(text)['compound']

    async def get_stock_data(self, symbol):
        df = None
        financial_extras = {
            'current_price': 'N/A', 'currency': 'N/A', 'market_cap': 'N/A',
            'ytd_change_percent': 'N/A', 'historical_data': None
        }
        financial_data = {}
        """
        Fetches stock data using StockDataFetcher and calculates TA indicators using pandas-ta.
        Returns a tuple: (DataFrame with TA indicators, dictionary with other financial data)
        or (None, None) if fetching or processing fails.
        """
        print(f"Fetching comprehensive stock data for {symbol} using StockDataFetcher.")
        raw_data_package = await self.stock_fetcher.get_stock_data(symbol)

        if raw_data_package and raw_data_package.get('success'):
            retrieved_data_payload = raw_data_package.get('data')
            if retrieved_data_payload:
                financial_data = retrieved_data_payload
                financial_extras['current_price'] = financial_data.get('current_price', financial_extras['current_price'])
                financial_extras['currency'] = financial_data.get('currency', financial_extras['currency'])
                financial_extras['market_cap'] = financial_data.get('market_cap', financial_extras['market_cap'])
                financial_extras['ytd_change_percent'] = financial_data.get('ytd_change_percent', financial_extras['ytd_change_percent'])
        else:
            error_msg = raw_data_package.get('error') if raw_data_package else 'Raw data package was None.'
            print(f"Failed to fetch data package or data missing for {symbol}. Error: {error_msg}")

        ohlcv_df = financial_data.get('historical_data')

        if ohlcv_df is not None and not ohlcv_df.empty:
            df = ohlcv_df.copy()
            df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'
            }, inplace=True)

            if 'close' not in df.columns:
                print(f"Critical 'close' column missing in historical data for {symbol}. Cannot calculate TA.")
                df = None
        else:
            print(f"No historical OHLCV data found for {symbol} in the fetched package. TA indicators will be skipped.")

        if df is not None and not df.empty:
            print(f"Calculating TA indicators for {symbol} using pandas-ta on {len(df)} data points.")
            try:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # SMAs and Bollinger Bands
                if len(df) >= 200:
                    df.ta.sma(length=200, append=True)
                    df.ta.sma(length=50, append=True)
                    df.ta.sma(length=20, append=True)
                    df.ta.bbands(length=20, append=True)
                elif len(df) >= 50:
                    df.ta.sma(length=50, append=True)
                    df.ta.sma(length=20, append=True)
                    df.ta.bbands(length=20, append=True)
                    df['SMA_200'] = np.nan
                elif len(df) >= 20:
                    df.ta.sma(length=20, append=True)
                    df.ta.bbands(length=20, append=True)
                    df['SMA_50'] = np.nan
                    df['SMA_200'] = np.nan
                else:
                    print(f"Warning: Insufficient data ({len(df)} points) for SMAs (20, 50, 200) and Bollinger Bands for {symbol}.")
                    df['SMA_20'] = np.nan
                    df['SMA_50'] = np.nan
                    df['SMA_200'] = np.nan
                    df['BBL_20_2.0'] = np.nan
                    df['BBM_20_2.0'] = np.nan
                    df['BBU_20_2.0'] = np.nan
                    df['BBB_20_2.0'] = np.nan
                    df['BBP_20_2.0'] = np.nan

                # MACD
                if len(df) >= 26:
                    df.ta.macd(append=True)
                else:
                    print(f"Warning: Insufficient data ({len(df)} points) for MACD for {symbol}.")
                    df['MACD_12_26_9'] = np.nan
                    df['MACDh_12_26_9'] = np.nan
                    df['MACDs_12_26_9'] = np.nan

                # RSI and Stochastic Oscillator
                if len(df) >= 14:
                    df.ta.rsi(append=True)
                    df.ta.stoch(append=True)
                else:
                    print(f"Warning: Insufficient data ({len(df)} points) for RSI and Stochastic Oscillator for {symbol}.")
                    df['RSI_14'] = np.nan
                    df['STOCHk_14_3_3'] = np.nan
                    df['STOCHd_14_3_3'] = np.nan

                # Volume SMA
                df['VOL_SMA_20'] = np.nan
                if 'volume' in df.columns and isinstance(df['volume'], pd.Series) and not df['volume'].isnull().all():
                    if len(df['volume']) >= 20:
                        try:
                            volume_series_copy = df['volume'].copy()
                            volume_series_copy.dropna(inplace=True)
                            if not volume_series_copy.empty and len(volume_series_copy) >= 20:
                                vol_sma_series = pta.sma(volume_series_copy, length=20)
                                if vol_sma_series is not None and not vol_sma_series.empty:
                                    df['VOL_SMA_20'] = vol_sma_series.reindex(df.index)
                                else:
                                    print(f"Warning: Volume SMA calculation (pta.sma) returned None or empty for {symbol} after dropna.")
                            else:
                                print(f"Warning: Volume series empty or too short for SMA after dropna for {symbol}.")
                        except Exception as vol_e:
                            print(f"Error during Volume SMA specific calculation for {symbol}: {vol_e}")
                    else:
                        print(f"Warning: Insufficient data length in df['volume'] ({len(df['volume'])} points) for Volume SMA for {symbol}.")
                else:
                    print(f"No 'volume' column, not a Series, or all NaN for {symbol}, skipping Volume SMA.")
                
                print(f"Successfully calculated TA for {symbol}. DataFrame columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error calculating TA indicators for {symbol} with pandas-ta: {e}")
                traceback.print_exc()
        else:
            print(f"Skipping TA calculation for {symbol} as DataFrame is None or empty.")
        financial_extras['historical_data'] = df

        return financial_extras.get('historical_data'), financial_extras

    async def process_symbol(self, symbol_input, sentiment_history_days=90, stock_data_days=90, historical_post_limit=3000):
        is_crypto = self._is_crypto(symbol_input)
        print(f"\nProcessing '{symbol_input}' as {'Crypto' if is_crypto else 'Stock'}.")
        print(f"Sentiment history: {sentiment_history_days} days. Historical post limit: {historical_post_limit}.")

        # Fetch Reddit data concurrently
        reddit_task = asyncio.gather(
            self.get_reddit_posts(symbol_input, limit=100, is_crypto_asset=is_crypto),
            self.get_historical_reddit_sentiment(symbol_input,
                                                 history_days=sentiment_history_days,
                                                 post_fetch_limit=historical_post_limit,
                                                 is_crypto_asset=is_crypto)
        )

        # Fetch stock data
        asset_data_df, financial_extras = await self.get_stock_data(symbol_input)

        # Await Reddit results
        recent_posts_df, historical_sentiment_df = await reddit_task

        prediction_text = "Neutral (Sentiment data unavailable)"
        avg_recent_sentiment_value = None
        num_recent_posts_with_sentiment = 0

        if not recent_posts_df.empty:
            recent_posts_df['sentiment'] = recent_posts_df['text'].apply(self.analyze_sentiment)
            valid_sentiments = recent_posts_df['sentiment'].dropna()
            num_recent_posts_with_sentiment = len(valid_sentiments)
            if not valid_sentiments.empty:
                 avg_recent_sentiment_value = valid_sentiments.mean()
                 if avg_recent_sentiment_value > 0.15: prediction_text = f"Bullish (Avg Recent Sentiment: {avg_recent_sentiment_value:.2f})"
                 elif avg_recent_sentiment_value < -0.15: prediction_text = f"Bearish (Avg Recent Sentiment: {avg_recent_sentiment_value:.2f})"
                 else: prediction_text = f"Neutral (Avg Recent Sentiment: {avg_recent_sentiment_value:.2f})"
            else: prediction_text = "Neutral (Recent sentiment data empty/all NaN)"
        else:
            recent_posts_df = pd.DataFrame(columns=['title', 'text', 'score', 'created_utc', 'sentiment', 'url'])

        if asset_data_df is None or asset_data_df.empty: 
            print(f"Warning: Could not fetch or process market data for TA for {symbol_input}.")
            if asset_data_df is None: asset_data_df = pd.DataFrame()

        if financial_extras is None:
            financial_extras = {
                'current_price': 'N/A', 'currency': 'N/A', 'market_cap': 'N/A',
                'ytd_change_percent': 'N/A'
            }
            print(f"Warning: Financial extras (current price, mkt cap, ytd) are unavailable for {symbol_input}.")

        return prediction_text, recent_posts_df, asset_data_df, historical_sentiment_df, avg_recent_sentiment_value, num_recent_posts_with_sentiment, financial_extras

    async def analyze_daily_log_returns_by_dom(self, symbol: str) -> dict:
        """
        Analyzes historical daily log returns for a stock symbol, grouped by the day of the month,
        and generates a buy/not-buy signal based on the historical mean return for the current day of the month.
        Also calculates monthly log returns.
        Uses the synchronous StockDataFetcher.get_historical_data via asyncio.to_thread.
        """
        to_date = datetime.now().date() - timedelta(days=1)
        from_date = to_date - timedelta(days=5*365)

        result = {
            'mean_log_returns_by_dom': pd.Series(dtype=float),
            'current_day_mean_log_return': None,
            'signal': 'Error: Initialization failed',
            'current_dom': datetime.now().day,
            'error_message': None,
            'monthly_log_returns_summary_df': pd.DataFrame(columns=['log_return']) # Initialize
        }

        try:
            print(f"Fetching 5-year daily aggregates for DOM analysis of {symbol} from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
            
            df = await self.stock_fetcher.get_historical_data(
                symbol=symbol.upper(),
                start_date_str=from_date.strftime('%Y-%m-%d'),
                end_date_str=to_date.strftime('%Y-%m-%d'),
                timespan='day'
            )

            if df is None or df.empty:
                error_msg = f"No data returned from StockDataFetcher.get_historical_data for DOM analysis of {symbol}."
                print(error_msg)
                result['error_message'] = error_msg
                result['signal'] = "Error: No data"
                return result

            if df is not None and not df.empty:
                df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'
                }, inplace=True, errors='ignore')
            if 'close' not in df.columns:
                 error_msg = f"'close' column missing in historical data for DOM analysis of {symbol}."
                 print(error_msg)
                 result['error_message'] = error_msg
                 result['signal'] = "Error: Missing 'close' data"
                 return result

            df.sort_index(inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)


            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df.dropna(subset=['log_return'], inplace=True)

            if df.empty:
                error_msg = f"DataFrame is empty after calculating log returns for DOM analysis of {symbol}."
                print(error_msg)
                result['error_message'] = error_msg
                result['signal'] = "Error: No data for log returns"
                return result

            # Monthly Log Returns Calculation
            if not df.empty and 'log_return' in df.columns and isinstance(df.index, pd.DatetimeIndex):
                monthly_log_returns_df = df['log_return'].resample('ME').sum().to_frame()
                result['monthly_log_returns_summary_df'] = monthly_log_returns_df
            else:
                result['monthly_log_returns_summary_df'] = pd.DataFrame(columns=['log_return'])


            df['day_of_month'] = df.index.day
            mean_returns_series = df.groupby('day_of_month')['log_return'].mean()
            result['mean_log_returns_by_dom'] = mean_returns_series

            current_dom_int = datetime.now().day
            result['current_dom'] = current_dom_int
            
            current_day_historical_mean_log_return = np.nan
            if current_dom_int in mean_returns_series.index:
                current_day_historical_mean_log_return = mean_returns_series.loc[current_dom_int]
            
            result['current_day_mean_log_return'] = current_day_historical_mean_log_return if pd.notna(current_day_historical_mean_log_return) else None

            if pd.notna(current_day_historical_mean_log_return):
                if current_day_historical_mean_log_return < 0:
                    result['signal'] = 'Buy Signal (Contrarian DOM)'
                elif current_day_historical_mean_log_return > 0:
                    result['signal'] = 'Not a Buy Signal (Contrarian DOM - Positive Hist. Return)'
                else:
                    result['signal'] = 'Neutral Signal (Contrarian DOM - Zero Hist. Return)'
            else:
                result['signal'] = 'Not a Buy Signal (DOM data N/A)'
                print(f"Current DOM {current_dom_int} not found in historical mean returns for {symbol}. Defaulting to 'Not a Buy Signal (DOM data N/A)'.")

        except Exception as e:
            error_msg = f"An error occurred while analyzing {symbol} for DOM: {e}"
            print(error_msg)
            traceback.print_exc()
            result['error_message'] = error_msg
            result['signal'] = f"Error in DOM: {str(e)}"
            if not isinstance(result.get('mean_log_returns_by_dom'), pd.Series):
                 result['mean_log_returns_by_dom'] = pd.Series(dtype=float)
            if not isinstance(result.get('current_day_mean_log_return'), (float, type(None))):
                result['current_day_mean_log_return'] = None
            if 'monthly_log_returns_summary_df' not in result or not isinstance(result['monthly_log_returns_summary_df'], pd.DataFrame):
                result['monthly_log_returns_summary_df'] = pd.DataFrame(columns=['log_return'])
        return result

    def _get_ta_metrics_analysis(self, latest_data, dom_analysis_results=None, overall_reddit_sentiment=None):
        """
        Analyzes the latest row of TA data (a pandas Series) and other sentiment/DOM results.
        Expects lowercase column names ('close', 'volume') and pandas-ta standard names.
        Returns a dictionary with metrics details, average score, and summary emoji.
        """
        ta_metrics_details = []
        ta_scores = []

        # 1. RSI (14) - pandas-ta default is RSI_14
        rsi_value = latest_data.get('RSI_14', np.nan)
        rsi_emoji = "🟡"
        rsi_score = 0.0
        if pd.notna(rsi_value):
            if rsi_value < 30: rsi_emoji, rsi_score = "🟢", 0.7
            elif rsi_value > 65: rsi_emoji, rsi_score = "🔴", -0.7 # Changed from 70 to 65
            else: rsi_emoji, rsi_score = "🟡", 0.0 # Neutral range is now 30 <= RSI <= 65
            ta_metrics_details.append({'name': 'RSI (14)', 'value': f"{rsi_value:.2f}", 'emoji': rsi_emoji, 'score': rsi_score, 'formatted_string': f"RSI (14): {rsi_value:.2f} {rsi_emoji} (Score: {rsi_score:.2f})"})
            ta_scores.append(rsi_score)
        else:
            ta_metrics_details.append({'name': 'RSI (14)', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "RSI (14): N/A ⚪ (Score: 0.00)"})

        # 2. MACD - pandas-ta defaults: MACD_12_26_9 (MACD line), MACDs_12_26_9 (Signal line)
        macd_value = latest_data.get('MACD_12_26_9', np.nan)
        macd_signal_value = latest_data.get('MACDs_12_26_9', np.nan)
        macd_emoji = "🟡"
        macd_score = 0.0
        if pd.notna(macd_value) and pd.notna(macd_signal_value):
            if macd_value > macd_signal_value: macd_emoji, macd_score = "🟢", 0.6
            elif macd_value < macd_signal_value: macd_emoji, macd_score = "🔴", -0.6
            else: macd_emoji, macd_score = "🟡", 0.0
            ta_metrics_details.append({'name': 'MACD', 'value': f"MACD: {macd_value:.2f}, Signal: {macd_signal_value:.2f}", 'emoji': macd_emoji, 'score': macd_score, 'formatted_string': f"MACD ({macd_value:.2f}) vs Signal ({macd_signal_value:.2f}) {macd_emoji} (Score: {macd_score:.2f})"})
            ta_scores.append(macd_score)
        else:
            ta_metrics_details.append({'name': 'MACD', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "MACD: N/A ⚪ (Score: 0.00)"})

        # 3. Price vs SMA 50 - pandas-ta default is SMA_50
        price = latest_data.get('close', np.nan)
        sma_50 = latest_data.get('SMA_50', np.nan)
        sma_50_emoji = "🟡"
        sma_50_score = 0.0
        if pd.notna(price) and pd.notna(sma_50):
            if price > sma_50: sma_50_emoji, sma_50_score = "🟢", 0.5
            elif price < sma_50: sma_50_emoji, sma_50_score = "🔴", -0.5
            else: sma_50_emoji, sma_50_score = "🟡", 0.0
            ta_metrics_details.append({'name': 'Price vs SMA 50', 'value': f"P: {price:.2f}, SMA50: {sma_50:.2f}", 'emoji': sma_50_emoji, 'score': sma_50_score, 'formatted_string': f"Price vs SMA 50 ({price:.2f} vs {sma_50:.2f}) {sma_50_emoji} (Score: {sma_50_score:.2f})"})
            ta_scores.append(sma_50_score)
        else:
            ta_metrics_details.append({'name': 'Price vs SMA 50', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "Price vs SMA 50: N/A ⚪ (Score: 0.00)"})

        # 4. Price vs SMA 200 - pandas-ta default is SMA_200
        sma_200 = latest_data.get('SMA_200', np.nan)
        sma_200_emoji = "🟡"
        sma_200_score = 0.0
        if pd.notna(price) and pd.notna(sma_200):
            if price > sma_200: sma_200_emoji, sma_200_score = "🟢", 0.8
            elif price < sma_200: sma_200_emoji, sma_200_score = "🔴", -0.8
            else: sma_200_emoji, sma_200_score = "🟡", 0.0
            ta_metrics_details.append({'name': 'Price vs SMA 200', 'value': f"P: {price:.2f}, SMA200: {sma_200:.2f}", 'emoji': sma_200_emoji, 'score': sma_200_score, 'formatted_string': f"Price vs SMA 200 ({price:.2f} vs {sma_200:.2f}) {sma_200_emoji} (Score: {sma_200_score:.2f})"})
            ta_scores.append(sma_200_score)
        else:
            ta_metrics_details.append({'name': 'Price vs SMA 200', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "Price vs SMA 200: N/A ⚪ (Score: 0.00)"})

        # 5. Volume vs SMA 20 (Volume) - pandas-ta default is VOL_SMA_20
        volume = latest_data.get('volume', np.nan)
        vol_sma_20 = latest_data.get('VOL_SMA_20', np.nan)
        vol_emoji = "🟡"
        vol_score = 0.0
        if pd.notna(volume) and pd.notna(vol_sma_20) and vol_sma_20 > 0:
            if volume > vol_sma_20 * 1.5: vol_emoji, vol_score = "🟢", 0.3
            elif volume < vol_sma_20 * 0.7: vol_emoji, vol_score = "🔴", -0.2
            else: vol_emoji, vol_score = "🟡", 0.0
            ta_metrics_details.append({'name': 'Volume vs SMA 20', 'value': f"Vol: {volume:,.0f}, SMA20: {vol_sma_20:,.0f}", 'emoji': vol_emoji, 'score': vol_score, 'formatted_string': f"Volume ({volume:,.0f}) vs SMA ({vol_sma_20:,.0f}) {vol_emoji} (Score: {vol_score:.2f})"})
            ta_scores.append(vol_score)
        else:
            ta_metrics_details.append({'name': 'Volume vs SMA 20', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "Volume vs SMA 20: N/A ⚪ (Score: 0.00)"})

        # 6. Stochastic Oscillator (%K) - pandas-ta defaults: STOCHk_14_3_3
        stoch_k = latest_data.get('STOCHk_14_3_3', np.nan)
        stoch_emoji = "🟡"
        stoch_score = 0.0
        if pd.notna(stoch_k):
            if stoch_k < 20: stoch_emoji, stoch_score = "🟢", 0.4
            elif stoch_k > 80: stoch_emoji, stoch_score = "🔴", -0.4
            else: stoch_emoji, stoch_score = "🟡", 0.0
            ta_metrics_details.append({'name': 'Stochastic %K (14,3,3)', 'value': f"{stoch_k:.2f}", 'emoji': stoch_emoji, 'score': stoch_score, 'formatted_string': f"Stochastic %K: {stoch_k:.2f} {stoch_emoji} (Score: {stoch_score:.2f})"})
            ta_scores.append(stoch_score)
        else:
            ta_metrics_details.append({'name': 'Stochastic %K (14,3,3)', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "Stochastic %K: N/A ⚪ (Score: 0.00)"})
        
        # 7. Bollinger Bands %B - pandas-ta default is BBP_20_2.0
        bbp = latest_data.get('BBP_20_2.0', np.nan)
        bb_emoji = "🟡"
        bb_score = 0.0
        if pd.notna(bbp):
            if bbp > 1.0: bb_emoji, bb_score = "🔴", -0.3
            elif bbp < 0.0: bb_emoji, bb_score = "🟢", 0.3
            elif bbp > 0.8: bb_emoji, bb_score = "🟠", -0.15 
            elif bbp < 0.2: bb_emoji, bb_score = "🟢", 0.15
            else: bb_emoji, bb_score = "🟡", 0.0 
            ta_metrics_details.append({'name': 'Bollinger Bands %B (20,2)', 'value': f"{bbp:.2f}", 'emoji': bb_emoji, 'score': bb_score, 'formatted_string': f"BB %B: {bbp:.2f} {bb_emoji} (Score: {bb_score:.2f})"})
            ta_scores.append(bb_score)
        else:
            ta_metrics_details.append({'name': 'Bollinger Bands %B (20,2)', 'value': "N/A", 'emoji': "⚪", 'score': 0.0, 'formatted_string': "BB %B: N/A ⚪ (Score: 0.00)"})

        # 8. Monthly Log Return (Previous Complete Month)
        monthly_log_return_value_raw = np.nan
        monthly_log_return_period_str = "N/A"
        monthly_lr_score = 0.0
        monthly_lr_emoji = "⚪"

        if dom_analysis_results and 'monthly_log_returns_summary_df' in dom_analysis_results:
            df_monthly = dom_analysis_results['monthly_log_returns_summary_df']
            if df_monthly is not None and not df_monthly.empty and 'log_return' in df_monthly.columns:
                df_monthly_sorted = df_monthly.sort_index()
                if not df_monthly_sorted.empty:
                    target_month_end = pd.Timestamp.now().normalize().replace(day=1) - pd.Timedelta(days=1)
                    if target_month_end in df_monthly_sorted.index:
                        monthly_log_return_value_raw = df_monthly_sorted.loc[target_month_end, 'log_return']
                        monthly_log_return_period_str = target_month_end.strftime('%Y-%m')
                    elif len(df_monthly_sorted.index) > 0 and df_monthly_sorted.index[-1] < pd.Timestamp.now().normalize().replace(day=1):
                        monthly_log_return_value_raw = df_monthly_sorted['log_return'].iloc[-1]
                        monthly_log_return_period_str = df_monthly_sorted.index[-1].strftime('%Y-%m')


        if pd.notna(monthly_log_return_value_raw):
            monthly_log_return_percent = monthly_log_return_value_raw * 100
            monthly_lr_score = min(1.0, max(-1.0, monthly_log_return_percent / 10.0))
            if monthly_lr_score > 0.05: monthly_lr_emoji = "🔴" # Positive return, negative signal
            elif monthly_lr_score < -0.05: monthly_lr_emoji = "🟢" # Negative return, positive signal
            else: monthly_lr_emoji = "🟡" # Neutral
            ta_metrics_details.append({
                'name': f"Monthly Log Return ({monthly_log_return_period_str})",
                'value': f"{monthly_log_return_percent:.2f}%",
                'emoji': monthly_lr_emoji, 'score': monthly_lr_score,
                'formatted_string': f"Monthly Log Ret ({monthly_log_return_period_str}): {monthly_log_return_percent:.2f}% {monthly_lr_emoji} (Score: {monthly_lr_score:.2f})"
            })
            ta_scores.append(monthly_lr_score)
        else:
            ta_metrics_details.append({
                'name': "Monthly Log Return (Prev)", 'value': "N/A", 'emoji': "⚪", 'score': 0.0,
                'formatted_string': "Monthly Log Ret (Prev): N/A ⚪ (Score: 0.00)"
            })

        # 9. Reddit Sentiment Score
        reddit_sentiment_val = np.nan
        reddit_score = 0.0
        reddit_emoji = "⚪"
        if overall_reddit_sentiment is not None and pd.notna(overall_reddit_sentiment):
            reddit_sentiment_val = overall_reddit_sentiment
            reddit_score = reddit_sentiment_val
            if reddit_score > 0.15: reddit_emoji = "🟢"
            elif reddit_score < -0.15: reddit_emoji = "🔴"
            else: reddit_emoji = "🟡"
            ta_metrics_details.append({
                'name': "Reddit Sentiment", 'value': f"{reddit_sentiment_val:.2f}",
                'emoji': reddit_emoji, 'score': reddit_score,
                'formatted_string': f"Reddit Sentiment: {reddit_sentiment_val:.2f} {reddit_emoji} (Score: {reddit_score:.2f})"
            })
            ta_scores.append(reddit_score)
        else:
            ta_metrics_details.append({
                'name': "Reddit Sentiment", 'value': "N/A", 'emoji': "⚪", 'score': 0.0,
                'formatted_string': "Reddit Sentiment: N/A ⚪ (Score: 0.00)"
            })


        # Calculate average score and overall emoji
        avg_score = sum(ta_scores) / len(ta_scores) if ta_scores else 0.0
        overall_emoji = "⚪"
        if avg_score > 0.20: overall_emoji = "🟢"
        elif avg_score > 0.05: overall_emoji = "🟢"
        elif avg_score < -0.20: overall_emoji = "🔴"
        elif avg_score < -0.05: overall_emoji = "🟠"
        else: overall_emoji = "🟡"


        return {
            'details': ta_metrics_details,
            'average_score': avg_score,
            'summary_emoji': overall_emoji
        }

    def _format_market_cap(self, mc):
        if mc == 'N/A' or not isinstance(mc, (int, float)) or pd.isna(mc):
            return 'N/A'
        if mc >= 1_000_000_000_000:
            return f"{mc / 1_000_000_000_000:.2f}T"
        if mc >= 1_000_000_000:
            return f"{mc / 1_000_000_000:.2f}B"
        if mc >= 1_000_000:
            return f"{mc / 1_000_000:.2f}M"
        return f"{mc:,.0f}"

# --- Plotting Functions ---
def plot_price_and_volume_on_ax(ax, asset_df, symbol):
    """Plots price, SMAs, Bollinger Bands, and volume."""
    ax.set_title(f'{symbol.upper()} Price, Volume & TA', fontsize=14)
    ax.plot(asset_df.index, asset_df['close'], label='Close Price', color='blue', alpha=0.7)
    
    # Plot SMAs if they exist
    if 'SMA_20' in asset_df.columns and not asset_df['SMA_20'].isnull().all():
        ax.plot(asset_df.index, asset_df['SMA_20'], label='SMA 20', color='orange', linestyle='--', alpha=0.7)
    if 'SMA_50' in asset_df.columns and not asset_df['SMA_50'].isnull().all():
        ax.plot(asset_df.index, asset_df['SMA_50'], label='SMA 50', color='green', linestyle='--', alpha=0.7)
    if 'SMA_200' in asset_df.columns and not asset_df['SMA_200'].isnull().all():
        ax.plot(asset_df.index, asset_df['SMA_200'], label='SMA 200', color='red', linestyle='--', alpha=0.7)
        
    # Plot Bollinger Bands if they exist
    if 'BBU_20_2.0' in asset_df.columns and 'BBL_20_2.0' in asset_df.columns and \
       not asset_df['BBU_20_2.0'].isnull().all() and not asset_df['BBL_20_2.0'].isnull().all():
        ax.plot(asset_df.index, asset_df['BBU_20_2.0'], label='Upper BB', color='gray', linestyle=':', alpha=0.5)
        ax.plot(asset_df.index, asset_df['BBL_20_2.0'], label='Lower BB', color='gray', linestyle=':', alpha=0.5)
        ax.fill_between(asset_df.index, asset_df['BBL_20_2.0'], asset_df['BBU_20_2.0'], color='gray', alpha=0.1)

    ax.set_ylabel('Price', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Volume subplot
    ax2 = ax.twinx()
    ax2.bar(asset_df.index, asset_df['volume'], label='Volume', color='grey', alpha=0.3)
    # Plot Volume SMA if it exists
    if 'VOL_SMA_20' in asset_df.columns and not asset_df['VOL_SMA_20'].isnull().all():
        ax2.plot(asset_df.index, asset_df['VOL_SMA_20'], label='Volume SMA 20', color='purple', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Volume', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim(bottom=0) # Ensure volume starts at 0
    ax2.legend(loc='upper right')

def plot_sentiment_dist_on_ax(ax, posts_df, symbol):
    """Plots the distribution of recent Reddit sentiment scores."""
    ax.set_title(f'{symbol.upper()} Recent Reddit Sentiment Distribution', fontsize=14)
    if 'sentiment' in posts_df.columns and not posts_df['sentiment'].isnull().all():
        sns.histplot(posts_df['sentiment'], bins=20, kde=True, ax=ax, color='skyblue')
        avg_sent = posts_df['sentiment'].mean()
        ax.axvline(avg_sent, color='red', linestyle='--', label=f'Avg: {avg_sent:.2f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Sentiment Data", ha='center', va='center')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.grid(True)

def plot_rsi_on_ax(ax, asset_df, symbol):
    """Plots the RSI indicator."""
    ax.set_title(f'{symbol.upper()} RSI (14)', fontsize=14)
    if 'RSI_14' in asset_df.columns and not asset_df['RSI_14'].isnull().all():
        ax.plot(asset_df.index, asset_df['RSI_14'], label='RSI', color='purple')
        ax.axhline(65, color='red', linestyle='--', alpha=0.5, label='Overbought (65)') # Changed from 70 to 65
        ax.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax.set_ylim(0, 100)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "RSI N/A", ha='center', va='center')
    ax.set_ylabel('RSI Value')
    ax.grid(True)

def plot_macd_on_ax(ax, asset_df, symbol):
    """Plots the MACD indicator."""
    ax.set_title(f'{symbol.upper()} MACD (12, 26, 9)', fontsize=14)
    if 'MACD_12_26_9' in asset_df.columns and 'MACDs_12_26_9' in asset_df.columns and 'MACDh_12_26_9' in asset_df.columns and \
       not asset_df['MACD_12_26_9'].isnull().all():
        ax.plot(asset_df.index, asset_df['MACD_12_26_9'], label='MACD', color='blue')
        ax.plot(asset_df.index, asset_df['MACDs_12_26_9'], label='Signal', color='red', linestyle='--')
        # Plot histogram using bars
        colors = ['green' if x > 0 else 'red' for x in asset_df['MACDh_12_26_9']]
        ax.bar(asset_df.index, asset_df['MACDh_12_26_9'], label='Histogram', color=colors, alpha=0.5)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "MACD N/A", ha='center', va='center')
    ax.set_ylabel('MACD Value')
    ax.grid(True)

def plot_historical_sentiment_on_ax(ax, sentiment_df, symbol, target_history_days=90):
    """Plots historical average daily sentiment with a rolling average."""
    ax.set_title(f'{symbol.upper()} Historical Sentiment ({target_history_days} days)', fontsize=14)
    if not sentiment_df.empty and 'Avg_Sentiment' in sentiment_df.columns:
        # Plot raw daily average
        ax.plot(sentiment_df.index, sentiment_df['Avg_Sentiment'], label='Daily Avg Sentiment', color='lightblue', alpha=0.6, marker='.', linestyle='None')
        
        # Calculate and plot rolling average (e.g., 7-day)
        rolling_avg = sentiment_df['Avg_Sentiment'].rolling(window=7, min_periods=3).mean() # min_periods to show earlier
        ax.plot(rolling_avg.index, rolling_avg, label='7-Day Rolling Avg', color='orange', linewidth=2)
        
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.set_ylim(-1, 1) # VADER compound score range
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Historical Sentiment N/A", ha='center', va='center')
    ax.set_ylabel('Avg Sentiment Score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

def plot_stochastic_on_ax(ax, asset_df, symbol):
    """Plots the Stochastic Oscillator (%K and %D)."""
    ax.set_title(f'{symbol.upper()} Stochastic Oscillator (14, 3, 3)', fontsize=14)
    if 'STOCHk_14_3_3' in asset_df.columns and 'STOCHd_14_3_3' in asset_df.columns and \
       not asset_df['STOCHk_14_3_3'].isnull().all():
        ax.plot(asset_df.index, asset_df['STOCHk_14_3_3'], label='%K', color='blue')
        ax.plot(asset_df.index, asset_df['STOCHd_14_3_3'], label='%D', color='orange', linestyle='--')
        ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='Overbought (80)')
        ax.axhline(20, color='green', linestyle='--', alpha=0.5, label='Oversold (20)')
        ax.set_ylim(0, 100)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Stochastic N/A", ha='center', va='center')
    ax.set_ylabel('Stochastic Value')
    ax.grid(True)

def plot_dom_log_returns_on_ax(ax, dom_log_returns_series, symbol):
    """Plots the mean log returns by day of the month."""
    ax.set_title(f'{symbol.upper()} Mean Daily Log Return by Day of Month', fontsize=14)
    if dom_log_returns_series is not None and not dom_log_returns_series.empty:
        # Ensure index is sorted for plotting
        dom_log_returns_series_sorted = dom_log_returns_series.sort_index()
        
        # Create colors based on positive/negative returns
        colors = ['green' if x >= 0 else 'red' for x in dom_log_returns_series_sorted]
        
        dom_log_returns_series_sorted.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
        
        # Highlight current day of month if data exists
        current_dom = datetime.now().day
        if current_dom in dom_log_returns_series_sorted.index:
            ax.patches[dom_log_returns_series_sorted.index.get_loc(current_dom)].set_edgecolor('black')
            ax.patches[dom_log_returns_series_sorted.index.get_loc(current_dom)].set_linewidth(1.5)
            # Add text annotation for the current day's mean return
            current_dom_mean = dom_log_returns_series_sorted.loc[current_dom]
            ax.text(dom_log_returns_series_sorted.index.get_loc(current_dom), current_dom_mean, f'{current_dom_mean:.4f}', ha='center', va='bottom' if current_dom_mean >= 0 else 'top', fontsize=9)

        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Day of Month')
        ax.set_ylabel('Mean Log Return')
        ax.tick_params(axis='x', rotation=0) # Keep day numbers horizontal
    else:
        ax.text(0.5, 0.5, "DOM Log Returns N/A", ha='center', va='center')
    ax.grid(axis='y', linestyle='--', alpha=0.6) # Grid lines for y-axis only


# --- Main Report Generation ---
async def generate_analysis_report(symbol_input, sentiment_history_days=90, stock_data_days=90, historical_post_limit=3000):
    start_time = time.time()
    analyzer = StockSentimentAnalyzer()
    unique_id = uuid.uuid4().hex[:8] # Unique ID for this run
    temp_img_dir = "temp_images"
    os.makedirs(temp_img_dir, exist_ok=True) # Ensure temp directory exists
    
    # Initialize results
    grok_data_payload = {'SYMBOL': symbol_input.upper()} # Start payload for Grok
    image_path = None
    dom_analysis_results = {}
    avg_recent_sentiment = None

    async with analyzer.reddit as reddit_session: # Use context manager for praw session
        try:
            print(f"--- Starting Analysis for {symbol_input} ---")
            
            # Fetch all data concurrently where possible
            process_task = analyzer.process_symbol(symbol_input, sentiment_history_days, stock_data_days, historical_post_limit)
            dom_task = analyzer.analyze_daily_log_returns_by_dom(symbol_input)
            
            # Await results
            (prediction_text, recent_posts_df, asset_data_df, historical_sentiment_df,
             avg_recent_sentiment, num_recent_posts, financial_extras) = await process_task
            dom_analysis_results = await dom_task # Fetch DOM results

            # Populate Grok payload with basic info
            grok_data_payload['AVG_RECENT_SENTIMENT_SCORE'] = avg_recent_sentiment if avg_recent_sentiment is not None else np.nan
            grok_data_payload['NUM_RECENT_POSTS'] = num_recent_posts
            initial_simple_prediction_for_grok_prompt = prediction_text # Store the simple prediction
            
            # Add sample posts details to payload
            sample_posts_list = []
            if recent_posts_df is not None and not recent_posts_df.empty:
                 # Ensure 'sentiment' column exists before sorting
                 if 'sentiment' not in recent_posts_df.columns:
                     recent_posts_df['sentiment'] = recent_posts_df['text'].apply(analyzer.analyze_sentiment)
                 # Sort by score (descending) and take top N (e.g., 5)
                 top_posts_df = recent_posts_df.nlargest(5, 'score')
                 sample_posts_list = top_posts_df[['title', 'sentiment']].to_dict('records') # Get title and score
            grok_data_payload['SAMPLE_POSTS_DETAILS'] = sample_posts_list

            # --- Financial Extras Formatting ---
            current_price_str = "N/A"
            market_cap_str = "N/A"
            ytd_change_str = "N/A"
            currency_symbol_emoji = ""
            ytd_emoji = "⚪"
            current_price_val_num = None # To store numeric price for TA override
            latest_volume_val_str = "N/A" # Initialize volume string

            if financial_extras:
                cp_raw = financial_extras.get('current_price', 'N/A')
                if isinstance(cp_raw, (int, float)):
                    current_price_val_num = cp_raw # Store numeric value
                    current_price_str = f"{cp_raw:.2f}"
                else:
                    current_price_str = str(cp_raw)

                mc_raw = financial_extras.get('market_cap', 'N/A')
                if isinstance(mc_raw, (int, float)):
                    market_cap_str = analyzer._format_market_cap(mc_raw)
                else:
                    market_cap_str = str(mc_raw)

                ytd_raw = financial_extras.get('ytd_change_percent', 'N/A')
                if isinstance(ytd_raw, (int, float)):
                    ytd_change_str = f"{ytd_raw:+.2f}%"
                    if ytd_raw > 0.01: ytd_emoji = "📈"
                    elif ytd_raw < -0.01: ytd_emoji = "📉"
                else:
                    ytd_change_str = str(ytd_raw)
                
                fetched_currency = financial_extras.get('currency', 'USD')
                if fetched_currency == "USD": currency_symbol_emoji = "$"
                elif fetched_currency == "EUR": currency_symbol_emoji = "€"
                # Only add currency symbol if price is numeric
                current_price_str = f"{currency_symbol_emoji}{current_price_str}" if current_price_val_num is not None else "N/A"
            
            grok_data_payload['LATEST_CLOSE_PRICE'] = current_price_str
            grok_data_payload['YTD_CHANGE_PERCENT'] = ytd_change_str
            grok_data_payload['MARKET_CAP'] = market_cap_str

            # --- TA Metrics Analysis ---
            ta_metrics_details_list = []
            average_ta_score_val = 0.0
            ta_overall_summary_emoji = "⚪"
            
            latest_data_for_ta = {} # Initialize as empty dict, will become Series
            if asset_data_df is not None and not asset_data_df.empty:
                latest_data_for_ta = asset_data_df.iloc[-1].copy() # Get last row as Series
                # Override 'close' in the series if we have a more current one from financial_extras
                if current_price_val_num is not None: 
                    latest_data_for_ta['close'] = current_price_val_num
                
                # Extract latest volume for summary block
                lv_raw = latest_data_for_ta.get('volume', np.nan) # 'volume' should be lowercase
                if pd.notna(lv_raw): latest_volume_val_str = f"{lv_raw:,.0f}"
                
                # Perform TA analysis on the latest data Series, now including DOM and Reddit sentiment
                ta_analysis_results_dict = analyzer._get_ta_metrics_analysis(
                    latest_data_for_ta,
                    dom_analysis_results=dom_analysis_results, # Pass the full dict
                    overall_reddit_sentiment=avg_recent_sentiment # Pass the calculated sentiment score
                )
                ta_metrics_details_list = ta_analysis_results_dict.get('details', []) # Changed 'metrics' to 'details'
                average_ta_score_val = ta_analysis_results_dict.get('average_score', 0.0) # Already a float
                ta_overall_summary_emoji = ta_analysis_results_dict.get('summary_emoji', "⚪") # Changed 'average_summary' to 'summary_emoji'

                # Populate TA fields in grok_data_payload using pandas-ta names
                grok_data_payload['LATEST_VOLUME'] = latest_data_for_ta.get('volume', np.nan)
                grok_data_payload['LATEST_SMA_20'] = latest_data_for_ta.get('SMA_20', np.nan)
                grok_data_payload['LATEST_SMA_50'] = latest_data_for_ta.get('SMA_50', np.nan)
                grok_data_payload['LATEST_SMA_200'] = latest_data_for_ta.get('SMA_200', np.nan)
                grok_data_payload['LATEST_BB_UPPER'] = latest_data_for_ta.get('BBU_20_2.0', np.nan)
                grok_data_payload['LATEST_BB_MIDDLE'] = latest_data_for_ta.get('BBM_20_2.0', np.nan)
                grok_data_payload['LATEST_BB_LOWER'] = latest_data_for_ta.get('BBL_20_2.0', np.nan)
                grok_data_payload['LATEST_VOLUME_AVG_20'] = latest_data_for_ta.get('VOL_SMA_20', np.nan)
                
                # Conditional TA strings
                latest_close_for_cond = latest_data_for_ta.get('close', np.nan)
                sma_20_for_cond = latest_data_for_ta.get('SMA_20', np.nan)
                bb_upper_for_cond = latest_data_for_ta.get('BBU_20_2.0', np.nan)
                bb_lower_for_cond = latest_data_for_ta.get('BBL_20_2.0', np.nan)
                bb_middle_for_cond = latest_data_for_ta.get('BBM_20_2.0', np.nan)
                latest_volume_val_for_cond = latest_data_for_ta.get('volume', np.nan)
                volume_sma_20_for_cond = latest_data_for_ta.get('VOL_SMA_20', np.nan)

                if pd.notna(latest_close_for_cond) and pd.notna(sma_20_for_cond):
                    if latest_close_for_cond > sma_20_for_cond * 1.01: grok_data_payload['LATEST_PRICE_VS_SMA20'] = "above"
                    elif latest_close_for_cond < sma_20_for_cond * 0.99: grok_data_payload['LATEST_PRICE_VS_SMA20'] = "below"
                    else: grok_data_payload['LATEST_PRICE_VS_SMA20'] = "testing"
                else: grok_data_payload['LATEST_PRICE_VS_SMA20'] = "N/A"

                if pd.notna(latest_close_for_cond) and pd.notna(bb_upper_for_cond) and pd.notna(bb_lower_for_cond) and pd.notna(bb_middle_for_cond):
                    band_range = bb_upper_for_cond - bb_lower_for_cond
                    if pd.notna(band_range) and band_range > 1e-5: 
                        if latest_close_for_cond > bb_upper_for_cond * 0.995: grok_data_payload['LATEST_PRICE_VS_BB'] = "near_upper"
                        elif latest_close_for_cond < bb_lower_for_cond * 1.005: grok_data_payload['LATEST_PRICE_VS_BB'] = "near_lower"
                        elif abs(latest_close_for_cond - bb_middle_for_cond) < (band_range * 0.15): grok_data_payload['LATEST_PRICE_VS_BB'] = "mid_range"
                        elif latest_close_for_cond > bb_middle_for_cond: grok_data_payload['LATEST_PRICE_VS_BB'] = "upper_half"
                        else: grok_data_payload['LATEST_PRICE_VS_BB'] = "lower_half"
                    elif pd.notna(band_range) and band_range <= 1e-5 : grok_data_payload['LATEST_PRICE_VS_BB'] = "flat_bands"
                    else: grok_data_payload['LATEST_PRICE_VS_BB'] = "N/A"
                else: grok_data_payload['LATEST_PRICE_VS_BB'] = "N/A"

                if pd.notna(latest_volume_val_for_cond) and pd.notna(volume_sma_20_for_cond) and volume_sma_20_for_cond > 0:
                    if latest_volume_val_for_cond > volume_sma_20_for_cond * 1.5: grok_data_payload['LATEST_VOLUME_VS_AVG'] = "high"
                    elif latest_volume_val_for_cond < volume_sma_20_for_cond * 0.7: grok_data_payload['LATEST_VOLUME_VS_AVG'] = "low"
                    else: grok_data_payload['LATEST_VOLUME_VS_AVG'] = "average"
                else: grok_data_payload['LATEST_VOLUME_VS_AVG'] = "N/A"
            else: # asset_data_df is None or empty
                print(f"Asset data for {symbol_input} is None or empty. TA metrics and related grok fields will be N/A.")
                for key in ['LATEST_VOLUME', 'LATEST_SMA_20', 'LATEST_SMA_50', 'LATEST_SMA_200',
                            'LATEST_BB_UPPER', 'LATEST_BB_MIDDLE', 'LATEST_BB_LOWER', 'LATEST_VOLUME_AVG_20',
                            'LATEST_PRICE_VS_SMA20', 'LATEST_PRICE_VS_BB', 'LATEST_VOLUME_VS_AVG']:
                    grok_data_payload[key] = "N/A" if key.endswith(('_VS_AVG', '_VS_BB', '_VS_SMA20')) else np.nan
            
            grok_data_payload['TA_METRICS_DETAILS'] = ta_metrics_details_list # This is now the list of dicts
            grok_data_payload['TA_AVERAGE_SCORE'] = f"{average_ta_score_val:.2f}" # Store as string
            grok_data_payload['TA_AVERAGE_SUMMARY_EMOJI'] = ta_overall_summary_emoji
            
            # --- Construct Formatted TA Summary Block ---
            formatted_ta_lines = [f"📊 TA Summary for {symbol_input.upper()}:"]
            # Basic info moved outside the TA block in the final summary_parts
            formatted_ta_lines.append("--- Scored TA Metrics ---")

            if ta_metrics_details_list: # This list now comes from the 'details' key
                for metric_detail in ta_metrics_details_list:
                    if isinstance(metric_detail, dict) and 'formatted_string' in metric_detail:
                        formatted_ta_lines.append(metric_detail['formatted_string'])
                    # No need for string fallback as new structure is dict
            else:
                formatted_ta_lines.append("No detailed TA metrics available.")
            
            formatted_ta_lines.append(f"--- Avg TA Score: {average_ta_score_val:.2f} {ta_overall_summary_emoji} ---")
            
            # The overall report string for Telegram / Grok prompt
            telegram_report_string = "\n".join(formatted_ta_lines)
            grok_data_payload['FORMATTED_TA_SUMMARY_BLOCK'] = telegram_report_string
            
            # --- Grok Prompt Formatting ---
            def _format_posts_for_grok_prompt(posts_details_list):
                if not posts_details_list: return "  - No recent posts data available."
                formatted_items = []
                for post_idx, post in enumerate(posts_details_list):
                    if post_idx >= 2: break 
                    title = post.get('title', 'N/A')
                    sentiment_score_val = post.get('sentiment', np.nan) # Use 'sentiment' key
                    sentiment_score_str = f"{sentiment_score_val:.2f}" if pd.notna(sentiment_score_val) else "N/A"
                    formatted_items.append(f"    - Title: {title} (Sentiment: {sentiment_score_str})")
                return "\n".join(formatted_items) if formatted_items else "  - No recent posts data available."

            symbol_for_prompt = grok_data_payload.get('SYMBOL', 'N/A')
            price_for_prompt = grok_data_payload.get('LATEST_CLOSE_PRICE', "N/A")
            avg_sentiment_for_prompt_val = grok_data_payload.get('AVG_RECENT_SENTIMENT_SCORE', np.nan)
            avg_sentiment_for_prompt = f"{avg_sentiment_for_prompt_val:.2f}" if pd.notna(avg_sentiment_for_prompt_val) and isinstance(avg_sentiment_for_prompt_val, (float, int)) else "N/A"
            num_recent_posts_for_prompt = grok_data_payload.get('NUM_RECENT_POSTS', 0)
            sample_posts_for_prompt = _format_posts_for_grok_prompt(grok_data_payload.get('SAMPLE_POSTS_DETAILS', []))
            ta_summary_block_for_prompt = grok_data_payload.get('FORMATTED_TA_SUMMARY_BLOCK', 'Technical Analysis Summary not available.')
            
            grok_main_prompt = f"""You are Grok, a sharp and savvy crypto degen analyst. Your goal is to synthesize market data, sentiment, and technical analysis into a concise, actionable, and entertaining degen-style insight.

**Analysis Request for Ticker: {symbol_for_prompt}**

**Provided Context:**
*   **Current Price:** {price_for_prompt}
*   **YTD Change:** {grok_data_payload.get('YTD_CHANGE_PERCENT', 'N/A')}
*   **Market Cap:** {grok_data_payload.get('MARKET_CAP', 'N/A')}
*   **Overall Recent Sentiment:** {initial_simple_prediction_for_grok_prompt} (Derived from {num_recent_posts_for_prompt} posts, Average Score: {avg_sentiment_for_prompt})
*   **Sample Recent Reddit Posts:**
{sample_posts_for_prompt}
*   **Pre-calculated Technical Analysis Summary Block (MUST BE OUTPUTTED VERBATIM):**
```text
{ta_summary_block_for_prompt}
```

**Your Mission:**
1.  **Output Verbatim TA Summary:** You MUST begin your response by outputting the *exact* "Pre-calculated Technical Analysis Summary Block" provided above (the content within the ```text ... ``` markers). Do not alter, summarize, or rephrase it in any way.
2.  **Deliver Grok's Take:** Immediately following the verbatim TA summary, provide your "Grok's take:". This section is your unique degen analysis. It should be insightful, consider all the provided context (price, YTD, Market Cap, sentiment, posts, and the TA summary you just outputted), and conclude with a prediction. Maintain your degen persona: be witty, bold, and use appropriate slang.

**Required Output Structure:**
[The verbatim content of "Pre-calculated Technical Analysis Summary Block" from the context above]

Grok's take:
[Your degen analysis, insights, and prediction here. Make it spicy!]
"""
            grok_data_payload['PREDICTION_TEXT'] = grok_main_prompt

            # --- Final Summary Construction for Telegram (using the generated TA block) ---
            summary_parts = [
                f"*{symbol_input.upper()} Analysis Report*",
                f"Current Price: {current_price_str} {currency_symbol_emoji}",
                f"Market Cap: {market_cap_str}",
                f"YTD Change: {ytd_change_str} {ytd_emoji}",
                f"Reddit Activity: {num_recent_posts} recent posts analyzed.",
                # prediction_text already includes avg_recent_sentiment if available
                f"Sentiment Prediction: {prediction_text}", 
                # The telegram_report_string (which is FORMATTED_TA_SUMMARY_BLOCK) now contains the new metrics.
                f"\n{grok_data_payload['FORMATTED_TA_SUMMARY_BLOCK']}\n" 
            ]
            final_report_string = "\n".join(summary_parts)

            # --- Plotting ---
            plot_filenames_dict = {} 
            fig = plt.figure(figsize=(18, 24), constrained_layout=True) 
            gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1, 1]) 

            ax_price_vol = fig.add_subplot(gs[0, :]) 
            ax_sentiment_dist = fig.add_subplot(gs[1, 0])
            ax_rsi = fig.add_subplot(gs[1, 1])
            ax_macd = fig.add_subplot(gs[2, 0])
            ax_stochastic = fig.add_subplot(gs[2, 1])
            ax_historical_sentiment = fig.add_subplot(gs[3, 0])
            ax_dom_log_returns = fig.add_subplot(gs[3, 1])

            plot_start_time = time.time()

            # Plotting calls need the DataFrame with TA indicators (asset_data_df)
            if asset_data_df is not None and not asset_data_df.empty:
                plot_price_and_volume_on_ax(ax_price_vol, asset_data_df, symbol_input)
                plot_filenames_dict['price_volume'] = "Price/Volume Chart"
                if 'RSI_14' in asset_data_df.columns: plot_rsi_on_ax(ax_rsi, asset_data_df, symbol_input); plot_filenames_dict['rsi'] = "RSI Plot"
                else: ax_rsi.text(0.5, 0.5, "RSI N/A", ha='center'); plot_filenames_dict['rsi'] = "RSI Plot: N/A"
                if 'MACD_12_26_9' in asset_data_df.columns: plot_macd_on_ax(ax_macd, asset_data_df, symbol_input); plot_filenames_dict['macd'] = "MACD Plot"
                else: ax_macd.text(0.5, 0.5, "MACD N/A", ha='center'); plot_filenames_dict['macd'] = "MACD Plot: N/A"
                if 'STOCHk_14_3_3' in asset_data_df.columns: plot_stochastic_on_ax(ax_stochastic, asset_data_df, symbol_input); plot_filenames_dict['stochastic'] = "Stochastic Plot"
                else: ax_stochastic.text(0.5, 0.5, "Stochastic N/A", ha='center'); plot_filenames_dict['stochastic'] = "Stochastic Plot: N/A"
            else: 
                for ax, name in [(ax_price_vol, 'price_volume'), (ax_rsi, 'rsi'), (ax_macd, 'macd'), (ax_stochastic, 'stochastic')]:
                    ax.text(0.5, 0.5, f"{name.replace('_',' ').title()} N/A", ha='center'); plot_filenames_dict[name] = f"{name.replace('_',' ').title()}: N/A"
            
            # Plot sentiment data if available
            if not recent_posts_df.empty and 'sentiment' in recent_posts_df.columns:
                plot_sentiment_dist_on_ax(ax_sentiment_dist, recent_posts_df, symbol_input); plot_filenames_dict['sentiment_dist'] = "Recent Sentiment Distribution"
            else: ax_sentiment_dist.text(0.5, 0.5, "Recent Sentiment N/A", ha='center'); plot_filenames_dict['sentiment_dist'] = "Recent Sentiment Distribution: N/A"
            
            if not historical_sentiment_df.empty:
                plot_historical_sentiment_on_ax(ax_historical_sentiment, historical_sentiment_df, symbol_input, target_history_days=sentiment_history_days); plot_filenames_dict['historical_sentiment'] = "Historical Sentiment Trend"
            else: ax_historical_sentiment.text(0.5, 0.5, "Hist. Sentiment N/A", ha='center'); plot_filenames_dict['historical_sentiment'] = "Historical Sentiment Trend: N/A"

            # Plot DOM data if available (using dom_analysis_results directly now)
            if dom_analysis_results and 'mean_log_returns_by_dom' in dom_analysis_results and not dom_analysis_results['mean_log_returns_by_dom'].empty:
                plot_dom_log_returns_on_ax(ax_dom_log_returns, dom_analysis_results['mean_log_returns_by_dom'], symbol_input); plot_filenames_dict['dom_log_returns'] = "DOM Mean Log Returns"
            else: ax_dom_log_returns.text(0.5, 0.5, "DOM Log Returns N/A", ha='center'); plot_filenames_dict['dom_log_returns'] = "DOM Log Returns: N/A"

            fig.suptitle(f"Comprehensive Analysis for {symbol_input.upper()}", fontsize=20, fontweight='bold')
            image_filename = f"analysis_{symbol_input.lower().replace('/', '_')}_{unique_id}.png"
            image_path = os.path.join(temp_img_dir, image_filename)
            
            plt.savefig(image_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            plotting_time = time.time() - plot_start_time
            print(f"Plotting for {symbol_input} took {plotting_time:.2f} seconds. Saved to {image_path}")
            
            grok_data_payload['IMAGE_PATH'] = image_path
            grok_data_payload['PLOT_FILENAMES'] = plot_filenames_dict # Store dict of plot statuses

        except Exception as e:
            print(f"An unexpected error occurred generating report for {symbol_input}: {str(e)}")
            traceback.print_exc()
            image_path = None 
            final_report_string = f"Error generating report for {symbol_input}: {e}"
            # Ensure grok_data_payload still has essential keys even on error
            grok_data_payload.setdefault('PREDICTION_TEXT', f"Error: Could not generate analysis for {symbol_input}")
            grok_data_payload.setdefault('IMAGE_PATH', None)

        finally:
            # Ensure reddit session is closed if opened outside context manager (though it shouldn't be needed with 'async with')
            # await analyzer.reddit.close() # Not strictly necessary with async with
            pass

    end_time = time.time()
    total_time = end_time - start_time
    print(f"--- Analysis for {symbol_input} completed in {total_time:.2f} seconds ---")
    
    # Return the grok payload and image path
    return grok_data_payload, image_path


# --- CLI Execution ---
async def main_cli():
    """Handles command-line execution."""
    symbols_input = input("Enter stock symbol(s) separated by comma (e.g., TSLA,AAPL,BTC): ")
    symbols = [s.strip().upper() for s in symbols_input.split(',')]
    
    if not symbols:
        print("No symbols entered. Exiting.")
        return

    # Define parameters (could be made CLI args later)
    sentiment_days = 90
    stock_days = 90 # Note: stock_data_days is not directly used by get_stock_data anymore
    hist_post_limit = 500 # Reduced default limit for CLI testing

    print(f"\nConfiguration: Sentiment History={sentiment_days}d, Post Fetch Limit={hist_post_limit}")

    tasks = [generate_analysis_report(symbol, sentiment_days, stock_days, hist_post_limit) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    for i, symbol in enumerate(symbols):
        grok_payload, img_path = results[i] # Unpack result
        print("\n" + "="*40)
        print(f"Report for {symbol}:")
        # For CLI, we might want a simpler output than the full grok payload.
        # Let's construct a summary similar to what was in final_report_string
        # or print the FORMATTED_TA_SUMMARY_BLOCK and some key fields.
        
        cli_summary_parts = [
            f"*{symbol.upper()} Analysis Report*",
            f"Current Price: {grok_payload.get('LATEST_CLOSE_PRICE', 'N/A')}",
            f"Market Cap: {grok_payload.get('MARKET_CAP', 'N/A')}",
            f"YTD Change: {grok_payload.get('YTD_CHANGE_PERCENT', 'N/A')}",
            f"Reddit Activity: {grok_payload.get('NUM_RECENT_POSTS', 0)} recent posts analyzed.",
            f"Sentiment Prediction (Avg Recent): {grok_payload.get('AVG_RECENT_SENTIMENT_SCORE', 'N/A')}",
            f"\n{grok_payload.get('FORMATTED_TA_SUMMARY_BLOCK', 'TA Summary N/A')}\n"
        ]
        print("\n".join(cli_summary_parts))

        if img_path:
            print(f"Image saved to: {img_path}")
        else:
            print("Image generation failed or was skipped.")
        print("="*40)
        # print(f"Grok Payload for {symbol}: {grok_payload}") # Optionally print full payload for debugging

if __name__ == "__main__":
    asyncio.run(main_cli())
