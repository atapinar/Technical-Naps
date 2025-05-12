import os
import time
import asyncio
import httpx # Added for CoinAPI
import pandas as pd # Ensure pandas is imported for DataFrame creation
from polygon import RESTClient
from polygon.exceptions import BadResponse
from datetime import datetime, date, timedelta, timezone # Added timezone

# --- Configuration for Asset Type Inference (Copied from stock_sentiment.py for self-containment) ---
COMMON_CRYPTO_SYMBOLS = {"BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "DOGE", "SOL", "SHIB"}
# ---

class StockDataFetcher:
    def __init__(self, max_retries=3, retry_delay=5):
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set.")
        self.client = RESTClient(self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # CoinAPI Setup
        self.coinapi_key = os.getenv('COIN_API_KEY')
        self.coinapi_base_url = "https://rest.coinapi.io"
        if not self.coinapi_key:
            print("Warning: COIN_API_KEY environment variable not set. Crypto data fetching will fail.")
        # Async HTTP client for CoinAPI
        self.http_client = httpx.AsyncClient()

    def _make_request(self, request_func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    time.sleep(self.retry_delay * (2 ** (attempt -1))) # Exponential backoff
                return request_func(*args, **kwargs)
            except BadResponse as e:
                # Handles 4xx/5xx errors from Polygon
                print(f"Polygon API request failed (attempt {attempt + 1}/{self.max_retries}): {e.status} - {e.message}")
                if e.status == 429: # Rate limit
                    print(f"Rate limit hit, sleeping for {self.retry_delay * (2 ** attempt)}s")
                    time.sleep(self.retry_delay * (2 ** attempt)) # Longer sleep for rate limits
                    if attempt == self.max_retries -1:
                         print("Rate limit still active after max retries.")
                         return None
                    continue
                elif e.status == 401: # Unauthorized
                    print("Polygon API key is invalid or unauthorized.")
                    return None # Stop retrying for auth errors
                elif e.status == 404: # Not Found (e.g. invalid ticker for some endpoints)
                    print(f"Resource not found for {args} with {request_func.__name__}.")
                    return None # Stop retrying for 404
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e: # Catch other potential exceptions (network issues, etc.)
                print(f"An unexpected error occurred (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
        return None

    # --- Crypto Detection Helper ---
    def _is_crypto(self, symbol):
        """Checks if a symbol is likely a cryptocurrency based on a predefined list."""
        return symbol.upper() in COMMON_CRYPTO_SYMBOLS

    # --- CoinAPI Helper Methods ---
    def _get_coinapi_symbol_id(self, symbol):
        """Maps common crypto symbols to a default CoinAPI symbol ID (e.g., from Bitstamp)."""
        # Basic mapping, can be expanded
        mapping = {
            "BTC": "BINANCE_SPOT_BTC_USDT", # Changed to Binance USDT pair
            "ETH": "BINANCE_SPOT_ETH_USDT", # Example for ETH USDT pair
            # Add other mappings as needed
        }
        default_symbol_id = mapping.get(symbol.upper())
        if not default_symbol_id:
            print(f"Warning: No specific CoinAPI symbol ID mapping found for {symbol}. Using generic construction (may fail).")
            # Attempt a generic construction (less reliable)
            return f"COINBASE_SPOT_{symbol.upper()}_USD" # Or another major exchange
        return default_symbol_id

    async def _make_coinapi_request(self, endpoint):
        """Makes an authenticated GET request to CoinAPI."""
        if not self.coinapi_key:
            print("Error: CoinAPI key is not configured.")
            return None

        url = f"{self.coinapi_base_url}{endpoint}"
        headers = {'X-CoinAPI-Key': self.coinapi_key}
        retries = 3
        delay = 5

        for attempt in range(retries):
            try:
                response = await self.http_client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except httpx.TimeoutException:
                print(f"CoinAPI request timed out (attempt {attempt + 1}/{retries}): {url}")
            except httpx.HTTPStatusError as e:
                print(f"CoinAPI HTTP error (attempt {attempt + 1}/{retries}): {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 429: # Rate limit
                     print(f"CoinAPI rate limit hit. Waiting {delay * (2**attempt)}s...")
                     await asyncio.sleep(delay * (2**attempt))
                elif e.response.status_code in [400, 401, 403]: # Bad request, auth error - don't retry
                    return None
                # Other errors might be retryable
            except httpx.RequestError as e:
                print(f"CoinAPI request error (attempt {attempt + 1}/{retries}): {e}")
            except Exception as e:
                 print(f"Unexpected error during CoinAPI request (attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                 await asyncio.sleep(delay * (2**attempt)) # Exponential backoff
            else:
                print(f"Failed to fetch data from CoinAPI after {retries} attempts: {url}")
                return None
        return None

    async def _fetch_coinapi_current_rate(self, symbol):
        """Fetches the current exchange rate for a crypto symbol vs USD."""
        endpoint = f"/v1/exchangerate/{symbol.upper()}/USDT" # Changed target quote to USDT
        data = await self._make_coinapi_request(endpoint)
        if data and 'rate' in data:
            return data['rate']
        return None

    async def _fetch_coinapi_historical_ohlcv(self, symbol, start_date_str, end_date_str, period_id="1DAY"):
        """Fetches historical OHLCV data from CoinAPI and returns a pandas DataFrame."""
        coinapi_symbol_id = self._get_coinapi_symbol_id(symbol)
        if not coinapi_symbol_id:
            return pd.DataFrame() # Return empty if no valid ID

        # CoinAPI uses ISO 8601 format
        start_iso = datetime.fromisoformat(start_date_str).isoformat()
        end_iso = datetime.fromisoformat(end_date_str).isoformat()

        # Adjust limit based on period and date range if necessary, default is 100
        # For longer ranges, might need multiple requests or adjust limit (if API allows)
        limit = 10000 # Example: Increase limit if possible/needed

        endpoint = (f"/v1/ohlcv/{coinapi_symbol_id}/history"
                    f"?period_id={period_id}"
                    f"&time_start={start_iso}"
                    f"&time_end={end_iso}"
                    f"&limit={limit}")

        data = await self._make_coinapi_request(endpoint)

        if data and isinstance(data, list) and len(data) > 0:
            try:
                df = pd.DataFrame(data)
                # Rename columns to match the expected format
                df.rename(columns={
                    'time_period_start': 'Date', # Or time_period_end / time_open / time_close
                    'price_open': 'Open',
                    'price_high': 'High',
                    'price_low': 'Low',
                    'price_close': 'Close',
                    'volume_traded': 'Volume'
                }, inplace=True)

                # Convert Date column to datetime objects and set as index
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Select only the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df[required_cols]

                # Add 'Adj Close' if needed (using 'Close' as fallback)
                if 'Adj Close' not in df.columns and 'Close' in df.columns:
                     df['Adj Close'] = df['Close']

                return df.sort_index() # Ensure data is sorted by date
            except Exception as e:
                 print(f"Error processing CoinAPI historical data into DataFrame for {symbol}: {e}")
                 return pd.DataFrame()
        else:
            print(f"No historical data returned from CoinAPI for {symbol} ({coinapi_symbol_id})")
            return pd.DataFrame()

    # --- Main Data Fetching Logic ---

    async def get_stock_data(self, symbol): # Changed name and made async
        current_price = None
        market_cap = None
        currency = 'USD' # Default currency
        ytd_change_percent = None
        historical_data_df = None # Will hold the DataFrame
        error_message = None
        success = False

        is_crypto_asset = self._is_crypto(symbol)

        if is_crypto_asset:
            # --- Fetch Crypto Data using CoinAPI ---
            print(f"Fetching crypto data for {symbol} using CoinAPI...")
            try:
                # Fetch current price
                current_price = await self._fetch_coinapi_current_rate(symbol)
                currency = 'USDT' # Set currency to USDT for crypto
                market_cap = 'N/A' # CoinAPI basic endpoints don't provide this easily
                ytd_change_percent = 'N/A' # Requires historical fetch, calculate below if needed

                # Fetch historical data (e.g., last 2 years for TA)
                today_dt = datetime.now(timezone.utc)
                two_years_ago_dt = today_dt - timedelta(days=365 * 2)
                historical_data_df = await self._fetch_coinapi_historical_ohlcv(
                    symbol,
                    two_years_ago_dt.strftime('%Y-%m-%d'),
                    today_dt.strftime('%Y-%m-%d')
                )

                if historical_data_df is None or historical_data_df.empty:
                     print(f"Warning: Could not fetch historical data from CoinAPI for {symbol}.")
                     # Attempt to get price is still made, success depends on price
                
                # Optional: Calculate YTD if needed from historical_data_df
                # (Add logic similar to Polygon's YTD calculation if required)

                success = current_price is not None # Success depends on getting a price

            except Exception as e:
                print(f"Error fetching crypto data for {symbol} from CoinAPI: {e}")
                error_message = f"Failed to fetch crypto data for {symbol} using CoinAPI."
                success = False # Ensure success is false on error

        else:
            # --- Fetch Stock Data using Polygon.io (Existing Logic) ---
            print(f"Fetching stock data for {symbol} using Polygon.io...")
            stock_symbol = symbol # Use original symbol for Polygon

        # Fetch Ticker Details for Market Cap and Currency
        try:
            ticker_details = await asyncio.to_thread(self._make_request, self.client.get_ticker_details, stock_symbol)
            if ticker_details and hasattr(ticker_details, 'market_cap') and hasattr(ticker_details, 'currency_name'):
                market_cap = ticker_details.market_cap
                currency = ticker_details.currency_name.upper() if ticker_details.currency_name else 'USD'
            else:
                print(f"Could not retrieve market cap or currency for {stock_symbol} from ticker details.")
        except Exception as e:
            print(f"Error fetching ticker details for {stock_symbol}: {str(e)}")

        # Fetch Previous Day's Close for Current Price
        try:
            prev_close_data = await asyncio.to_thread(self._make_request, self.client.get_previous_close_agg, stock_symbol)
            if prev_close_data and isinstance(prev_close_data, list) and len(prev_close_data) > 0:
                # If prev_close_data is a list of Agg-like objects
                if hasattr(prev_close_data[0], 'close'):
                    current_price = prev_close_data[0].close
                else:
                    print(f"First item in prev_close_data list for {stock_symbol} does not have 'close' attribute.")
            elif prev_close_data and hasattr(prev_close_data, 'results') and prev_close_data.results and len(prev_close_data.results) > 0:
                # Original logic, if prev_close_data is an Aggs object as documented
                if hasattr(prev_close_data.results[0], 'close'):
                    current_price = prev_close_data.results[0].close
                else:
                    print(f"First item in prev_close_data.results for {stock_symbol} does not have 'close' attribute.")
            else:
                # Fallback: try to get the latest trade if previous close is not available
                print(f"Previous close not available for {stock_symbol}, trying latest trade.")
                latest_trade = await asyncio.to_thread(self._make_request, self.client.get_last_trade, stock_symbol)
                if latest_trade and hasattr(latest_trade, 'results') and latest_trade.results:
                     current_price = latest_trade.results.price
                else:
                    print(f"Could not retrieve current price for {stock_symbol} from previous close or last trade.")
        except Exception as e:
            print(f"Error fetching current price for {stock_symbol}: {str(e)}")


        # Calculate YTD change
        if current_price:
            try:
                today = date.today()
                start_of_year_dt = datetime(today.year, 1, 1)
                # Polygon API expects dates as YYYY-MM-DD strings
                start_of_year_str = start_of_year_dt.strftime('%Y-%m-%d')
                # Fetch data from start of year up to yesterday to ensure we get a close price
                # If today is Jan 1st, YTD is 0 or needs special handling
                if today.month == 1 and today.day == 1:
                    ytd_change_percent = 0.0
                else:
                    # Fetch data up to a recent date to get the start of year price
                    # We need the first trading day of the year.
                    # Fetch a small window from start of year to find the first close.
                    end_of_first_week_str = (start_of_year_dt + timedelta(days=7)).strftime('%Y-%m-%d')
                    year_start_aggs = await asyncio.to_thread(
                        self._make_request,
                        self.client.get_aggs,
                        ticker=stock_symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start_of_year_str,
                        to=end_of_first_week_str, # Check first few days
                        limit=5 # Limit to a few results
                    )

                    if year_start_aggs and len(year_start_aggs) > 0:
                        # Sort by timestamp to ensure we get the earliest
                        sorted_aggs = sorted(year_start_aggs, key=lambda x: x.timestamp)
                        start_year_price = sorted_aggs[0].close
                        if start_year_price and current_price: # current_price might be None
                            ytd_change_percent = ((current_price - start_year_price) / start_year_price) * 100
                        else:
                            print(f"Could not calculate YTD: Start year price or current price is missing for {stock_symbol}.")
                    else:
                        print(f"No historical data found for YTD calculation for {stock_symbol} near {start_of_year_str}")
            except Exception as ytd_e:
                print(f"Could not calculate YTD change for {stock_symbol}: {str(ytd_e)}")
        else:
            print(f"Skipping YTD calculation for {stock_symbol} as current price is unavailable.")

        # Fetch historical data for indicators (e.g., 200 days for SMA 200)
        # This part needs to be adapted based on how indicators are calculated in stock_sentiment.py
        # For now, let's fetch a reasonable amount of data (e.g., 1 year for daily)
        try:
            today_str = date.today().strftime('%Y-%m-%d')
            one_year_ago_str = (date.today() - timedelta(days=365 * 2)).strftime('%Y-%m-%d') # Fetch 2 years for 200 day SMA
            
            aggs = await asyncio.to_thread(
                self._make_request,
                self.client.get_aggs,
                ticker=stock_symbol,
                multiplier=1,
                timespan="day",
                from_=one_year_ago_str,
                to=today_str,
                limit=50000 # Max limit, adjust as needed
            )

            if aggs and len(aggs) > 0:
                # Convert to DataFrame if needed by stock_sentiment.py
                # Assuming stock_sentiment.py expects a DataFrame similar to yfinance
                import pandas as pd
                historical_data_df = pd.DataFrame([{
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                    'Date': datetime.fromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') # or .date()
                } for agg in aggs])
                historical_data_df['Date'] = pd.to_datetime(historical_data_df['Date'])
                historical_data_df = historical_data_df.set_index('Date')
                # Ensure columns match what yfinance provided if necessary, e.g. 'Adj Close'
                # Polygon doesn't provide 'Adj Close' directly in simple aggregates.
                # If 'Adj Close' is critical, further logic or a different endpoint might be needed.
                # For now, we'll assume 'Close' is sufficient or can be used as 'Adj Close'.
                if 'Adj Close' not in historical_data_df.columns and 'Close' in historical_data_df.columns:
                    historical_data_df['Adj Close'] = historical_data_df['Close']

            else:
                print(f"Could not fetch sufficient historical data for {stock_symbol}")
        except Exception as hist_e:
            print(f"Error fetching historical data for {stock_symbol} from Polygon: {str(hist_e)}")
            # End of Polygon-specific logic block


        # Determine success based on whether we got a current price at least
        success = current_price is not None

        return {
            'success': success,
            'data': {
                'current_price': current_price if current_price is not None else 'N/A',
                'currency': currency if currency else 'N/A',
                'market_cap': market_cap if market_cap is not None else 'N/A',
                'ytd_change_percent': ytd_change_percent if ytd_change_percent is not None else 'N/A',
                'historical_data': historical_data_df # This will be None if fetching failed
            },
            'error': error_message # Use the captured error message
        }

    async def get_historical_data(self, symbol, start_date_str, end_date_str, timespan="day", multiplier=1, limit=50000): # Made async
        """
        Fetches historical OHLCV data for a given stock symbol and date range.
        Returns a pandas DataFrame.
        """
        is_crypto_asset = self._is_crypto(symbol)

        if is_crypto_asset:
            print(f"Fetching historical crypto data for {symbol} using CoinAPI ({start_date_str} to {end_date_str})...")
            # CoinAPI uses period_id, map common timespans if needed
            period_id = "1DAY" # Default to daily
            if timespan == "minute": period_id = "1MIN"
            elif timespan == "hour": period_id = "1HRS"
            # Add other mappings as necessary

            try:
                # Ensure dates are in ISO format if needed by helper
                 start_iso = datetime.fromisoformat(start_date_str).isoformat()
                 end_iso = datetime.fromisoformat(end_date_str).isoformat()
            except ValueError:
                 print(f"Invalid date format for CoinAPI historical fetch: {start_date_str}, {end_date_str}. Using as is.")
                 start_iso = start_date_str # Use original if not ISO format
                 end_iso = end_date_str

            return await self._fetch_coinapi_historical_ohlcv(symbol, start_iso, end_iso, period_id)
        else:
            # --- Fetch Stock Historical Data using Polygon.io (Existing Logic) ---
            print(f"Fetching historical stock data for {symbol} using Polygon.io ({start_date_str} to {end_date_str})...")
            stock_symbol = symbol # Use original symbol for Polygon
        try:
            aggs = self._make_request(self.client.get_aggs,
                                     ticker=stock_symbol,
                                     multiplier=multiplier,
                                     timespan=timespan,
                                     from_=start_date_str,
                                     to=end_date_str,
                                     limit=limit)

            if aggs and len(aggs) > 0:
                import pandas as pd
                df = pd.DataFrame([{
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                    # Polygon timestamps are in milliseconds
                    'Date': datetime.fromtimestamp(agg.timestamp / 1000) #.strftime('%Y-%m-%d %H:%M:%S')
                } for agg in aggs])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                # Add 'Adj Close' if not present, assuming 'Close' is sufficient
                if 'Adj Close' not in df.columns and 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                return df
            else:
                print(f"No historical data found for {stock_symbol} between {start_date_str} and {end_date_str}")
                return pd.DataFrame() # Return empty DataFrame
        except Exception as e:
            print(f"Error fetching historical data for {stock_symbol} from Polygon: {str(e)}")
            import pandas as pd
            return pd.DataFrame() # Return empty DataFrame on error
            # End of Polygon-specific logic block

async def close(self):
        """Closes the underlying httpx.AsyncClient."""
        if hasattr(self, 'http_client') and isinstance(self.http_client, httpx.AsyncClient):
            await self.http_client.aclose()
            print("StockDataFetcher's httpx.AsyncClient closed.")
# Example usage (optional, for testing)
if __name__ == '__main__':
    # Example usage needs to be async now
    async def run_example():
        fetcher = StockDataFetcher()
        # Make sure POLYGON_API_KEY and COINAPI_KEY are set in your environment

        print("--- Testing Stock (AAPL) ---")
        stock_info = await fetcher.get_stock_data("AAPL") # await needed
        if stock_info['success']:
            print("AAPL Stock Info:")
            print(f"  Current Price: {stock_info['data']['current_price']} {stock_info['data']['currency']}")
            print(f"  Market Cap: {stock_info['data']['market_cap']}")
            print(f"  YTD Change: {stock_info['data']['ytd_change_percent']}%")
            if stock_info['data']['historical_data'] is not None:
                 print(f"  Historical Data (last 5 days for AAPL):")
                 print(stock_info['data']['historical_data'].tail())
            else:
                 print("  Historical Data: Not available")
        else:
            print(f"Error fetching AAPL data: {stock_info['error']}")

        print("\n" + "="*30 + "\n")

        print("--- Testing Crypto (BTC) ---")
        crypto_info = await fetcher.get_stock_data("BTC") # await needed
        if crypto_info['success']:
             print("BTC Crypto Info:")
             print(f"  Current Price: {crypto_info['data']['current_price']} {crypto_info['data']['currency']}")
             print(f"  Market Cap: {crypto_info['data']['market_cap']}") # Expected N/A
             print(f"  YTD Change: {crypto_info['data']['ytd_change_percent']}%") # Expected N/A
             if crypto_info['data']['historical_data'] is not None:
                  print(f"  Historical Data (last 5 days for BTC):")
                  print(crypto_info['data']['historical_data'].tail())
             else:
                  print("  Historical Data: Not available")
        else:
             print(f"Error fetching BTC data: {crypto_info['error']}")
        
        print("\n" + "="*30 + "\n")

        print("--- Testing Historical (ETH) ---")
        today = date.today()
        one_month_ago = today - timedelta(days=30)
        hist_df_eth = await fetcher.get_historical_data("ETH", one_month_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')) # await needed
        if not hist_df_eth.empty:
            print("ETH Historical Data (last month):")
            print(hist_df_eth.tail())
        else:
            print("Could not fetch ETH historical data.")
            
        # Close the http client when done
        await fetcher.http_client.aclose()

    asyncio.run(run_example())
    # Make sure POLYGON_API_KEY is set in your environment
    
    # Test get_stock_data
    # stock_info = fetcher.get_stock_data("AAPL")
    # if stock_info['success']:
    #     print("AAPL Stock Info:")
    #     print(f"  Current Price: {stock_info['data']['current_price']} {stock_info['data']['currency']}")
    #     print(f"  Market Cap: {stock_info['data']['market_cap']}")
    #     print(f"  YTD Change: {stock_info['data']['ytd_change_percent']}%")
    #     if stock_info['data']['historical_data'] is not None:
    #         print(f"  Historical Data (last 5 days for AAPL):")
    #         print(stock_info['data']['historical_data'].tail())
    #     else:
    #         print("  Historical Data: Not available")
    # else:
    #     print(f"Error fetching AAPL data: {stock_info['error']}")

    # print("\n" + "="*30 + "\n")

    # stock_info_msft = fetcher.get_stock_data("MSFT")
    # if stock_info_msft['success']:
    #     print("MSFT Stock Info:")
    #     print(f"  Current Price: {stock_info_msft['data']['current_price']} {stock_info_msft['data']['currency']}")
    #     print(f"  Market Cap: {stock_info_msft['data']['market_cap']}")
    #     print(f"  YTD Change: {stock_info_msft['data']['ytd_change_percent']}%")
    # else:
    #     print(f"Error fetching MSFT data: {stock_info_msft['error']}")

    # print("\n" + "="*30 + "\n")
    
    # Test get_historical_data
    # today = date.today()
    # one_month_ago = today - timedelta(days=30)
    # hist_df = fetcher.get_historical_data("GOOGL", one_month_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    # if not hist_df.empty:
    #     print("GOOGL Historical Data (last month):")
    #     print(hist_df.tail())
    # else:
    #     print("Could not fetch GOOGL historical data.")

    # Test with a potentially problematic ticker
    # stock_info_invalid = fetcher.get_stock_data("INVALIDTICKERXYZ")
    # if not stock_info_invalid['success']:
    #     print(f"Error for INVALIDTICKERXYZ as expected: {stock_info_invalid['error']}")

    pass # Keep the pass for when example usage is commented out
