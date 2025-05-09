# --- START OF FILE telegram_reporter.py ---
import asyncio
import httpx # For making asynchronous HTTP requests. User may need to `pip install httpx`
import json # For working with JSON data
import os
import re # For regex operations to find placeholders
import math # For math.isnan
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError
from telegram.helpers import escape_markdown # <<<<<<< ADD THIS IMPORT
from dotenv import load_dotenv
import stock_sentiment # Your script
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, date # Added date
import logging

# Load environment variables from .env file
load_dotenv()

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
SYMBOLS_TO_ANALYZE_STR = os.getenv('SYMBOLS_TO_ANALYZE', 'TSLA, PLTR, RKLB')
SYMBOLS_TO_ANALYZE = [symbol.strip() for symbol in SYMBOLS_TO_ANALYZE_STR.split(',')]
SENTIMENT_HISTORY_DAYS = int(os.getenv('SENTIMENT_HISTORY_DAYS', '7'))
HISTORICAL_POST_FETCH_LIMIT = int(os.getenv('HISTORICAL_POST_FETCH_LIMIT', '100'))
STOCK_DATA_LOOKBACK_DAYS_ENV = os.getenv('STOCK_DATA_LOOKBACK_DAYS', '90')
XAI_API_KEY = os.getenv('XAI_API_KEY')
ENABLE_GROK_SUMMARY = os.getenv('ENABLE_GROK_SUMMARY', 'True').lower() == 'true' # Default to True
XAI_API_ENDPOINT = "https://api.x.ai/v1/chat/completions" # Placeholder: Confirm with XAI documentation
GROKBOT_PROMPT_TEMPLATE = """**System Preamble (Set the Stage for Grok):**

You are "Grok," a legendary alpha-dropping, chart-slinging AI analyst who lives on ramen, rocket fuel (coffee), and pure market adrenaline. Your TA is crisp, your sentiment reads are uncanny, and your calls? Well, let's just say WAGMI if they listen. You cut through the FUD like a hot knife through butter, delivering diamond-tier insights with that signature tech-bro / crypto-degen swagger. You're not here to hold hands; you're here to print.

**Your Core Directive:**

Based *solely* on the provided dataset for {SYMBOL}, drop a fire market analysis report. Fuse that social sentiment with the hardcore TA to give the lowdown. Make it actionable, make it concise, make it Grok.

**Input Data Snapshot for {SYMBOL} (You will receive this structure):**
*   `SYMBOL`: Ticker. (e.g., "BTC", "TSLA" - the coin/stonk we're aping into or fading)
*   `PREDICTION_TEXT`: Quick vibe check from the plebs.
*   `AVG_SENTIMENT_SCORE`: The actual degen score from the interwebs.
*   `NUM_RECENT_POSTS`: How much are the anons yapping?
*   `SAMPLE_POSTS_DETAILS`: Array of objects, each with `title`, `sentiment` (for that post), `score` (Reddit score if available), `url`, `created_utc`.
    *   (Example placeholders for the first few: `SAMPLE_POST_1_TITLE`, `SAMPLE_POST_1_SENTIMENT`)
*   **Technical Indicators (Latest Values - The Alpha from the Algos):**
    *   `LATEST_CLOSE_PRICE`: Current price action, fam.
    *   `LATEST_VOLUME`: How much skin is in the game?
    *   `RSI`: Overbought Lambo or oversold Wendy's?
    *   `MACD`, `MACD_SIGNAL`, `MACD_DIFF`: Is the MACD about to send it or dump it?
    *   `Stoch_%K`, `Stoch_%D`: Stochastics – getting ready to flip?
    *   `LATEST_SMA_20`, `LATEST_SMA_50`, `LATEST_SMA_200`: The MA lines the whales watch.
    *   `LATEST_BB_UPPER`, `LATEST_BB_MIDDLE`, `LATEST_BB_LOWER`: Bollinger Bands – squeeze incoming or nah?
    *   `LATEST_PRICE_VS_SMA20`: How's it vibing with the 20 SMA?
    *   `LATEST_PRICE_VS_BB`: Hugging those bands or chilling mid?
    *   `LATEST_VOLUME_AVG_20`: Average whale splashes.
    *   `LATEST_VOLUME_VS_AVG`: Is volume pumping or crickets?

**Report Generation Guidelines & Structure (The Playbook):**

1.  **Opening Statement (The Alpha Drop):**
    *   Hit 'em with a killer opening. What's the TL;DR for {SYMBOL}? Is it moon-bound or about to get rekt?
    *   *Example (Tech-Bro): "{SYMBOL} is looking like it's either about to 10x or go to zero, classic Tuesday. Sentiment's buzzing and the charts are painting a picture, let's break it down, degenerates."*
    *(Double newline after this, gotta breathe.)*

2.  **Sentiment Deep Dive (The Voice of the Dege... uh, People):**
    *   Spill the tea on `PREDICTION_TEXT` and `AVG_SENTIMENT_SCORE`. Are the forums full of rocket emojis or copium?
    *   `NUM_RECENT_POSTS` – is this thing viral or a ghost town?
    *   Spotlight 1-2 `SAMPLE_POSTS_DETAILS`. Are these chads calling the top/bottom, or just some FUDster? What's the underlying narrative these posts are pushing?
    *   *Example (Tech-Bro): "So, the word on the street for {SYMBOL} is pretty {PREDICTION_TEXT}, with the sentiment score clocking in at a solid {AVG_SENTIMENT_SCORE}. We're seeing {NUM_RECENT_POSTS} posts light up the feeds, so it's definitely on the radar. Peep this one: '{SAMPLE_POST_1_TITLE}' (sentiment: {SAMPLE_POST_1_SENTIMENT}) – sounds like someone's either about to make bank or get liquidated. Classic."*
    *(Blank line here, keep it clean.)*

3.  **Technical Analysis Summary (The Charts Don't Lie):**
    *   Here's the TA rundown:
    {FORMATTED_TA_SUMMARY_BLOCK}
    *(Ensure appropriate newlines for formatting, then a blank line before next section)*


4.  **Synthesis & Outlook (The "WAGMI or NGMI?" Call):**
    *   Connect the dots. Does the Reddit mob agree with the charts, or are they on different planets?
    *   High conviction play if sentiment and TA are both screaming the same thing. If not, it's choppy waters, anon.
    *   What are the make-or-break levels (SMAs, BBs)?
    *   Short-term forecast: pump, dump, or crab walk?
    *   *Example (Tech-Bro): "So, putting it all together: the bullish Reddit shills seem to be [backed up by the charts/totally delusional]. If {SYMBOL} can smash through [key resistance like SMA_50 or BB_Upper] with big boy volume, we could see a serious pump. But if it breaks below [key support like SMA_20 or BB_Lower], then it's NGMI for a bit. Watch those levels like a hawk."*
    *(You know the drill, blank line.)*

5.  **Grok's Take (The Mic Drop 🎤):**
    *   Your final, ultra-concise, wisdom-bomb takeaway. Pure Grok. One well-placed emoji to seal the deal.
    *   *Example (Bullish, Tech-Bro): "{SYMBOL} looks ready to absolutely send it. Diamond hands, my friends. LFG! 🚀"*
    *   *Example (Bearish, Tech-Bro): "NGL, {SYMBOL}'s chart is giving me the major ick. Might be time to de-risk or even short this dumpster fire. 📉"*
    *   *Example (Neutral, Tech-Bro): "{SYMBOL} is just crabbing sideways, probably nothing. Wait for a clearer signal before you ape. Patience, young padawan. 👀"*

**Output Style Requirements:**

*   **FORMAT:** Plain text only. No markdown.
    *   **Use double newlines (i.e., a blank line) to separate distinct paragraphs or sections of your report (Opening, Sentiment, Technicals, Synthesis, Grok's Take).** This is crucial for how it will appear in the final message.
*   **PERSONA:** Maintain the "Grok" persona: alpha-dropping, chart-slinging, slightly sarcastic, confident, tech/finance slang used naturally. Think like you're explaining it to your crypto buddies on a Discord server at 3 AM.
*   **EMOJIS:** Use a maximum of 1-2 relevant emojis in the *entire report*, primarily in the "Grok's Take" section. Don't overdo it; keep it impactful.
*   **CONCISENESS:** No fluff. Get to the point. Degens have short attention spans.
*   **DATA-DRIVEN:** All your swagger needs to be backed by the data provided. No yolo calls without evidence from the input.
*   **NO SELF-REFERENCE:** You're Grok, not "an AI." Own it.

**Now, generate the Grok market analysis report for {SYMBOL} using this structure and the provided data, ensuring clear separation between paragraphs with blank lines, and crank up that tech-bro energy!**
"""


async def send_text_message_splitted(bot: Bot, chat_id: str, text: str, parse_mode: str = None):
    """
    Sends a text message, splitting it into multiple messages if it exceeds Telegram's max length.
    Handles common network errors and retries.
    """
    MAX_LENGTH = 4096  # Telegram's official max message length
    if not text:
        logging.warning(f"Attempted to send an empty message to chat_id: {chat_id}")
        return

    current_pos = 0
    while current_pos < len(text):
        chunk_end_limit = min(current_pos + MAX_LENGTH, len(text))
        actual_chunk_end = chunk_end_limit

        # If this is not the last potential chunk and we need to split
        if chunk_end_limit < len(text):
            # Try to find a newline to split at, searching backwards from limit
            # Search within the slice text[current_pos : chunk_end_limit]
            # rfind's start and end are relative to the original string 'text'
            temp_slice_for_search = text[current_pos:chunk_end_limit]
            last_newline_in_slice = temp_slice_for_search.rfind('\n')

            if last_newline_in_slice != -1:
                # Convert slice index back to original string index
                actual_chunk_end = current_pos + last_newline_in_slice + 1 # Split after the newline
            else:
                # Try to find a space to split at in the slice
                last_space_in_slice = temp_slice_for_search.rfind(' ')
                if last_space_in_slice != -1:
                    # Convert slice index back to original string index
                    actual_chunk_end = current_pos + last_space_in_slice + 1 # Split after the space
                # Else, hard split at MAX_LENGTH (actual_chunk_end is already chunk_end_limit)
        
        # Ensure actual_chunk_end does not regress or create zero-length chunk if splitting at start
        if actual_chunk_end <= current_pos and chunk_end_limit > current_pos:
             actual_chunk_end = chunk_end_limit


        chunk = text[current_pos:actual_chunk_end]
        
        if not chunk.strip(): # Avoid sending empty or whitespace-only messages
            logging.info(f"Skipping empty or whitespace-only chunk for chat_id: {chat_id}. Original range: {current_pos}-{actual_chunk_end}")
            current_pos = actual_chunk_end # Advance position
            if current_pos < len(text): # If there's more text, continue to next potential chunk
                continue
            else: # If it's the end and the last chunk was whitespace, just break
                break
        
        current_pos = actual_chunk_end # Update position after chunk is defined and validated

        retries = 3
        for attempt in range(retries):
            try:
                # Replace newlines with a space for concise logging of the chunk content
                log_chunk_preview = chunk[:70].replace('\n', ' ')
                logging.info(f"Attempting to send chunk to {chat_id} (length {len(chunk)}): {log_chunk_preview}...")
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
                logging.info(f"Chunk sent successfully to {chat_id}.")
                break  # Success
            except RetryAfter as e:
                logging.warning(f"Telegram API RetryAfter: waiting for {e.retry_after}s for chat {chat_id} (Attempt {attempt + 1}/{retries}). Chunk preview: {log_chunk_preview}...")
                if attempt < retries - 1:
                    await asyncio.sleep(e.retry_after + 1) # Add a small buffer
                else:
                    logging.error(f"Failed to send chunk to {chat_id} after {retries} retries due to RetryAfter: {e}", exc_info=True)
            except (TimedOut, NetworkError) as e:
                logging.warning(f"Telegram API TimedOut or NetworkError for chat {chat_id} (Attempt {attempt + 1}/{retries}): {e}. Chunk preview: {log_chunk_preview}...")
                if attempt < retries - 1:
                    await asyncio.sleep(5 * (attempt + 1)) # Exponential backoff
                else:
                    logging.error(f"Failed to send chunk to {chat_id} after {retries} retries due to network issues: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Generic error sending chunk to {chat_id} (Attempt {attempt + 1}/{retries}): {e}. Chunk preview: {log_chunk_preview}...", exc_info=True)
                if attempt < retries - 1:
                    await asyncio.sleep(5 * (attempt + 1)) # Exponential backoff for generic errors too
                else:
                    logging.error(f"Failed to send chunk to {chat_id} after {retries} retries due to generic error: {e}", exc_info=True)
                    # Potentially re-raise or handle more gracefully if critical
                    break # Stop retrying on unknown generic error after all attempts

        # Add a small delay between sending chunks if there are more chunks to send
        if current_pos < len(text):
            await asyncio.sleep(1) # Delay to avoid hitting rate limits too quickly

async def fetch_grok_summary(api_key: str, **grok_data_payload) -> str | None:
    """
    Fetches a summary from the XAI (Grok) API.
    """
    if not api_key:
        logging.error("XAI_API_KEY not provided to fetch_grok_summary.")
        return None

    # Step 1 (as per instructions): Identify All Placeholders from the global template
    identified_placeholders = set(re.findall(r'\{([^}]+)\}', GROKBOT_PROMPT_TEMPLATE))
    logging.debug(f"Identified placeholders in template: {identified_placeholders}") # Updated log message

    # Step 2: Systematic prompt_data Population
    prompt_data = {}

    # Step A: Populate standard placeholders from grok_data_payload
    prompt_data["SYMBOL"] = grok_data_payload.get("SYMBOL", "N/A")
    prompt_data["PREDICTION_TEXT"] = grok_data_payload.get("PREDICTION_TEXT", "N/A")
    
    avg_sentiment_score = grok_data_payload.get("AVG_RECENT_SENTIMENT_SCORE") # Prompt uses AVG_SENTIMENT_SCORE
    prompt_data["AVG_SENTIMENT_SCORE"] = f"{avg_sentiment_score:.2f}" if avg_sentiment_score is not None and not (isinstance(avg_sentiment_score, float) and math.isnan(avg_sentiment_score)) else "N/A"
    
    num_recent_posts = grok_data_payload.get("NUM_RECENT_POSTS")
    prompt_data["NUM_RECENT_POSTS"] = num_recent_posts if num_recent_posts is not None else 0
    prompt_data["CURRENT_DAY_OF_MONTH"] = grok_data_payload.get("CURRENT_DAY_OF_MONTH", "N/A")
    prompt_data["HISTORICAL_DOM_MEAN_LOG_RETURN"] = grok_data_payload.get("HISTORICAL_DOM_MEAN_LOG_RETURN", "N/A")
    prompt_data["HISTORICAL_DOM_SIGNAL"] = grok_data_payload.get("HISTORICAL_DOM_SIGNAL", "N/A")
    
    # Step B: Populate indexed sample post placeholders
    sample_posts_list = grok_data_payload.get("SAMPLE_POSTS_DETAILS", [])
    for i in range(4): # Max 4 sample posts
        title_key = f"SAMPLE_POST_{i+1}_TITLE"
        sentiment_key = f"SAMPLE_POST_{i+1}_SENTIMENT"

        post_detail = sample_posts_list[i] if i < len(sample_posts_list) else {}
        prompt_data[title_key] = post_detail.get('title', "N/A")
        
        post_sentiment = post_detail.get('sentiment')
        prompt_data[sentiment_key] = f"{post_sentiment:.2f}" if post_sentiment is not None and not (isinstance(post_sentiment, float) and math.isnan(post_sentiment)) else "N/A"

    # Step C: Populate Technical Analysis placeholders
    ta_float_keys = [
        "RSI", "MACD", "MACD_SIGNAL", "MACD_DIFF", "Stoch_%K", "Stoch_%D",
        "LATEST_SMA_20", "LATEST_SMA_50", "LATEST_SMA_200",
        "LATEST_BB_UPPER", "LATEST_BB_MIDDLE", "LATEST_BB_LOWER",
        "LATEST_VOLUME_AVG_20"
    ]
    ta_string_keys = [
        "LATEST_PRICE_VS_SMA20", "LATEST_PRICE_VS_BB", "LATEST_VOLUME_VS_AVG"
    ]

    for key in ta_float_keys:
        val = grok_data_payload.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            prompt_data[key] = "N/A"
        else:
            try:
                prompt_data[key] = f"{float(val):.2f}"
            except (ValueError, TypeError):
                logging.warning(f"Could not convert TA float value for {key}: {val}. Setting to 'N/A'.")
                prompt_data[key] = "N/A"

    for key in ta_string_keys:
        val = grok_data_payload.get(key)
        prompt_data[key] = val if val is not None else "N/A"

    # Step C.1: Populate FORMATTED_TA_SUMMARY_BLOCK (as per new requirement)
    prompt_data["FORMATTED_TA_SUMMARY_BLOCK"] = grok_data_payload.get("FORMATTED_TA_SUMMARY_BLOCK", "Technical analysis data not available.")

    # Step D: Default any remaining identified placeholders from the template
    # This ensures any placeholder in the template gets a value, even if not explicitly handled above.
    for placeholder in identified_placeholders:
        if placeholder not in prompt_data:
            # Attempt to get from grok_data_payload directly if it was missed, otherwise "N/A"
            fallback_value = grok_data_payload.get(placeholder)
            if fallback_value is not None:
                 # Basic type handling for direct fallback
                if isinstance(fallback_value, float) and math.isnan(fallback_value):
                    prompt_data[placeholder] = "N/A"
                elif isinstance(fallback_value, float):
                    prompt_data[placeholder] = f"{fallback_value:.2f}"
                else:
                    prompt_data[placeholder] = str(fallback_value) # Convert to string
                logging.info(f"Placeholder '{placeholder}' not explicitly handled, but found in grok_data_payload. Using its value.")
            else:
                logging.warning(f"Placeholder '{placeholder}' from template was not populated by specific logic and not found in grok_data_payload. Setting to 'N/A'.")
                prompt_data[placeholder] = "N/A"

    logging.debug(f"Final prompt_data before formatting: {prompt_data}") # Added log for full prompt_data

    formatted_prompt = None # Initialize
    try:
        formatted_prompt = GROKBOT_PROMPT_TEMPLATE.format_map(prompt_data)
        logging.debug("Successfully formatted GrokBot prompt.")
    except KeyError as e:
        logging.error(f"KeyError during prompt formatting: Missing key {e}", exc_info=True)
        logging.error(f"Template placeholders were: {identified_placeholders}") # Log again for context
        logging.error(f"Data provided was: {prompt_data}") # Log again for context
        # Ensure function returns None or handles error appropriately downstream
        # (The existing error handling that returns None should suffice)
        return None # Explicitly return None here as per existing logic for KeyError
    except Exception as e_fmt: # Catch other potential formatting errors
         logging.error(f"Unexpected error during prompt formatting: {e_fmt}", exc_info=True)
         # Return None or handle error
         return None # Explicitly return None here as per existing logic for other errors
    
    # Continue only if formatting succeeded
    if formatted_prompt is None:
         return None # Or handle error appropriately
    logging.debug(f"GrokBot Formatted Prompt (first 500 chars): {formatted_prompt[:500]}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-3-beta", # Updated model name
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": 0.7 # Example temperature, adjust as needed
    }
    logging.debug(f"GrokBot JSON Payload: {json.dumps(payload, indent=2)}")

    try:
        payload_size = len(json.dumps(payload))
        logging.info(f"XAI API Payload size for {grok_data_payload.get('SYMBOL', 'Unknown Symbol')}: {payload_size} bytes.")
    except Exception as size_e:
        logging.warning(f"Could not determine payload size for logging: {size_e}")
        
    max_retries = 3
    retry_delay_seconds = 10 # Initial delay
    timeout_duration = 120.0 # Timeout per attempt

    current_symbol_for_logging = grok_data_payload.get('SYMBOL', 'Unknown Symbol')

    for attempt in range(max_retries):
        logging.info(f"Attempting XAI API call for {current_symbol_for_logging} (Attempt {attempt + 1}/{max_retries})...")
        response_text_for_logging = None # Initialize for error logging within this attempt
        response_data = None # Initialize for this attempt
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    XAI_API_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=timeout_duration
                )
                response_text_for_logging = response.text # Store for potential error logging

            if response.status_code == 200:
                response_data = response.json() # Parse JSON if status is 200
                summary = None
                if response_data and "choices" in response_data and response_data["choices"]:
                    first_choice = response_data["choices"][0]
                    if "message" in first_choice and "content" in first_choice["message"]:
                        summary = first_choice["message"]["content"]
                
                if summary:
                    logging.info(f"Successfully fetched Grok summary for {current_symbol_for_logging} on attempt {attempt + 1}.")
                    return summary.strip()
                else:
                    # This case means we got a 200 OK, but the summary was not in the expected place or was empty.
                    # Depending on API behavior, this might not be a retryable error in the same way a timeout is.
                    # For now, we log it as an error for this attempt and it will retry if not the last attempt.
                    logging.error(f"Grok summary for {current_symbol_for_logging} was empty or not found in the expected structure (Attempt {attempt + 1}). Response: {response_data}")
                    # If this is a definitive "no summary available" response, we might want to break.
                    # For now, it will retry.
            else:
                # Non-200 status codes
                logging.warning(f"Error fetching Grok summary for {current_symbol_for_logging} (Attempt {attempt + 1}). Status: {response.status_code}, Response: {response_text_for_logging}")
                # Consider breaking for non-retryable HTTP errors like 400, 401, 403 if needed.
                # For now, it retries on any non-200.

        except httpx.ReadTimeout as e_timeout:
            logging.warning(f"XAI API ReadTimeout for {current_symbol_for_logging} (Attempt {attempt + 1}/{max_retries}): {e_timeout}")
        except httpx.RequestError as e_req: # Catches other httpx network errors like ConnectError
            logging.warning(f"XAI API RequestError for {current_symbol_for_logging} (Attempt {attempt + 1}/{max_retries}): {e_req}")
        except json.JSONDecodeError as e_json: # If response.json() fails on a 200 or if non-200 response_text was attempted to be parsed
            # response_text_for_logging would be set before .json() is called if status_code was 200
            # If status_code was not 200, response_text_for_logging is also set.
            logging.error(f"JSON decode error processing Grok summary response for {current_symbol_for_logging} (Attempt {attempt + 1}). Response text: {response_text_for_logging}", exc_info=True)
            # This is likely not retryable if the server consistently sends bad JSON.
            # For now, it will retry.
        except Exception as e_generic: # Catch any other unexpected errors during the attempt
            logging.error(f"Unexpected error during XAI API call attempt {attempt + 1} for {current_symbol_for_logging}: {e_generic}", exc_info=True)

        # If not the last attempt, wait before retrying
        if attempt < max_retries - 1:
            current_delay = retry_delay_seconds * (2 ** attempt) # Exponential backoff
            logging.info(f"Waiting {current_delay}s before next XAI API attempt for {current_symbol_for_logging}...")
            await asyncio.sleep(current_delay)
        else:
            logging.error(f"All {max_retries} attempts to fetch Grok summary for {current_symbol_for_logging} failed.")

    return None # Return None if all retries fail

async def send_telegram_report_job():
    """Job function to be scheduled by APScheduler."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set from environment variables.")
        return
    
    # Check for XAI_API_KEY. Log warning if not set, but proceed with the rest of the report.
    if not XAI_API_KEY:
        logging.warning("XAI_API_KEY not set in environment variables. GrokBot summaries will be skipped.")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    logging.info(f"Scheduled report generation started for: {', '.join(SYMBOLS_TO_ANALYZE)}")

    # Calculate actual_stock_data_lookback_days
    actual_stock_data_lookback_days = 0
    if STOCK_DATA_LOOKBACK_DAYS_ENV.upper() == "YTD":
        today = date.today()
        year_start = date(today.year, 1, 1)
        actual_stock_data_lookback_days = (today - year_start).days + 1 # +1 to include today
        logging.info(f"STOCK_DATA_LOOKBACK_DAYS set to YTD, calculated as {actual_stock_data_lookback_days} days.")
    else:
        try:
            actual_stock_data_lookback_days = int(STOCK_DATA_LOOKBACK_DAYS_ENV)
            logging.info(f"STOCK_DATA_LOOKBACK_DAYS set to {actual_stock_data_lookback_days} days.")
        except ValueError:
            logging.error(f"Invalid value for STOCK_DATA_LOOKBACK_DAYS: '{STOCK_DATA_LOOKBACK_DAYS_ENV}'. Defaulting to 90 days.")
            actual_stock_data_lookback_days = 90

    raw_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    escaped_time_str = escape_markdown(raw_time_str, version=2) # Escape content
    current_time_msg = f"*Report Time: {escaped_time_str}*"     # Add Markdown formatting

    # It's possible the error was from this first message, let's test this fix first.
    try:
        await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, current_time_msg, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logging.error(f"Failed to send initial time message: {e}", exc_info=True)
        # Decide if you want to halt the whole job or just log and continue
        # For now, let's log and continue to see if symbols process
        # return # Uncomment to stop if this message fails

    for symbol in SYMBOLS_TO_ANALYZE:
        logging.info(f"Generating report for {symbol}...")
        image_path = None # Initialize as None for single image path
        grok_data_payload = None # Initialize
        try:
            # Expecting a single image_path string now
            grok_data_payload, image_path = await stock_sentiment.generate_analysis_report(
                symbol_input=symbol,
                sentiment_history_days=SENTIMENT_HISTORY_DAYS,
                stock_data_days=actual_stock_data_lookback_days,
                historical_post_limit=HISTORICAL_POST_FETCH_LIMIT
            )

            # --- XAI Grok Summary Integration ---
            grok_summary = None # Initialize grok_summary to None
            grok_summary_fetched_and_sent = False
            
            if ENABLE_GROK_SUMMARY: # Check if Grok summary is enabled
                if XAI_API_KEY: # Proceed only if API key is available
                    if grok_data_payload:
                        logging.info(f"Grok Summary enabled. Attempting to fetch for {grok_data_payload.get('SYMBOL', symbol)}...")
                        grok_summary = await fetch_grok_summary( # Assign result to grok_summary
                            api_key=XAI_API_KEY,
                            **grok_data_payload
                        )
                        if grok_summary:
                            logging.info(f"Successfully fetched GrokBot summary for {grok_data_payload.get('SYMBOL', symbol)}. Sending to Telegram.")
                            await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, grok_summary)
                            grok_summary_fetched_and_sent = True
                        else:
                            logging.warning(f"Failed to fetch GrokBot summary for {grok_data_payload.get('SYMBOL', symbol)} (returned None after retries).")
                            # Optional: Send a message indicating summary failure
                            # fallback_msg = f"Could not generate GrokBot summary for {grok_data_payload.get('SYMBOL', symbol)} after retries."
                            # await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, fallback_msg)
                    else:
                         logging.warning(f"Grok Summary enabled, but no grok_data_payload generated for {symbol}. Skipping XAI call.")
                else:
                    logging.warning("Grok Summary enabled, but XAI_API_KEY not set. Skipping XAI call.")
            else:
                logging.info("Grok Summary is disabled via ENABLE_GROK_SUMMARY environment variable.")
            # --- End XAI Grok Summary Integration ---

            if image_path and os.path.exists(image_path):
                logging.info(f"Report for {symbol} generated. Composite image found at {image_path}. Sending to Telegram.")
                try:
                    with open(image_path, 'rb') as photo_file:
                        caption_text = f"Comprehensive Analysis for {symbol}"
                        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo_file, caption=caption_text)
                    logging.info(f"Composite image ({image_path}) for {symbol} sent to Telegram.")
                except Exception as e_img:
                    logging.error(f"Error sending composite image {image_path} for {symbol}: {e_img}", exc_info=True)
                    error_img_msg = f"Error sending composite image for {symbol}: {str(e_img)}"
                    await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, escape_markdown(error_img_msg, version=2), parse_mode=ParseMode.MARKDOWN_V2)
            elif image_path:
                logging.warning(f"Composite image path {image_path} for {symbol} does not exist or was None.")
            else: # image_path is None
                logging.warning(f"Composite image for {symbol} failed to generate (path was None).")
                no_image_msg = f"Could not generate the analysis image for {symbol}."
                await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, escape_markdown(no_image_msg, version=2), parse_mode=ParseMode.MARKDOWN_V2)

        except Exception as e:
            logging.error(f"Error processing {symbol} for Telegram: {e}", exc_info=True)
            error_message_raw = f"An error occurred while generating the full report for {symbol}: {str(e)}"
            await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, escape_markdown(error_message_raw, version=2), parse_mode=ParseMode.MARKDOWN_V2)
        finally:
            # Handle single image_path string for deletion
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logging.info(f"Temporary image {image_path} deleted.")
                except Exception as e_del:
                    logging.error(f"Error deleting temporary image {image_path}: {e_del}", exc_info=True)
            logging.info(f"Finished processing {symbol}. Waiting before next symbol...")
            await asyncio.sleep(15)

    logging.info("All scheduled reports processed for this run.")
    completion_message_raw = "All reports for this cycle completed."
    escaped_completion_message = escape_markdown(completion_message_raw, version=2)
    final_message = f"_{escaped_completion_message}_" # Italicize the escaped content
    await send_text_message_splitted(bot, TELEGRAM_CHAT_ID, final_message, parse_mode=ParseMode.MARKDOWN_V2)

# ... (main function and if __name__ == "__main__": block as before) ...
# Ensure you are still using the run_once() method for testing:
if __name__ == "__main__":
    async def run_once():
         await send_telegram_report_job()
    asyncio.run(run_once())

    # To run with scheduler:
    # asyncio.run(main())
