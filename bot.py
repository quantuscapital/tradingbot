import os
import csv
import requests
import yaml
import time
import logging
import asyncio
import json
import httpx
import signal
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.request import HTTPXRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
from message_formatting import (
    escape_markdown_v2_custom,
    format_ai_analysis_output,
    format_markdown_v2,
    prepare_telegram_message,
    format_compact_analysis,
    format_new_token_alert
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    logger.info("Successfully loaded config.yaml")
except Exception as e:
    logger.error(f"Error loading config.yaml: {str(e)}")
    raise

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables")
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment variables")

# Check required files exist
required_files = ['active_chats.json', 'shared_tokens_log.csv']
for file in required_files:
    if not Path(file).exists():
        logger.error(f"Required file {file} is missing")
        raise FileNotFoundError(f"Required file {file} is missing")

# Shared token log
SHARED_TOKEN_LOG = 'shared_tokens_log.csv'
LAST_FETCH_TIME_FILE = 'last_fetch_time.txt'

# Active chats file
ACTIVE_CHATS_FILE = 'active_chats.json'

# Rate limiting configuration
RATE_LIMIT_DELAY = 0.2  # 0.2 seconds between calls = 300 requests per minute
MAX_RETRIES = 3

async def rate_limited_api_call(url: str, params: dict = None):
    """Make API calls with rate limiting and retries"""
    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES):
            try:
                # Respect rate limit
                await asyncio.sleep(RATE_LIMIT_DELAY)

                # Make the request
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected API error: {str(e)}")
                raise

async def get_token_pairs_data(token_addresses: list):
    """Fetch token pairs data from DexScreener"""
    try:
        if not token_addresses:
            return {}

        # DexScreener API endpoint for token pairs (max 30 addresses per request)
        url = 'https://api.dexscreener.com/latest/dex/tokens/'

        # Process addresses in batches of 30
        pairs_dict = {}
        for i in range(0, len(token_addresses), 30):
            batch = token_addresses[i:i + 30]

            # Normalize addresses in batch
            normalized_batch = batch

            # Make API call with rate limiting
            response = await rate_limited_api_call(url + ','.join(normalized_batch))

            if response.status_code == 200:
                data = response.json()
                if 'pairs' in data and isinstance(data['pairs'], list):
                    for pair in data['pairs']:
                        if not isinstance(pair, dict):
                            continue

                        base_token = pair.get('baseToken', {})
                        if not isinstance(base_token, dict):
                            continue

                        base_address = base_token.get('address', '')
                        if base_address:
                            # Store the first pair for each token (usually the most liquid)
                            if base_address not in pairs_dict:
                                # Ensure required fields exist with proper types
                                if all(key in pair for key in ['chainId', 'dexId', 'pairAddress']):
                                    pairs_dict[base_address] = pair
                                    logger.debug(f"Stored pair data for {base_address}")
            else:
                logger.error(f"Failed to fetch token pairs batch: {response.status_code}")

        return pairs_dict

    except Exception as e:
        logger.error(f"Error fetching token pairs: {str(e)}")
        return {}

async def send_token_alert(bot, chat_id, token, message_thread_id=None):
    """Send alert about a viable token"""
    try:
        token_address = token['tokenAddress']
        pair = token['pair']

        # Prepare token data for new format
        token_data = {
            'name': pair['baseToken'].get('name', 'Unknown'),
            'ticker': pair['baseToken'].get('symbol', 'Unknown'),
            'address': token_address,
            'chain_id': pair.get('chainId', 'Unknown'),
            'dex_id': pair.get('dexId', 'Unknown'),
            'market_cap': pair.get('fdv', '0'),
            'liquidity': pair.get('liquidity', {}).get('usd', '0'),
            'volume': pair.get('volume', {}).get('h24', 0),
            'buys': pair.get('txns', {}).get('h24', {}).get('buys', 0),
            'sells': pair.get('txns', {}).get('h24', {}).get('sells', 0),
            'price_change': pair.get('priceChange', {}).get('h24', 0),
            'initial_price': pair.get('priceUsd', '0.0000'),
            'created_at': datetime.fromtimestamp(int(pair.get('pairCreatedAt', 0)/1000)).strftime('%Y-%m-%d %H:%M:%S') if pair.get('pairCreatedAt') else None,
            'info': pair.get('info', {}),
            'socials': [
                f"Website (https://www.eva-labs.tech/)",
                f"Website (https://medium.com/@evasolana00/eva-the-blade-of-dawn-phase-1-f18feec1634a)"
            ]
        }

        # Format the message using new format
        message = format_new_token_alert(token_data)
        message = message.replace('websites:', 'socials:')

        # Log the final message for debugging
        logger.info(f"Prepared message:\n{message}")

        # Log topic details
        logger.info(f"Sending message to chat_id: {chat_id}, message_thread_id: {message_thread_id}")

        # Send the message
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
            message_thread_id=message_thread_id
        )

        # Log token to prevent duplicates
        log_token(token_address)

    except Exception as e:
        logger.error(f"Error sending token alert: {str(e)}")
        if chat_id:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ An error occurred while sending the token alert. Please try again."
            )

# Track processed tokens and updates
processed_tokens = set()
SHARED_TOKEN_LOG = 'shared_tokens_log.csv'

def log_token(token_address):
    """Log token to prevent duplicates and track sharing"""
    if token_address in processed_tokens:
        logger.debug(f"Token {token_address} already logged, skipping.")
        return

    processed_tokens.add(token_address)

    # Check if token is already logged in shared_tokens_log.csv
    token_already_logged = False
    try:
        with open(SHARED_TOKEN_LOG, 'r', newline='') as csvfile: # Added newline='' for reading consistency
            reader = csv.reader(csvfile)
            for row in reader:
                # Check if row is not empty and has at least 2 columns
                if row and len(row) >= 2:
                    if row[1] == token_address:
                        logger.debug(f"Token {token_address} already logged in shared_tokens_log.csv, skipping write.")
                        token_already_logged = True
                        break
                else:
                    logger.warning(f"Skipping malformed row in {SHARED_TOKEN_LOG}: {row}")
    except FileNotFoundError:
        logger.warning(f"File {SHARED_TOKEN_LOG} not found. It will be created.")
    except Exception as e:
        logger.error(f"Error reading {SHARED_TOKEN_LOG}: {e}")
        # Decide if we should proceed to write or not. For now, let's proceed.

    # Log to CSV file only if not already found
    if not token_already_logged:
        try:
            with open(SHARED_TOKEN_LOG, 'a', newline='') as csvfile: # Added newline='' for writing consistency
                writer = csv.writer(csvfile)
                writer.writerow([datetime.now().isoformat(), token_address])
                logger.debug(f"Logged token {token_address} to {SHARED_TOKEN_LOG}")
        except Exception as e:
            logger.error(f"Error writing to {SHARED_TOKEN_LOG}: {e}")

def initialize_processed_tokens():
    """Initialize processed tokens with time-based filtering"""
    logger.debug("Initializing processed tokens")
    processed_tokens.clear()  # Clear existing set
    cutoff_time = datetime.now() - timedelta(hours=24)  # Only load tokens from last 24 hours
    file_corrupt = False
    try:
        with open(SHARED_TOKEN_LOG, 'r', newline='') as f: # Added newline=''
            reader = csv.reader(f)
            header = next(reader, None) # Skip header if exists (optional, depends on file structure)
            # If you expect a header and want to validate it, add checks here.
            # For now, we just skip the first line if it exists.

            for i, row in enumerate(reader):
                if len(row) >= 2:  # Ensure row has timestamp and address
                    try:
                        timestamp_str = row[0]
                        token_address = row[1]
                        
                        # Basic validation
                        if not timestamp_str or not token_address:
                            raise ValueError("Empty timestamp or address")
                            
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp > cutoff_time:  # Only add recent tokens
                            processed_tokens.add(token_address)
                            
                    except (ValueError, IndexError, TypeError) as e:
                        logger.error(f"Error processing row {i+1} in {SHARED_TOKEN_LOG}: {e}. Row: {row}")
                        # Mark file as potentially corrupt if errors occur during processing
                        file_corrupt = True
                        # Optionally break here if one error means the whole file is untrustworthy
                        # break 
                else:
                    logger.warning(f"Skipping malformed row {i+1} in {SHARED_TOKEN_LOG} (expected 2+ columns): {row}")
                    file_corrupt = True # Consider short rows as corruption indicator

    except FileNotFoundError:
        logger.warning(f"File {SHARED_TOKEN_LOG} not found. Starting with empty processed tokens.")
        # No need to set file_corrupt = True here, file just doesn't exist yet.
    except csv.Error as e:
        logger.error(f"CSV formatting error in {SHARED_TOKEN_LOG}: {e}")
        file_corrupt = True
    except Exception as e:
        logger.error(f"Unexpected error reading {SHARED_TOKEN_LOG}: {e}")
        file_corrupt = True

    # If file was corrupt, clear the set, log it, and remove the bad file
    if file_corrupt:
        logger.warning(f"{SHARED_TOKEN_LOG} appears corrupt or improperly formatted. Resetting token history.")
        processed_tokens.clear()
        try:
            os.remove(SHARED_TOKEN_LOG)
            logger.info(f"Removed corrupt file: {SHARED_TOKEN_LOG}")
        except OSError as e:
            logger.error(f"Error removing corrupt file {SHARED_TOKEN_LOG}: {e}")
            
    logger.info(f"Initialized with {len(processed_tokens)} processed tokens from the last 24h.") # Changed level to INFO

def load_active_chats():
    """Load active chats from file"""
    try:
        if Path(ACTIVE_CHATS_FILE).exists():
            with open(ACTIVE_CHATS_FILE, 'r') as f:
                data = json.load(f)
                # Ensure we always return a dictionary
                if isinstance(data, dict):
                    return data
                logger.warning(f"Invalid active_chats format, expected dict but got {type(data)}")
        return {}
    except json.JSONDecodeError:
        logger.warning("active_chats.json contains invalid JSON, resetting to empty dict")
        return {}

def save_active_chats(active_chats):
    """Save active chats to file"""
    try:
        with open(ACTIVE_CHATS_FILE, 'w') as f:
            json.dump(active_chats, f)
    except Exception as e:
        logger.error(f"Error saving active chats: {str(e)}")

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command for both private chats and groups"""
    try:
        chat = update.effective_chat
        chat_id = chat.id
        chat_type = chat.type

        # Load existing active chats
        active_chats = load_active_chats()

        # Initialize chat if not already active
        if str(chat_id) not in active_chats:
            active_chats[str(chat_id)] = {
                'type': chat_type,
                'active': True,
                'last_activity': datetime.now().isoformat()
            }
            save_active_chats(active_chats)
            logger.info(f"Added new chat: {chat_id} ({chat_type})")

        # Send welcome message
        if chat_type == 'private':
            message = (
                "🤖 *Welcome to the Crypto Trading Bot!*\n\n"
                "I'll help you track and analyze cryptocurrency tokens.\n\n"
                "Add me to your group to start receiving token alerts\\!"
            )
        else:
            message = (
                "🤖 *Crypto Trading Bot Activated!*\n\n"
                "I'll monitor and analyze cryptocurrency tokens for this group.\n\n"
                "You'll receive alerts when we find promising tokens\\!"
            )

        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    except Exception as e:
        logger.error(f"Error in handle_start: {str(e)}")
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An error occurred while processing the command. Please try again."
            )

async def get_latest_tokens():
    """Fetch latest tokens from DexScreener"""
    try:
        url = 'https://api.dexscreener.com/token-profiles/latest/v1'
        response = await rate_limited_api_call(url)

        if not response or response.status_code != 200:
            status = response.status_code if response else "No response"
            logger.error(f"Failed to fetch latest tokens: {status}")
            return []

        data = response.json()
        logger.debug(f"Raw API response: {data}")

        # Extract tokens list
        tokens = []
        if isinstance(data, list):
            tokens = data
        elif isinstance(data, dict) and 'tokens' in data:
            tokens = data['tokens']
        else:
            logger.error(f"Unexpected API response structure: {type(data)}")
            return []

        if not isinstance(tokens, list):
            logger.error(f"Invalid tokens type: {type(tokens)}")
            return []

        # Process and validate each token
        valid_tokens = []
        seen_addresses = set()
        for token in tokens:
            try:
                if not isinstance(token, dict):
                    logger.warning(f"Invalid token format (not a dict): {token}")
                    continue

                token_address = token.get('tokenAddress')
                if not token_address or not isinstance(token_address, str):
                    logger.warning(f"Invalid or missing token address: {token}")
                    continue

                # Normalize token address
                
                # More aggressive duplicate prevention
                if (token_address in processed_tokens or 
                    token_address in seen_addresses):
                    logger.debug(f"Skipping already processed token: {token_address}")
                    continue

                token['tokenAddress'] = token_address
                valid_tokens.append(token)
                seen_addresses.add(token_address)
                logger.debug(f"Added valid token: {token_address}")

            except Exception as e:
                logger.error(f"Error processing token: {str(e)}")
                continue

        logger.info(f"Found {len(valid_tokens)} valid tokens")
        return valid_tokens

    except Exception as e:
        logger.error(f"Error fetching latest tokens: {str(e)}")
        return []

async def filter_tokens_by_criteria(tokens):
    """Filter tokens based on configured criteria"""
    viable_tokens = []
    criteria = config['token_criteria']
    seen_viable_addresses = set()

    if not tokens:
        logger.warning("No tokens fetched from Dexscreener")
        return viable_tokens

    try:
        # Process tokens in batches
        for i in range(0, len(tokens), config['max_tokens_per_request']):
            batch = tokens[i:i + config['max_tokens_per_request']]

            # Validate batch structure
            if not isinstance(batch, list):
                logger.error(f"Invalid batch type: {type(batch)}")
                continue

            # Extract token addresses safely with additional validation
            token_addresses = []
            for token in batch:
                try:
                    if isinstance(token, dict) and 'tokenAddress' in token:
                        if isinstance(token['tokenAddress'], str):
                            token_addresses.append(token['tokenAddress'])
                        else:
                            logger.warning(f"Invalid tokenAddress type: {type(token['tokenAddress'])}")
                    else:
                        logger.warning(f"Invalid token format: {token}")
                except Exception as e:
                    logger.error(f"Error processing token: {str(e)}")
                    continue

            if not token_addresses:
                continue

            # Fetch data for the batch
            pairs_data = await get_token_pairs_data(token_addresses)

            for token in batch:
                if not isinstance(token, dict) or 'tokenAddress' not in token:
                    logger.debug(f"Skipping invalid token: {token}")
                    continue

                token_address = token['tokenAddress']  # Normalize address
                logger.debug(f"Processing token: {token_address}")

                # Add additional validation for token address
                if not isinstance(token_address, str):
                    logger.error(f"Invalid token address type: {type(token_address)}")
                    continue

                # Check only for immediate duplicates and recent alerts
                if token_address in seen_viable_addresses:
                    logger.info(f"Token {token_address} filtered out: Duplicate in current batch")
                    continue

                # Safely get pair data with additional validation
                pair = pairs_data.get(token_address)
                if not pair:
                    logger.debug(f"No pair data found for token: {token_address}")
                    continue

                if not isinstance(pair, dict):
                    logger.error(f"Invalid pair data type for token {token_address}: {type(pair)}")
                    continue

                if pair:
                    logger.debug(f"Pair data type: {type(pair)}")
                    if isinstance(pair, dict):
                        try:
                            # Apply filtering criteria
                            if meets_criteria(pair, criteria):
                                # Store both token address and pair data
                                viable_tokens.append({
                                    'tokenAddress': token_address,
                                    'pair': pair
                                })
                                seen_viable_addresses.add(token_address)
                                logger.info(f"Token {token_address} meets criteria")
                        except Exception as e:
                            logger.error(f"Error processing token {token_address}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in filter_tokens_by_criteria: {str(e)}")

    return viable_tokens

def meets_criteria(pair, criteria):
    """Check if token pair meets all criteria"""
    # First check if it's a Solana token
    chain_id = pair.get('chainId', '').lower()
    if chain_id != 'solana':
        return False

    liquidity = pair.get('liquidity', {}).get('usd', 0)
    volume = pair.get('volume', {}).get('h24', 0)
    buys = pair.get('txns', {}).get('h24', {}).get('buys', 0)
    sells = pair.get('txns', {}).get('h24', {}).get('sells', 0)
    price_change = pair.get('priceChange', {}).get('h24', 0)

    return (liquidity >= criteria['min_liquidity'] and
            volume >= criteria['min_volume'] and
            buys >= criteria['min_buy_transactions'] and
            sells >= criteria['min_sell_transactions'] and
            criteria['min_price_change'] <= price_change <= criteria['max_price_change'])

async def notify_viable_tokens(bot, chat_id, viable_tokens):
    """Notify about viable tokens"""
    last_sent_time = 0
    rate_limit = 15  # 15 seconds

    for token_data in viable_tokens:
        try:
            token_address = token_data['tokenAddress']
            pair = token_data['pair']
            logger.debug(f"Processing notification for token: {token_address}")

            # Check if rate limit has been exceeded
            current_time = time.time()
            if current_time - last_sent_time < rate_limit:
                await asyncio.sleep(rate_limit - (current_time - last_sent_time))

            # Log viable token details
            logger.info(f"Sending alert for viable token: {token_address}")
            logger.info(f"Token pair data: {pair}")
            
            # Send initial alert using the passed chat_id
            await send_token_alert(bot, chat_id, token_data, message_thread_id=824)

            # Update the last sent time
            last_sent_time = time.time()

        except Exception as e:
            logger.error(f"Error processing token notification: {str(e)}")
            continue

async def scan_loop(bot):
    """Continuous scanning loop for new tokens"""
    active_chats = load_active_chats()

    while True:
        try:
            # Get latest tokens from DexScreener
            tokens = await get_latest_tokens()

            # Filter tokens based on criteria
            viable_tokens = await filter_tokens_by_criteria(tokens)

            # Skip notification if no viable tokens
            if not viable_tokens:
                logger.debug("No viable tokens to notify")
                await asyncio.sleep(config['scan_interval'])
                continue

            # Load active chats
            active_chats = load_active_chats()
            if not active_chats:
                logger.debug("No active chats to notify")
                await asyncio.sleep(config['scan_interval'])
                continue

            # Notify all active chats about viable tokens
            for chat_id_str, chat_data in active_chats.items():
                try:
                    if isinstance(chat_data, dict) and chat_data.get('active', False):
                        await notify_viable_tokens(bot, int(chat_id_str), viable_tokens)
                except Exception as e:
                    logger.error(f"Error notifying chat {chat_id_str}: {str(e)}")
                    continue

            # Wait for next scan
            await asyncio.sleep(config['scan_interval'])

        except Exception as e:
            logger.error(f"Error in scan loop: {str(e)}")
            await asyncio.sleep(60)  # Wait before retrying after error

async def handle_get_topic_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /get_topic_id command to log topic details"""
    try:
        chat = update.effective_chat
        chat_id = chat.id
        message_thread_id = update.message.message_thread_id if update.message.message_thread_id else None

        # Log topic details
        logger.info(f"Received /get_topic_id command in chat_id: {chat_id}, message_thread_id: {message_thread_id}")

        # Send response with topic details
        response = f"Chat ID: {chat_id}\nTopic ID: {message_thread_id}"
        await context.bot.send_message(chat_id=chat_id, text=response, message_thread_id=message_thread_id)

    except Exception as e:
        logger.error(f"Error handling /get_topic_id command: {str(e)}")
        if update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An error occurred while processing the command. Please try again."
            )

async def main():
    """Main application entry point"""
    try:
        logger.info("Starting bot application")

        # Initialize bot
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", handle_start))
        application.add_handler(CommandHandler("get_topic_id", handle_get_topic_id))

        logger.info("Bot setup complete, starting polling")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

        # Initialize processed tokens
        initialize_processed_tokens()

        # Send initialization message to the alerts group
        await send_initialization_message(application.bot)

        # Start scanning loop
        asyncio.create_task(scan_loop(application.bot))

        # Keep the application running until interrupted
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        logger.info("Shutting down bot application")
        await application.stop()
        await application.updater.stop()
        await application.shutdown()

async def send_initialization_message(bot):
    """Send initialization message to the alerts group"""
    try:
        chat_id = -1002440374107  # Replace with the actual chat ID of the alerts group
        message = "🚀 Bot has started and token history has been initialized. 🚀"
        await bot.send_message(chat_id=chat_id, text=message)
        logger.info("Initialization message sent to the alerts group")
    except Exception as e:
        logger.error(f"Error sending initialization message: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())
