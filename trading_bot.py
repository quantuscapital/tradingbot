import os
import asyncio
import logging
import yaml
import time
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

from paper_trading import PaperTradingEngine
from tui import run_tui_with_bot
from config_manager import ConfigManager
from price_feeds import get_sol_price

# Try to import live trading - it's optional for paper trading only
try:
    from live_trading import LiveTradingEngine
    LIVE_TRADING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Live trading not available: {e}")
    logger.info("You can still use paper trading mode. For live trading, install: pip install solana solders")
    LiveTradingEngine = None
    LIVE_TRADING_AVAILABLE = False
from telegram import Bot
from telegram.constants import ParseMode

# Import existing bot functions
from bot import (
    get_latest_tokens, filter_tokens_by_criteria, get_token_pairs_data,
    rate_limited_api_call, meets_criteria, load_active_chats
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot that combines scanning, paper trading, and TUI"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Initialize config manager
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize trading engines
        self.paper_engine = PaperTradingEngine(self.config)
        self.live_engine: Optional[LiveTradingEngine] = None
        self.current_engine = self.paper_engine  # Default to paper trading
        
        # Initialize Telegram bot if token is available
        self.telegram_bot = None
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if telegram_token:
            self.telegram_bot = Bot(token=telegram_token)
            logger.info("Telegram bot initialized")
        else:
            logger.warning("No Telegram bot token found - notifications disabled")
        
        # Track processed tokens to avoid duplicates
        self.processed_tokens = set()
        
        # Store current prices for position monitoring
        self.current_prices = {}
        
        # Initialize trading mode
        self._initialize_trading_mode()
        
        logger.info("Trading bot initialized")
    
    def _initialize_trading_mode(self):
        """Initialize the appropriate trading mode"""
        try:
            is_paper_mode = self.config_manager.get('paper_trading.enabled', True)
            
            if is_paper_mode:
                self.current_engine = self.paper_engine
                logger.info("Trading mode: PAPER (simulation)")
            else:
                # Initialize live trading
                private_key = self.config_manager.get('wallet.private_key', '')
                rpc_url = self.config_manager.get('wallet.rpc_url', '')
                
                if not private_key:
                    logger.warning("Live trading mode requested but no private key configured. Falling back to paper trading.")
                    self.current_engine = self.paper_engine
                    return
                
                if not LIVE_TRADING_AVAILABLE:
                    logger.warning("Live trading requested but dependencies not available. Install: pip install solana solders")
                    logger.info("Falling back to paper trading mode")
                    self.current_engine = self.paper_engine
                    return
                
                try:
                    self.live_engine = LiveTradingEngine(
                        config=self.config,
                        private_key=private_key,
                        rpc_url=rpc_url or None
                    )
                    self.current_engine = self.live_engine
                    logger.info("Trading mode: LIVE (real money)")
                    logger.warning("⚠️  LIVE TRADING ACTIVE - Real funds at risk!")
                except Exception as e:
                    logger.error(f"Failed to initialize live trading: {e}")
                    logger.info("Falling back to paper trading mode")
                    self.current_engine = self.paper_engine
                    
        except Exception as e:
            logger.error(f"Error initializing trading mode: {e}")
            self.current_engine = self.paper_engine
    
    async def switch_trading_mode(self, live_mode: bool):
        """Switch between paper and live trading modes"""
        try:
            if live_mode and not self.live_engine:
                # Check if live trading is available
                if not LIVE_TRADING_AVAILABLE:
                    raise ValueError("Live trading dependencies not available. Install: pip install solana solders")
                
                # Initialize live trading engine
                private_key = self.config_manager.get('wallet.private_key', '')
                rpc_url = self.config_manager.get('wallet.rpc_url', '')
                
                if not private_key:
                    raise ValueError("No private key configured for live trading")
                
                self.live_engine = LiveTradingEngine(
                    config=self.config,
                    private_key=private_key,
                    rpc_url=rpc_url or None
                )
                await self.live_engine.initialize()
            
            # Switch engines
            if live_mode:
                self.current_engine = self.live_engine
                logger.warning("⚠️  SWITCHED TO LIVE TRADING - Real funds at risk!")
            else:
                self.current_engine = self.paper_engine
                logger.info("Switched to paper trading (simulation)")
                
        except Exception as e:
            logger.error(f"Failed to switch trading mode: {e}")
            self.current_engine = self.paper_engine
            raise
    
    def is_live_trading(self) -> bool:
        """Check if currently in live trading mode"""
        return isinstance(self.current_engine, LiveTradingEngine)
    
    async def update_config(self, new_config: dict):
        """Update configuration and potentially switch trading modes"""
        try:
            # Update config manager
            self.config_manager.config.update(new_config)
            
            # Check if trading mode changed
            new_paper_mode = new_config.get('paper_trading', {}).get('enabled', True)
            current_is_paper = not self.is_live_trading()
            
            if new_paper_mode != current_is_paper:
                logger.info(f"Trading mode changing: {'paper' if new_paper_mode else 'live'}")
                await self.switch_trading_mode(not new_paper_mode)
            
            # Update current engine config
            if hasattr(self.current_engine, 'update_config'):
                self.current_engine.update_config(new_config.get('paper_trading', {}))
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise
    
    async def scan_and_trade(self) -> List[dict]:
        """Main scanning and trading logic"""
        try:
            # Get latest tokens from DexScreener
            tokens = await get_latest_tokens()
            if not tokens:
                logger.debug("No tokens fetched from DexScreener")
                return []
            
            # Filter tokens based on criteria
            viable_tokens = await filter_tokens_by_criteria(tokens)
            if not viable_tokens:
                logger.debug("No viable tokens found")
                return []
            
            logger.info(f"Found {len(viable_tokens)} viable tokens")
            
            # Process each viable token for trading
            for token_data in viable_tokens:
                await self.process_token_for_trading(token_data)
            
            # Note: Position monitoring is now handled by the TUI price updater
            
            return viable_tokens
            
        except Exception as e:
            logger.error(f"Error in scan_and_trade: {e}")
            return []
    
    async def process_token_for_trading(self, token_data: dict):
        """Process a viable token for potential trading"""
        try:
            token_address = token_data['tokenAddress']
            pair = token_data['pair']
            
            # Skip if already processed recently
            if token_address in self.processed_tokens:
                return
            
            # Check if we already have a position in this token
            if hasattr(self.current_engine, 'positions') and token_address in self.current_engine.positions:
                logger.debug(f"Already have position in {token_address}")
                return
            
            # Execute trade (paper or live)
            if self.is_live_trading():
                trade_record = await self.current_engine.execute_buy(token_address, token_data)
            else:
                trade_record = self.current_engine.simulate_buy(token_address, token_data)
            
            if trade_record:
                # Mark as processed
                self.processed_tokens.add(token_address)
                
                # Send Telegram notification
                await self.send_trade_notification(trade_record)
                
                # Update current price for monitoring
                current_price = float(pair.get('priceUsd', 0))
                if current_price > 0:
                    self.current_prices[token_address] = current_price
                
                logger.info(f"Executed paper for {token_address}")
            
        except Exception as e:
            logger.error(f"Error processing token {token_data.get('tokenAddress', 'unknown')}: {e}")
    
    async def monitor_positions(self):
        """Monitor existing positions for sell conditions"""
        try:
            if not hasattr(self.current_engine, 'positions') or not self.current_engine.positions:
                return
            
            # Get current prices for all positions
            token_addresses = list(self.current_engine.positions.keys())
            
            # Fetch current prices in batches
            for i in range(0, len(token_addresses), 30):
                batch = token_addresses[i:i + 30]
                pairs_data = await get_token_pairs_data(batch)
                
                for token_address in batch:
                    if token_address in pairs_data:
                        pair = pairs_data[token_address]
                        current_price = float(pair.get('priceUsd', 0))
                        
                        if current_price > 0:
                            self.current_prices[token_address] = current_price
                            
                            # Check sell conditions
                            should_sell, reason, is_partial = self.current_engine.check_sell_conditions(
                                token_address, current_price
                            )
                            
                            if should_sell:
                                # Get sell percentage from profit tier if applicable
                                sell_percent = None
                                if "Profit Tier" in reason and is_partial:
                                    # Extract tier number from reason
                                    tier_num = int(reason.split("Tier ")[1].split(" ")[0]) - 1
                                    profit_tiers = self.current_engine.config.get('profit_tiers', [])
                                    if tier_num < len(profit_tiers):
                                        sell_percent = profit_tiers[tier_num]['sell_percent']
                                
                                # Execute sell trade (paper or live)
                                if self.is_live_trading():
                                    trade_record = await self.current_engine.execute_sell(
                                        token_address, current_price, reason, is_partial, sell_percent
                                    )
                                else:
                                    trade_record = self.current_engine.simulate_sell(
                                        token_address, current_price, reason, is_partial, sell_percent
                                    )
                                
                                if trade_record:
                                    # Send Telegram notification
                                    await self.send_trade_notification(trade_record)
                                    
                                    logger.info(f"Executed sell for {token_address}: {reason}")
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def send_trade_notification(self, trade_record: dict):
        """Send trade notification via Telegram"""
        if not self.telegram_bot:
            return
        
        try:
            trade_type = trade_record['type']
            symbol = trade_record['token_symbol']
            
            if trade_type == 'BUY':
                message = self.format_buy_notification(trade_record)
            elif trade_type == 'SELL':
                message = self.format_sell_notification(trade_record)
            else:
                return
            
            # Send to active chats
            active_chats = load_active_chats()
            for chat_id_str, chat_data in active_chats.items():
                try:
                    if isinstance(chat_data, dict) and chat_data.get('active', False):
                        await self.telegram_bot.send_message(
                            chat_id=int(chat_id_str),
                            text=message,
                            parse_mode=ParseMode.MARKDOWN_V2,
                            disable_web_page_preview=True
                        )
                except Exception as e:
                    logger.error(f"Error sending notification to chat {chat_id_str}: {e}")
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
    
    def format_buy_notification(self, trade_record: dict) -> str:
        """Format buy notification message"""
        symbol = trade_record['token_symbol']
        quantity = trade_record['quantity']
        price = trade_record['price_usd']
        sol_amount = trade_record['sol_amount']
        
        message = f"""
🟢 *BUY EXECUTED*

*Token:* {symbol}
*Quantity:* {quantity:.6f}
*Price:* ${price:.8f}
*SOL Amount:* {sol_amount:.4f} SOL
*Time:* {datetime.now().strftime('%H:%M:%S')}

📊 *Portfolio Status:*
*SOL Balance:* {getattr(self.current_engine, 'sol_balance', 0):.4f} SOL
*Open Positions:* {len(getattr(self.current_engine, 'positions', {}))}
*Total Trades:* {getattr(self.current_engine, 'metrics', {}).get('total_trades', 0)}
*Mode:* {'🔴 LIVE' if self.is_live_trading() else '📄 PAPER'}
        """.strip()
        
        # Escape markdown
        return message.replace('.', '\\.').replace('-', '\\-').replace('(', '\\(').replace(')', '\\)')
    
    def format_sell_notification(self, trade_record: dict) -> str:
        """Format sell notification message"""
        symbol = trade_record['token_symbol']
        quantity = trade_record['quantity']
        price = trade_record['price_usd']
        sol_proceeds = trade_record['sol_proceeds']
        pnl_percent = trade_record['pnl_percent']
        pnl_usd = trade_record['pnl_usd']
        reason = trade_record['sell_reason']
        partial = trade_record['partial']
        
        sell_type = "PARTIAL SELL" if partial else "FULL SELL"
        emoji = "🟢" if pnl_percent > 0 else "🔴"
        
        message = f"""
{emoji} *PAPER {sell_type} EXECUTED*

*Token:* {symbol}
*Quantity:* {quantity:.6f}
*Price:* ${price:.8f}
*SOL Proceeds:* {sol_proceeds:.4f} SOL
*P&L:* {pnl_percent:+.2f}% (${pnl_usd:+.2f})
*Reason:* {reason}
*Time:* {datetime.now().strftime('%H:%M:%S')}

📊 *Portfolio Status:*
*SOL Balance:* {getattr(self.current_engine, 'sol_balance', 0):.4f} SOL
*Total P&L:* ${getattr(self.current_engine, 'metrics', {}).get('total_pnl_usd', 0):+.2f}
*Win Rate:* {(getattr(self.current_engine, 'metrics', {}).get('winning_trades', 0) / max(getattr(self.current_engine, 'metrics', {}).get('total_trades', 1), 1)) * 100:.1f}%
*Mode:* {'🔴 LIVE' if self.is_live_trading() else '📄 PAPER'}
        """.strip()
        
        # Escape markdown
        return message.replace('.', '\\.').replace('-', '\\-').replace('(', '\\(').replace(')', '\\)')
    
    async def run_with_tui(self):
        """Run the trading bot with TUI interface"""
        logger.info("Starting trading bot with TUI")
        
        # Create the scan function for TUI
        async def scan_function():
            return await self.scan_and_trade()
        
        # Run TUI with bot (pass the current active engine)
        await run_tui_with_bot(self.current_engine, scan_function, self.config_manager)
    
    async def run_headless(self):
        """Run the trading bot without TUI (headless mode)"""
        logger.info("Starting trading bot in headless mode")
        
        while True:
            try:
                viable_tokens = await self.scan_and_trade()
                
                if viable_tokens:
                    logger.info(f"Processed {len(viable_tokens)} viable tokens")
                
                # Wait before next scan
                await asyncio.sleep(self.config['scan_interval'])
                
            except Exception as e:
                logger.error(f"Error in headless scan loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--headless', action='store_true', help='Run without TUI interface')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    try:
        # Create trading bot
        bot = TradingBot(args.config)
        
        if args.headless:
            await bot.run_headless()
        else:
            await bot.run_with_tui()
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
