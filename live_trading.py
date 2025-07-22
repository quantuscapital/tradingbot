import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from decimal import Decimal
from market_analysis import analyze_token_comprehensive, ComprehensiveAnalysis

try:
    from solana.transaction import Transaction
    from solana.publickey import PublicKey
except ImportError:
    # Try alternative import structure
    try:
        from solders.transaction import Transaction
        from solders.pubkey import Pubkey as PublicKey
    except ImportError:
        # Fallback to basic functionality without solana imports
        Transaction = None
        PublicKey = None

from solana_wallet import SolanaWallet, TransactionResult
from jupiter_client import JupiterClient, SwapQuote, SwapTransaction
from price_feeds import PriceFeedManager, get_sol_price
from paper_trading import PaperTradingEngine  # For shared logic

logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """Live trading engine that executes real trades on Solana"""
    
    def __init__(self, config: dict, private_key: str, rpc_url: str = None):
        self.config = config.get('paper_trading', {})  # Reuse paper trading config structure
        self.private_key = private_key
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        
        # State files
        self.state_file = 'live_trading_state.json'
        self.trades_file = 'live_trades_log.json'
        
        # Runtime control
        self.enabled = True
        self.emergency_stop = False
        
        # Trading state
        self.sol_balance = 0.0
        self.positions = {}  # {token_address: position_data}
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl_sol': 0.0,
            'total_pnl_usd': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'total_fees_sol': 0.0,
            'failed_transactions': 0,
            'average_hold_time': 0.0
        }
        
        # Safety limits
        self.max_trade_size_sol = self.config.get('max_trade_size_sol', 1.0)
        self.max_slippage_bps = self.config.get('max_slippage_bps', 300)  # 3%
        self.min_sol_balance = self.config.get('min_sol_balance', 0.01)
        
        # Initialize components
        self.wallet: Optional[SolanaWallet] = None
        self.jupiter_client: Optional[JupiterClient] = None
        self.price_manager: Optional[PriceFeedManager] = None
        
        # Load existing state
        self.load_state()
    
    async def initialize(self):
        """Initialize all async components"""
        try:
            # Check if required modules are available
            if Transaction is None or PublicKey is None:
                raise ImportError("solana-py or solders is required for live trading. Install with: pip install solana solders")
            
            # Initialize wallet
            self.wallet = SolanaWallet(self.private_key, self.rpc_url)
            await self.wallet.connect()
            
            # Initialize Jupiter client
            self.jupiter_client = JupiterClient()
            
            # Initialize price manager
            self.price_manager = PriceFeedManager()
            
            # Update SOL balance
            await self.update_sol_balance()
            
            logger.info(f"Live trading engine initialized. SOL Balance: {self.sol_balance:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize live trading engine: {e}")
            raise
    
    async def close(self):
        """Close all connections"""
        if self.wallet:
            await self.wallet.close()
        if self.jupiter_client:
            await self.jupiter_client.close()
        if self.price_manager:
            await self.price_manager.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def load_state(self):
        """Load trading state from file"""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
                    self.metrics = data.get('metrics', self.metrics)
                    logger.info(f"Loaded live trading state: {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error loading live trading state: {e}")
            
        try:
            if Path(self.trades_file).exists():
                with open(self.trades_file, 'r') as f:
                    self.trade_history = json.load(f)
                    logger.info(f"Loaded {len(self.trade_history)} trade records")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
    
    def save_state(self):
        """Save trading state to file"""
        try:
            state_data = {
                'sol_balance': self.sol_balance,
                'positions': self.positions,
                'metrics': self.metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            # Atomic write
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            Path(temp_file).rename(self.state_file)
            
            # Save trade history
            temp_trades = f"{self.trades_file}.tmp"
            with open(temp_trades, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            Path(temp_trades).rename(self.trades_file)
            
        except Exception as e:
            logger.error(f"Error saving live trading state: {e}")
    
    async def update_sol_balance(self):
        """Update SOL balance from blockchain"""
        try:
            if self.wallet:
                self.sol_balance = await self.wallet.get_sol_balance()
        except Exception as e:
            logger.error(f"Failed to update SOL balance: {e}")
    
    def calculate_trade_amount(self, token_score: float = None) -> float:
        """Calculate SOL amount to use for a trade with safety checks"""
        base_allocation = self.config.get('allocation_percentage', 15.0) / 100.0
        base_amount = self.sol_balance * base_allocation
        
        # Apply position sizing multipliers based on token score (same logic as paper trading)
        if token_score is not None:
            multipliers = self.config.get('position_size_multipliers', {})
            high_threshold = multipliers.get('high_conviction_threshold', 80)
            medium_threshold = multipliers.get('medium_conviction_threshold', 60)
            high_multiplier = multipliers.get('high_conviction_multiplier', 1.5)
            medium_multiplier = multipliers.get('medium_conviction_multiplier', 1.2)
            
            if token_score >= high_threshold:
                base_amount *= high_multiplier
            elif token_score >= medium_threshold:
                base_amount *= medium_multiplier
        
        # Apply safety limits
        max_amount = min(base_amount, self.max_trade_size_sol)
        
        # Ensure we keep minimum SOL balance
        available_sol = self.sol_balance - self.min_sol_balance
        final_amount = min(max_amount, available_sol)
        
        return max(0, final_amount)
    
    def analyze_token_comprehensive(self, token_data: dict) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis of a token using advanced market analysis"""
        try:
            return analyze_token_comprehensive(token_data)
        except Exception as e:
            logger.error(f"Error performing comprehensive token analysis: {e}")
            # Return a neutral analysis as fallback
            pair = token_data.get('pair', {})
            symbol = pair.get('baseToken', {}).get('symbol', 'Unknown')
            current_price = float(pair.get('priceUsd', 0))
            
            from market_analysis import (
                ComprehensiveAnalysis, TrendAnalysis, MomentumAnalysis, 
                FallingKnifeSignals, MarketStructure, TrendDirection, MomentumStrength
            )
            
            # Create neutral fallback analysis
            neutral_trend = TrendAnalysis(
                direction=TrendDirection.SIDEWAYS,
                strength=50.0,
                confidence=0.5,
                timeframe="1h",
                duration_hours=1.0,
                trend_angle=0.0
            )
            
            neutral_momentum = MomentumAnalysis(
                strength=MomentumStrength.MODERATE,
                velocity=0.0,
                acceleration=0.0,
                rsi_14=50.0,
                volume_momentum=50.0,
                sustainability_score=50.0
            )
            
            neutral_falling_knife = FallingKnifeSignals(
                is_falling_knife=False,
                confidence=0.0,
                signals=[],
                volume_divergence=False,
                recent_high_proximity=0.5,
                deceleration_rate=0.0,
                bounce_sustainability=0.5
            )
            
            neutral_structure = MarketStructure(
                support_levels=[current_price * 0.95],
                resistance_levels=[current_price * 1.05],
                volume_profile={'volume_24h': 0, 'liquidity': 0},
                order_flow_imbalance=0.0,
                market_depth_score=50.0
            )
            
            return ComprehensiveAnalysis(
                symbol=symbol,
                current_price=current_price,
                timestamp=datetime.now(),
                trend_1h=neutral_trend,
                trend_4h=neutral_trend,
                trend_24h=neutral_trend,
                trend_alignment_score=50.0,
                dominant_trend=TrendDirection.SIDEWAYS,
                momentum=neutral_momentum,
                falling_knife=neutral_falling_knife,
                structure=neutral_structure,
                buy_confidence=50.0,
                sell_pressure=50.0,
                risk_score=50.0,
                buy_reasoning=["Neutral market conditions"],
                warning_signals=["Analysis failed - using fallback"]
            )
    
    def calculate_token_score(self, pair: dict) -> float:
        """Legacy method for backward compatibility - now uses comprehensive analysis"""
        try:
            # Create token_data structure expected by comprehensive analysis
            token_data = {'pair': pair}
            
            # Get comprehensive analysis
            analysis = self.analyze_token_comprehensive(token_data)
            
            # Convert buy_confidence to legacy score format
            return analysis.buy_confidence
            
        except Exception as e:
            logger.error(f"Error calculating token score: {e}")
            return 50  # Default neutral score
    
    async def execute_buy(self, token_address: str, token_data: dict) -> Optional[dict]:
        """Execute a real buy order through Jupiter with comprehensive market analysis"""
        if not self.can_trade():
            logger.debug("Live trading is disabled or emergency stopped")
            return None
        
        try:
            pair = token_data['pair']
            token_mint = pair['baseToken']['address']
            
            # Perform comprehensive market analysis
            analysis = self.analyze_token_comprehensive(token_data)
            
            # Check if token should be avoided (falling knife detection)
            should_avoid, avoid_reason = analysis.falling_knife.is_falling_knife and analysis.falling_knife.confidence > 0.7, ""
            if analysis.falling_knife.is_falling_knife and analysis.falling_knife.confidence > 0.7:
                avoid_reason = f"High probability falling knife ({analysis.falling_knife.confidence:.1%} confidence)"
            elif analysis.risk_score > 80:
                should_avoid, avoid_reason = True, f"Extremely high risk score ({analysis.risk_score:.0f}/100)"
            elif analysis.buy_confidence < 20:
                should_avoid, avoid_reason = True, f"Very low buy confidence ({analysis.buy_confidence:.0f}/100)"
            elif len(analysis.warning_signals) >= 4:
                should_avoid, avoid_reason = True, f"Too many warning signals ({len(analysis.warning_signals)})"
            
            if should_avoid:
                logger.info(f"AVOIDING token {token_address}: {avoid_reason}")
                logger.info(f"Warning signals: {', '.join(analysis.warning_signals)}")
                return None
            
            # Get entry timing recommendation
            from market_analysis import get_market_analyzer
            analyzer = get_market_analyzer()
            timing_rec = analyzer.get_entry_timing_recommendation(analysis)
            
            if timing_rec['avoid']:
                logger.info(f"AVOIDING token {token_address}: {', '.join(timing_rec['reasoning'])}")
                return None
            elif timing_rec['wait_for_pullback']:
                logger.info(f"WAITING for better entry on {token_address}: {', '.join(timing_rec['reasoning'])}")
                logger.info(f"Patience score: {timing_rec['patience_score']}/100")
                return None
            elif timing_rec['immediate_buy']:
                logger.info(f"IMMEDIATE BUY signal for {token_address}: {', '.join(timing_rec['reasoning'])}")
            
            # Calculate trade amount using enhanced analysis
            token_score = analysis.buy_confidence
            sol_amount = self.calculate_trade_amount(token_score)
            
            # Log detailed analysis
            logger.info(f"Live token {token_address} analysis:")
            logger.info(f"  Buy Confidence: {analysis.buy_confidence:.1f}/100")
            logger.info(f"  Risk Score: {analysis.risk_score:.1f}/100")
            logger.info(f"  Trend Alignment: {analysis.trend_alignment_score:.1f}/100")
            logger.info(f"  Momentum: {analysis.momentum.strength.value}")
            logger.info(f"  SOL allocation: {sol_amount:.4f}")
            if analysis.buy_reasoning:
                logger.info(f"  Buy reasoning: {', '.join(analysis.buy_reasoning)}")
            if analysis.warning_signals:
                logger.info(f"  Warnings: {', '.join(analysis.warning_signals)}")
            
            if sol_amount <= 0:
                logger.warning(f"Invalid trade amount: {sol_amount}")
                return None
            
            # Update balance to ensure we have enough SOL
            await self.update_sol_balance()
            
            if sol_amount > self.sol_balance - self.min_sol_balance:
                logger.warning(f"Insufficient SOL balance for trade: {sol_amount} > {self.sol_balance - self.min_sol_balance}")
                return None
            
            # Get quote from Jupiter
            sol_lamports = self.jupiter_client.sol_to_lamports(sol_amount)
            wsol_mint = self.jupiter_client.WSOL_MINT
            
            quote = await self.jupiter_client.get_quote(
                input_mint=wsol_mint,
                output_mint=token_mint,
                amount=sol_lamports,
                slippage_bps=min(self.config.get('slippage_tolerance', 2.0) * 100, self.max_slippage_bps)
            )
            
            if not quote:
                logger.warning(f"Failed to get quote for {token_address}")
                return None
            
            # Check price impact
            if quote.price_impact_pct > 5.0:  # 5% max price impact
                logger.warning(f"Price impact too high: {quote.price_impact_pct}%")
                return None
            
            # Get swap transaction
            swap_tx = await self.jupiter_client.get_swap_transaction(
                quote=quote,
                user_public_key=str(self.wallet.public_key),
                wrap_and_unwrap_sol=True,
                use_shared_accounts=True,
                dynamic_compute_unit_limit=True
            )
            
            if not swap_tx:
                logger.warning(f"Failed to get swap transaction for {token_address}")
                return None
            
            # Decode and simulate transaction
            import base64
            tx_bytes = base64.b64decode(swap_tx.swap_transaction)
            transaction = Transaction.deserialize(tx_bytes)
            
            # Simulate before sending
            simulation = await self.wallet.simulate_transaction(transaction)
            if not simulation['success']:
                logger.error(f"Transaction simulation failed: {simulation['error']}")
                return None
            
            # Execute the trade
            result = await self.wallet.send_transaction(transaction)
            
            if result and result.success:
                # Calculate actual token amount received (estimate)
                token_amount = self.jupiter_client.lamports_to_sol(quote.output_amount)  # Simplified
                
                # Record the position
                position_data = {
                    'quantity': token_amount,
                    'buy_price_sol': sol_amount / token_amount,
                    'buy_price_usd': float(pair.get('priceUsd', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'token_name': pair['baseToken'].get('name', 'Unknown'),
                    'token_symbol': pair['baseToken'].get('symbol', 'Unknown'),
                    'token_mint': token_mint,
                    'transaction_signature': result.signature,
                    'highest_price_reached': float(pair.get('priceUsd', 0)),
                    'trailing_stop_percent': self.config.get('stop_loss_percent', 10.0),
                    'triggered_tiers': [],
                    'jupiter_quote': {
                        'input_amount': quote.input_amount,
                        'output_amount': quote.output_amount,
                        'price_impact_pct': quote.price_impact_pct
                    }
                }
                
                self.positions[token_address] = position_data
                
                # Record trade
                trade_record = {
                    'type': 'BUY',
                    'token_address': token_address,
                    'token_name': pair['baseToken'].get('name', 'Unknown'),
                    'token_symbol': pair['baseToken'].get('symbol', 'Unknown'),
                    'token_mint': token_mint,
                    'quantity': token_amount,
                    'sol_amount': sol_amount,
                    'price_usd': float(pair.get('priceUsd', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'transaction_signature': result.signature,
                    'slippage_bps': quote.slippage_bps,
                    'price_impact_pct': quote.price_impact_pct,
                    'route_summary': self._summarize_route(quote.route_plan)
                }
                
                self.trade_history.append(trade_record)
                self.metrics['total_trades'] += 1
                
                # Update balance
                await self.update_sol_balance()
                self.save_state()
                
                logger.info(f"Live BUY executed: {token_amount:.6f} {pair['baseToken'].get('symbol', 'Unknown')} for {sol_amount:.4f} SOL | TX: {result.signature}")
                return trade_record
            
            else:
                logger.error(f"Transaction failed: {result.error if result else 'Unknown error'}")
                self.metrics['failed_transactions'] += 1
                self.save_state()
                return None
                
        except Exception as e:
            logger.error(f"Error executing buy for {token_address}: {e}")
            self.metrics['failed_transactions'] += 1
            self.save_state()
            return None
    
    async def execute_sell(self, token_address: str, current_price_usd: float, sell_reason: str, partial: bool = False, sell_percent: float = None) -> Optional[dict]:
        """Execute a real sell order through Jupiter"""
        if not self.enabled:
            logger.debug("Live trading is disabled")
            return None
        
        try:
            if token_address not in self.positions:
                logger.warning(f"No position found for token {token_address}")
                return None
            
            position = self.positions[token_address]
            token_mint = position['token_mint']
            
            # Determine quantity to sell
            if partial and sell_percent is not None:
                sell_quantity = position['quantity'] * (sell_percent / 100.0)
            elif partial:
                sell_quantity = position['quantity'] * 0.5  # Default 50%
            else:
                sell_quantity = position['quantity']
            
            # Get current token balance to verify
            token_balance = await self.wallet.get_token_balance(token_mint)
            if not token_balance or token_balance.ui_amount < sell_quantity:
                logger.warning(f"Insufficient token balance for sell: {sell_quantity} > {token_balance.ui_amount if token_balance else 0}")
                return None
            
            # Convert to smallest unit for Jupiter
            token_decimals = token_balance.decimals if token_balance else 9
            sell_amount_raw = int(sell_quantity * (10 ** token_decimals))
            
            # Get quote from Jupiter for selling
            quote = await self.jupiter_client.get_quote(
                input_mint=token_mint,
                output_mint=self.jupiter_client.WSOL_MINT,
                amount=sell_amount_raw,
                slippage_bps=min(self.config.get('slippage_tolerance', 2.0) * 100, self.max_slippage_bps)
            )
            
            if not quote:
                logger.warning(f"Failed to get sell quote for {token_address}")
                return None
            
            # Get swap transaction
            swap_tx = await self.jupiter_client.get_swap_transaction(
                quote=quote,
                user_public_key=str(self.wallet.public_key),
                wrap_and_unwrap_sol=True,
                use_shared_accounts=True,
                dynamic_compute_unit_limit=True
            )
            
            if not swap_tx:
                logger.warning(f"Failed to get sell swap transaction for {token_address}")
                return None
            
            # Decode and simulate transaction
            import base64
            tx_bytes = base64.b64decode(swap_tx.swap_transaction)
            transaction = Transaction.deserialize(tx_bytes)
            
            # Simulate before sending
            simulation = await self.wallet.simulate_transaction(transaction)
            if not simulation['success']:
                logger.error(f"Sell transaction simulation failed: {simulation['error']}")
                return None
            
            # Execute the sell trade
            result = await self.wallet.send_transaction(transaction)
            
            if result and result.success:
                # Calculate proceeds
                sol_proceeds = self.jupiter_client.lamports_to_sol(quote.output_amount)
                
                # Calculate P&L
                cost_basis = sell_quantity * position['buy_price_sol']
                pnl_sol = sol_proceeds - cost_basis
                pnl_percent = (pnl_sol / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Get current SOL price for USD P&L
                sol_price_usd = await get_sol_price() or 100.0
                pnl_usd = pnl_sol * sol_price_usd
                
                # Update position
                if partial:
                    self.positions[token_address]['quantity'] -= sell_quantity
                else:
                    del self.positions[token_address]
                
                # Update metrics
                if pnl_usd > 0:
                    self.metrics['winning_trades'] += 1
                    if pnl_usd > self.metrics['largest_win']:
                        self.metrics['largest_win'] = pnl_usd
                else:
                    if pnl_usd < self.metrics['largest_loss']:
                        self.metrics['largest_loss'] = pnl_usd
                
                self.metrics['total_pnl_sol'] += pnl_sol
                self.metrics['total_pnl_usd'] += pnl_usd
                
                # Record trade
                trade_record = {
                    'type': 'SELL',
                    'token_address': token_address,
                    'token_name': position['token_name'],
                    'token_symbol': position['token_symbol'],
                    'token_mint': token_mint,
                    'quantity': sell_quantity,
                    'sol_proceeds': sol_proceeds,
                    'price_usd': current_price_usd,
                    'pnl_sol': pnl_sol,
                    'pnl_usd': pnl_usd,
                    'pnl_percent': pnl_percent,
                    'sell_reason': sell_reason,
                    'partial': partial,
                    'timestamp': datetime.now().isoformat(),
                    'transaction_signature': result.signature,
                    'slippage_bps': quote.slippage_bps,
                    'price_impact_pct': quote.price_impact_pct
                }
                
                self.trade_history.append(trade_record)
                
                # Update balance
                await self.update_sol_balance()
                self.save_state()
                
                logger.info(f"Live SELL executed: {sell_quantity:.6f} {position['token_symbol']} for {sol_proceeds:.4f} SOL (P&L: {pnl_percent:.2f}%) | TX: {result.signature}")
                return trade_record
            
            else:
                logger.error(f"Sell transaction failed: {result.error if result else 'Unknown error'}")
                self.metrics['failed_transactions'] += 1
                self.save_state()
                return None
                
        except Exception as e:
            logger.error(f"Error executing sell for {token_address}: {e}")
            self.metrics['failed_transactions'] += 1
            self.save_state()
            return None
    
    def check_sell_conditions(self, token_address: str, current_price_usd: float) -> Tuple[bool, str, bool]:
        """Check if sell conditions are met (same logic as paper trading)"""
        if token_address not in self.positions:
            return False, "", False
        
        position = self.positions[token_address]
        buy_price = position['buy_price_usd']
        
        # Update highest price reached
        highest_price = max(position.get('highest_price_reached', buy_price), current_price_usd)
        self.positions[token_address]['highest_price_reached'] = highest_price
        
        # Calculate current P&L percentage
        pnl_percent = ((current_price_usd - buy_price) / buy_price) * 100
        
        # Calculate trailing stop loss from highest price
        trailing_stop_percent = position.get('trailing_stop_percent', self.config.get('stop_loss_percent', 10.0))
        trailing_stop_price = highest_price * (1 - trailing_stop_percent / 100.0)
        
        # Check trailing stop loss
        if current_price_usd <= trailing_stop_price:
            return True, f"Trailing Stop Loss (-{trailing_stop_percent:.1f}% from peak)", False
        
        # Check profit tiers
        profit_tiers = self.config.get('profit_tiers', [])
        triggered_tiers = position.get('triggered_tiers', [])
        
        for i, tier in enumerate(profit_tiers):
            tier_profit = tier['profit_percent']
            tier_sell_percent = tier['sell_percent']
            tier_trailing_stop = tier['trailing_stop']
            
            if pnl_percent >= tier_profit and i not in triggered_tiers:
                self.positions[token_address]['triggered_tiers'].append(i)
                self.positions[token_address]['trailing_stop_percent'] = tier_trailing_stop
                
                remaining_tiers = len([t for t in profit_tiers if profit_tiers.index(t) > i])
                is_partial = remaining_tiers > 0
                
                return True, f"Profit Tier {i+1} (+{tier_profit}%)", is_partial
        
        # Fallback stop loss
        if not profit_tiers and pnl_percent <= -self.config.get('stop_loss_percent', 10.0):
            return True, f"Stop Loss (-{self.config.get('stop_loss_percent', 10.0)}%)", False
        
        return False, "", False
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.enabled and not self.emergency_stop and self.sol_balance > self.min_sol_balance
    
    def set_enabled(self, enabled: bool):
        """Enable or disable trading"""
        old_state = self.enabled
        self.enabled = enabled
        
        if old_state != enabled:
            status = "enabled" if enabled else "disabled"
            logger.info(f"Live trading {status}")
    
    def emergency_stop_all(self):
        """Emergency stop all trading"""
        self.emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
    
    def resume_trading(self):
        """Resume trading after emergency stop"""
        self.emergency_stop = False
        logger.info("Trading resumed after emergency stop")
    
    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary"""
        total_positions = len(self.positions)
        win_rate = (self.metrics['winning_trades'] / max(self.metrics['total_trades'], 1)) * 100
        
        return {
            'sol_balance': self.sol_balance,
            'total_positions': total_positions,
            'total_trades': self.metrics['total_trades'],
            'winning_trades': self.metrics['winning_trades'],
            'win_rate': win_rate,
            'total_pnl_usd': self.metrics['total_pnl_usd'],
            'total_pnl_sol': self.metrics['total_pnl_sol'],
            'largest_win': self.metrics['largest_win'],
            'largest_loss': self.metrics['largest_loss'],
            'total_fees_sol': self.metrics['total_fees_sol'],
            'failed_transactions': self.metrics['failed_transactions'],
            'allocation_per_trade': self.calculate_trade_amount()
        }
    
    def _summarize_route(self, route_plan: List[Dict]) -> str:
        """Summarize the swap route"""
        if not route_plan:
            return "Direct swap"
        
        steps = []
        for step in route_plan:
            if 'swapInfo' in step:
                swap_info = step['swapInfo']
                label = swap_info.get('label', 'Unknown')
                steps.append(label)
        
        return " → ".join(steps) if steps else "Direct swap"