import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
from market_analysis import analyze_token_comprehensive, ComprehensiveAnalysis

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    """Paper trading engine for simulating cryptocurrency trades"""
    
    def __init__(self, config: dict):
        self.config = config['paper_trading']
        self.state_file = 'paper_trading_state.json'
        self.trades_file = 'paper_trades_log.json'
        
        # Runtime control
        self.enabled = self.config.get('enabled', True)
        
        # Initialize state
        self.sol_balance = self.config['initial_balance']
        self.positions = {}  # {token_address: {quantity, buy_price_usd, buy_price_sol, timestamp}}
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl_sol': 0.0,
            'total_pnl_usd': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_hold_time': 0.0
        }
        
        # Load existing state if available
        self.load_state()
        
    def load_state(self):
        """Load paper trading state from file"""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.sol_balance = data.get('sol_balance', self.config['initial_balance'])
                    self.positions = data.get('positions', {})
                    self.metrics = data.get('metrics', self.metrics)
                    logger.info(f"Loaded paper trading state: {self.sol_balance} SOL, {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error loading paper trading state: {e}")
            
        try:
            if Path(self.trades_file).exists():
                with open(self.trades_file, 'r') as f:
                    self.trade_history = json.load(f)
                    logger.info(f"Loaded {len(self.trade_history)} trade records")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
    
    def save_state(self):
        """Save paper trading state to file"""
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
            logger.error(f"Error saving paper trading state: {e}")
    
    def calculate_trade_amount(self, token_score: float = None) -> float:
        """Calculate SOL amount to use for a trade with optional score-based sizing"""
        base_allocation = self.config['allocation_percentage'] / 100.0
        base_amount = self.sol_balance * base_allocation
        
        # Apply position sizing multipliers based on token score
        if token_score is not None:
            multipliers = self.config.get('position_size_multipliers', {})
            high_threshold = multipliers.get('high_conviction_threshold', 80)
            medium_threshold = multipliers.get('medium_conviction_threshold', 60)
            high_multiplier = multipliers.get('high_conviction_multiplier', 1.5)
            medium_multiplier = multipliers.get('medium_conviction_multiplier', 1.2)
            
            if token_score >= high_threshold:
                return base_amount * high_multiplier
            elif token_score >= medium_threshold:
                return base_amount * medium_multiplier
        
        return base_amount
    
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
    
    def simulate_buy(self, token_address: str, token_data: dict) -> Optional[dict]:
        """Simulate buying a token with comprehensive market analysis"""
        try:
            # Check if paper trading is enabled
            if not self.can_trade():
                logger.debug("Paper trading is disabled, skipping buy simulation")
                return None
            
            pair = token_data['pair']
            current_price_usd = float(pair.get('priceUsd', 0))
            
            if current_price_usd <= 0:
                logger.warning(f"Invalid price for token {token_address}: {current_price_usd}")
                return None
            
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
            
            # Calculate token score and trade amount using enhanced analysis
            token_score = analysis.buy_confidence
            sol_amount = self.calculate_trade_amount(token_score)
            
            # Log detailed analysis
            logger.info(f"Token {token_address} analysis:")
            logger.info(f"  Buy Confidence: {analysis.buy_confidence:.1f}/100")
            logger.info(f"  Risk Score: {analysis.risk_score:.1f}/100")
            logger.info(f"  Trend Alignment: {analysis.trend_alignment_score:.1f}/100")
            logger.info(f"  Momentum: {analysis.momentum.strength.value}")
            logger.info(f"  SOL allocation: {sol_amount:.4f}")
            if analysis.buy_reasoning:
                logger.info(f"  Buy reasoning: {', '.join(analysis.buy_reasoning)}")
            if analysis.warning_signals:
                logger.info(f"  Warnings: {', '.join(analysis.warning_signals)}")
            
            if sol_amount > self.sol_balance:
                logger.warning(f"Insufficient SOL balance for trade: {sol_amount} > {self.sol_balance}")
                return None
            
            # Apply slippage
            slippage = 1 + (self.config['slippage_tolerance'] / 100.0)
            effective_price = current_price_usd * slippage
            
            # Assume 1 SOL = $100 for simulation (in real implementation, get from price feed)
            sol_price_usd = 100.0  # This should be fetched from a price oracle
            usd_amount = sol_amount * sol_price_usd
            token_quantity = usd_amount / effective_price
            
            # Update balances
            self.sol_balance -= sol_amount
            
            # Add to positions
            if token_address in self.positions:
                # Average down
                existing = self.positions[token_address]
                total_quantity = existing['quantity'] + token_quantity
                total_cost_usd = (existing['quantity'] * existing['buy_price_usd']) + (token_quantity * effective_price)
                avg_price_usd = total_cost_usd / total_quantity
                
                self.positions[token_address] = {
                    'quantity': total_quantity,
                    'buy_price_usd': avg_price_usd,
                    'buy_price_sol': sol_amount / token_quantity,  # This is approximate
                    'timestamp': datetime.now().isoformat(),
                    'token_name': pair['baseToken'].get('name', 'Unknown'),
                    'token_symbol': pair['baseToken'].get('symbol', 'Unknown'),
                    'highest_price_reached': max(existing.get('highest_price_reached', effective_price), effective_price),
                    'trailing_stop_percent': self.config['stop_loss_percent'],
                    'triggered_tiers': existing.get('triggered_tiers', [])
                }
            else:
                self.positions[token_address] = {
                    'quantity': token_quantity,
                    'buy_price_usd': effective_price,
                    'buy_price_sol': sol_amount / token_quantity,
                    'timestamp': datetime.now().isoformat(),
                    'token_name': pair['baseToken'].get('name', 'Unknown'),
                    'token_symbol': pair['baseToken'].get('symbol', 'Unknown'),
                    'highest_price_reached': effective_price,
                    'trailing_stop_percent': self.config['stop_loss_percent'],
                    'triggered_tiers': []
                }
            
            # Record trade
            trade_record = {
                'type': 'BUY',
                'token_address': token_address,
                'token_name': pair['baseToken'].get('name', 'Unknown'),
                'token_symbol': pair['baseToken'].get('symbol', 'Unknown'),
                'quantity': token_quantity,
                'price_usd': effective_price,
                'sol_amount': sol_amount,
                'timestamp': datetime.now().isoformat(),
                'slippage_applied': self.config['slippage_tolerance']
            }
            
            self.trade_history.append(trade_record)
            self.metrics['total_trades'] += 1
            self.save_state()
            
            logger.info(f"Simulated BUY: {token_quantity:.6f} {pair['baseToken'].get('symbol', 'Unknown')} for {sol_amount:.4f} SOL")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error simulating buy for {token_address}: {e}")
            return None
    
    def simulate_sell(self, token_address: str, current_price_usd: float, sell_reason: str, partial: bool = False, sell_percent: float = None) -> Optional[dict]:
        """Simulate selling a token"""
        try:
            # Check if paper trading is enabled (allow selling existing positions even when disabled)
            if not self.enabled:
                logger.debug("Paper trading is disabled, but allowing sell of existing position")
            
            if token_address not in self.positions:
                logger.warning(f"No position found for token {token_address}")
                return None
            
            position = self.positions[token_address]
            
            # Determine quantity to sell
            if partial and sell_percent is not None:
                # Use specific percentage from profit tier
                sell_quantity = position['quantity'] * (sell_percent / 100.0)
            elif partial:
                # Fallback to config default
                sell_quantity = position['quantity'] * self.config.get('partial_sell_fraction', 0.5)
            else:
                sell_quantity = position['quantity']
            
            # Apply slippage (negative for selling)
            slippage = 1 - (self.config['slippage_tolerance'] / 100.0)
            effective_price = current_price_usd * slippage
            
            # Calculate proceeds
            usd_proceeds = sell_quantity * effective_price
            sol_price_usd = 100.0  # This should be fetched from a price oracle
            sol_proceeds = usd_proceeds / sol_price_usd
            
            # Calculate P&L
            cost_basis = sell_quantity * position['buy_price_usd']
            pnl_usd = usd_proceeds - cost_basis
            pnl_percent = (pnl_usd / cost_basis) * 100
            
            # Update balances
            self.sol_balance += sol_proceeds
            
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
            
            self.metrics['total_pnl_usd'] += pnl_usd
            self.metrics['total_pnl_sol'] += (sol_proceeds - (sell_quantity * position['buy_price_sol']))
            
            # Record trade
            trade_record = {
                'type': 'SELL',
                'token_address': token_address,
                'token_name': position['token_name'],
                'token_symbol': position['token_symbol'],
                'quantity': sell_quantity,
                'price_usd': effective_price,
                'sol_proceeds': sol_proceeds,
                'pnl_usd': pnl_usd,
                'pnl_percent': pnl_percent,
                'sell_reason': sell_reason,
                'partial': partial,
                'timestamp': datetime.now().isoformat(),
                'slippage_applied': self.config['slippage_tolerance']
            }
            
            self.trade_history.append(trade_record)
            self.save_state()
            
            logger.info(f"Simulated SELL: {sell_quantity:.6f} {position['token_symbol']} for {sol_proceeds:.4f} SOL (P&L: {pnl_percent:.2f}%)")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error simulating sell for {token_address}: {e}")
            return None
    
    def check_sell_conditions(self, token_address: str, current_price_usd: float) -> Tuple[bool, str, bool]:
        """Check if sell conditions are met for a position using tiered profit-taking"""
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
        trailing_stop_percent = position.get('trailing_stop_percent', self.config['stop_loss_percent'])
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
            
            # Check if this tier should be triggered
            if pnl_percent >= tier_profit and i not in triggered_tiers:
                # Mark this tier as triggered
                self.positions[token_address]['triggered_tiers'].append(i)
                
                # Update trailing stop to this tier's level
                self.positions[token_address]['trailing_stop_percent'] = tier_trailing_stop
                
                # Determine if this is a partial or full sell
                remaining_tiers = len([t for t in profit_tiers if profit_tiers.index(t) > i])
                is_partial = remaining_tiers > 0
                
                return True, f"Profit Tier {i+1} (+{tier_profit}%)", is_partial
        
        # Fallback to original stop loss if no tiers configured
        if not profit_tiers and pnl_percent <= -self.config['stop_loss_percent']:
            return True, f"Stop Loss (-{self.config['stop_loss_percent']}%)", False
        
        return False, "", False
    
    def get_position_pnl(self, token_address: str, current_price_usd: float) -> dict:
        """Get current P&L for a position"""
        if token_address not in self.positions:
            return {}
        
        position = self.positions[token_address]
        buy_price = position['buy_price_usd']
        quantity = position['quantity']
        
        current_value_usd = quantity * current_price_usd
        cost_basis = quantity * buy_price
        pnl_usd = current_value_usd - cost_basis
        pnl_percent = (pnl_usd / cost_basis) * 100
        
        return {
            'token_address': token_address,
            'token_symbol': position['token_symbol'],
            'quantity': quantity,
            'buy_price_usd': buy_price,
            'current_price_usd': current_price_usd,
            'current_value_usd': current_value_usd,
            'cost_basis': cost_basis,
            'pnl_usd': pnl_usd,
            'pnl_percent': pnl_percent,
            'timestamp': position['timestamp']
        }
    
    def get_portfolio_summary(self) -> dict:
        """Get overall portfolio summary"""
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
            'allocation_per_trade': self.calculate_trade_amount()
        }
    
    def reset_state(self):
        """Reset paper trading state"""
        self.sol_balance = self.config['initial_balance']
        self.positions = {}
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl_sol': 0.0,
            'total_pnl_usd': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_hold_time': 0.0
        }
        self.save_state()
        logger.info("Paper trading state reset")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable paper trading at runtime"""
        old_state = self.enabled
        self.enabled = enabled
        
        if old_state != enabled:
            status = "enabled" if enabled else "disabled"
            logger.info(f"Paper trading {status}")
            
            # If disabling, optionally close all positions
            if not enabled and self.positions:
                logger.info("Paper trading disabled - positions remain open but no new trades will be executed")
    
    def is_enabled(self) -> bool:
        """Check if paper trading is currently enabled"""
        return self.enabled
    
    def update_config(self, new_config: dict):
        """Update paper trading configuration at runtime"""
        try:
            old_balance = self.config.get('initial_balance', 10.0)
            
            # Update configuration
            self.config.update(new_config)
            
            # Update enabled state
            self.enabled = self.config.get('enabled', True)
            
            # Update initial balance if changed and no trades yet
            new_balance = self.config.get('initial_balance', 10.0)
            if new_balance != old_balance and self.metrics['total_trades'] == 0:
                self.sol_balance = new_balance
                logger.info(f"Updated initial balance to {new_balance} SOL")
            
            # Save updated state
            self.save_state()
            logger.info("Paper trading configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating paper trading config: {e}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed (enabled and sufficient balance)"""
        return self.enabled and self.sol_balance > 0
