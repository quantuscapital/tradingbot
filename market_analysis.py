import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    SIDEWAYS = "sideways"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"

class MomentumStrength(Enum):
    """Momentum strength classification"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

class MarketRegime(Enum):
    """Overall market condition"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"

@dataclass
class TrendAnalysis:
    """Results of trend analysis"""
    direction: TrendDirection
    strength: float  # 0-100
    confidence: float  # 0-1
    timeframe: str
    duration_hours: float
    trend_angle: float  # degrees from horizontal
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None

@dataclass
class MomentumAnalysis:
    """Results of momentum analysis"""
    strength: MomentumStrength
    velocity: float  # Price change per hour
    acceleration: float  # Change in velocity
    rsi_14: float
    volume_momentum: float
    sustainability_score: float  # 0-100

@dataclass
class FallingKnifeSignals:
    """Signals indicating a potential falling knife"""
    is_falling_knife: bool
    confidence: float  # 0-1
    signals: List[str]  # List of warning signals
    volume_divergence: bool
    recent_high_proximity: float  # 0-1, how close to recent highs
    deceleration_rate: float
    bounce_sustainability: float

@dataclass
class MarketStructure:
    """Market microstructure analysis"""
    support_levels: List[float]
    resistance_levels: List[float]
    volume_profile: Dict[str, float]
    order_flow_imbalance: float
    market_depth_score: float

@dataclass
class ComprehensiveAnalysis:
    """Complete market analysis for a token"""
    symbol: str
    current_price: float
    timestamp: datetime
    
    # Trend analysis for different timeframes
    trend_1h: TrendAnalysis
    trend_4h: TrendAnalysis
    trend_24h: TrendAnalysis
    
    # Overall trend alignment
    trend_alignment_score: float  # 0-100
    dominant_trend: TrendDirection
    
    # Momentum analysis
    momentum: MomentumAnalysis
    
    # Falling knife detection
    falling_knife: FallingKnifeSignals
    
    # Market structure
    structure: MarketStructure
    
    # Overall scores
    buy_confidence: float  # 0-100
    sell_pressure: float  # 0-100
    risk_score: float  # 0-100 (higher = riskier)
    
    # Reasoning
    buy_reasoning: List[str]
    warning_signals: List[str]

class MarketAnalyzer:
    """Advanced market analysis for crypto tokens"""
    
    def __init__(self):
        self.price_history: Dict[str, List[Dict]] = {}  # Cache for price data
        self.analysis_cache: Dict[str, ComprehensiveAnalysis] = {}
        self.cache_duration = 300  # 5 minutes
        
    def _get_price_changes(self, data: Dict) -> Dict[str, float]:
        """Extract price changes from DexScreener data"""
        price_change = data.get('priceChange', {})
        return {
            '5m': float(price_change.get('m5', 0)),
            '1h': float(price_change.get('h1', 0)),
            '6h': float(price_change.get('h6', 0)),
            '24h': float(price_change.get('h24', 0))
        }
    
    def _calculate_velocity(self, price_changes: Dict[str, float]) -> float:
        """Calculate price velocity (change per hour)"""
        if price_changes['1h'] != 0:
            return price_changes['1h']
        elif price_changes['6h'] != 0:
            return price_changes['6h'] / 6
        elif price_changes['24h'] != 0:
            return price_changes['24h'] / 24
        return 0
    
    def _calculate_acceleration(self, price_changes: Dict[str, float]) -> float:
        """Calculate price acceleration (change in velocity)"""
        try:
            # Compare short-term vs longer-term velocity
            velocity_1h = price_changes['1h']
            velocity_6h = price_changes['6h'] / 6
            
            if velocity_6h != 0:
                return (velocity_1h - velocity_6h) / velocity_6h
            return 0
        except:
            return 0
    
    def _analyze_trend_timeframe(self, price_changes: Dict[str, float], timeframe: str) -> TrendAnalysis:
        """Analyze trend for a specific timeframe"""
        change_key_map = {
            '1h': '1h',
            '4h': '6h',  # Use 6h as proxy for 4h
            '24h': '24h'
        }
        
        change_key = change_key_map.get(timeframe, '24h')
        price_change = price_changes.get(change_key, 0)
        
        # Determine trend direction and strength
        abs_change = abs(price_change)
        
        if price_change > 50:
            direction = TrendDirection.STRONG_UP
            strength = min(100, abs_change * 2)
        elif price_change > 15:
            direction = TrendDirection.WEAK_UP
            strength = min(80, abs_change * 3)
        elif price_change < -50:
            direction = TrendDirection.STRONG_DOWN
            strength = min(100, abs_change * 2)
        elif price_change < -15:
            direction = TrendDirection.WEAK_DOWN
            strength = min(80, abs_change * 3)
        else:
            direction = TrendDirection.SIDEWAYS
            strength = max(0, 50 - abs_change * 2)
        
        # Calculate confidence based on consistency with other timeframes
        confidence = min(1.0, abs_change / 100 + 0.3)
        
        # Estimate trend angle (simplified)
        hours_map = {'1h': 1, '4h': 4, '24h': 24}
        hours = hours_map.get(timeframe, 24)
        trend_angle = np.arctan(price_change / hours) * 180 / np.pi
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            timeframe=timeframe,
            duration_hours=hours,
            trend_angle=trend_angle
        )
    
    def _analyze_momentum(self, data: Dict, price_changes: Dict[str, float]) -> MomentumAnalysis:
        """Analyze momentum characteristics"""
        
        # Calculate velocity and acceleration
        velocity = self._calculate_velocity(price_changes)
        acceleration = self._calculate_acceleration(price_changes)
        
        # Get volume data
        volume_data = data.get('volume', {})
        volume_24h = float(volume_data.get('h24', 0))
        
        # Calculate volume momentum (simplified)
        volume_momentum = min(100, volume_24h / 100000)  # Normalize volume
        
        # Calculate RSI approximation using price changes
        recent_changes = [price_changes['5m'], price_changes['1h'], price_changes['6h']]
        gains = [max(0, change) for change in recent_changes]
        losses = [max(0, -change) for change in recent_changes]
        
        avg_gain = statistics.mean(gains) if gains else 0.01
        avg_loss = statistics.mean(losses) if losses else 0.01
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
        
        # Determine momentum strength
        momentum_score = abs(velocity) + volume_momentum * 0.3
        
        if momentum_score > 80:
            strength = MomentumStrength.VERY_STRONG
        elif momentum_score > 50:
            strength = MomentumStrength.STRONG
        elif momentum_score > 25:
            strength = MomentumStrength.MODERATE
        elif momentum_score > 10:
            strength = MomentumStrength.WEAK
        else:
            strength = MomentumStrength.VERY_WEAK
        
        # Calculate sustainability score
        sustainability_score = self._calculate_sustainability(velocity, acceleration, volume_momentum)
        
        return MomentumAnalysis(
            strength=strength,
            velocity=velocity,
            acceleration=acceleration,
            rsi_14=rsi_14,
            volume_momentum=volume_momentum,
            sustainability_score=sustainability_score
        )
    
    def _calculate_sustainability(self, velocity: float, acceleration: float, volume_momentum: float) -> float:
        """Calculate how sustainable current momentum is"""
        sustainability = 50  # Base score
        
        # Positive velocity with positive acceleration is good
        if velocity > 0 and acceleration > 0:
            sustainability += 30
        elif velocity > 0 and acceleration < -0.5:  # Decelerating
            sustainability -= 40
        
        # Volume momentum support
        if volume_momentum > 50:
            sustainability += 20
        elif volume_momentum < 20:
            sustainability -= 15
        
        return max(0, min(100, sustainability))
    
    def _detect_falling_knife(self, data: Dict, price_changes: Dict[str, float], momentum: MomentumAnalysis) -> FallingKnifeSignals:
        """Detect if this is a falling knife scenario"""
        
        signals = []
        is_falling_knife = False
        confidence = 0.0
        
        # Get current price and volume
        current_price = float(data.get('priceUsd', 0))
        volume_24h = float(data.get('volume', {}).get('h24', 0))
        
        # Signal 1: Rapid deceleration from high positive momentum
        if (price_changes['24h'] > 25 and 
            price_changes['1h'] < 5 and 
            momentum.acceleration < -0.3):
            signals.append("Rapid deceleration from pump")
            confidence += 0.4
        
        # Signal 2: Volume divergence (price up but volume declining)
        volume_divergence = False
        if price_changes['1h'] > 0 and momentum.volume_momentum < 30:
            volume_divergence = True
            signals.append("Volume divergence - price up, volume weak")
            confidence += 0.3
        
        # Signal 3: Very high recent gains suggest exhaustion
        if price_changes['24h'] > 100:
            signals.append("Extreme gains suggest exhaustion")
            confidence += 0.2
        
        # Signal 4: RSI overbought with momentum weakening
        if momentum.rsi_14 > 70 and momentum.sustainability_score < 40:
            signals.append("Overbought with weakening momentum")
            confidence += 0.3
        
        # Signal 5: Price action shows signs of topping
        if (price_changes['5m'] < 0 and 
            price_changes['1h'] > 10 and 
            price_changes['6h'] > 30):
            signals.append("Recent price action showing weakness")
            confidence += 0.2
        
        # Calculate deceleration rate
        deceleration_rate = abs(momentum.acceleration) if momentum.acceleration < 0 else 0
        
        # Estimate proximity to recent highs (simplified)
        recent_high_proximity = min(1.0, price_changes['24h'] / 100)
        
        # Bounce sustainability
        bounce_sustainability = momentum.sustainability_score / 100
        
        # Final determination
        if confidence > 0.6 or len(signals) >= 3:
            is_falling_knife = True
        
        return FallingKnifeSignals(
            is_falling_knife=is_falling_knife,
            confidence=min(1.0, confidence),
            signals=signals,
            volume_divergence=volume_divergence,
            recent_high_proximity=recent_high_proximity,
            deceleration_rate=deceleration_rate,
            bounce_sustainability=bounce_sustainability
        )
    
    def _analyze_market_structure(self, data: Dict, current_price: float) -> MarketStructure:
        """Analyze market microstructure"""
        
        # Calculate support/resistance levels based on recent price action
        price_24h_high = current_price * (1 + data.get('priceChange', {}).get('h24', 0) / 100)
        price_24h_low = current_price * (1 - abs(data.get('priceChange', {}).get('h24', 0)) / 200)
        
        support_levels = [price_24h_low, current_price * 0.95]
        resistance_levels = [current_price * 1.05, price_24h_high]
        
        # Volume profile analysis (simplified)
        volume_24h = float(data.get('volume', {}).get('h24', 0))
        liquidity = float(data.get('liquidity', {}).get('usd', 0))
        
        volume_profile = {
            'high_volume_node': current_price,
            'volume_24h': volume_24h,
            'liquidity': liquidity
        }
        
        # Market depth score based on liquidity
        market_depth_score = min(100, liquidity / 50000)  # Normalize
        
        return MarketStructure(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            volume_profile=volume_profile,
            order_flow_imbalance=0,  # Would need order book data
            market_depth_score=market_depth_score
        )
    
    def _calculate_trend_alignment(self, trend_1h: TrendAnalysis, trend_4h: TrendAnalysis, trend_24h: TrendAnalysis) -> Tuple[float, TrendDirection]:
        """Calculate how well trends align across timeframes"""
        
        trends = [trend_1h, trend_4h, trend_24h]
        weights = [0.2, 0.3, 0.5]  # Weight longer timeframes more
        
        # Convert trends to numeric values
        trend_values = []
        for trend in trends:
            if trend.direction == TrendDirection.STRONG_UP:
                trend_values.append(2)
            elif trend.direction == TrendDirection.WEAK_UP:
                trend_values.append(1)
            elif trend.direction == TrendDirection.SIDEWAYS:
                trend_values.append(0)
            elif trend.direction == TrendDirection.WEAK_DOWN:
                trend_values.append(-1)
            elif trend.direction == TrendDirection.STRONG_DOWN:
                trend_values.append(-2)
        
        # Calculate weighted average
        weighted_trend = sum(v * w for v, w in zip(trend_values, weights))
        
        # Calculate alignment score
        trend_variance = statistics.variance(trend_values) if len(trend_values) > 1 else 0
        alignment_score = max(0, 100 - trend_variance * 50)
        
        # Determine dominant trend
        if weighted_trend > 1.5:
            dominant = TrendDirection.STRONG_UP
        elif weighted_trend > 0.5:
            dominant = TrendDirection.WEAK_UP
        elif weighted_trend < -1.5:
            dominant = TrendDirection.STRONG_DOWN
        elif weighted_trend < -0.5:
            dominant = TrendDirection.WEAK_DOWN
        else:
            dominant = TrendDirection.SIDEWAYS
        
        return alignment_score, dominant
    
    def _generate_buy_reasoning(self, analysis: 'ComprehensiveAnalysis') -> List[str]:
        """Generate reasoning for buy recommendation"""
        reasoning = []
        
        # Trend alignment
        if analysis.trend_alignment_score > 70:
            reasoning.append(f"Strong trend alignment across timeframes ({analysis.trend_alignment_score:.0f}/100)")
        
        # Momentum strength
        if analysis.momentum.strength in [MomentumStrength.STRONG, MomentumStrength.VERY_STRONG]:
            reasoning.append(f"Strong momentum with {analysis.momentum.sustainability_score:.0f}% sustainability")
        
        # No falling knife signals
        if not analysis.falling_knife.is_falling_knife:
            reasoning.append("No falling knife signals detected")
        
        # Volume support
        if analysis.momentum.volume_momentum > 50:
            reasoning.append("Strong volume supporting price action")
        
        # Market structure
        if analysis.structure.market_depth_score > 60:
            reasoning.append("Good market depth and liquidity")
        
        return reasoning
    
    def _generate_warnings(self, analysis: 'ComprehensiveAnalysis') -> List[str]:
        """Generate warning signals"""
        warnings = []
        
        # Falling knife warnings
        if analysis.falling_knife.is_falling_knife:
            warnings.extend(analysis.falling_knife.signals)
        
        # Trend misalignment
        if analysis.trend_alignment_score < 30:
            warnings.append("Poor trend alignment across timeframes")
        
        # Weak momentum
        if analysis.momentum.strength == MomentumStrength.VERY_WEAK:
            warnings.append("Very weak momentum")
        
        # Overbought conditions
        if analysis.momentum.rsi_14 > 80:
            warnings.append("Severely overbought conditions")
        
        # Poor market structure
        if analysis.structure.market_depth_score < 30:
            warnings.append("Poor market depth and liquidity")
        
        return warnings
    
    def analyze_token(self, token_data: Dict) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis of a token"""
        
        pair = token_data.get('pair', {})
        symbol = pair.get('baseToken', {}).get('symbol', 'Unknown')
        current_price = float(pair.get('priceUsd', 0))
        
        # Extract price changes
        price_changes = self._get_price_changes(pair)
        
        # Analyze trends for different timeframes
        trend_1h = self._analyze_trend_timeframe(price_changes, '1h')
        trend_4h = self._analyze_trend_timeframe(price_changes, '4h')
        trend_24h = self._analyze_trend_timeframe(price_changes, '24h')
        
        # Calculate trend alignment
        trend_alignment_score, dominant_trend = self._calculate_trend_alignment(trend_1h, trend_4h, trend_24h)
        
        # Analyze momentum
        momentum = self._analyze_momentum(pair, price_changes)
        
        # Detect falling knife
        falling_knife = self._detect_falling_knife(pair, price_changes, momentum)
        
        # Analyze market structure
        structure = self._analyze_market_structure(pair, current_price)
        
        # Calculate overall scores
        buy_confidence = self._calculate_buy_confidence(trend_alignment_score, momentum, falling_knife, structure)
        sell_pressure = self._calculate_sell_pressure(falling_knife, momentum)
        risk_score = self._calculate_risk_score(falling_knife, momentum, structure)
        
        # Create comprehensive analysis
        analysis = ComprehensiveAnalysis(
            symbol=symbol,
            current_price=current_price,
            timestamp=datetime.now(),
            trend_1h=trend_1h,
            trend_4h=trend_4h,
            trend_24h=trend_24h,
            trend_alignment_score=trend_alignment_score,
            dominant_trend=dominant_trend,
            momentum=momentum,
            falling_knife=falling_knife,
            structure=structure,
            buy_confidence=buy_confidence,
            sell_pressure=sell_pressure,
            risk_score=risk_score,
            buy_reasoning=[],
            warning_signals=[]
        )
        
        # Generate reasoning and warnings
        analysis.buy_reasoning = self._generate_buy_reasoning(analysis)
        analysis.warning_signals = self._generate_warnings(analysis)
        
        return analysis
    
    def _calculate_buy_confidence(self, trend_alignment: float, momentum: MomentumAnalysis, 
                                falling_knife: FallingKnifeSignals, structure: MarketStructure) -> float:
        """Calculate overall buy confidence score"""
        
        confidence = 0
        
        # Trend alignment (40% weight)
        confidence += trend_alignment * 0.4
        
        # Momentum strength (30% weight)
        momentum_score = momentum.sustainability_score
        confidence += momentum_score * 0.3
        
        # No falling knife (20% weight)
        if not falling_knife.is_falling_knife:
            confidence += 20
        else:
            confidence -= falling_knife.confidence * 30
        
        # Market structure (10% weight)
        confidence += structure.market_depth_score * 0.1
        
        return max(0, min(100, confidence))
    
    def _calculate_sell_pressure(self, falling_knife: FallingKnifeSignals, momentum: MomentumAnalysis) -> float:
        """Calculate sell pressure score"""
        
        pressure = 0
        
        if falling_knife.is_falling_knife:
            pressure += falling_knife.confidence * 50
        
        if momentum.rsi_14 > 70:
            pressure += (momentum.rsi_14 - 70) * 2
        
        if momentum.acceleration < -0.3:
            pressure += 20
        
        return min(100, pressure)
    
    def _calculate_risk_score(self, falling_knife: FallingKnifeSignals, momentum: MomentumAnalysis, 
                            structure: MarketStructure) -> float:
        """Calculate overall risk score"""
        
        risk = 30  # Base risk
        
        if falling_knife.is_falling_knife:
            risk += falling_knife.confidence * 40
        
        if momentum.sustainability_score < 30:
            risk += 20
        
        if structure.market_depth_score < 30:
            risk += 15
        
        if momentum.rsi_14 > 80:
            risk += 15
        
        return min(100, risk)
    
    def should_avoid_token(self, analysis: ComprehensiveAnalysis) -> Tuple[bool, str]:
        """Determine if token should be avoided and why"""
        
        # Critical avoidance criteria
        if analysis.falling_knife.is_falling_knife and analysis.falling_knife.confidence > 0.7:
            return True, f"High probability falling knife ({analysis.falling_knife.confidence:.1%} confidence)"
        
        if analysis.risk_score > 80:
            return True, f"Extremely high risk score ({analysis.risk_score:.0f}/100)"
        
        if analysis.buy_confidence < 20:
            return True, f"Very low buy confidence ({analysis.buy_confidence:.0f}/100)"
        
        if len(analysis.warning_signals) >= 4:
            return True, f"Too many warning signals ({len(analysis.warning_signals)})"
        
        return False, ""
    
    def get_entry_timing_recommendation(self, analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Get recommendations for entry timing"""
        
        recommendations = {
            'immediate_buy': False,
            'wait_for_pullback': False,
            'avoid': False,
            'patience_score': 0,  # 0-100, higher means wait longer
            'reasoning': []
        }
        
        # Check if we should avoid completely
        should_avoid, avoid_reason = self.should_avoid_token(analysis)
        if should_avoid:
            recommendations['avoid'] = True
            recommendations['reasoning'].append(avoid_reason)
            return recommendations
        
        # Strong confidence with good momentum - can buy immediately
        if (analysis.buy_confidence > 70 and 
            analysis.momentum.sustainability_score > 60 and
            not analysis.falling_knife.is_falling_knife):
            recommendations['immediate_buy'] = True
            recommendations['reasoning'].append("Strong confidence with sustainable momentum")
            return recommendations
        
        # Good setup but wait for better entry
        if analysis.buy_confidence > 50:
            recommendations['wait_for_pullback'] = True
            recommendations['patience_score'] = min(100, (100 - analysis.buy_confidence) * 2)
            recommendations['reasoning'].append("Good setup but patience recommended for better entry")
        
        return recommendations

# Global analyzer instance
_market_analyzer = None

def get_market_analyzer() -> MarketAnalyzer:
    """Get global market analyzer instance"""
    global _market_analyzer
    if _market_analyzer is None:
        _market_analyzer = MarketAnalyzer()
    return _market_analyzer

def analyze_token_comprehensive(token_data: Dict) -> ComprehensiveAnalysis:
    """Convenience function for comprehensive token analysis"""
    analyzer = get_market_analyzer()
    return analyzer.analyze_token(token_data)