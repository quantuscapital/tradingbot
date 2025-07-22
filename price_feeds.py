import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Represents price data for a token"""
    symbol: str
    price: float
    source: str
    timestamp: datetime
    confidence: float = 1.0  # 0-1, how reliable this price is
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None

class PriceFeedManager:
    """Manages multiple price feed sources with fallback"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, PriceData] = {}
        self.cache_duration = 30  # seconds
        
        # Price feed sources (in order of preference)
        self.sources = [
            JupiterPriceFeed(),
            CoinGeckoPriceFeed(),
            CoinMarketCapPriceFeed(),
            DexScreenerPriceFeed()
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        for source in self.sources:
            if hasattr(source, '__aenter__'):
                await source.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        for source in self.sources:
            if hasattr(source, '__aexit__'):
                await source.__aexit__(exc_type, exc_val, exc_tb)
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached price is still valid"""
        if symbol not in self.cache:
            return False
        
        price_data = self.cache[symbol]
        age = (datetime.now() - price_data.timestamp).total_seconds()
        return age < self.cache_duration
    
    async def get_price(self, symbol: str, force_refresh: bool = False) -> Optional[PriceData]:
        """Get price from best available source with caching"""
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(symbol):
            return self.cache[symbol]
        
        # Try each source until we get a price
        for source in self.sources:
            try:
                price_data = await source.get_price(symbol, self.session)
                if price_data and price_data.price > 0:
                    # Cache the result
                    self.cache[symbol] = price_data
                    return price_data
            except Exception as e:
                logger.warning(f"Price source {source.__class__.__name__} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All price sources failed for {symbol}")
        return None
    
    async def get_sol_price(self, force_refresh: bool = False) -> Optional[float]:
        """Get SOL price in USD"""
        price_data = await self.get_price("SOL", force_refresh)
        return price_data.price if price_data else None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[PriceData]]:
        """Get prices for multiple symbols concurrently"""
        tasks = [self.get_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else None
            for symbol, result in zip(symbols, results)
        }

class JupiterPriceFeed:
    """Jupiter price feed - most accurate for Solana tokens"""
    
    def __init__(self):
        self.base_url = "https://quote-api.jup.ag/v6"
        self.wsol_mint = "So11111111111111111111111111111111111111112"
        self.usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Optional[PriceData]:
        """Get price from Jupiter"""
        try:
            # Map symbols to mint addresses
            mint_map = {
                "SOL": self.wsol_mint,
                "WSOL": self.wsol_mint,
                "USDC": self.usdc_mint
            }
            
            mint = mint_map.get(symbol.upper(), symbol)
            
            url = f"{self.base_url}/price"
            params = {
                'ids': mint,
                'vsToken': 'USDC'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    price_info = data.get('data', {}).get(mint)
                    
                    if price_info and 'price' in price_info:
                        return PriceData(
                            symbol=symbol.upper(),
                            price=float(price_info['price']),
                            source="Jupiter",
                            timestamp=datetime.now(),
                            confidence=0.95
                        )
        except Exception as e:
            logger.debug(f"Jupiter price feed error for {symbol}: {e}")
        
        return None

class CoinGeckoPriceFeed:
    """CoinGecko price feed - good fallback"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.symbol_map = {
            "SOL": "solana",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDC": "usd-coin",
            "USDT": "tether"
        }
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Optional[PriceData]:
        """Get price from CoinGecko"""
        try:
            coin_id = self.symbol_map.get(symbol.upper())
            if not coin_id:
                return None
            
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    coin_data = data.get(coin_id)
                    
                    if coin_data and 'usd' in coin_data:
                        return PriceData(
                            symbol=symbol.upper(),
                            price=float(coin_data['usd']),
                            source="CoinGecko",
                            timestamp=datetime.now(),
                            confidence=0.85,
                            volume_24h=coin_data.get('usd_24h_vol'),
                            change_24h=coin_data.get('usd_24h_change')
                        )
        except Exception as e:
            logger.debug(f"CoinGecko price feed error for {symbol}: {e}")
        
        return None

class CoinMarketCapPriceFeed:
    """CoinMarketCap price feed"""
    
    def __init__(self):
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.symbol_map = {
            "SOL": "SOL",
            "BTC": "BTC",
            "ETH": "ETH",
            "USDC": "USDC",
            "USDT": "USDT"
        }
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Optional[PriceData]:
        """Get price from CoinMarketCap (requires API key)"""
        # Note: This would require an API key for production use
        # For now, we'll return None to fall back to other sources
        return None

class DexScreenerPriceFeed:
    """DexScreener price feed - good for newer tokens"""
    
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Optional[PriceData]:
        """Get price from DexScreener"""
        try:
            # For SOL specifically
            if symbol.upper() in ["SOL", "WSOL"]:
                url = f"{self.base_url}/search"
                params = {'q': 'SOL/USDC'}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        # Find SOL/USDC pair
                        for pair in pairs:
                            if (pair.get('baseToken', {}).get('symbol') == 'SOL' and
                                pair.get('quoteToken', {}).get('symbol') == 'USDC'):
                                
                                price_usd = pair.get('priceUsd')
                                if price_usd:
                                    return PriceData(
                                        symbol=symbol.upper(),
                                        price=float(price_usd),
                                        source="DexScreener",
                                        timestamp=datetime.now(),
                                        confidence=0.75,
                                        volume_24h=pair.get('volume', {}).get('h24'),
                                        change_24h=pair.get('priceChange', {}).get('h24')
                                    )
        except Exception as e:
            logger.debug(f"DexScreener price feed error for {symbol}: {e}")
        
        return None

# Global price manager instance
_price_manager = None

async def get_price_manager() -> PriceFeedManager:
    """Get global price manager instance"""
    global _price_manager
    if _price_manager is None:
        _price_manager = PriceFeedManager()
    return _price_manager

async def get_sol_price() -> Optional[float]:
    """Convenience function to get SOL price"""
    async with PriceFeedManager() as manager:
        return await manager.get_sol_price()

async def get_token_price(symbol: str) -> Optional[float]:
    """Convenience function to get token price"""
    async with PriceFeedManager() as manager:
        price_data = await manager.get_price(symbol)
        return price_data.price if price_data else None

class PriceMonitor:
    """Monitor prices with alerts and caching"""
    
    def __init__(self, price_manager: PriceFeedManager):
        self.price_manager = price_manager
        self.monitoring = False
        self.callbacks = {}
        self.price_history = {}
        
    def add_price_alert(self, symbol: str, threshold: float, callback, above: bool = True):
        """Add price alert callback
        
        Args:
            symbol: Token symbol to monitor
            threshold: Price threshold
            callback: Function to call when threshold is hit
            above: True for alert when price goes above, False for below
        """
        key = f"{symbol}_{threshold}_{above}"
        self.callbacks[key] = {
            'symbol': symbol,
            'threshold': threshold,
            'callback': callback,
            'above': above,
            'triggered': False
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start monitoring prices"""
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Get all symbols we're monitoring
                symbols = set(alert['symbol'] for alert in self.callbacks.values())
                
                if symbols:
                    prices = await self.price_manager.get_multiple_prices(list(symbols))
                    
                    for symbol, price_data in prices.items():
                        if price_data:
                            # Store price history
                            if symbol not in self.price_history:
                                self.price_history[symbol] = []
                            
                            self.price_history[symbol].append({
                                'price': price_data.price,
                                'timestamp': price_data.timestamp
                            })
                            
                            # Keep only last 100 prices
                            if len(self.price_history[symbol]) > 100:
                                self.price_history[symbol] = self.price_history[symbol][-100:]
                            
                            # Check alerts
                            await self._check_alerts(symbol, price_data.price)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Price monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _check_alerts(self, symbol: str, current_price: float):
        """Check if any price alerts should be triggered"""
        for key, alert in list(self.callbacks.items()):
            if alert['symbol'] == symbol and not alert['triggered']:
                if (alert['above'] and current_price >= alert['threshold']) or \
                   (not alert['above'] and current_price <= alert['threshold']):
                    
                    alert['triggered'] = True
                    
                    try:
                        await alert['callback'](symbol, current_price, alert['threshold'])
                    except Exception as e:
                        logger.error(f"Price alert callback error: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring prices"""
        self.monitoring = False
    
    def get_price_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent price history for a symbol"""
        return self.price_history.get(symbol, [])[-limit:]
    
    def remove_alert(self, symbol: str, threshold: float, above: bool = True):
        """Remove a price alert"""
        key = f"{symbol}_{threshold}_{above}"
        if key in self.callbacks:
            del self.callbacks[key]