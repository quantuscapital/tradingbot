import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class SwapQuote:
    """Represents a swap quote from Jupiter"""
    input_mint: str
    output_mint: str
    input_amount: int  # In smallest unit (lamports for SOL)
    output_amount: int  # In smallest unit
    other_amount_threshold: int
    swap_mode: str
    slippage_bps: int
    platform_fee: Optional[Dict] = None
    price_impact_pct: float = 0.0
    route_plan: List[Dict] = None
    context_slot: Optional[int] = None
    time_taken: Optional[float] = None

@dataclass
class SwapTransaction:
    """Represents a swap transaction from Jupiter"""
    swap_transaction: str  # Base64 encoded transaction
    last_valid_block_height: int
    priority_fee_lamports: Optional[int] = None

class JupiterClient:
    """Client for Jupiter DEX aggregator API"""
    
    def __init__(self, base_url: str = "https://quote-api.jup.ag/v6"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Common Solana token mints
        self.WSOL_MINT = "So11111111111111111111111111111111111111112"  # Wrapped SOL
        self.USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        self.USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"  # USDT
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to Jupiter API"""
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Jupiter API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Jupiter API request failed: {str(e)}")
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,  # 0.5% default slippage
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False
    ) -> SwapQuote:
        """Get a swap quote from Jupiter
        
        Args:
            input_mint: The mint address of input token
            output_mint: The mint address of output token  
            amount: Amount in smallest unit (lamports for SOL)
            slippage_bps: Slippage in basis points (50 = 0.5%)
            swap_mode: "ExactIn" or "ExactOut"
            only_direct_routes: Only use direct routes (faster but potentially worse price)
            as_legacy_transaction: Use legacy transaction format
        """
        params = {
            'inputMint': input_mint,
            'outputMint': output_mint,
            'amount': str(amount),
            'slippageBps': str(slippage_bps),
            'swapMode': swap_mode,
            'onlyDirectRoutes': str(only_direct_routes).lower(),
            'asLegacyTransaction': str(as_legacy_transaction).lower()
        }
        
        response = await self._make_request('GET', '/quote', params=params)
        
        return SwapQuote(
            input_mint=response['inputMint'],
            output_mint=response['outputMint'],
            input_amount=int(response['inAmount']),
            output_amount=int(response['outAmount']),
            other_amount_threshold=int(response['otherAmountThreshold']),
            swap_mode=response['swapMode'],
            slippage_bps=int(response['slippageBps']),
            platform_fee=response.get('platformFee'),
            price_impact_pct=float(response.get('priceImpactPct', 0)),
            route_plan=response.get('routePlan', []),
            context_slot=response.get('contextSlot'),
            time_taken=response.get('timeTaken')
        )
    
    async def get_swap_transaction(
        self,
        quote: SwapQuote,
        user_public_key: str,
        wrap_and_unwrap_sol: bool = True,
        use_shared_accounts: bool = True,
        fee_account: Optional[str] = None,
        tracking_account: Optional[str] = None,
        as_legacy_transaction: bool = False,
        use_token_ledger: bool = False,
        destination_token_account: Optional[str] = None,
        dynamic_compute_unit_limit: bool = False,
        skip_user_accounts_rpc_calls: bool = False
    ) -> SwapTransaction:
        """Get a swap transaction from Jupiter
        
        Args:
            quote: The swap quote to execute
            user_public_key: The user's wallet public key
            wrap_and_unwrap_sol: Automatically wrap/unwrap SOL
            use_shared_accounts: Use shared token accounts
            fee_account: Fee account for platform fees
            tracking_account: Tracking account for analytics
            as_legacy_transaction: Use legacy transaction format
            use_token_ledger: Use token ledger for token accounts
            destination_token_account: Specific destination token account
            dynamic_compute_unit_limit: Use dynamic compute unit limits
            skip_user_accounts_rpc_calls: Skip RPC calls for user accounts
        """
        payload = {
            'quoteResponse': {
                'inputMint': quote.input_mint,
                'inAmount': str(quote.input_amount),
                'outputMint': quote.output_mint,
                'outAmount': str(quote.output_amount),
                'otherAmountThreshold': str(quote.other_amount_threshold),
                'swapMode': quote.swap_mode,
                'slippageBps': quote.slippage_bps,
                'platformFee': quote.platform_fee,
                'priceImpactPct': str(quote.price_impact_pct),
                'routePlan': quote.route_plan,
                'contextSlot': quote.context_slot,
                'timeTaken': quote.time_taken
            },
            'userPublicKey': user_public_key,
            'wrapAndUnwrapSol': wrap_and_unwrap_sol,
            'useSharedAccounts': use_shared_accounts,
            'asLegacyTransaction': as_legacy_transaction,
            'useTokenLedger': use_token_ledger,
            'dynamicComputeUnitLimit': dynamic_compute_unit_limit,
            'skipUserAccountsRpcCalls': skip_user_accounts_rpc_calls
        }
        
        if fee_account:
            payload['feeAccount'] = fee_account
        if tracking_account:
            payload['trackingAccount'] = tracking_account
        if destination_token_account:
            payload['destinationTokenAccount'] = destination_token_account
        
        response = await self._make_request('POST', '/swap', json=payload)
        
        return SwapTransaction(
            swap_transaction=response['swapTransaction'],
            last_valid_block_height=response['lastValidBlockHeight'],
            priority_fee_lamports=response.get('priorityFeeInLamports')
        )
    
    async def get_tokens_list(self) -> List[Dict[str, Any]]:
        """Get list of all supported tokens"""
        return await self._make_request('GET', '/tokens')
    
    async def get_token_price(self, mint: str, vs_token: str = "USDC") -> Optional[Dict[str, Any]]:
        """Get token price
        
        Args:
            mint: Token mint address
            vs_token: Token to price against (USDC, SOL, etc.)
        """
        try:
            params = {
                'ids': mint,
                'vsToken': vs_token
            }
            response = await self._make_request('GET', '/price', params=params)
            return response.get('data', {}).get(mint)
        except Exception as e:
            logger.warning(f"Failed to get price for {mint}: {e}")
            return None
    
    async def get_sol_price(self) -> Optional[float]:
        """Get SOL price in USD"""
        try:
            price_data = await self.get_token_price(self.WSOL_MINT, "USDC")
            if price_data and 'price' in price_data:
                return float(price_data['price'])
        except Exception as e:
            logger.warning(f"Failed to get SOL price: {e}")
        return None
    
    def lamports_to_sol(self, lamports: int) -> float:
        """Convert lamports to SOL"""
        return lamports / 1_000_000_000
    
    def sol_to_lamports(self, sol: float) -> int:
        """Convert SOL to lamports"""
        return int(sol * 1_000_000_000)
    
    def calculate_minimum_received(self, output_amount: int, slippage_bps: int) -> int:
        """Calculate minimum tokens received after slippage"""
        slippage_multiplier = (10000 - slippage_bps) / 10000
        return int(output_amount * slippage_multiplier)
    
    def calculate_price_impact(self, input_amount_usd: float, output_amount_usd: float) -> float:
        """Calculate price impact percentage"""
        if input_amount_usd <= 0:
            return 0.0
        return ((input_amount_usd - output_amount_usd) / input_amount_usd) * 100
    
    async def find_best_route(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        max_slippage_bps: int = 50
    ) -> Optional[SwapQuote]:
        """Find the best route for a swap with multiple slippage tolerances"""
        slippage_levels = [10, 25, 50, 100]  # 0.1%, 0.25%, 0.5%, 1%
        
        best_quote = None
        best_output = 0
        
        for slippage in slippage_levels:
            if slippage > max_slippage_bps:
                continue
                
            try:
                quote = await self.get_quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=amount,
                    slippage_bps=slippage
                )
                
                if quote.output_amount > best_output:
                    best_output = quote.output_amount
                    best_quote = quote
                    
            except Exception as e:
                logger.warning(f"Failed to get quote with {slippage} bps slippage: {e}")
                continue
        
        return best_quote
    
    async def simulate_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> Dict[str, Any]:
        """Simulate a swap and return detailed information"""
        try:
            quote = await self.get_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                slippage_bps=slippage_bps
            )
            
            minimum_received = self.calculate_minimum_received(quote.output_amount, slippage_bps)
            
            # Get token prices for USD calculations
            input_price = await self.get_token_price(input_mint)
            output_price = await self.get_token_price(output_mint)
            
            simulation = {
                'quote': quote,
                'minimum_received': minimum_received,
                'price_impact_pct': quote.price_impact_pct,
                'estimated_gas': None,  # Would need RPC call to estimate
                'route_summary': self._summarize_route(quote.route_plan),
                'input_price_usd': input_price.get('price', 0) if input_price else 0,
                'output_price_usd': output_price.get('price', 0) if output_price else 0,
            }
            
            return simulation
            
        except Exception as e:
            logger.error(f"Swap simulation failed: {e}")
            raise
    
    def _summarize_route(self, route_plan: List[Dict]) -> str:
        """Summarize the swap route"""
        if not route_plan:
            return "Direct swap"
        
        steps = []
        for step in route_plan:
            if 'swapInfo' in step:
                swap_info = step['swapInfo']
                amm_key = swap_info.get('ammKey', 'Unknown')
                label = swap_info.get('label', amm_key[:8])
                steps.append(label)
        
        return " → ".join(steps) if steps else "Direct swap"

# Utility functions
async def get_jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = 50
) -> Optional[SwapQuote]:
    """Convenience function to get a Jupiter quote"""
    async with JupiterClient() as client:
        return await client.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=slippage_bps
        )

async def get_sol_price_usd() -> Optional[float]:
    """Convenience function to get SOL price in USD"""
    async with JupiterClient() as client:
        return await client.get_sol_price()