import asyncio
import base58
import logging
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass

try:
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Commitment
    from solana.rpc.types import TxOpts
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    from solana.transaction import Transaction
    from solana.system_program import transfer, TransferParams
    SOLANA_AVAILABLE = True
except ImportError:
    # Fallback when solana-py is not available
    AsyncClient = None
    Commitment = None
    TxOpts = None
    Keypair = None
    PublicKey = None
    Transaction = None
    transfer = None
    TransferParams = None
    SOLANA_AVAILABLE = False

try:
    from spl.token.async_client import AsyncToken
    from spl.token.constants import TOKEN_PROGRAM_ID
    from spl.token.instructions import get_associated_token_address
    SPL_AVAILABLE = True
except ImportError:
    AsyncToken = None
    TOKEN_PROGRAM_ID = None
    get_associated_token_address = None
    SPL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TokenBalance:
    """Represents a token balance"""
    mint: str
    amount: int  # Raw amount in smallest unit
    decimals: int
    ui_amount: float  # Human readable amount
    symbol: Optional[str] = None
    name: Optional[str] = None

@dataclass
class TransactionResult:
    """Result of a transaction"""
    signature: str
    success: bool
    error: Optional[str] = None
    slot: Optional[int] = None
    confirmation_status: Optional[str] = None

class SolanaWallet:
    """Solana wallet operations using private key"""
    
    def __init__(self, private_key: str, rpc_url: str = None):
        """Initialize wallet with private key and RPC endpoint
        
        Args:
            private_key: Base58 encoded private key
            rpc_url: Custom RPC URL, defaults to public mainnet
        """
        if not SOLANA_AVAILABLE:
            raise ImportError("solana-py is required for live trading. Install with: pip install solana")
        
        self.private_key_str = private_key
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        
        # Initialize keypair from private key
        try:
            private_key_bytes = base58.b58decode(private_key)
            self.keypair = Keypair.from_secret_key(private_key_bytes)
            self.public_key = self.keypair.public_key
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")
        
        # Initialize RPC client
        self.client: Optional[AsyncClient] = None
        
        # Common token mints
        self.WSOL_MINT = PublicKey("So11111111111111111111111111111111111111112")
        self.USDC_MINT = PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        self.USDT_MINT = PublicKey("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Connect to Solana RPC"""
        if self.client is None:
            self.client = AsyncClient(self.rpc_url)
        
        # Test connection
        try:
            await self.client.get_slot()
            logger.info(f"Connected to Solana RPC: {self.rpc_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Solana RPC: {e}")
            raise
    
    async def close(self):
        """Close RPC connection"""
        if self.client:
            await self.client.close()
    
    async def get_sol_balance(self) -> float:
        """Get SOL balance in SOL (not lamports)"""
        try:
            response = await self.client.get_balance(self.public_key)
            if response.value is not None:
                return response.value / 1_000_000_000  # Convert lamports to SOL
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get SOL balance: {e}")
            return 0.0
    
    async def get_token_balance(self, mint_address: str) -> Optional[TokenBalance]:
        """Get balance of a specific SPL token
        
        Args:
            mint_address: Token mint address as string
        """
        try:
            mint_pubkey = PublicKey(mint_address)
            token_account = get_associated_token_address(self.public_key, mint_pubkey)
            
            # Check if token account exists
            account_info = await self.client.get_account_info(token_account)
            if account_info.value is None:
                # Account doesn't exist, balance is 0
                return TokenBalance(
                    mint=mint_address,
                    amount=0,
                    decimals=0,
                    ui_amount=0.0
                )
            
            # Get token account balance
            response = await self.client.get_token_account_balance(token_account)
            if response.value:
                return TokenBalance(
                    mint=mint_address,
                    amount=int(response.value.amount),
                    decimals=response.value.decimals,
                    ui_amount=float(response.value.ui_amount or 0)
                )
            
            return None
        except Exception as e:
            logger.error(f"Failed to get token balance for {mint_address}: {e}")
            return None
    
    async def get_all_token_balances(self) -> List[TokenBalance]:
        """Get all non-zero SPL token balances"""
        try:
            response = await self.client.get_token_accounts_by_owner(
                self.public_key,
                {"programId": TOKEN_PROGRAM_ID},
                commitment=Commitment("confirmed")
            )
            
            balances = []
            for account in response.value:
                try:
                    # Parse token account data
                    account_data = account.account.data
                    if len(account_data) >= 64:  # Minimum token account size
                        # Extract mint (first 32 bytes after account type)
                        mint_bytes = account_data[0:32]
                        mint_pubkey = PublicKey(mint_bytes)
                        
                        # Get balance from RPC
                        balance_response = await self.client.get_token_account_balance(
                            PublicKey(account.pubkey)
                        )
                        
                        if balance_response.value and balance_response.value.ui_amount > 0:
                            balances.append(TokenBalance(
                                mint=str(mint_pubkey),
                                amount=int(balance_response.value.amount),
                                decimals=balance_response.value.decimals,
                                ui_amount=float(balance_response.value.ui_amount)
                            ))
                except Exception as e:
                    logger.warning(f"Failed to parse token account: {e}")
                    continue
            
            return balances
        except Exception as e:
            logger.error(f"Failed to get all token balances: {e}")
            return []
    
    async def send_sol(self, to_address: str, amount_sol: float) -> Optional[TransactionResult]:
        """Send SOL to another address
        
        Args:
            to_address: Recipient address
            amount_sol: Amount in SOL
        """
        try:
            to_pubkey = PublicKey(to_address)
            lamports = int(amount_sol * 1_000_000_000)
            
            # Create transfer transaction
            transaction = Transaction()
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=self.public_key,
                    to_pubkey=to_pubkey,
                    lamports=lamports
                )
            )
            transaction.add(transfer_instruction)
            
            # Send transaction
            response = await self.client.send_transaction(
                transaction,
                self.keypair,
                opts=TxOpts(skip_confirmation=False, preflight_commitment=Commitment("confirmed"))
            )
            
            return TransactionResult(
                signature=str(response.value),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send SOL: {e}")
            return TransactionResult(
                signature="",
                success=False,
                error=str(e)
            )
    
    async def send_transaction(self, transaction: Transaction) -> Optional[TransactionResult]:
        """Send a pre-built transaction
        
        Args:
            transaction: Solana transaction to send
        """
        try:
            # Sign and send transaction
            response = await self.client.send_transaction(
                transaction,
                self.keypair,
                opts=TxOpts(skip_confirmation=False, preflight_commitment=Commitment("confirmed"))
            )
            
            signature = str(response.value)
            
            # Wait for confirmation
            confirmation = await self.client.confirm_transaction(signature)
            
            return TransactionResult(
                signature=signature,
                success=not confirmation.value[0].err,
                error=str(confirmation.value[0].err) if confirmation.value[0].err else None,
                slot=confirmation.value[0].slot,
                confirmation_status="confirmed"
            )
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            return TransactionResult(
                signature="",
                success=False,
                error=str(e)
            )
    
    async def simulate_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Simulate a transaction before sending
        
        Args:
            transaction: Transaction to simulate
        """
        try:
            response = await self.client.simulate_transaction(transaction)
            
            return {
                'success': not response.value.err,
                'error': str(response.value.err) if response.value.err else None,
                'logs': response.value.logs,
                'units_consumed': response.value.units_consumed,
                'accounts': response.value.accounts
            }
            
        except Exception as e:
            logger.error(f"Failed to simulate transaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'logs': [],
                'units_consumed': None,
                'accounts': []
            }
    
    async def get_recent_blockhash(self) -> Optional[str]:
        """Get recent blockhash for transactions"""
        try:
            response = await self.client.get_recent_blockhash()
            return str(response.value.blockhash)
        except Exception as e:
            logger.error(f"Failed to get recent blockhash: {e}")
            return None
    
    async def get_transaction_status(self, signature: str) -> Dict[str, Any]:
        """Get transaction status and details
        
        Args:
            signature: Transaction signature
        """
        try:
            response = await self.client.get_signature_statuses([signature])
            
            if response.value and response.value[0]:
                status = response.value[0]
                return {
                    'confirmed': True,
                    'slot': status.slot,
                    'confirmations': status.confirmations,
                    'error': str(status.err) if status.err else None,
                    'confirmation_status': status.confirmation_status
                }
            else:
                return {
                    'confirmed': False,
                    'slot': None,
                    'confirmations': None,
                    'error': None,
                    'confirmation_status': None
                }
                
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {
                'confirmed': False,
                'error': str(e)
            }
    
    async def wait_for_confirmation(
        self, 
        signature: str, 
        timeout: int = 60,
        commitment: str = "confirmed"
    ) -> bool:
        """Wait for transaction confirmation
        
        Args:
            signature: Transaction signature to wait for
            timeout: Timeout in seconds
            commitment: Confirmation commitment level
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                status = await self.get_transaction_status(signature)
                if status['confirmed'] and not status.get('error'):
                    return True
                elif status.get('error'):
                    logger.error(f"Transaction failed: {status['error']}")
                    return False
                
                # Wait before checking again
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error checking transaction status: {e}")
                await asyncio.sleep(2)
        
        logger.warning(f"Transaction confirmation timeout: {signature}")
        return False
    
    def lamports_to_sol(self, lamports: int) -> float:
        """Convert lamports to SOL"""
        return lamports / 1_000_000_000
    
    def sol_to_lamports(self, sol: float) -> int:
        """Convert SOL to lamports"""
        return int(sol * 1_000_000_000)
    
    async def get_minimum_balance_for_rent_exemption(self, data_length: int = 0) -> int:
        """Get minimum balance needed for rent exemption
        
        Args:
            data_length: Size of account data in bytes
        """
        try:
            response = await self.client.get_minimum_balance_for_rent_exemption(data_length)
            return response.value
        except Exception as e:
            logger.error(f"Failed to get minimum balance for rent exemption: {e}")
            return 890880  # Default minimum for empty account
    
    async def check_wallet_validity(self) -> Dict[str, Any]:
        """Check if wallet is valid and get basic info"""
        try:
            # Test connection
            await self.connect()
            
            # Get basic account info
            sol_balance = await self.get_sol_balance()
            account_info = await self.client.get_account_info(self.public_key)
            
            return {
                'valid': True,
                'public_key': str(self.public_key),
                'sol_balance': sol_balance,
                'account_exists': account_info.value is not None,
                'rpc_connected': True,
                'rpc_url': self.rpc_url
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'public_key': str(self.public_key) if hasattr(self, 'public_key') else None,
                'rpc_connected': False,
                'rpc_url': self.rpc_url
            }

# Utility functions
async def validate_private_key(private_key: str, rpc_url: str = None) -> Dict[str, Any]:
    """Validate a private key and return wallet info"""
    try:
        async with SolanaWallet(private_key, rpc_url) as wallet:
            return await wallet.check_wallet_validity()
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

async def get_wallet_balances(private_key: str, rpc_url: str = None) -> Dict[str, Any]:
    """Get all balances for a wallet"""
    try:
        async with SolanaWallet(private_key, rpc_url) as wallet:
            sol_balance = await wallet.get_sol_balance()
            token_balances = await wallet.get_all_token_balances()
            
            return {
                'success': True,
                'sol_balance': sol_balance,
                'token_balances': [
                    {
                        'mint': balance.mint,
                        'amount': balance.amount,
                        'ui_amount': balance.ui_amount,
                        'decimals': balance.decimals,
                        'symbol': balance.symbol,
                        'name': balance.name
                    }
                    for balance in token_balances
                ]
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }