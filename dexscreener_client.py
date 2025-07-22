import requests
import asyncio
import httpx
from typing import Dict, List, Optional

class DexScreenerClient:
    BASE_URL = "https://api.dexscreener.com/latest/dex/tokens/"
    TOKENS_API_URL = "https://api.dexscreener.com/tokens/v1/solana/"
    
    def __init__(self):
        self.session = requests.Session()
    
    async def get_token_prices(self, token_addresses: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple tokens using the new API endpoint"""
        if not token_addresses:
            return {}
        
        try:
            # Process addresses in batches of 30 (API limit)
            prices = {}
            for i in range(0, len(token_addresses), 30):
                batch = token_addresses[i:i + 30]
                addresses_str = ','.join(batch)
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.TOKENS_API_URL}{addresses_str}")
                    response.raise_for_status()
                    data = response.json()
                    
                    # Process the response - it should be a list of token data
                    if isinstance(data, list):
                        for token_data in data:
                            if isinstance(token_data, dict):
                                base_token = token_data.get('baseToken', {})
                                token_address = base_token.get('address')
                                price_usd = token_data.get('priceUsd')
                                
                                if token_address and price_usd:
                                    try:
                                        prices[token_address] = float(price_usd)
                                    except (ValueError, TypeError):
                                        continue
                
                # Rate limiting - respect 300 requests per minute
                await asyncio.sleep(0.2)
            
            return prices
            
        except Exception as e:
            print(f"Error fetching token prices: {e}")
            return {}
    
    def get_token_info(self, token_address: str) -> Optional[Dict]:
        """Fetch token information from DexScreener API"""
        try:
            response = self.session.get(f"{self.BASE_URL}{token_address}")
            response.raise_for_status()
            data = response.json()

            # Ensure 'pairs' exists, is a list, and is not empty before accessing the first element
            pairs_list = data.get("pairs")
            if not isinstance(pairs_list, list) or not pairs_list:
                print(f"No valid pairs found for token {token_address}")
                return None

            # Get the first pair (most relevant)
            pair = pairs_list[0]
            
            # Extract and validate websites and socials
            info = pair.get("info", {})
            
            # Process websites - only include valid URLs
            websites = []
            for website in info.get("websites", []):
                if isinstance(website, dict) and website.get("url"):
                    url = website["url"]
                    if url.startswith(("http://", "https://")):
                        websites.append(url)
            
            # Process socials - ensure proper structure
            socials = []
            for social in info.get("socials", []):
                if isinstance(social, dict):
                    platform = social.get("platform", "").strip()
                    handle = social.get("handle", "").strip()
                    url = social.get("url", "").strip()
                    
                    # Only include if we have valid data
                    if platform and (handle or url):
                        socials.append({
                            "platform": platform,
                            "handle": handle,
                            "url": url if url.startswith(("http://", "https://")) else ""
                        })
            
            return {
                "websites": websites,
                "socials": socials,
                "pair_url": pair.get("url"),
                "image_url": info.get("imageUrl")
            }
            
        except requests.RequestException as e:
            print(f"Error fetching token info: {e}")
            return None

def get_token_links(token_address: str) -> Dict:
    """Get websites and socials for a token"""
    client = DexScreenerClient()
    info = client.get_token_info(token_address)
    
    if not info:
        return {"websites": [], "socials": []}
        
    return {
        "websites": info["websites"],
        "socials": info["socials"]
    }
