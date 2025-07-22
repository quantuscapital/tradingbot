import re
import logging
from datetime import datetime
from dexscreener_client import get_token_links

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def escape_markdown_v2_custom(text):
    """
    Escapes special characters for Markdown V2 while preserving link formatting.
    """
    if not text:
        return ""
        
    text = str(text)
    
    # Characters that need escaping in MarkdownV2 (including . which needs escaping)
    special_chars = '_*[]()~>#+-=|{}.!\\'  # Includes . but excludes backtick
    
    # Regex pattern to match markdown links
    link_pattern = r'(\[.*?\]\(.*?\))'
    
    # Split text into parts, preserving links
    parts = re.split(link_pattern, text)
    
    escaped_text = ''
    for part in parts:
        # Process markdown links to escape display text
        link_match = re.match(r'\[(.*?)\]\((.*?)\)', part)
        if link_match:
            display_text = link_match.group(1)
            url = link_match.group(2)
            
            # Escape special characters in display text
            escaped_display = ''
            for char in display_text:
                if char in special_chars:
                    escaped_display += f'\\{char}'
                else:
                    escaped_display += char
            
            escaped_text += f'[{escaped_display}]({url})'
        else:
            # Escape non-link text
            for char in part:
                if char in special_chars:
                    escaped_text += f'\\{char}'
                else:
                    escaped_text += char
    
    logger.debug(f"Final escaped text: {escaped_text}")
    return escaped_text

def format_new_token_alert(token_data):
    """
    Formats new token alert in the specified style.
    Only processes Solana network tokens.
    """
    if not token_data:
        return "No token data available."
        
    # Check if token is on Solana network
    chain_id = token_data.get('chain_id', '').lower()
    if chain_id != 'solana':
        logger.info(f"Skipping non-Solana token: {token_data.get('name', 'Unknown')}")
        return None
    
    # Get token details
    name = token_data.get('name', 'Unknown')
    ticker = token_data.get('ticker', 'Unknown')
    address = token_data.get('address', 'Unknown')
    chain_id = token_data.get('chain_id', 'Unknown')
    dex_id = token_data.get('dex_id', 'Unknown')
    market_cap = token_data.get('market_cap', '0')
    liquidity = token_data.get('liquidity', '0')
    volume = token_data.get('volume', '0')
    buys = token_data.get('buys', '0')
    sells = token_data.get('sells', '0')
    price_change = token_data.get('price_change', '0')
    initial_price = token_data.get('initial_price', '0')
    created_at = token_data.get('created_at', None)
    dexscreener_url = token_data.get('dexscreener_url', '#')
    
    # Get token links from DexScreener
    links = get_token_links(token_data.get('address', ''))
    websites = links.get('websites', [])
    socials = links.get('socials', [])
    
    # Format creation time - handle both string and timestamp formats
    if created_at:
        try:
            if isinstance(created_at, str):
                # If it's already a formatted string, use as-is
                creation_info = f"\n• Created: {created_at}"
            else:
                # Handle timestamp in milliseconds
                timestamp = int(created_at) if created_at else 0
                if timestamp > 0:
                    # Convert from milliseconds to seconds
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    creation_info = f"\n• Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    creation_info = ""
        except Exception as e:
            logger.error(f"Error formatting creation time: {str(e)}")
            creation_info = ""
    else:
        creation_info = ""
    
    # Format websites and socials using proper markdown V2 syntax
    website_links = []
    for url in websites:
        if url:
            # Use raw markdown V2 link format
            website_links.append(f"[Website]({url})")
    
    social_links = []
    if socials:
        for social in socials:
            platform = social.get('platform', '')
            handle = social.get('handle', '')
            url = social.get('url', '')
            if platform and handle:
                if url:
                    # Use raw markdown V2 link format
                    social_links.append(f"• [{platform}]({url})")
                else:
                    # Just show platform and handle without link
                    social_links.append(f"• {platform}: {handle}")
    
    # Helper function to format large numbers
    def format_number(num):
        try:
            num = float(num)
            if num >= 1000:
                return f"{num/1000:.1f}K".rstrip('0').rstrip('.') if num < 10000 else f"{round(num/1000)}K"
            return str(round(num))
        except:
            return num

    # Truncate address for display
    truncated_address = address

    # Enhanced number formatting with M for millions
    def format_number(num):
        try:
            num = float(num)
            if num >= 1_000_000:
                return f"{num/1_000_000:.1f}M".rstrip('0').rstrip('.')
            if num >= 1000:
                return f"{num/1000:.1f}K".rstrip('0').rstrip('.') if num < 10_000 else f"{round(num/1000)}K"
            return str(round(num))
        except:
            return num

    # Build the message with compact multi-column layout
    website_link = website_links[0] if website_links else ""
    message = f"""🚨 *{name.upper()} ({ticker})* 🚨

*Token Details:*
🌐 {chain_id.capitalize()} | 🏦 {dex_id.capitalize()}
🔗 [`{truncated_address}`](tg://copy?text={address}) | [Explorer](https://explorer.solana.com/address/{address}){creation_info}

*Market Data:*
📈 ${format_number(market_cap)} | 💧 ${format_number(liquidity)} | 📊 ${format_number(volume)}
🛒 {format_number(buys)}/{format_number(sells)} | 🚀 {'-' if float(price_change) < 0 else '+'}{abs(float(price_change)):.0f}% | 🏁 ${float(initial_price):.6f}"""
    
    # Add chart link to websites section first
    if dexscreener_url and dexscreener_url != '#':
        website_links.insert(0, f"[Chart]({dexscreener_url})")
    
    # Add website section only if there are websites
    if website_links:
        message += "\n\n" + "\n".join(website_links)
    
    # Add socials section only if there are socials
    if social_links:
        message += "\n\n*Socials:*\n" + "\n".join(social_links)
    
    return escape_markdown_v2_custom(message)

def format_compact_analysis(data):
    """
    Formats analysis in compact, emoji-rich style with links.
    """
    if not data:
        return "No analysis data available."
    
    # Get raw values
    price = data.get('price', '0.0000')
    liquidity = data.get('liquidity', '0')
    volume = data.get('volume', '0')
    ratio = f"{float(data.get('volume_liquidity_ratio', 0)):.2f}"
    top_holders = f"{float(data.get('top_holders_percent', 0)):.1f}"
    holder1 = f"{float(data.get('holder1_percent', 0)):.1f}"
    holder2 = f"{float(data.get('holder2_percent', 0)):.1f}"
    holder3 = f"{float(data.get('holder3_percent', 0)):.1f}"
    dexscreener_url = data.get('dexscreener_url', '#')
    rank_url = data.get('rank_url', '#')
    
    # Build the message with unescaped text
    message = f"""🆕 [Token]({dexscreener_url})
💰 USD: ${price}
💦 Liq: ${liquidity}
📊 Vol: ${volume} (24h)

📈 Volume/Liquidity Ratio: {ratio}x
👥 TH: Top 3 holders control {top_holders}%
- Holder 1: {holder1}%
- Holder 2: {holder2}%
- Holder 3: {holder3}%

⚠️ High Risk Indicators!
- Low liquidity (<$50k) creates high slippage risk
- High concentration in top wallets
- Very high volume compared to liquidity suggests potential manipulation
- Lack of verified contract address is concerning

Immediate Action Advice!
- Exercise extreme caution
- Small position sizes only if trading
- Set tight stop losses
- Watch for sudden liquidity removal

💨 Not seen in a group yet!
🎖️ NEW: Test /rank for group rankings [here]({rank_url})"""
    
    # Escape the entire message at once
    output = escape_markdown_v2_custom(message)
    
    return output

def format_ai_analysis_output(ai_analysis):
    """
    Formats the AI analysis output for readability in Telegram messages.
    """
    logger.info(f"Raw AI analysis output: {ai_analysis}")
    
    if not ai_analysis:
        return "*AI Analysis Results:*\n\nNo analysis data available\\."
    
    formatted_output = ["*AI Analysis Results:*\n"]
    
    # Split the analysis into sections
    sections = ai_analysis.split('\n\n')
    
    for section in sections:
        lines = section.split('\n')
        if lines:
            # Escape the header
            header = escape_markdown_v2_custom(lines[0])
            formatted_output.append(f"*{header}*\n")
            
            # Process and escape the content
            for line in lines[1:]:
                escaped_line = escape_markdown_v2_custom(line)
                formatted_output.append(f"{escaped_line}\n")
            
            formatted_output.append("\n")  # Add an extra newline between sections
    
    result = "".join(formatted_output).strip()
    logger.info(f"Formatted AI analysis output: {result}")
    return result

def format_markdown_v2(text):
    """
    Formats text for Markdown V2 by escaping special characters.
    """
    return escape_markdown_v2_custom(text)

def prepare_telegram_message(message):
    """Prepare message for Telegram by escaping markdown characters"""
    # Convert **text** to *text* for bold
    message = message.replace('**', '*')
    
    # Use our improved escape function
    escaped_message = escape_markdown_v2_custom(message)
    
    logger.info(f"Prepared message: {escaped_message}")
    return escaped_message
