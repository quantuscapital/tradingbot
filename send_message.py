import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def send_telegram_message(chat_id, message, message_thread_id=None):
    """
    Send a message to a Telegram chat using the Telegram Bot API.
    
    Args:
        chat_id (int): The ID of the chat to send the message to.
        message (str): The message text to send.
        message_thread_id (int, optional): The ID of the thread to send the message to (for supergroups).
    
    Returns:
        dict: The response from the Telegram API.
    """
    # Get the bot token from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Construct the API URL
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Prepare the payload
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'MarkdownV2'  # Use MarkdownV2 for formatting
    }
    
    # Add message_thread_id if provided (for supergroups with topics)
    if message_thread_id:
        payload['message_thread_id'] = message_thread_id
    
    # Send the request
    response = requests.post(url, json=payload)
    
    # Parse and return the response
    return response.json()

if __name__ == "__main__":
    # Example usage
    chat_id = -1002440374107  # Replace with your chat ID
    message = "🚀 *Test Message* 🚀\n\nThis is a test message sent using the Telegram Bot API\\."
    
    # Optional: Specify a message_thread_id for supergroups with topics
    message_thread_id = 824  # Replace with your topic ID or remove if not needed
    
    # Send the message
    response = send_telegram_message(chat_id, message, message_thread_id)
    
    # Print the response
    print(json.dumps(response, indent=2))
