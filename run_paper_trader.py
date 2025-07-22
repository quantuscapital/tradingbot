#!/usr/bin/env python3
"""
Crypto Paper Trading Bot Launcher
Simple launcher script for the paper trading bot with TUI
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from trading_bot import main

if __name__ == "__main__":
    print("🚀 Starting Crypto Paper Trading Bot...")
    print("📊 Paper trading mode - No real funds at risk")
    print("🎮 Use 's' to start/pause scanning, 'q' to quit")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
