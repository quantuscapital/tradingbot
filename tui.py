import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, DataTable, Log, Button, 
    ProgressBar, Label, Rule, RichLog
)
from textual.reactive import reactive
from textual.timer import Timer
from textual.screen import Screen
from textual.binding import Binding
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.columns import Columns
from rich.box import ROUNDED, DOUBLE, HEAVY
from rich import box

from paper_trading import PaperTradingEngine
from dexscreener_client import DexScreenerClient
from config_manager import ConfigManager
from settings_screen import SettingsScreen

# Suppress all stdout/stderr to prevent terminal interference
class SuppressOutput:
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def __enter__(self):
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

# Custom logging handler that only sends to TUI
class TUILogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.tui_app = None
        self.buffer = []
        
    def set_tui_app(self, app):
        self.tui_app = app
        # Flush buffered messages
        for record in self.buffer:
            self.emit(record)
        self.buffer.clear()
        
    def emit(self, record):
        if self.tui_app and hasattr(self.tui_app, 'activity_log'):
            try:
                level = record.levelname
                message = record.getMessage()
                self.tui_app.activity_log.log_event(message, level)
            except:
                pass  # Silently ignore logging errors
        else:
            # Buffer messages until TUI is ready
            self.buffer.append(record)

# Global TUI log handler
tui_log_handler = TUILogHandler()

class MarketFeedWidget(Static):
    """Enhanced market feed widget with better formatting"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viable_tokens = []
        self.last_update = None
    
    def update_tokens(self, tokens: List[dict]):
        """Update the list of viable tokens"""
        self.viable_tokens = tokens
        self.last_update = datetime.now()
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the display with current tokens"""
        if not self.viable_tokens:
            content = Panel(
                Align.center(
                    "[bold blue]🔍 SCANNING FOR OPPORTUNITIES[/bold blue]\n\n"
                    "[dim]Analyzing market data...[/dim]\n"
                    "[dim]Looking for viable tokens...[/dim]",
                    vertical="middle"
                ),
                title="[bold cyan]Market Scanner[/bold cyan]",
                border_style="bright_blue",
                box=ROUNDED,
                padding=(1, 2)
            )
        else:
            table = Table(
                show_header=True, 
                header_style="bold bright_cyan",
                box=box.ROUNDED,
                border_style="bright_blue",
                expand=True
            )
            table.add_column("Symbol", style="bright_cyan", width=10, no_wrap=True)
            table.add_column("Price", style="bright_green", width=12, justify="right")
            table.add_column("Liquidity", style="bright_yellow", width=10, justify="right")
            table.add_column("Volume 24h", style="bright_blue", width=10, justify="right")
            table.add_column("Change 24h", style="bright_magenta", width=10, justify="right")
            table.add_column("Score", style="bright_white", width=8, justify="center")
            
            # Show last 8 tokens to fit better
            for token in self.viable_tokens[-8:]:
                pair = token.get('pair', {})
                base_token = pair.get('baseToken', {})
                symbol = base_token.get('symbol', 'Unknown')[:8]
                
                price = float(pair.get('priceUsd', 0))
                price_str = f"${price:.6f}" if price < 1 else f"${price:.4f}"
                
                liquidity = float(pair.get('liquidity', {}).get('usd', 0))
                liq_str = f"${liquidity/1000:.0f}K" if liquidity >= 1000 else f"${liquidity:.0f}"
                
                volume = float(pair.get('volume', {}).get('h24', 0))
                vol_str = f"${volume/1000:.0f}K" if volume >= 1000 else f"${volume:.0f}"
                
                change = float(pair.get('priceChange', {}).get('h24', 0))
                change_str = f"{change:+.1f}%"
                change_style = "bright_green" if change > 0 else "bright_red"
                
                # Calculate score (simplified)
                score = min(100, max(0, (liquidity/1000 + volume/1000 + abs(change)) / 3))
                score_str = f"{score:.0f}"
                score_style = "bright_green" if score >= 70 else "bright_yellow" if score >= 50 else "bright_red"
                
                table.add_row(
                    symbol,
                    price_str,
                    liq_str,
                    vol_str,
                    Text(change_str, style=change_style),
                    Text(score_str, style=score_style)
                )
            
            update_time = self.last_update.strftime("%H:%M:%S") if self.last_update else "Never"
            content = Panel(
                table,
                title=f"[bold cyan]Market Feed[/bold cyan] [dim]({len(self.viable_tokens)} found)[/dim]",
                subtitle=f"[dim]Last update: {update_time}[/dim]",
                border_style="bright_blue",
                box=ROUNDED
            )
        
        self.update(content)

class PositionsWidget(Static):
    """Enhanced positions widget with better P&L visualization"""
    
    def __init__(self, paper_engine: PaperTradingEngine, **kwargs):
        super().__init__(**kwargs)
        self.paper_engine = paper_engine
        self.current_prices = {}
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for positions"""
        self.current_prices.update(prices)
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the positions display"""
        positions = self.paper_engine.positions
        
        if not positions:
            content = Panel(
                Align.center(
                    "[bold yellow]📊 NO OPEN POSITIONS[/bold yellow]\n\n"
                    "[dim]Ready to trade when opportunities arise[/dim]",
                    vertical="middle"
                ),
                title="[bold green]Active Positions[/bold green]",
                border_style="bright_green",
                box=ROUNDED,
                padding=(1, 2)
            )
        else:
            table = Table(
                show_header=True, 
                header_style="bold bright_green",
                box=box.ROUNDED,
                border_style="bright_green",
                expand=True
            )
            table.add_column("Symbol", style="bright_cyan", width=8, no_wrap=True)
            table.add_column("Qty", style="white", width=10, justify="right")
            table.add_column("Entry", style="bright_yellow", width=10, justify="right")
            table.add_column("Current", style="bright_blue", width=10, justify="right")
            table.add_column("P&L %", style="bold", width=8, justify="center")
            table.add_column("P&L $", style="bold", width=10, justify="right")
            table.add_column("Status", style="bright_white", width=8, justify="center")
            
            total_pnl = 0
            for token_address, position in positions.items():
                symbol = position.get('token_symbol', 'Unknown')[:6]
                quantity = position['quantity']
                qty_str = f"{quantity:.2f}" if quantity >= 1 else f"{quantity:.4f}"
                
                buy_price = position['buy_price_usd']
                buy_str = f"${buy_price:.6f}" if buy_price < 1 else f"${buy_price:.4f}"
                
                current_price = self.current_prices.get(token_address, buy_price)
                current_str = f"${current_price:.6f}" if current_price < 1 else f"${current_price:.4f}"
                
                # Calculate P&L
                pnl_data = self.paper_engine.get_position_pnl(token_address, current_price)
                if pnl_data:
                    pnl_percent = pnl_data['pnl_percent']
                    pnl_usd = pnl_data['pnl_usd']
                    total_pnl += pnl_usd
                    
                    pnl_percent_str = f"{pnl_percent:+.1f}%"
                    pnl_usd_str = f"${pnl_usd:+.2f}"
                    
                    # Enhanced color coding with emojis
                    if pnl_percent > 10:
                        pnl_style = "bold bright_green"
                        status = "🚀"
                    elif pnl_percent > 0:
                        pnl_style = "bright_green"
                        status = "📈"
                    elif pnl_percent > -5:
                        pnl_style = "bright_yellow"
                        status = "⚖️"
                    elif pnl_percent > -15:
                        pnl_style = "bright_red"
                        status = "📉"
                    else:
                        pnl_style = "bold bright_red"
                        status = "🔻"
                else:
                    pnl_percent_str = "N/A"
                    pnl_usd_str = "N/A"
                    pnl_style = "dim"
                    status = "❓"
                
                table.add_row(
                    symbol,
                    qty_str,
                    buy_str,
                    current_str,
                    Text(pnl_percent_str, style=pnl_style),
                    Text(pnl_usd_str, style=pnl_style),
                    status
                )
            
            total_pnl_str = f"${total_pnl:+.2f}"
            total_style = "bright_green" if total_pnl > 0 else "bright_red"
            
            content = Panel(
                table,
                title=f"[bold green]Active Positions[/bold green] [dim]({len(positions)})[/dim]",
                subtitle=f"[bold]Total P&L: [{total_style}]{total_pnl_str}[/{total_style}][/bold]",
                border_style="bright_green",
                box=ROUNDED
            )
        
        self.update(content)

class MetricsWidget(Static):
    """Enhanced metrics widget with better visual hierarchy"""
    
    def __init__(self, paper_engine: PaperTradingEngine, **kwargs):
        super().__init__(**kwargs)
        self.paper_engine = paper_engine
    
    def refresh_display(self):
        """Refresh the metrics display"""
        summary = self.paper_engine.get_portfolio_summary()
        
        # Create metrics in a structured layout
        balance_section = f"[bold bright_cyan]💰 SOL Balance:[/bold bright_cyan] [bright_white]{summary['sol_balance']:.4f} SOL[/bright_white]"
        allocation_section = f"[bold bright_yellow]📊 Per Trade:[/bold bright_yellow] [bright_white]{summary['allocation_per_trade']:.4f} SOL[/bright_white]"
        
        trades_section = f"[bold bright_blue]📈 Total Trades:[/bold bright_blue] [bright_white]{summary['total_trades']}[/bright_white]"
        winrate_color = "bright_green" if summary['win_rate'] >= 60 else "bright_yellow" if summary['win_rate'] >= 40 else "bright_red"
        winrate_section = f"[bold bright_green]🎯 Win Rate:[/bold bright_green] [{winrate_color}]{summary['win_rate']:.1f}%[/{winrate_color}]"
        
        pnl_color = "bright_green" if summary['total_pnl_usd'] > 0 else "bright_red"
        pnl_section = f"[bold bright_magenta]💵 Total P&L:[/bold bright_magenta] [{pnl_color}]${summary['total_pnl_usd']:+.2f}[/{pnl_color}]"
        
        win_section = f"[bold bright_green]🏆 Best Win:[/bold bright_green] [bright_white]${summary['largest_win']:.2f}[/bright_white]"
        loss_section = f"[bold bright_red]💥 Worst Loss:[/bold bright_red] [bright_white]${summary['largest_loss']:.2f}[/bright_white]"
        
        positions_section = f"[bold bright_white]📋 Open Positions:[/bold bright_white] [bright_cyan]{summary['total_positions']}[/bright_cyan]"
        
        metrics_text = f"""
{balance_section}
{allocation_section}

{trades_section}
{winrate_section}

{pnl_section}

{win_section}
{loss_section}

{positions_section}
        """.strip()
        
        content = Panel(
            metrics_text,
            title="[bold yellow]Portfolio Metrics[/bold yellow]",
            border_style="bright_yellow",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.update(content)

class ActivityLogWidget(RichLog):
    """Enhanced activity log with better formatting and filtering"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "[bold white]Activity Log[/bold white]"
        self.border_subtitle = "[dim]Live Events[/dim]"
        self.max_lines = 1000
    
    def log_event(self, message: str, level: str = "INFO"):
        """Log an event with enhanced formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Enhanced level styling with emojis
        if level == "ERROR":
            icon = "❌"
            style = "bold bright_red"
        elif level == "WARNING":
            icon = "⚠️"
            style = "bold bright_yellow"
        elif level == "SUCCESS":
            icon = "✅"
            style = "bold bright_green"
        elif level == "INFO":
            icon = "ℹ️"
            style = "bright_blue"
        else:
            icon = "📝"
            style = "white"
        
        # Format message with better structure
        formatted_message = f"[dim]{timestamp}[/dim] {icon} [{style}]{message}[/{style}]"
        
        self.write(formatted_message)
        
        # Auto-scroll to bottom
        self.scroll_end()

class TradingBotTUI(App):
    """Industry-grade TUI application for the trading bot"""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 4;
        grid-gutter: 1;
        background: $background;
    }
    
    #market_feed {
        column-span: 1;
        row-span: 2;
    }
    
    #positions {
        column-span: 1;
        row-span: 2;
    }
    
    #metrics {
        column-span: 1;
        row-span: 2;
    }
    
    #activity_log {
        column-span: 3;
        row-span: 2;
    }
    
    .status_running {
        color: $success;
    }
    
    .status_paused {
        color: $error;
    }
    
    Header {
        background: $primary;
    }
    
    Footer {
        background: $primary;
    }
    
    Static {
        border: solid $primary;
        margin: 1;
    }
    
    RichLog {
        border: solid $accent;
        margin: 1;
    }
    """
    
    BINDINGS = [
        Binding("s", "toggle_scanning", "Start/Pause", priority=True),
        Binding("r", "reset_state", "Reset", priority=True),
        Binding("d", "dump_state", "Save", priority=True),
        Binding("o", "show_settings", "Settings", priority=True),
        Binding("t", "toggle_theme", "Theme", priority=True),
        Binding("f", "toggle_fullscreen", "Fullscreen", priority=True),
        Binding("ctrl+c,q", "quit", "Quit", priority=True),
    ]
    
    def __init__(self, paper_engine: PaperTradingEngine, config_manager: ConfigManager, **kwargs):
        super().__init__(**kwargs)
        self.paper_engine = paper_engine
        self.config_manager = config_manager
        self.scanning = False
        self.scan_timer: Optional[Timer] = None
        self.fullscreen_mode = False
        
        # Set up enhanced logging
        self.setup_logging()
        
        # Set up config callbacks
        self.setup_config_callbacks()
        
        # Clear terminal and hide cursor
        os.system('clear')
        print('\033[?25l', end='')  # Hide cursor
    
    def setup_logging(self):
        """Set up comprehensive logging control"""
        # Remove all existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set TUI app reference
        tui_log_handler.set_tui_app(self)
        
        # Add only our TUI handler
        root_logger.addHandler(tui_log_handler)
        root_logger.setLevel(logging.INFO)
        
        # Suppress specific noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    def compose(self) -> ComposeResult:
        """Create the enhanced TUI layout"""
        yield Header(show_clock=True, name="Crypto Paper Trader")
        
        # Market feed widget
        self.market_feed = MarketFeedWidget(id="market_feed")
        yield self.market_feed
        
        # Positions widget
        self.positions = PositionsWidget(self.paper_engine, id="positions")
        yield self.positions
        
        # Metrics widget
        self.metrics = MetricsWidget(self.paper_engine, id="metrics")
        yield self.metrics
        
        # Activity log widget
        self.activity_log = ActivityLogWidget(id="activity_log")
        yield self.activity_log
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted"""
        self.title = "Crypto Paper Trader"
        self.sub_title = "⏸ Paused - Press 's' to start scanning"
        
        # Initial refresh
        self.refresh_all_widgets()
        
        # Welcome message
        self.activity_log.log_event("🚀 Crypto Paper Trader initialized", "SUCCESS")
        self.activity_log.log_event("📊 Paper trading mode active", "INFO")
        self.activity_log.log_event("⌨️  Press 's' to start/pause scanning", "INFO")
        
        # Set up refresh timer (every 2 seconds for better responsiveness)
        self.set_interval(2.0, self.refresh_all_widgets)
    
    def on_unmount(self) -> None:
        """Called when the app is unmounted"""
        # Show cursor again
        print('\033[?25h', end='')
    
    def refresh_all_widgets(self):
        """Refresh all widgets with current data"""
        try:
            self.positions.refresh_display()
            self.metrics.refresh_display()
        except Exception as e:
            if hasattr(self, 'activity_log'):
                self.activity_log.log_event(f"Widget refresh error: {str(e)}", "ERROR")
    
    def update_market_feed(self, viable_tokens: List[dict]):
        """Update market feed with new viable tokens"""
        try:
            self.market_feed.update_tokens(viable_tokens)
            if viable_tokens:
                self.activity_log.log_event(f"📈 Market scan complete: {len(viable_tokens)} viable tokens found", "SUCCESS")
            else:
                self.activity_log.log_event("🔍 Market scan complete: No viable tokens found", "INFO")
        except Exception as e:
            self.activity_log.log_event(f"Market feed update error: {str(e)}", "ERROR")
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update position prices"""
        try:
            self.positions.update_prices(prices)
        except Exception as e:
            self.activity_log.log_event(f"Price update error: {str(e)}", "ERROR")
    
    def log_trade(self, trade_record: dict):
        """Log a trade event with enhanced formatting"""
        try:
            trade_type = trade_record['type']
            symbol = trade_record['token_symbol']
            
            if trade_type == 'BUY':
                sol_amount = trade_record['sol_amount']
                price = trade_record['price_usd']
                message = f"💰 BUY ORDER: {symbol} | {sol_amount:.4f} SOL @ ${price:.6f}"
                self.activity_log.log_event(message, "SUCCESS")
            elif trade_type == 'SELL':
                pnl_percent = trade_record['pnl_percent']
                pnl_usd = trade_record['pnl_usd']
                reason = trade_record['sell_reason']
                partial = trade_record.get('partial', False)
                
                order_type = "PARTIAL SELL" if partial else "SELL ORDER"
                message = f"💸 {order_type}: {symbol} | P&L: {pnl_percent:+.1f}% (${pnl_usd:+.2f}) | {reason}"
                level = "SUCCESS" if pnl_percent > 0 else "WARNING"
                self.activity_log.log_event(message, level)
        except Exception as e:
            self.activity_log.log_event(f"Trade logging error: {str(e)}", "ERROR")
    
    def action_toggle_scanning(self) -> None:
        """Toggle scanning on/off"""
        self.scanning = not self.scanning
        
        if self.scanning:
            self.sub_title = "⏵ Running - Scanning for opportunities"
            self.activity_log.log_event("▶️ Market scanning STARTED", "SUCCESS")
        else:
            self.sub_title = "⏸ Paused - Press 's' to resume"
            self.activity_log.log_event("⏸️ Market scanning PAUSED", "WARNING")
    
    def action_reset_state(self) -> None:
        """Reset paper trading state"""
        try:
            self.paper_engine.reset_state()
            self.refresh_all_widgets()
            self.activity_log.log_event("🔄 Paper trading state RESET - All positions cleared", "WARNING")
        except Exception as e:
            self.activity_log.log_event(f"Reset error: {str(e)}", "ERROR")
    
    def action_dump_state(self) -> None:
        """Save current state to file"""
        try:
            self.paper_engine.save_state()
            self.activity_log.log_event("💾 Trading state SAVED to file", "SUCCESS")
        except Exception as e:
            self.activity_log.log_event(f"Save error: {str(e)}", "ERROR")
    
    def action_toggle_theme(self) -> None:
        """Toggle between light and dark theme"""
        self.dark = not self.dark
        theme = "dark" if self.dark else "light"
        self.activity_log.log_event(f"🎨 Theme switched to {theme} mode", "SUCCESS")
    
    def action_toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode"""
        # This is a placeholder - actual fullscreen would require terminal manipulation
        self.fullscreen_mode = not self.fullscreen_mode
        mode = "fullscreen" if self.fullscreen_mode else "windowed"
        self.activity_log.log_event(f"🖥️ Display mode: {mode}", "INFO")
    
    def action_show_settings(self) -> None:
        """Show settings screen"""
        try:
            settings_screen = SettingsScreen(self.config_manager)
            self.push_screen(settings_screen)
            self.activity_log.log_event("⚙️ Settings screen opened", "INFO")
        except Exception as e:
            self.activity_log.log_event(f"Error opening settings: {str(e)}", "ERROR")
    
    def setup_config_callbacks(self):
        """Set up callbacks for configuration changes"""
        def on_paper_trading_toggle(key: str, old_value: Any, new_value: bool):
            """Handle paper trading enable/disable"""
            if hasattr(self, 'activity_log'):
                status = "enabled" if new_value else "disabled"
                self.activity_log.log_event(f"📊 Paper trading {status}", "SUCCESS" if new_value else "WARNING")
                
                # Update paper engine state
                if hasattr(self.paper_engine, 'set_enabled'):
                    self.paper_engine.set_enabled(new_value)
        
        def on_theme_change(key: str, old_value: Any, new_value: str):
            """Handle theme changes"""
            if hasattr(self, 'activity_log'):
                self.activity_log.log_event(f"🎨 Theme changed to {new_value}", "INFO")
                # Apply theme change
                self.dark = (new_value == "dark")
        
        # Register callbacks
        self.config_manager.register_callback('paper_trading.enabled', on_paper_trading_toggle)
        self.config_manager.register_callback('ui.theme', on_theme_change)
    
    def on_settings_changed(self, message: SettingsScreen.SettingsChanged) -> None:
        """Handle settings changes from settings screen"""
        try:
            # Log the settings change
            self.activity_log.log_event("⚙️ Settings updated successfully", "SUCCESS")
            
            # Refresh widgets to reflect new settings
            self.refresh_all_widgets()
            
            # Update paper engine configuration
            if hasattr(self.paper_engine, 'update_config'):
                pt_config = self.config_manager.get_paper_trading_config()
                self.paper_engine.update_config(pt_config)
                
        except Exception as e:
            self.activity_log.log_event(f"Error applying settings: {str(e)}", "ERROR")
    
    def on_settings_cancel(self, message: SettingsScreen.SettingsCancel) -> None:
        """Handle settings cancellation"""
        self.activity_log.log_event("⚙️ Settings changes cancelled", "INFO")

async def run_tui_with_bot(paper_engine: PaperTradingEngine, bot_scan_function, config_manager: ConfigManager = None):
    """Run the enhanced TUI alongside the bot scanning function"""
    
    # Suppress all external output
    with SuppressOutput():
        # Create or use provided config manager
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Create TUI app
        tui_app = TradingBotTUI(paper_engine, config_manager)
        
        # Create DexScreener client for price updates
        dex_client = DexScreenerClient()
        
        # Shared state for communication
        shared_state = {
            'viable_tokens': [],
            'position_prices': {},
            'tui_app': tui_app,
            'last_scan_time': None,
            'scan_count': 0
        }
        
        async def bot_wrapper():
            """Enhanced bot scanning wrapper with better error handling"""
            scan_interval = 30  # seconds
            error_count = 0
            max_errors = 5
            
            while True:
                try:
                    if tui_app.scanning:
                        # Run the bot scan function
                        viable_tokens = await bot_scan_function()
                        
                        # Update shared state
                        shared_state['viable_tokens'] = viable_tokens or []
                        shared_state['last_scan_time'] = datetime.now()
                        shared_state['scan_count'] += 1
                        
                        # Update TUI
                        tui_app.update_market_feed(viable_tokens or [])
                        
                        # Reset error count on successful scan
                        error_count = 0
                        
                        # Log scan statistics
                        if shared_state['scan_count'] % 10 == 0:  # Every 10 scans
                            tui_app.activity_log.log_event(
                                f"📊 Scan #{shared_state['scan_count']} complete | "
                                f"Total viable tokens found: {len(shared_state['viable_tokens'])}",
                                "INFO"
                            )
                    
                    # Dynamic scan interval based on activity
                    await asyncio.sleep(scan_interval)
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"Scan error #{error_count}: {str(e)}"
                    tui_app.activity_log.log_event(error_msg, "ERROR")
                    
                    if error_count >= max_errors:
                        tui_app.activity_log.log_event(
                            f"⚠️ Too many scan errors ({max_errors}). Increasing scan interval.",
                            "WARNING"
                        )
                        scan_interval = min(scan_interval * 2, 300)  # Max 5 minutes
                        error_count = 0
                    
                    await asyncio.sleep(min(60, scan_interval))  # Wait at least 1 minute on error
        
        async def price_updater():
            """Enhanced price updater with better performance"""
            update_interval = 2.0  # seconds
            error_count = 0
            
            while True:
                try:
                    # Get all open position addresses
                    position_addresses = list(paper_engine.positions.keys())
                    
                    if position_addresses:
                        # Fetch current prices
                        current_prices = await dex_client.get_token_prices(position_addresses)
                        
                        if current_prices:
                            # Update TUI with new prices
                            tui_app.update_position_prices(current_prices)
                            
                            # Check for sell conditions
                            for token_address, current_price in current_prices.items():
                                if token_address in paper_engine.positions:
                                    should_sell, reason, is_partial = paper_engine.check_sell_conditions(
                                        token_address, current_price
                                    )
                                    
                                    if should_sell:
                                        # Execute paper sell
                                        trade_record = paper_engine.simulate_sell(
                                            token_address, current_price, reason, is_partial
                                        )
                                        
                                        if trade_record:
                                            tui_app.log_trade(trade_record)
                            
                            # Reset error count on success
                            error_count = 0
                    
                    await asyncio.sleep(update_interval)
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:  # Only log first few errors
                        tui_app.activity_log.log_event(f"Price update error: {str(e)}", "ERROR")
                    
                    # Exponential backoff on errors
                    await asyncio.sleep(min(update_interval * (2 ** min(error_count, 4)), 30))
        
        async def system_monitor():
            """Monitor system health and performance"""
            while True:
                try:
                    # Log system status every 5 minutes
                    await asyncio.sleep(300)
                    
                    positions_count = len(paper_engine.positions)
                    scan_count = shared_state['scan_count']
                    
                    tui_app.activity_log.log_event(
                        f"🔧 System Status: {positions_count} positions, {scan_count} scans completed",
                        "INFO"
                    )
                    
                except Exception as e:
                    tui_app.activity_log.log_event(f"Monitor error: {str(e)}", "ERROR")
                    await asyncio.sleep(60)
        
        # Run all components concurrently
        try:
            await asyncio.gather(
                tui_app.run_async(),
                bot_wrapper(),
                price_updater(),
                system_monitor()
            )
        except Exception as e:
            tui_app.activity_log.log_event(f"Critical error: {str(e)}", "ERROR")
            raise

if __name__ == "__main__":
    # For testing the TUI standalone
    config_manager = ConfigManager()
    paper_engine = PaperTradingEngine(config_manager.config)
    
    async def dummy_scan():
        return []
    
    asyncio.run(run_tui_with_bot(paper_engine, dummy_scan, config_manager))
