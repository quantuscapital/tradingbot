import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Label, Rule, 
    Switch, Input, Select, Collapsible
)
from textual.binding import Binding
from textual.message import Message
from textual.validation import Number, ValidationResult, Validator
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

from config_manager import ConfigManager

logger = logging.getLogger(__name__)

class PercentageValidator(Validator):
    """Validator for percentage values (0-100)"""
    
    def validate(self, value: str) -> ValidationResult:
        try:
            num = float(value)
            if 0 <= num <= 100:
                return self.success()
            else:
                return self.failure("Must be between 0 and 100")
        except ValueError:
            return self.failure("Must be a valid number")

class PositiveNumberValidator(Validator):
    """Validator for positive numbers"""
    
    def validate(self, value: str) -> ValidationResult:
        try:
            num = float(value)
            if num > 0:
                return self.success()
            else:
                return self.failure("Must be greater than 0")
        except ValueError:
            return self.failure("Must be a valid number")

class IntervalValidator(Validator):
    """Validator for time intervals"""
    
    def __init__(self, min_value: int = 1):
        super().__init__()
        self.min_value = min_value
    
    def validate(self, value: str) -> ValidationResult:
        try:
            num = int(value)
            if num >= self.min_value:
                return self.success()
            else:
                return self.failure(f"Must be at least {self.min_value}")
        except ValueError:
            return self.failure("Must be a valid integer")

class PrivateKeyValidator(Validator):
    """Validator for Solana private key (base58 encoded)"""
    
    def validate(self, value: str) -> ValidationResult:
        if not value:  # Allow empty for optional field
            return self.success()
        
        # Basic validation for base58 private key format
        if len(value) < 80 or len(value) > 90:
            return self.failure("Invalid private key length")
        
        # Check if it contains only base58 characters
        import re
        if not re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', value):
            return self.failure("Invalid private key format")
        
        return self.success()

class RPCUrlValidator(Validator):
    """Validator for RPC URL format"""
    
    def validate(self, value: str) -> ValidationResult:
        if not value:  # Allow empty for default
            return self.success()
        
        import re
        # Basic URL validation for HTTP/HTTPS
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', value):
            return self.failure("Invalid URL format")
        
        # Check if it looks like a Solana RPC endpoint
        if not any(keyword in value.lower() for keyword in ['solana', 'rpc', 'mainnet', 'devnet', 'helius', 'quicknode', 'alchemy']):
            return self.failure("URL doesn't appear to be a Solana RPC endpoint")
        
        return self.success()

class SettingsScreen(Screen):
    """Settings screen for configuring the trading bot"""
    
    CSS = """
    SettingsScreen {
        layout: vertical;
        background: $background;
    }
    
    Header {
        dock: top;
    }
    
    Footer {
        dock: bottom;
    }
    
    .settings-container {
        width: 100%;
        height: 1fr;
        padding: 2;
    }
    
    .settings-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary;
        border-title-align: left;
    }
    
    .settings-row {
        height: auto;
        margin: 1 0;
        padding: 0 1;
    }
    
    .setting-label {
        width: 30%;
        margin-right: 2;
    }
    
    .setting-input {
        width: 20;
        margin-right: 2;
    }
    
    .setting-switch {
        margin-right: 2;
    }
    
    .setting-select {
        width: 20;
        margin-right: 2;
    }
    
    .button-row {
        height: 5;
        margin: 2 0;
        padding: 1;
    }
    
    Button {
        margin: 0 2;
        min-width: 20;
    }
    
    Collapsible {
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("ctrl+r", "reset", "Reset", priority=True),
    ]
    
    class SettingsChanged(Message):
        """Message sent when settings are saved"""
        def __init__(self, settings: Dict[str, Any]) -> None:
            self.settings = settings
            super().__init__()
    
    class SettingsCancel(Message):
        """Message sent when settings are cancelled"""
        pass
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.original_settings = {}
        self.current_settings = {}
        self.widgets_initialized = False
    
    def compose(self) -> ComposeResult:
        """Create the settings screen layout"""
        yield Header(show_clock=True, name="⚙️ Settings Configuration")
        
        with ScrollableContainer(classes="settings-container"):
            yield Static("Configure your trading bot preferences below:", id="settings-intro")
            yield Rule()
            
            # Trading Mode Section
            with Collapsible(title="⚡ Trading Mode", collapsed=False):
                with Horizontal(classes="settings-row"):
                    yield Label("Trading Mode:", classes="setting-label")
                    yield Select(
                        [("Paper Trading", "paper"), ("Live Trading", "live")],
                        id="trading_mode",
                        classes="setting-select"
                    )
                    yield Label("Paper = Simulation, Live = Real money", id="trading_mode_help")
            
            # Wallet Configuration Section
            with Collapsible(title="🔐 Wallet & RPC Configuration", collapsed=False):
                with Horizontal(classes="settings-row"):
                    yield Label("Private Key:", classes="setting-label")
                    yield Input(
                        id="private_key",
                        placeholder="Enter your Solana private key (base58)",
                        password=True,
                        validators=[PrivateKeyValidator()],
                        classes="setting-input"
                    )
                    yield Label("Required for live trading", id="pk_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("RPC URL:", classes="setting-label")
                    yield Input(
                        id="rpc_url",
                        placeholder="https://api.mainnet-beta.solana.com (leave empty for default)",
                        validators=[RPCUrlValidator()],
                        classes="setting-input"
                    )
                    yield Label("Custom Solana RPC endpoint (optional)", id="rpc_help")
            
            # Paper Trading Section
            with Collapsible(title="📊 Paper Trading Configuration", collapsed=False):
                with Horizontal(classes="settings-row"):
                    yield Label("Enable Paper Trading:", classes="setting-label")
                    yield Switch(id="paper_trading_enabled", classes="setting-switch")
                    yield Label("Simulate trades without real money", id="paper_trading_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("Initial Balance (SOL):", classes="setting-label")
                    yield Input(
                        id="initial_balance", 
                        placeholder="10.0",
                        validators=[PositiveNumberValidator()],
                        classes="setting-input"
                    )
                    yield Label("Starting SOL balance for paper trading", id="balance_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("Allocation per Trade (%):", classes="setting-label")
                    yield Input(
                        id="allocation_percentage",
                        placeholder="15.0",
                        validators=[PercentageValidator()],
                        classes="setting-input"
                    )
                    yield Label("Percentage of balance used per trade", id="allocation_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("Stop Loss (%):", classes="setting-label")
                    yield Input(
                        id="stop_loss_percent",
                        placeholder="10.0",
                        validators=[PercentageValidator()],
                        classes="setting-input"
                    )
                    yield Label("Maximum loss before auto-sell", id="stop_loss_help")
            
            # Trading Behavior Section
            with Collapsible(title="🤖 Trading Behavior", collapsed=True):
                with Horizontal(classes="settings-row"):
                    yield Label("Scan Interval (seconds):", classes="setting-label")
                    yield Input(
                        id="scan_interval",
                        placeholder="300",
                        validators=[IntervalValidator(10)],
                        classes="setting-input"
                    )
                    yield Label("Time between market scans", id="scan_interval_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("Message Interval (seconds):", classes="setting-label")
                    yield Input(
                        id="message_interval",
                        placeholder="5",
                        validators=[IntervalValidator(1)],
                        classes="setting-input"
                    )
                    yield Label("Minimum time between messages", id="message_interval_help")
            
            # UI Settings Section
            with Collapsible(title="🎨 User Interface", collapsed=True):
                with Horizontal(classes="settings-row"):
                    yield Label("Theme:", classes="setting-label")
                    yield Select(
                        [("Dark", "dark"), ("Light", "light")],
                        id="theme",
                        classes="setting-select"
                    )
                    yield Label("Color theme for the interface", id="theme_help")
                
                with Horizontal(classes="settings-row"):
                    yield Label("Auto-save:", classes="setting-label")
                    yield Switch(id="auto_save", classes="setting-switch")
                    yield Label("Automatically save settings changes", id="auto_save_help")
            
            yield Rule()
            
            # Button Row
            with Horizontal(classes="button-row"):
                yield Button("💾 Save Settings", id="save_button", variant="success")
                yield Button("❌ Cancel", id="cancel_button", variant="error")
                yield Button("🔄 Reset to Defaults", id="reset_button", variant="warning")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the settings screen"""
        self.title = "Settings"
        self.sub_title = "Configure your trading bot preferences"
        
        # Load current settings
        self.load_current_settings()
        
        # Populate widgets with current values
        self.populate_widgets()
        
        self.widgets_initialized = True
    
    def load_current_settings(self):
        """Load current settings from config manager"""
        self.current_settings = self.config_manager.get_ui_settings()
        self.original_settings = self.current_settings.copy()
    
    def populate_widgets(self):
        """Populate widgets with current settings values"""
        try:
            # Trading mode
            self.query_one("#trading_mode", Select).value = self.current_settings.get('trading_mode', 'paper')
            
            # Wallet settings
            self.query_one("#private_key", Input).value = self.current_settings.get('private_key', '')
            self.query_one("#rpc_url", Input).value = self.current_settings.get('rpc_url', '')
            
            # Paper trading settings
            self.query_one("#paper_trading_enabled", Switch).value = self.current_settings.get('paper_trading_enabled', True)
            self.query_one("#initial_balance", Input).value = str(self.current_settings.get('initial_balance', 10.0))
            self.query_one("#allocation_percentage", Input).value = str(self.current_settings.get('allocation_percentage', 15.0))
            self.query_one("#stop_loss_percent", Input).value = str(self.current_settings.get('stop_loss_percent', 10.0))
            
            # Trading behavior settings
            self.query_one("#scan_interval", Input).value = str(self.current_settings.get('scan_interval', 300))
            self.query_one("#message_interval", Input).value = str(self.current_settings.get('message_interval', 5))
            
            # UI settings
            self.query_one("#theme", Select).value = self.current_settings.get('theme', 'dark')
            self.query_one("#auto_save", Switch).value = self.current_settings.get('auto_save', True)
            
        except Exception as e:
            logger.error(f"Error populating settings widgets: {e}")
    
    def collect_current_values(self) -> Dict[str, Any]:
        """Collect current values from all input widgets"""
        try:
            return {
                'trading_mode': self.query_one("#trading_mode", Select).value,
                'private_key': self.query_one("#private_key", Input).value or "",
                'rpc_url': self.query_one("#rpc_url", Input).value or "",
                'paper_trading_enabled': self.query_one("#paper_trading_enabled", Switch).value,
                'initial_balance': float(self.query_one("#initial_balance", Input).value or "10.0"),
                'allocation_percentage': float(self.query_one("#allocation_percentage", Input).value or "15.0"),
                'stop_loss_percent': float(self.query_one("#stop_loss_percent", Input).value or "10.0"),
                'scan_interval': int(self.query_one("#scan_interval", Input).value or "300"),
                'message_interval': int(self.query_one("#message_interval", Input).value or "5"),
                'theme': self.query_one("#theme", Select).value,
                'auto_save': self.query_one("#auto_save", Switch).value
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error collecting settings values: {e}")
            return self.current_settings.copy()
    
    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """Validate all settings and return error messages"""
        errors = {}
        
        try:
            # Validate live trading requirements
            if settings.get('trading_mode') == 'live':
                if not settings.get('private_key', '').strip():
                    errors['private_key'] = "Private key is required for live trading"
                
                # Validate private key format if provided
                private_key = settings.get('private_key', '').strip()
                if private_key and (len(private_key) < 80 or len(private_key) > 90):
                    errors['private_key'] = "Invalid private key length"
            
            # Validate initial balance
            if settings['initial_balance'] <= 0:
                errors['initial_balance'] = "Must be greater than 0"
            
            # Validate allocation percentage
            if not (0 < settings['allocation_percentage'] <= 100):
                errors['allocation_percentage'] = "Must be between 0 and 100"
            
            # Validate stop loss percentage
            if not (0 < settings['stop_loss_percent'] < 100):
                errors['stop_loss_percent'] = "Must be between 0 and 100"
            
            # Validate scan interval
            if settings['scan_interval'] < 10:
                errors['scan_interval'] = "Must be at least 10 seconds"
            
            # Validate message interval
            if settings['message_interval'] < 1:
                errors['message_interval'] = "Must be at least 1 second"
                
        except (KeyError, TypeError, ValueError) as e:
            errors['general'] = f"Validation error: {e}"
        
        return errors
    
    def show_validation_errors(self, errors: Dict[str, str]):
        """Display validation errors to the user"""
        error_text = "\\n".join([f"• {field}: {message}" for field, message in errors.items()])
        self.app.notify(f"Settings validation failed:\\n{error_text}", severity="error", timeout=5)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        if event.button.id == "save_button":
            self.action_save()
        elif event.button.id == "cancel_button":
            self.action_cancel()
        elif event.button.id == "reset_button":
            self.action_reset()
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes"""
        if not self.widgets_initialized:
            return
            
        # Update current settings immediately for switches
        if event.switch.id == "paper_trading_enabled":
            self.current_settings['paper_trading_enabled'] = event.value
        elif event.switch.id == "auto_save":
            self.current_settings['auto_save'] = event.value
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        if not self.widgets_initialized:
            return
            
        # Update current settings for valid inputs
        try:
            if event.input.id == "private_key":
                self.current_settings['private_key'] = event.value or ""
            elif event.input.id == "rpc_url":
                self.current_settings['rpc_url'] = event.value or ""
            elif event.input.id == "initial_balance":
                self.current_settings['initial_balance'] = float(event.value or "10.0")
            elif event.input.id == "allocation_percentage":
                self.current_settings['allocation_percentage'] = float(event.value or "15.0")
            elif event.input.id == "stop_loss_percent":
                self.current_settings['stop_loss_percent'] = float(event.value or "10.0")
            elif event.input.id == "scan_interval":
                self.current_settings['scan_interval'] = int(event.value or "300")
            elif event.input.id == "message_interval":
                self.current_settings['message_interval'] = int(event.value or "5")
        except (ValueError, TypeError):
            pass  # Invalid input, ignore for now
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes"""
        if not self.widgets_initialized:
            return
            
        if event.select.id == "theme":
            self.current_settings['theme'] = event.value
        elif event.select.id == "trading_mode":
            self.current_settings['trading_mode'] = event.value
    
    def action_save(self) -> None:
        """Save settings action"""
        try:
            # Collect current values
            new_settings = self.collect_current_values()
            
            # Validate settings
            errors = self.validate_settings(new_settings)
            if errors:
                self.show_validation_errors(errors)
                return
            
            # Update config manager
            success = self.config_manager.update_ui_settings(new_settings)
            
            if success:
                self.app.notify("Settings saved successfully!", severity="information")
                
                # Send settings changed message
                self.post_message(self.SettingsChanged(new_settings))
                
                # Close screen
                self.app.pop_screen()
            else:
                self.app.notify("Failed to save settings. Please try again.", severity="error")
                
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self.app.notify(f"Error saving settings: {str(e)}", severity="error")
    
    def action_cancel(self) -> None:
        """Cancel settings action"""
        # Check if settings have changed
        current_values = self.collect_current_values()
        if current_values != self.original_settings:
            # Could add a confirmation dialog here
            self.app.notify("Settings changes discarded", severity="warning")
        
        # Send cancel message and close
        self.post_message(self.SettingsCancel())
        self.app.pop_screen()
    
    def action_reset(self) -> None:
        """Reset to defaults action"""
        try:
            # Reset config to defaults
            self.config_manager.reset_to_defaults(save=False)
            
            # Reload settings
            self.load_current_settings()
            
            # Repopulate widgets
            self.populate_widgets()
            
            self.app.notify("Settings reset to defaults", severity="information")
            
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            self.app.notify(f"Error resetting settings: {str(e)}", severity="error")