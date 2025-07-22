import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

class ConfigManager:
    """Runtime configuration manager for the trading bot"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.lock = Lock()  # Thread-safe config updates
        self.callbacks = {}  # Callbacks for config changes
        
        # Load initial config
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        try:
            with self.lock:
                if Path(self.config_path).exists():
                    with open(self.config_path, 'r') as f:
                        self.config = yaml.safe_load(f) or {}
                    logger.info(f"Configuration loaded from {self.config_path}")
                    return True
                else:
                    logger.warning(f"Config file {self.config_path} not found, using defaults")
                    self.config = self._get_default_config()
                    return False
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self._get_default_config()
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to YAML file"""
        try:
            with self.lock:
                # Create backup
                backup_path = f"{self.config_path}.backup"
                if Path(self.config_path).exists():
                    Path(self.config_path).rename(backup_path)
                
                # Save new config atomically
                temp_path = f"{self.config_path}.tmp"
                with open(temp_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                
                Path(temp_path).rename(self.config_path)
                
                # Remove backup on success
                if Path(backup_path).exists():
                    Path(backup_path).unlink()
                
                logger.info(f"Configuration saved to {self.config_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            # Restore backup if it exists
            backup_path = f"{self.config_path}.backup"
            if Path(backup_path).exists():
                Path(backup_path).rename(self.config_path)
                logger.info("Config backup restored due to save error")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'paper_trading.enabled')"""
        with self.lock:
            keys = key.split('.')
            value = self.config
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """Set a configuration value using dot notation"""
        with self.lock:
            keys = key.split('.')
            config = self.config
            
            try:
                # Navigate to the parent dictionary
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                
                # Set the final value
                old_value = config.get(keys[-1])
                config[keys[-1]] = value
                
                # Trigger callbacks
                self._trigger_callbacks(key, old_value, value)
                
                # Save to file if requested
                if save:
                    return self.save_config()
                return True
                
            except Exception as e:
                logger.error(f"Error setting config key '{key}': {e}")
                return False
    
    def update(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """Update multiple configuration values"""
        success = True
        for key, value in updates.items():
            if not self.set(key, value, save=False):
                success = False
        
        if save and success:
            success = self.save_config()
        
        return success
    
    def register_callback(self, key: str, callback):
        """Register a callback for when a specific config key changes"""
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
    
    def _trigger_callbacks(self, key: str, old_value: Any, new_value: Any):
        """Trigger callbacks for a configuration change"""
        if key in self.callbacks:
            for callback in self.callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in config callback for {key}: {e}")
    
    def get_paper_trading_config(self) -> Dict[str, Any]:
        """Get paper trading configuration"""
        return self.get('paper_trading', {})
    
    def set_paper_trading_enabled(self, enabled: bool, save: bool = True) -> bool:
        """Enable or disable paper trading"""
        return self.set('paper_trading.enabled', enabled, save)
    
    def is_paper_trading_enabled(self) -> bool:
        """Check if paper trading is enabled"""
        return self.get('paper_trading.enabled', False)
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-specific settings"""
        return {
            'trading_mode': 'paper' if self.is_paper_trading_enabled() else 'live',
            'private_key': self.get('wallet.private_key', ''),
            'rpc_url': self.get('wallet.rpc_url', ''),
            'paper_trading_enabled': self.is_paper_trading_enabled(),
            'initial_balance': self.get('paper_trading.initial_balance', 10.0),
            'allocation_percentage': self.get('paper_trading.allocation_percentage', 15.0),
            'stop_loss_percent': self.get('paper_trading.stop_loss_percent', 10.0),
            'scan_interval': self.get('scan_interval', 300),
            'message_interval': self.get('message_interval', 5),
            'theme': self.get('ui.theme', 'dark'),
            'auto_save': self.get('ui.auto_save', True)
        }
    
    def update_ui_settings(self, settings: Dict[str, Any], save: bool = True) -> bool:
        """Update UI settings"""
        updates = {}
        
        # Map UI settings to config keys
        if 'trading_mode' in settings:
            # Convert trading mode to paper_trading_enabled boolean
            updates['paper_trading.enabled'] = (settings['trading_mode'] == 'paper')
        if 'private_key' in settings:
            updates['wallet.private_key'] = settings['private_key']
        if 'rpc_url' in settings:
            updates['wallet.rpc_url'] = settings['rpc_url']
        if 'paper_trading_enabled' in settings:
            updates['paper_trading.enabled'] = settings['paper_trading_enabled']
        if 'initial_balance' in settings:
            updates['paper_trading.initial_balance'] = float(settings['initial_balance'])
        if 'allocation_percentage' in settings:
            updates['paper_trading.allocation_percentage'] = float(settings['allocation_percentage'])
        if 'stop_loss_percent' in settings:
            updates['paper_trading.stop_loss_percent'] = float(settings['stop_loss_percent'])
        if 'scan_interval' in settings:
            updates['scan_interval'] = int(settings['scan_interval'])
        if 'message_interval' in settings:
            updates['message_interval'] = int(settings['message_interval'])
        if 'theme' in settings:
            updates['ui.theme'] = settings['theme']
        if 'auto_save' in settings:
            updates['ui.auto_save'] = bool(settings['auto_save'])
        
        return self.update(updates, save)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'token_criteria': {
                'min_liquidity': 10000,
                'min_volume': 5000,
                'min_buy_transactions': 10,
                'min_sell_transactions': 5,
                'min_price_change': 25,
                'max_price_change': 1000,
                'multiplier_time_window': 86400,
                'multiplier_thresholds': {
                    '2x': 100,
                    '3x': 200,
                    '5x': 400,
                    '10x': 900
                },
                'pnl_threshold': 50
            },
            'scoring_weights': {
                'liquidity': 20,
                'volume': 20,
                'transactions': 15,
                'price_change': 20,
                'buy_sell_ratio': 10,
                'positive_price_change': 10,
                'multiplier': 5
            },
            'message_interval': 5,
            'scan_interval': 300,
            'price_update_interval': 1,
            'max_time_without_alert': 3600,
            'max_tokens_per_request': 50,
            'paper_trading': {
                'enabled': True,
                'initial_balance': 10.0,
                'allocation_percentage': 15.0,
                'stop_loss_percent': 10.0,
                'slippage_tolerance': 2.0,
                'max_time_without_trade': 7200,
                'profit_tiers': [
                    {
                        'profit_percent': 30.0,
                        'sell_percent': 50.0,
                        'trailing_stop': 20.0
                    },
                    {
                        'profit_percent': 50.0,
                        'sell_percent': 25.0,
                        'trailing_stop': 30.0
                    },
                    {
                        'profit_percent': 80.0,
                        'sell_percent': 25.0,
                        'trailing_stop': 40.0
                    }
                ],
                'position_size_multipliers': {
                    'high_conviction_threshold': 80,
                    'high_conviction_multiplier': 1.5,
                    'medium_conviction_threshold': 60,
                    'medium_conviction_multiplier': 1.2
                }
            },
            'ui': {
                'theme': 'dark',
                'auto_save': True
            },
            'wallet': {
                'private_key': '',
                'rpc_url': ''
            }
        }
    
    def reset_to_defaults(self, save: bool = True) -> bool:
        """Reset configuration to defaults"""
        with self.lock:
            self.config = self._get_default_config()
            if save:
                return self.save_config()
            return True
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        errors = {}
        warnings = {}
        
        try:
            # Validate paper trading config
            pt_config = self.get_paper_trading_config()
            
            if 'initial_balance' in pt_config:
                balance = pt_config['initial_balance']
                if not isinstance(balance, (int, float)) or balance <= 0:
                    errors['paper_trading.initial_balance'] = "Must be a positive number"
            
            if 'allocation_percentage' in pt_config:
                alloc = pt_config['allocation_percentage']
                if not isinstance(alloc, (int, float)) or alloc <= 0 or alloc > 100:
                    errors['paper_trading.allocation_percentage'] = "Must be between 0 and 100"
            
            if 'stop_loss_percent' in pt_config:
                stop_loss = pt_config['stop_loss_percent']
                if not isinstance(stop_loss, (int, float)) or stop_loss <= 0 or stop_loss >= 100:
                    errors['paper_trading.stop_loss_percent'] = "Must be between 0 and 100"
            
            # Validate intervals
            scan_interval = self.get('scan_interval', 300)
            if not isinstance(scan_interval, int) or scan_interval < 10:
                errors['scan_interval'] = "Must be at least 10 seconds"
            
            msg_interval = self.get('message_interval', 5)
            if not isinstance(msg_interval, int) or msg_interval < 1:
                errors['message_interval'] = "Must be at least 1 second"
                
        except Exception as e:
            errors['general'] = f"Config validation error: {e}"
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }