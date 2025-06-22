"""
Configuration management for healthcare benefits dashboard.
Handles client-specific settings and parameters.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Manages configuration for different clients and environments."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        path = Path(config_path)
        
        if not path.exists():
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            self.load_default_config()
            return
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    self.config = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.load_default_config()
    
    def load_default_config(self) -> None:
        """Load default configuration."""
        self.config = {
            "client_id": "default",
            "client_name": "Default Client",
            
            "data_processing": {
                "date_format": "%Y-%m-%d",
                "encoding": "utf-8",
                "decimal_places": 2
            },
            
            "pharmacy_rebates": {
                "enabled": True,
                "flat_percentage": 0.20,  # 20% default rebate
                "tiers": {
                    "Generic": 0.75,      # 75% rebate on generics
                    "Preferred": 0.25,    # 25% on preferred brands
                    "Non-Preferred": 0.15, # 15% on non-preferred
                    "Specialty": 0.10     # 10% on specialty drugs
                }
            },
            
            "stop_loss": {
                "enabled": True,
                "attachment_point": 250000,
                "corridor": 50000,
                "reimbursement_percentage": 1.0,  # 100% above attachment
                "reimbursement_lag_months": 2,
                "aggregating_specific": True
            },
            
            "fixed_costs": {
                "admin_fee_pepm": 35.00,
                "stop_loss_premium_pepm": 65.00,
                "network_access_pmpm": 2.50,
                "wellness_program_pepm": 5.00,
                "disease_management_pepm": 8.00
            },
            
            "budget": {
                "pmpm": 500.00,
                "annual_trend": 0.08,  # 8% annual trend
                "confidence_interval": 0.95
            },
            
            "high_cost_threshold": 50000,
            
            "reporting": {
                "include_member_detail": False,  # HIPAA compliance
                "anonymize_ids": True,
                "rolling_months": 24,
                "comparison_periods": ["prior_month", "prior_year", "ytd"]
            },
            
            "validation": {
                "outlier_threshold": 3.0,  # Z-score threshold
                "max_variance_warning": 0.20,  # 20% variance triggers warning
                "required_data_completeness": 0.95  # 95% completeness required
            },
            
            "visualization": {
                "color_scheme": "professional_blue",
                "chart_style": "modern",
                "logo_path": None,
                "brand_colors": {
                    "primary": "#1E4D8B",
                    "secondary": "#16A085",
                    "accent": "#3498DB",
                    "positive": "#27AE60",
                    "negative": "#E74C3C",
                    "neutral": "#95A5A6"
                }
            }
        }
        
        self.logger.info("Loaded default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        path = Path(output_path)
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
            self.logger.info(f"Saved configuration to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration completeness and correctness."""
        issues = {
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = [
            "client_id",
            "pharmacy_rebates.enabled",
            "stop_loss.enabled",
            "fixed_costs.admin_fee_pepm"
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                issues["errors"].append(f"Missing required field: {field}")
        
        # Validate ranges
        if self.get("pharmacy_rebates.flat_percentage", 0) > 1:
            issues["warnings"].append("Pharmacy rebate percentage > 100%")
        
        if self.get("stop_loss.attachment_point", 0) < 50000:
            issues["warnings"].append("Stop-loss attachment point unusually low")
        
        if self.get("budget.pmpm", 0) < 100:
            issues["warnings"].append("Budget PMPM seems unrealistically low")
        
        return issues
    
    def get_client_branding(self) -> Dict[str, str]:
        """Get client-specific branding configuration."""
        return {
            "client_name": self.get("client_name", "Client"),
            "logo_path": self.get("visualization.logo_path"),
            "primary_color": self.get("visualization.brand_colors.primary", "#1E4D8B"),
            "secondary_color": self.get("visualization.brand_colors.secondary", "#16A085"),
            "font_family": self.get("visualization.font_family", "Roboto")
        }
    
    def get_calculation_parameters(self) -> Dict[str, Any]:
        """Get all calculation-related parameters."""
        return {
            "pharmacy_rebates": self.get("pharmacy_rebates", {}),
            "stop_loss": self.get("stop_loss", {}),
            "fixed_costs": self.get("fixed_costs", {}),
            "budget": self.get("budget", {}),
            "high_cost_threshold": self.get("high_cost_threshold", 50000)
        }