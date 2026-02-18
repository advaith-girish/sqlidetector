"""Configuration management for SQL injection detector"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager that loads settings from YAML file"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        
        # If config doesn't exist in current dir, try workspace root
        if not config_file.exists():
            workspace_root = Path(__file__).parent.parent.parent
            config_file = workspace_root / self.config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'stage0.bloom_filter.capacity')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_stage0_config(self) -> Dict[str, Any]:
        """Get Stage 0 (Bloom filter) configuration"""
        return self._config.get('stage0', {})
    
    def get_stage1_config(self) -> Dict[str, Any]:
        """Get Stage 1 (SVM) configuration"""
        return self._config.get('stage1', {})
    
    def get_stage2_config(self) -> Dict[str, Any]:
        """Get Stage 2 (DistilBERT) configuration"""
        return self._config.get('stage2', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return self._config.get('pipeline', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self._config.get('training', {})


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance