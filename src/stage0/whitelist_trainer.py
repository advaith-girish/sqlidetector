"""Training script for Stage 0 Bloom filter whitelist"""

from pathlib import Path
from typing import List, Optional
from .bloom_filter import BloomFilter
from ..pipeline.query_normalizer import QueryNormalizer
from ..utils.config import get_config


class WhitelistTrainer:
    """Trainer for building Bloom filter whitelist from safe queries"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize whitelist trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.normalizer = QueryNormalizer()
        self.stage0_config = self.config.get_stage0_config()
    
    def train(self, safe_queries: List[str], output_path: Optional[str] = None) -> BloomFilter:
        """
        Train Bloom filter from safe queries
        
        Args:
            safe_queries: List of safe SQL query strings
            output_path: Optional path to save the trained filter
        
        Returns:
            Trained BloomFilter instance
        """
        if not safe_queries:
            raise ValueError("No safe queries provided")
        
        # Normalize all queries
        print(f"Normalizing {len(safe_queries)} safe queries...")
        normalized_queries = self.normalizer.normalize_batch(safe_queries)
        
        # Get unique normalized patterns
        unique_patterns = list(set(normalized_queries))
        print(f"Found {len(unique_patterns)} unique query patterns")
        
        # Get Bloom filter configuration
        bloom_config = self.stage0_config.get('bloom_filter', {})
        capacity = bloom_config.get('capacity', len(unique_patterns) * 2)  # 2x for growth
        false_positive_rate = bloom_config.get('false_positive_rate', 0.001)
        
        # Create and populate Bloom filter
        print(f"Building Bloom filter (capacity={capacity}, fpr={false_positive_rate})...")
        bloom_filter = BloomFilter(capacity, false_positive_rate)
        bloom_filter.add_batch(unique_patterns)
        
        # Print statistics
        stats = bloom_filter.get_stats()
        print(f"Bloom filter created:")
        print(f"  - Elements added: {stats['count']}")
        print(f"  - Bit array size: {stats['num_bits']}")
        print(f"  - Hash functions: {stats['num_hash_functions']}")
        print(f"  - Load factor: {stats['load_factor']:.2%}")
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            bloom_filter.save(output_path)
            print(f"Bloom filter saved to {output_path}")
        else:
            # Use default path from config
            model_path = bloom_config.get('model_path', 'models/stage0_bloom.bin')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            bloom_filter.save(model_path)
            print(f"Bloom filter saved to {model_path}")
        
        return bloom_filter
    
    def load_trained_filter(self, model_path: Optional[str] = None) -> BloomFilter:
        """
        Load a trained Bloom filter
        
        Args:
            model_path: Optional path to model file
        
        Returns:
            Loaded BloomFilter instance
        """
        if model_path is None:
            bloom_config = self.stage0_config.get('bloom_filter', {})
            model_path = bloom_config.get('model_path', 'models/stage0_bloom.bin')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Bloom filter not found at {model_path}")
        
        return BloomFilter.load(model_path)