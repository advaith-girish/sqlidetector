"""Training script for Stage 0 Bloom filter whitelist"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage0.whitelist_trainer import WhitelistTrainer
from training.data_loader import load_training_data


def main():
    parser = argparse.ArgumentParser(description='Train Stage 0 Bloom filter whitelist')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (file or directory)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for trained model (default: from config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Stage 0: Bloom Filter Whitelist")
    print("=" * 60)
    
    # Load training data (only safe queries needed for Stage 0)
    print(f"\nLoading training data from {args.data}...")
    safe_queries, _ = load_training_data(args.data)
    
    if not safe_queries:
        print("Error: No safe queries found in training data")
        return
    
    print(f"Loaded {len(safe_queries)} safe queries")
    
    # Train
    trainer = WhitelistTrainer(config_path=args.config)
    trainer.train(safe_queries, output_path=args.output)
    
    print("\nStage 0 training completed!")


if __name__ == '__main__':
    main()