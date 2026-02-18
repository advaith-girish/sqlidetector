"""Training script for Stage 1 Linear SVM"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1.trainer import Stage1Trainer
from training.data_loader import load_training_data


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 Linear SVM classifier')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (file or directory)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for trained model (default: from config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Stage 1: Linear SVM Classifier")
    print("=" * 60)
    
    # Load training data
    print(f"\nLoading training data from {args.data}...")
    safe_queries, injection_queries = load_training_data(args.data)
    
    if not safe_queries or not injection_queries:
        print("Error: Both safe and injection queries are required for Stage 1")
        return
    
    print(f"Loaded {len(safe_queries)} safe and {len(injection_queries)} injection queries")
    
    # Train
    trainer = Stage1Trainer(config_path=args.config)
    trainer.train(
        safe_queries,
        injection_queries,
        validation_split=args.val_split,
        test_split=args.test_split,
        output_path=args.output
    )
    
    print("\nStage 1 training completed!")


if __name__ == '__main__':
    main()