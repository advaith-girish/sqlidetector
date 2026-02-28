"""Training script for Stage 2 DistilBERT"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage2.trainer import Stage2Trainer
from training.data_loader import load_training_data


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2 DistilBERT model')
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
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Stage 2: Quantized DistilBERT")
    print("=" * 60)
    
    # Load training data
    print(f"\nLoading training data from {args.data}...")
    safe_queries, injection_queries = load_training_data(args.data)
    
    if not safe_queries or not injection_queries:
        print("Error: Both safe and injection queries are required for Stage 2")
        return
    
    print(f"Loaded {len(safe_queries)} safe and {len(injection_queries)} injection queries")
    
    # Train
    trainer = Stage2Trainer(config_path=args.config)
    trainer.train(
        safe_queries,
        injection_queries,
        validation_split=args.val_split,
        test_split=args.test_split,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_path=args.output
    )
    
    print("\nStage 2 training completed!")


if __name__ == '__main__':
    main()