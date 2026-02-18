"""Training script for Stage 1 Linear SVM"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .tokenizer import SQLTokenizer
from .feature_hasher import FeatureHasher
from .svm_classifier import SVMClassifier
from ..utils.config import get_config


class Stage1Trainer:
    """Trainer for Stage 1 Linear SVM classifier"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Stage 1 trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.stage1_config = self.config.get_stage1_config()
        
        # Initialize components
        tokenizer_config = self.stage1_config.get('tokenizer', {})
        self.tokenizer = SQLTokenizer(
            case_sensitive=tokenizer_config.get('case_sensitive', False)
        )
        
        hasher_config = self.stage1_config.get('feature_hashing', {})
        self.feature_hasher = FeatureHasher(
            vector_size=hasher_config.get('vector_size', 65536),
            num_hash_functions=hasher_config.get('num_hash_functions', 2)
        )
        
        svm_config = self.stage1_config.get('svm', {})
        self.svm = SVMClassifier(
            threshold=svm_config.get('threshold', 0.5)
        )
    
    def prepare_features(self, queries: List[str]) -> np.ndarray:
        """
        Prepare feature vectors from queries
        
        Args:
            queries: List of SQL query strings
        
        Returns:
            Feature matrix (shape: [n_samples, vector_size])
        """
        print(f"Tokenizing {len(queries)} queries...")
        token_lists = self.tokenizer.tokenize_batch(queries)
        
        print("Hashing tokens to feature vectors...")
        feature_vectors = self.feature_hasher.hash_batch(token_lists)
        
        return feature_vectors
    
    def train(self, safe_queries: List[str], injection_queries: List[str],
              validation_split: float = 0.2, test_split: float = 0.1,
              output_path: Optional[str] = None) -> SVMClassifier:
        """
        Train Stage 1 SVM classifier
        
        Args:
            safe_queries: List of safe SQL queries
            injection_queries: List of SQL injection queries
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            output_path: Optional path to save trained model
        
        Returns:
            Trained SVMClassifier instance
        """
        # Prepare labels
        safe_labels = [0] * len(safe_queries)
        injection_labels = [1] * len(injection_queries)
        
        all_queries = safe_queries + injection_queries
        all_labels = safe_labels + injection_labels
        
        print(f"Training on {len(safe_queries)} safe and {len(injection_queries)} injection queries")
        
        # Prepare features
        X = self.prepare_features(all_queries)
        y = np.array(all_labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split / (1 - test_split),
            random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
        
        # Train model
        self.svm.train(X_train, y_train, calibrate=True)
        
        # Evaluate on validation set
        print("\nValidation Set Performance:")
        y_val_pred, y_val_proba = self.svm.predict_with_threshold(X_val)
        print(classification_report(y_val, y_val_pred, target_names=['Safe', 'Injection']))
        
        # Find optimal threshold using validation set
        print("\nFinding optimal threshold...")
        best_threshold = self._find_optimal_threshold(X_val, y_val)
        self.svm.set_threshold(best_threshold)
        print(f"Optimal threshold: {best_threshold:.4f}")
        
        # Evaluate on test set with optimal threshold
        print("\nTest Set Performance:")
        y_test_pred, y_test_proba = self.svm.predict_with_threshold(X_test)
        print(classification_report(y_test, y_test_pred, target_names=['Safe', 'Injection']))
        
        try:
            auc = roc_auc_score(y_test, y_test_proba)
            print(f"ROC-AUC Score: {auc:.4f}")
        except:
            pass
        
        # Save model
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.svm.save(output_path)
            print(f"\nModel saved to {output_path}")
        else:
            svm_config = self.stage1_config.get('svm', {})
            model_path = svm_config.get('model_path', 'models/stage1_svm.pkl')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.svm.save(model_path)
            print(f"\nModel saved to {model_path}")
        
        return self.svm
    
    def _find_optimal_threshold(self, X_val: np.ndarray, y_val: np.ndarray, 
                                 metric: str = 'f1') -> float:
        """
        Find optimal threshold using validation set
        
        Args:
            X_val: Validation feature vectors
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
        
        Returns:
            Optimal threshold value
        """
        proba = self.svm.predict_proba(X_val)
        prob_injection = proba[:, 1]
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (prob_injection >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_val, predictions)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, predictions, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, predictions, zero_division=0)
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def load_trained_model(self, model_path: Optional[str] = None) -> SVMClassifier:
        """
        Load a trained SVM model
        
        Args:
            model_path: Optional path to model file
        
        Returns:
            Loaded SVMClassifier instance
        """
        if model_path is None:
            svm_config = self.stage1_config.get('svm', {})
            model_path = svm_config.get('model_path', 'models/stage1_svm.pkl')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SVM model not found at {model_path}")
        
        self.svm = SVMClassifier()
        self.svm.load(model_path)
        return self.svm