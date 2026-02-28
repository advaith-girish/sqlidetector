"""Linear SVM classifier for Stage 1"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional, Tuple


class SVMClassifier:
    """
    Linear SVM classifier using SGD for fast training and inference
    
    Uses probability calibration for threshold tuning
    """
    
    def __init__(self, threshold: float = 0.5, model_path: Optional[str] = None):
        """
        Initialize SVM classifier
        
        Args:
            threshold: Probability threshold for classification (default: 0.5)
            model_path: Optional path to load pre-trained model
        """
        self.threshold = threshold
        self.model = None
        self.is_calibrated = False
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            # Initialize untrained model
            # Use hinge loss for SVM, L2 penalty, high learning rate for fast convergence
            self.model = SGDClassifier(
                loss='hinge',  # SVM loss
                penalty='l2',
                alpha=0.0001,  # Regularization strength
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True):
        """
        Train the SVM classifier
        
        Args:
            X: Feature vectors (shape: [n_samples, n_features])
            y: Labels (0 for safe, 1 for injection)
            calibrate: Whether to calibrate probabilities
        """
        if self.model is None:
            self.model = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1
            )
        
        # Train the model
        print(f"Training SVM on {len(X)} samples...")
        self.model.fit(X, y)
        
        # Calibrate probabilities if requested
        if calibrate:
            print("Calibrating probabilities...")
            self.model = CalibratedClassifierCV(self.model, cv=3, method='sigmoid')
            self.model.fit(X, y)
            self.is_calibrated = True
        
        print("SVM training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature vectors (shape: [n_samples, n_features])
        
        Returns:
            Predicted labels (0 for safe, 1 for injection)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature vectors (shape: [n_samples, n_features])
        
        Returns:
            Probability array (shape: [n_samples, 2])
            Column 0: probability of safe, Column 1: probability of injection
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if not self.is_calibrated:
            # If not calibrated, use decision function and convert to probability
            decision = self.model.decision_function(X)
            # Convert to probability using sigmoid
            prob_injection = 1 / (1 + np.exp(-decision))
            prob_safe = 1 - prob_injection
            return np.column_stack([prob_safe, prob_injection])
        
        return self.model.predict_proba(X)
    
    def predict_with_threshold(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with threshold-based decision
        
        Args:
            X: Feature vectors (shape: [n_samples, n_features])
        
        Returns:
            Tuple of (predictions, probabilities)
            - predictions: 0 for ALLOW, 1 for BLOCK
            - probabilities: probability of injection for each sample
        """
        proba = self.predict_proba(X)
        prob_injection = proba[:, 1]  # Probability of injection
        predictions = (prob_injection >= self.threshold).astype(int)
        
        return predictions, prob_injection
    
    def classify_single(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single query
        
        Args:
            feature_vector: Single feature vector (shape: [n_features])
        
        Returns:
            Tuple of (decision, confidence)
            - decision: 'ALLOW' or 'BLOCK'
            - confidence: Probability of injection
        """
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        predictions, prob_injection = self.predict_with_threshold(feature_vector)
        
        decision = 'BLOCK' if predictions[0] == 1 else 'ALLOW'
        confidence = float(prob_injection[0])
        
        return decision, confidence
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'is_calibrated': self.is_calibrated
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load the model from
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.threshold = model_data.get('threshold', 0.5)
        self.is_calibrated = model_data.get('is_calibrated', False)
    
    def set_threshold(self, threshold: float):
        """Set classification threshold"""
        self.threshold = threshold