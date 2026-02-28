"""Feature hashing using MurmurHash3 for Stage 1"""

import mmh3
import numpy as np
from typing import List, Optional


class FeatureHasher:
    """
    Feature hashing using MurmurHash3 to convert tokens to fixed-size vectors
    
    Uses ensemble hashing (multiple hash functions) to reduce collisions
    """
    
    def __init__(self, vector_size: int = 65536, num_hash_functions: int = 2):
        """
        Initialize feature hasher
        
        Args:
            vector_size: Size of the feature vector (default: 65536)
            num_hash_functions: Number of hash functions for ensemble hashing
        """
        self.vector_size = vector_size
        self.num_hash_functions = num_hash_functions
    
    def hash_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Hash a list of tokens into a fixed-size feature vector
        
        Args:
            tokens: List of token strings
        
        Returns:
            Feature vector of size vector_size
        """
        feature_vector = np.zeros(self.vector_size, dtype=np.float32)
        
        for token in tokens:
            # Use multiple hash functions to reduce collisions
            for hash_idx in range(self.num_hash_functions):
                # Use different seeds for different hash functions
                hash_value = mmh3.hash(token, seed=hash_idx) % self.vector_size
                # Use absolute value to ensure positive index
                hash_value = abs(hash_value)
                # Increment the feature (can use TF-IDF weighting later)
                feature_vector[hash_value] += 1.0
        
        # Normalize by L2 norm to prevent bias from query length
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
        
        return feature_vector
    
    def hash_batch(self, token_lists: List[List[str]]) -> np.ndarray:
        """
        Hash a batch of token lists
        
        Args:
            token_lists: List of token lists
        
        Returns:
            Array of feature vectors (shape: [batch_size, vector_size])
        """
        vectors = [self.hash_tokens(tokens) for tokens in token_lists]
        return np.array(vectors)
    
    def get_vector_size(self) -> int:
        """Get the feature vector size"""
        return self.vector_size