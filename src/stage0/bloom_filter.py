"""Bloom filter implementation for Stage 0 whitelist"""

import mmh3
import pickle
from bitarray import bitarray
from typing import Optional
import math


class BloomFilter:
    """
    Bloom filter for fast probabilistic membership testing
    
    Uses MurmurHash3 for hashing and bitarray for efficient bit storage
    """
    
    def __init__(self, capacity: int, false_positive_rate: float = 0.001):
        """
        Initialize Bloom filter
        
        Args:
            capacity: Expected number of elements to store
            false_positive_rate: Desired false positive rate (e.g., 0.001 for 0.1%)
        """
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal number of bits and hash functions
        # m = -n * ln(p) / (ln(2)^2)
        # k = m * ln(2) / n
        self.num_bits = int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))
        self.num_hash_functions = int((self.num_bits / capacity) * math.log(2))
        
        # Ensure at least 1 hash function
        if self.num_hash_functions < 1:
            self.num_hash_functions = 1
        
        # Initialize bit array
        self.bit_array = bitarray(self.num_bits)
        self.bit_array.setall(0)
        
        self.count = 0  # Track number of elements added
    
    def _get_hash_indices(self, item: str) -> list:
        """
        Get all hash indices for an item
        
        Args:
            item: String to hash
        
        Returns:
            List of bit indices
        """
        indices = []
        # Use double hashing technique: h_i(x) = (h1(x) + i * h2(x)) mod m
        hash1 = mmh3.hash(item, seed=0) % self.num_bits
        hash2 = mmh3.hash(item, seed=1) % self.num_bits
        
        for i in range(self.num_hash_functions):
            index = (hash1 + i * hash2) % self.num_bits
            indices.append(index)
        
        return indices
    
    def add(self, item: str):
        """
        Add an item to the Bloom filter
        
        Args:
            item: String to add
        """
        if not item:
            return
        
        indices = self._get_hash_indices(item)
        for index in indices:
            self.bit_array[index] = 1
        
        self.count += 1
    
    def contains(self, item: str) -> bool:
        """
        Check if item might be in the Bloom filter
        
        Args:
            item: String to check
        
        Returns:
            True if item might be present (may be false positive),
            False if item is definitely not present
        """
        if not item:
            return False
        
        indices = self._get_hash_indices(item)
        for index in indices:
            if not self.bit_array[index]:
                return False  # Definitely not present
        
        return True  # Might be present (could be false positive)
    
    def add_batch(self, items: list):
        """
        Add multiple items to the Bloom filter
        
        Args:
            items: List of strings to add
        """
        for item in items:
            self.add(item)
    
    def save(self, filepath: str):
        """
        Save Bloom filter to disk
        
        Args:
            filepath: Path to save the filter
        """
        data = {
            'capacity': self.capacity,
            'false_positive_rate': self.false_positive_rate,
            'num_bits': self.num_bits,
            'num_hash_functions': self.num_hash_functions,
            'bit_array': self.bit_array,
            'count': self.count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BloomFilter':
        """
        Load Bloom filter from disk
        
        Args:
            filepath: Path to load the filter from
        
        Returns:
            Loaded BloomFilter instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance
        instance = cls(data['capacity'], data['false_positive_rate'])
        instance.num_bits = data['num_bits']
        instance.num_hash_functions = data['num_hash_functions']
        instance.bit_array = data['bit_array']
        instance.count = data['count']
        
        return instance
    
    def get_stats(self) -> dict:
        """
        Get statistics about the Bloom filter
        
        Returns:
            Dictionary with filter statistics
        """
        return {
            'capacity': self.capacity,
            'count': self.count,
            'num_bits': self.num_bits,
            'num_hash_functions': self.num_hash_functions,
            'false_positive_rate': self.false_positive_rate,
            'load_factor': self.count / self.capacity if self.capacity > 0 else 0.0
        }