"""Data loading utilities for training"""

import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional


def load_from_csv(filepath: str, query_column: str = 'query', 
                  label_column: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Load training data from CSV file
    
    Args:
        filepath: Path to CSV file
        query_column: Name of column containing SQL queries
        label_column: Name of column containing labels (0=safe, 1=injection)
    
    Returns:
        Tuple of (queries, labels)
    """
    queries = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get(query_column, '').strip()
            label = int(row.get(label_column, 0))
            
            if query:
                queries.append(query)
                labels.append(label)
    
    return queries, labels


def load_from_json(filepath: str, query_key: str = 'query',
                   label_key: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Load training data from JSON file
    
    Args:
        filepath: Path to JSON file
        query_key: Key for SQL query in JSON objects
        label_key: Key for label in JSON objects
    
    Returns:
        Tuple of (queries, labels)
    """
    queries = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                query = item.get(query_key, '').strip()
                label = int(item.get(label_key, 0))
                
                if query:
                    queries.append(query)
                    labels.append(label)
        else:
            # Assume it's a dict with lists
            queries = [q.strip() for q in data.get(query_key, [])]
            labels = [int(l) for l in data.get(label_key, [])]
    
    return queries, labels


def load_from_text(filepath: str, label: int = 0) -> List[str]:
    """
    Load queries from text file (one per line)
    
    Args:
        filepath: Path to text file
        label: Label to assign to all queries
    
    Returns:
        List of queries
    """
    queries = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            query = line.strip()
            if query:
                queries.append(query)
    
    return queries


def split_safe_and_injection(queries: List[str], labels: List[int]) -> Tuple[List[str], List[str]]:
    """
    Split queries into safe and injection lists
    
    Args:
        queries: List of SQL queries
        labels: List of labels (0=safe, 1=injection)
    
    Returns:
        Tuple of (safe_queries, injection_queries)
    """
    safe_queries = [q for q, l in zip(queries, labels) if l == 0]
    injection_queries = [q for q, l in zip(queries, labels) if l == 1]
    
    return safe_queries, injection_queries


def load_training_data(data_path: str, format: str = 'auto') -> Tuple[List[str], List[str]]:
    """
    Load training data from directory or file
    
    Args:
        data_path: Path to data file or directory
        format: Data format ('csv', 'json', 'text', or 'auto' to detect)
    
    Returns:
        Tuple of (safe_queries, injection_queries)
    """
    data_path_obj = Path(data_path)
    
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    if data_path_obj.is_file():
        # Single file
        if format == 'auto':
            if data_path_obj.suffix == '.csv':
                format = 'csv'
            elif data_path_obj.suffix == '.json':
                format = 'json'
            else:
                format = 'text'
        
        if format == 'csv':
            queries, labels = load_from_csv(str(data_path_obj))
        elif format == 'json':
            queries, labels = load_from_json(str(data_path_obj))
        else:
            queries = load_from_text(str(data_path_obj))
            labels = [0] * len(queries)  # Assume all safe if no labels
        
        return split_safe_and_injection(queries, labels)
    
    else:
        # Directory - look for safe and injection files
        safe_file = data_path_obj / 'safe_queries.txt'
        injection_file = data_path_obj / 'injection_queries.txt'
        
        safe_queries = []
        injection_queries = []
        
        if safe_file.exists():
            safe_queries = load_from_text(str(safe_file), label=0)
        
        if injection_file.exists():
            injection_queries = load_from_text(str(injection_file), label=1)
        
        if not safe_queries and not injection_queries:
            raise ValueError(f"No training data found in {data_path}")
        
        return safe_queries, injection_queries