"""SQL query tokenization for Stage 1 feature extraction"""

import re
from typing import List


class SQLTokenizer:
    """Tokenizes SQL queries into words, operators, and symbols"""
    
    # SQL keywords (common ones)
    SQL_KEYWORDS = {
        'select', 'from', 'where', 'insert', 'update', 'delete', 'create', 'drop',
        'alter', 'table', 'database', 'index', 'view', 'procedure', 'function',
        'union', 'join', 'inner', 'left', 'right', 'outer', 'on', 'as', 'and',
        'or', 'not', 'in', 'like', 'between', 'is', 'null', 'order', 'by',
        'group', 'having', 'limit', 'offset', 'distinct', 'count', 'sum', 'avg',
        'max', 'min', 'case', 'when', 'then', 'else', 'end', 'if', 'else',
        'while', 'for', 'loop', 'begin', 'commit', 'rollback', 'transaction',
        'grant', 'revoke', 'exec', 'execute', 'declare', 'set', 'use', 'show',
        'describe', 'explain', 'truncate', 'backup', 'restore', 'sleep', 'waitfor',
        'benchmark', 'pg_sleep', 'delay', 'load_file', 'into', 'outfile', 'dumpfile'
    }
    
    # SQL operators
    SQL_OPERATORS = {
        '=', '!=', '<>', '<', '>', '<=', '>=', '<=>', '||', '&&',
        '+', '-', '*', '/', '%', '&', '|', '^', '~', '<<', '>>'
    }
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize tokenizer
        
        Args:
            case_sensitive: Whether to preserve case in tokens
        """
        self.case_sensitive = case_sensitive
    
    def tokenize(self, query: str) -> List[str]:
        """
        Tokenize SQL query into tokens
        
        Args:
            query: SQL query string
        
        Returns:
            List of tokens (keywords, identifiers, operators, literals, etc.)
        """
        if not query:
            return []
        
        tokens = []
        
        # Pattern to match:
        # - String literals (single or double quoted)
        # - Numbers (integers and floats)
        # - Identifiers (alphanumeric + underscore)
        # - Operators and punctuation
        # - Whitespace (to split on)
        
        # First, extract string literals
        string_pattern = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
        strings = string_pattern.findall(query)
        query_without_strings = string_pattern.sub(' __STRING__ ', query)
        
        # Extract numbers
        number_pattern = re.compile(r'\b\d+\.?\d*\b')
        numbers = number_pattern.findall(query_without_strings)
        query_without_numbers = number_pattern.sub(' __NUMBER__ ', query_without_strings)
        
        # Split by whitespace and punctuation
        # Keep operators and punctuation as separate tokens
        token_pattern = re.compile(r'(\w+|[^\w\s])')
        raw_tokens = token_pattern.findall(query_without_numbers)
        
        # Process tokens
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            
            # Normalize case if needed
            if not self.case_sensitive:
                token_lower = token.lower()
            else:
                token_lower = token
            
            # Check if it's a keyword
            if token_lower in self.SQL_KEYWORDS:
                tokens.append(f"KEYWORD_{token_lower}")
            # Check if it's an operator
            elif token in self.SQL_OPERATORS:
                tokens.append(f"OP_{token}")
            # Check if it's a placeholder
            elif token == '__STRING__':
                tokens.append('LITERAL_STRING')
            elif token == '__NUMBER__':
                tokens.append('LITERAL_NUMBER')
            # Otherwise it's an identifier or other token
            else:
                # Preserve original case for identifiers
                tokens.append(f"ID_{token_lower if not self.case_sensitive else token}")
        
        # Add special tokens for common injection patterns
        # Check for suspicious patterns
        query_lower = query.lower()
        if 'union' in query_lower and 'select' in query_lower:
            tokens.append('PATTERN_UNION_SELECT')
        if 'or' in query_lower and ('1=1' in query_lower or "'1'='1'" in query_lower):
            tokens.append('PATTERN_OR_TRUE')
        if '--' in query or '/*' in query:
            tokens.append('PATTERN_COMMENT')
        if ';' in query and ('drop' in query_lower or 'delete' in query_lower):
            tokens.append('PATTERN_MULTI_STMT')
        
        return tokens
    
    def tokenize_batch(self, queries: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of queries
        
        Args:
            queries: List of SQL query strings
        
        Returns:
            List of token lists
        """
        return [self.tokenize(q) for q in queries]