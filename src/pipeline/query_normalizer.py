"""SQL query normalization for Stage 0 whitelist matching"""

import re
from typing import Optional


class QueryNormalizer:
    """Normalizes SQL queries by replacing parameter values with placeholders"""
    
    # SQL string patterns (single and double quotes)
    STRING_PATTERN = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
    
    # Numeric patterns (integers and floats)
    NUMBER_PATTERN = re.compile(r'\b\d+\.?\d*\b')
    
    # IN clause pattern
    IN_CLAUSE_PATTERN = re.compile(r'\bIN\s*\([^)]+\)', re.IGNORECASE)
    # SQL comments (strip so normalization is consistent)
    LINE_COMMENT = re.compile(r'--[^\n]*', re.IGNORECASE)
    BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL | re.IGNORECASE)

    def normalize(self, query: str) -> str:
        """
        Normalize SQL query by replacing parameter values with placeholders.
        Strips SQL comments first for consistent whitelist matching.
        """
        if not query:
            return query

        normalized = query
        # Strip comments so evasion via comments doesn't change the template
        normalized = self.BLOCK_COMMENT.sub(" ", normalized)
        normalized = self.LINE_COMMENT.sub(" ", normalized)

        # Replace string literals
        normalized = self.STRING_PATTERN.sub("?", normalized)
        
        # Replace numeric literals (but preserve SQL keywords that might contain numbers)
        # We need to be careful not to replace numbers in SQL keywords
        normalized = self._replace_numbers_safely(normalized)
        
        # Normalize IN clauses: IN (1,2,3) -> IN (?)
        normalized = self.IN_CLAUSE_PATTERN.sub(lambda m: re.sub(
            r'\([^)]+\)', '(?)', m.group(), flags=re.IGNORECASE
        ), normalized)
        
        # Normalize whitespace (multiple spaces to single space)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Trim
        normalized = normalized.strip()
        
        return normalized
    
    def _replace_numbers_safely(self, query: str) -> str:
        """
        Replace numeric literals while preserving SQL keywords
        
        Args:
            query: SQL query string
        
        Returns:
            Query with numbers replaced by ?
        """
        # List of SQL keywords that might contain numbers (edge cases)
        # Most SQL keywords don't contain numbers, but we'll be conservative
        sql_keywords = {
            'sql', 'sql92', 'sql99', 'sql2003', 'sql2008', 'sql2011',
            'mysql', 'postgresql', 'mssql', 'oracle', 'sqlite'
        }
        
        def replace_number(match):
            num_str = match.group()
            # Check if this number is part of a word (likely a keyword)
            start, end = match.span()
            # Check context around the match
            if start > 0 and query[start-1].isalnum():
                return num_str  # Part of a word, don't replace
            if end < len(query) and query[end].isalnum():
                return num_str  # Part of a word, don't replace
            return '?'
        
        return self.NUMBER_PATTERN.sub(replace_number, query)
    
    def normalize_batch(self, queries: list) -> list:
        """
        Normalize a batch of queries
        
        Args:
            queries: List of SQL query strings
        
        Returns:
            List of normalized queries
        """
        return [self.normalize(q) for q in queries]