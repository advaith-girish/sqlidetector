"""Pre-filter: validate query before any ML stage."""

from typing import Tuple


class QueryValidator:
    """Rejects obviously invalid or unsafe inputs before ML."""

    def __init__(self, max_length: int = 65536, reject_null_bytes: bool = True):
        self.max_length = max_length
        self.reject_null_bytes = reject_null_bytes

    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Validate query. Returns (ok, reason).
        If ok is False, reason explains why (e.g. "empty", "too_long").
        """
        if not query or not isinstance(query, str):
            return False, "empty_or_invalid"
        q = query.strip()
        if not q:
            return False, "empty"
        if len(q) > self.max_length:
            return False, "too_long"
        if self.reject_null_bytes and "\x00" in q:
            return False, "null_byte"
        return True, ""