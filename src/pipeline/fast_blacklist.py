"""Fast blacklist: instant BLOCK for high-confidence attack patterns."""

import re
from typing import List, Tuple


class FastBlacklist:
    """Match high-confidence SQL injection patterns for immediate BLOCK."""

    # Normalized patterns (lowercase, we match after lowercasing query)
    # Keep patterns specific to avoid false positives (e.g. "or 1=1" in value is rare in safe queries)
    PATTERNS: List[Tuple[str, str]] = [
        (r"\bor\s*['\"]?\s*1\s*=\s*1\b", "or_1_equals_1"),
        (r"\bor\s*['\"]?\s*1\s*=\s*['\"]?\s*1\b", "or_1_equals_1_quoted"),
        (r"\bunion\s+select\b", "union_select"),
        (r"\bunion\s+all\s+select\b", "union_all_select"),
        (r";\s*drop\s+table\b", "semicolon_drop_table"),
        (r";\s*delete\s+from\b", "semicolon_delete_from"),
        (r"\bexec\s*\(\s*", "exec_paren"),
        (r"\bexecute\s+immediate\b", "execute_immediate"),
        (r"\bpg_sleep\s*\(", "pg_sleep"),
        (r"\bbenchmark\s*\(", "benchmark"),
        (r"\bsleep\s*\(", "sleep"),
        (r"\bwaitfor\s+delay\b", "waitfor_delay"),
        (r"\bload_file\s*\(", "load_file"),
        (r"\binto\s+outfile\b", "into_outfile"),
        (r"\binto\s+dumpfile\b", "into_dumpfile"),
        (r"\bconcat\s*\(\s*.*\s*\)\s*from\s+information_schema", "concat_information_schema"),
    ]

    def __init__(self, patterns: List[Tuple[str, str]] = None):
        if patterns is not None:
            self._compiled = [(re.compile(p, re.IGNORECASE | re.DOTALL), name) for p, name in patterns]
        else:
            self._compiled = [(re.compile(p, re.IGNORECASE | re.DOTALL), name) for p, name in self.PATTERNS]

    def match(self, query: str) -> Tuple[bool, str]:
        """
        Returns (is_blacklisted, trigger_name).
        If is_blacklisted is True, trigger_name is the matched pattern id.
        """
        if not query:
            return False, ""
        q = query
        for regex, name in self._compiled:
            if regex.search(q):
                return True, name
        return False, ""