"""Performance metrics and tracking for SQL injection detector"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StageMetrics:
    """Metrics for a single stage"""
    stage_name: str
    total_queries: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    decisions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_query(self, latency_ms: float, decision: Optional[str] = None):
        """Record a query processing event"""
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        if decision:
            self.decisions[decision] += 1
    
    def get_avg_latency_ms(self) -> float:
        """Get average latency in milliseconds"""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'stage_name': self.stage_name,
            'total_queries': self.total_queries,
            'avg_latency_ms': self.get_avg_latency_ms(),
            'min_latency_ms': self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            'max_latency_ms': self.max_latency_ms,
            'decisions': dict(self.decisions)
        }


class MetricsCollector:
    """Collects and aggregates metrics across all stages"""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize metrics collector
        
        Args:
            enabled: Whether to collect metrics
        """
        self.enabled = enabled
        self.stages: Dict[str, StageMetrics] = {}
        self.pipeline_metrics = {
            'total_queries': 0,
            'total_blocked': 0,
            'total_allowed': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    def get_stage_metrics(self, stage_name: str) -> StageMetrics:
        """Get or create metrics for a stage"""
        if not self.enabled:
            return None
        
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(stage_name)
        return self.stages[stage_name]
    
    def record_stage(self, stage_name: str, latency_ms: float, decision: Optional[str] = None):
        """Record a stage execution"""
        if not self.enabled:
            return
        
        metrics = self.get_stage_metrics(stage_name)
        metrics.add_query(latency_ms, decision)
    
    def record_pipeline_decision(self, decision: str, is_correct: Optional[bool] = None):
        """Record a pipeline-level decision"""
        if not self.enabled:
            return
        
        self.pipeline_metrics['total_queries'] += 1
        if decision == 'BLOCK':
            self.pipeline_metrics['total_blocked'] += 1
        elif decision == 'ALLOW':
            self.pipeline_metrics['total_allowed'] += 1
        
        if is_correct is False:
            if decision == 'BLOCK':
                self.pipeline_metrics['false_positives'] += 1
            else:
                self.pipeline_metrics['false_negatives'] += 1
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        if not self.enabled:
            return {}
        
        summary = {
            'pipeline': self.pipeline_metrics.copy(),
            'stages': {}
        }
        
        for stage_name, metrics in self.stages.items():
            summary['stages'][stage_name] = metrics.to_dict()
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.stages.clear()
        self.pipeline_metrics = {
            'total_queries': 0,
            'total_blocked': 0,
            'total_allowed': 0,
            'false_positives': 0,
            'false_negatives': 0
        }


class LatencyTimer:
    """Context manager for measuring latency"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None, 
                 stage_name: Optional[str] = None):
        """
        Initialize latency timer
        
        Args:
            metrics_collector: Optional metrics collector to record to
            stage_name: Name of stage being timed
        """
        self.metrics_collector = metrics_collector
        self.stage_name = stage_name
        self.start_time = None
        self.latency_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.latency_ms = (end_time - self.start_time) * 1000.0
        
        if self.metrics_collector and self.stage_name:
            self.metrics_collector.record_stage(self.stage_name, self.latency_ms)
        
        return False
    
    def get_latency_ms(self) -> float:
        """Get latency in milliseconds"""
        return self.latency_ms if self.latency_ms is not None else 0.0