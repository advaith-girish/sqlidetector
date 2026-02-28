"""Main pipeline orchestrator for SQL injection detection"""

from typing import Optional, Dict, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from ..pipeline.query_normalizer import QueryNormalizer
from ..pipeline.query_validator import QueryValidator
from ..pipeline.fast_blacklist import FastBlacklist
from ..stage0.bloom_filter import BloomFilter
from ..stage1.tokenizer import SQLTokenizer
from ..stage1.feature_hasher import FeatureHasher
from ..stage1.svm_classifier import SVMClassifier
from ..stage2.distilbert_model import QuantizedDistilBERT
from ..utils.config import get_config
from ..utils.metrics import MetricsCollector, LatencyTimer

# Tokens that suggest injection (for explainability when Stage 1 blocks)
SUSPICIOUS_TOKEN_PREFIXES = ("PATTERN_", "KEYWORD_union", "KEYWORD_benchmark", "KEYWORD_sleep", "KEYWORD_pg_sleep", "KEYWORD_load_file", "KEYWORD_exec")


class SQLInjectionDetector:
    """
    Main SQL injection detector with cascaded hierarchical inference
    
    Pipeline:
    1. Stage 0: Bloom filter whitelist (fast, handles 80-90% of traffic)
    2. Stage 1: Linear SVM (fast rejection of obvious attacks)
    3. Stage 2: Quantized DistilBERT (deep semantic analysis)
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 metrics_enabled: bool = True):
        """
        Initialize SQL injection detector
        
        Args:
            config_path: Path to configuration file
            metrics_enabled: Whether to collect performance metrics
        """
        self.config = get_config(config_path)
        self.pipeline_config = self.config.get_pipeline_config()
        self.metrics = MetricsCollector(enabled=metrics_enabled and
                                        self.pipeline_config.get('metrics_enabled', True))

        # Pre-filter and fast blacklist
        self.enable_prefilter = self.pipeline_config.get('enable_prefilter', True)
        self.enable_fast_blacklist = self.pipeline_config.get('enable_fast_blacklist', True)
        max_query_length = self.pipeline_config.get('max_query_length', 65536)
        self.query_validator = QueryValidator(max_length=max_query_length) if self.enable_prefilter else None
        self.fast_blacklist = FastBlacklist() if self.enable_fast_blacklist else None
        self.stage2_timeout_ms = self.pipeline_config.get('stage2_timeout_ms') or 0
        self.strict_mode = self.pipeline_config.get('strict_mode', False)
        self.include_trigger = self.pipeline_config.get('include_trigger', True)

        # Initialize components
        self.normalizer = QueryNormalizer()

        # Stage 0: Bloom filter
        self.stage0_enabled = self.pipeline_config.get('enable_stage0', True)
        self.bloom_filter = None
        if self.stage0_enabled:
            self._load_stage0()
        
        # Stage 1: SVM
        self.stage1_enabled = self.pipeline_config.get('enable_stage1', True)
        self.tokenizer = None
        self.feature_hasher = None
        self.svm_classifier = None
        if self.stage1_enabled:
            self._load_stage1()
        
        # Stage 2: DistilBERT
        self.stage2_enabled = self.pipeline_config.get('enable_stage2', True)
        self.distilbert = None
        if self.stage2_enabled:
            self._load_stage2()
    
    def _load_stage0(self):
        """Load Stage 0 Bloom filter"""
        try:
            stage0_config = self.config.get_stage0_config()
            bloom_config = stage0_config.get('bloom_filter', {})
            model_path = bloom_config.get('model_path', 'models/stage0_bloom.bin')
            
            if Path(model_path).exists():
                self.bloom_filter = BloomFilter.load(model_path)
                print(f"Loaded Stage 0 Bloom filter from {model_path}")
            else:
                print(f"Warning: Stage 0 Bloom filter not found at {model_path}")
                self.stage0_enabled = False
        except Exception as e:
            print(f"Warning: Could not load Stage 0: {e}")
            self.stage0_enabled = False
    
    def _load_stage1(self):
        """Load Stage 1 SVM classifier"""
        try:
            stage1_config = self.config.get_stage1_config()
            
            # Initialize tokenizer
            tokenizer_config = stage1_config.get('tokenizer', {})
            self.tokenizer = SQLTokenizer(
                case_sensitive=tokenizer_config.get('case_sensitive', False)
            )
            
            # Initialize feature hasher
            hasher_config = stage1_config.get('feature_hashing', {})
            self.feature_hasher = FeatureHasher(
                vector_size=hasher_config.get('vector_size', 65536),
                num_hash_functions=hasher_config.get('num_hash_functions', 2)
            )
            
            # Load SVM classifier
            svm_config = stage1_config.get('svm', {})
            model_path = svm_config.get('model_path', 'models/stage1_svm.pkl')
            
            if Path(model_path).exists():
                self.svm_classifier = SVMClassifier()
                self.svm_classifier.load(model_path)
                self.svm_classifier.set_threshold(svm_config.get('threshold', 0.5))
                print(f"Loaded Stage 1 SVM from {model_path}")
            else:
                print(f"Warning: Stage 1 SVM not found at {model_path}")
                self.stage1_enabled = False
        except Exception as e:
            print(f"Warning: Could not load Stage 1: {e}")
            self.stage1_enabled = False
    
    def _load_stage2(self):
        """Load Stage 2 DistilBERT model"""
        try:
            stage2_config = self.config.get_stage2_config()
            model_config = stage2_config.get('model', {})
            model_path = model_config.get('model_path', 'models/stage2_distilbert')
            
            if Path(model_path).exists():
                self.distilbert = QuantizedDistilBERT(
                    model_name=model_config.get('model_name', 'distilbert-base-uncased'),
                    model_path=model_path,
                    quantized=model_config.get('quantized', True),
                    max_length=model_config.get('max_length', 512)
                )
                print(f"Loaded Stage 2 DistilBERT from {model_path}")
            else:
                print(f"Warning: Stage 2 DistilBERT not found at {model_path}")
                self.stage2_enabled = False
        except Exception as e:
            print(f"Warning: Could not load Stage 2: {e}")
            self.stage2_enabled = False
    
    def _fallback_decision(self, result: Dict, stage: str, trigger: str) -> Dict:
        """Set fallback decision (e.g. timeout or error). Uses strict_mode."""
        result['stage'] = stage
        result['latencies'][stage] = 0.0
        if self.include_trigger:
            result['trigger'] = trigger
        if self.strict_mode:
            result['decision'] = 'BLOCK'
            result['confidence'] = 1.0
        else:
            result['decision'] = 'ALLOW'
            result['confidence'] = 0.0
        self.metrics.record_pipeline_decision(result['decision'])
        return result

    def detect(self, query: str) -> Dict:
        """
        Detect SQL injection in a query.

        Returns dict with: decision, confidence, stage, latencies, and optionally trigger, suspicious_tokens.
        """
        result = {
            'decision': 'ALLOW',
            'confidence': 0.0,
            'stage': None,
            'latencies': {},
            'trigger': '',
            'suspicious_tokens': []
        }

        # Pre-filter: reject invalid input
        if self.enable_prefilter and self.query_validator:
            ok, reason = self.query_validator.validate(query)
            if not ok:
                result['decision'] = 'BLOCK'
                result['confidence'] = 1.0
                result['stage'] = 'prefilter'
                if self.include_trigger:
                    result['trigger'] = reason
                self.metrics.record_pipeline_decision('BLOCK')
                return result

        # Fast blacklist: instant BLOCK for high-confidence patterns
        if self.enable_fast_blacklist and self.fast_blacklist:
            is_blacklisted, trigger_name = self.fast_blacklist.match(query)
            if is_blacklisted:
                result['decision'] = 'BLOCK'
                result['confidence'] = 1.0
                result['stage'] = 'fast_blacklist'
                if self.include_trigger:
                    result['trigger'] = trigger_name
                self.metrics.record_pipeline_decision('BLOCK')
                return result

        # Normalize query for Stage 0
        normalized_query = self.normalizer.normalize(query)

        # Stage 0: Bloom filter whitelist
        if self.stage0_enabled and self.bloom_filter:
            with LatencyTimer(self.metrics, 'stage0') as timer:
                if self.bloom_filter.contains(normalized_query):
                    result['decision'] = 'ALLOW'
                    result['confidence'] = 1.0
                    result['stage'] = 'stage0'
                    result['latencies']['stage0'] = timer.get_latency_ms()
                    if self.include_trigger:
                        result['trigger'] = 'stage0'
                    if self.pipeline_config.get('log_decisions', False):
                        print(f"Stage 0: ALLOW (whitelist match)")
                    self.metrics.record_pipeline_decision('ALLOW')
                    return result
            
            result['latencies']['stage0'] = timer.get_latency_ms()
        
        # Stage 1: Linear SVM
        if self.stage1_enabled and self.tokenizer and self.feature_hasher and self.svm_classifier:
            with LatencyTimer(self.metrics, 'stage1') as timer:
                tokens = self.tokenizer.tokenize(query)
                feature_vector = self.feature_hasher.hash_tokens(tokens)
                decision, confidence = self.svm_classifier.classify_single(feature_vector)
                
                result['latencies']['stage1'] = timer.get_latency_ms()
                
                if decision == 'ALLOW':
                    result['decision'] = 'ALLOW'
                    result['confidence'] = 1.0 - confidence
                    result['stage'] = 'stage1'
                    if self.include_trigger:
                        result['trigger'] = 'stage1_allow'
                    if self.pipeline_config.get('log_decisions', False):
                        print(f"Stage 1: ALLOW (confidence: {result['confidence']:.4f})")
                    self.metrics.record_pipeline_decision('ALLOW')
                    return result
                # Stage 1 blocked: add explainability
                if self.include_trigger and self.tokenizer:
                    tokens = self.tokenizer.tokenize(query)
                    result['suspicious_tokens'] = [
                        t for t in tokens
                        if t.startswith('PATTERN_') or any(t.startswith(p) for p in SUSPICIOUS_TOKEN_PREFIXES)
                    ]
                    result['trigger'] = 'stage1_suspicious'
                # Continue to Stage 2

        # Stage 2: DistilBERT (with optional timeout)
        if self.stage2_enabled and self.distilbert:
            stage2_config = self.config.get_stage2_config()
            threshold = stage2_config.get('threshold', 0.5)
            timeout_sec = (self.stage2_timeout_ms / 1000.0) if self.stage2_timeout_ms else None

            def _run_stage2() -> Tuple[str, float]:
                return self.distilbert.predict(query, threshold)

            with LatencyTimer(self.metrics, 'stage2') as timer:
                if timeout_sec and timeout_sec > 0:
                    try:
                        with ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(_run_stage2)
                            decision, confidence = fut.result(timeout=timeout_sec)
                    except (FuturesTimeoutError, Exception):
                        return self._fallback_decision(result, 'stage2_timeout', 'stage2_timeout')
                else:
                    decision, confidence = _run_stage2()

                result['decision'] = decision
                result['confidence'] = confidence if decision == 'BLOCK' else 1.0 - confidence
                result['stage'] = 'stage2'
                result['latencies']['stage2'] = timer.get_latency_ms()
                if self.include_trigger:
                    result['trigger'] = 'stage2'
                if self.pipeline_config.get('log_decisions', False):
                    print(f"Stage 2: {decision} (confidence: {result['confidence']:.4f})")
                self.metrics.record_pipeline_decision(decision)
                return result
        
        # Fallback: no stage handled it (e.g. models missing)
        result['decision'] = 'BLOCK' if self.strict_mode else 'ALLOW'
        result['confidence'] = 1.0 if self.strict_mode else 0.0
        result['stage'] = 'none'
        if self.include_trigger:
            result['trigger'] = 'no_stage'
        self.metrics.record_pipeline_decision(result['decision'])
        return result

    def detect_batch(self, queries: List[str]) -> List[Dict]:
        """
        Detect SQL injection in a batch of queries.
        Queries that need Stage 2 are run through Stage 2 in a single batch for efficiency.
        """
        if not queries:
            return []
        # First pass: run prefilter, blacklist, stage0, stage1 for each; collect indices needing stage2
        need_stage2: List[Tuple[int, str]] = []
        results: List[Dict] = [None] * len(queries)

        for i, query in enumerate(queries):
            r = {
                'decision': 'ALLOW',
                'confidence': 0.0,
                'stage': None,
                'latencies': {},
                'trigger': '',
                'suspicious_tokens': []
            }
            # Prefilter
            if self.enable_prefilter and self.query_validator:
                ok, reason = self.query_validator.validate(query)
                if not ok:
                    r['decision'] = 'BLOCK'
                    r['confidence'] = 1.0
                    r['stage'] = 'prefilter'
                    r['trigger'] = reason
                    results[i] = r
                    self.metrics.record_pipeline_decision('BLOCK')
                    continue
            # Fast blacklist
            if self.enable_fast_blacklist and self.fast_blacklist:
                is_bl, trig = self.fast_blacklist.match(query)
                if is_bl:
                    r['decision'] = 'BLOCK'
                    r['confidence'] = 1.0
                    r['stage'] = 'fast_blacklist'
                    r['trigger'] = trig
                    results[i] = r
                    self.metrics.record_pipeline_decision('BLOCK')
                    continue
            norm = self.normalizer.normalize(query)
            # Stage 0
            if self.stage0_enabled and self.bloom_filter and self.bloom_filter.contains(norm):
                r['decision'] = 'ALLOW'
                r['confidence'] = 1.0
                r['stage'] = 'stage0'
                r['trigger'] = 'stage0'
                results[i] = r
                self.metrics.record_pipeline_decision('ALLOW')
                continue
            # Stage 1
            if self.stage1_enabled and self.tokenizer and self.feature_hasher and self.svm_classifier:
                tokens = self.tokenizer.tokenize(query)
                fv = self.feature_hasher.hash_tokens(tokens)
                dec, conf = self.svm_classifier.classify_single(fv)
                if dec == 'ALLOW':
                    r['decision'] = 'ALLOW'
                    r['confidence'] = 1.0 - conf
                    r['stage'] = 'stage1'
                    r['trigger'] = 'stage1_allow'
                    results[i] = r
                    self.metrics.record_pipeline_decision('ALLOW')
                    continue
                if self.include_trigger and self.tokenizer:
                    r['suspicious_tokens'] = [t for t in tokens if t.startswith('PATTERN_') or any(t.startswith(p) for p in SUSPICIOUS_TOKEN_PREFIXES)]
                    r['trigger'] = 'stage1_suspicious'
            results[i] = r
            need_stage2.append((i, query))

        # Batch Stage 2 for all that need it
        if need_stage2 and self.stage2_enabled and self.distilbert:
            stage2_config = self.config.get_stage2_config()
            threshold = stage2_config.get('threshold', 0.5)
            indices = [t[0] for t in need_stage2]
            batch_queries = [t[1] for t in need_stage2]
            batch_results = self.distilbert.predict_batch(batch_queries, threshold)
            for (idx, _), (decision, confidence) in zip(need_stage2, batch_results):
                r = results[idx]
                if r is None:
                    r = {'decision': 'ALLOW', 'confidence': 0.0, 'stage': None, 'latencies': {}, 'trigger': '', 'suspicious_tokens': []}
                    results[idx] = r
                r['decision'] = decision
                r['confidence'] = confidence if decision == 'BLOCK' else 1.0 - confidence
                r['stage'] = 'stage2'
                r['trigger'] = 'stage2'
                self.metrics.record_pipeline_decision(decision)
        else:
            for idx, _ in need_stage2:
                r = results[idx]
                if r is None:
                    r = {'decision': 'ALLOW', 'confidence': 0.0, 'stage': None, 'latencies': {}, 'trigger': '', 'suspicious_tokens': []}
                    results[idx] = r
                if r['decision'] == 'ALLOW' and r['stage'] is None:
                    r['decision'] = 'BLOCK' if self.strict_mode else 'ALLOW'
                    r['confidence'] = 1.0 if self.strict_mode else 0.0
                    r['stage'] = 'none'
                    r['trigger'] = 'no_stage'
                    self.metrics.record_pipeline_decision(r['decision'])

        return results
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()