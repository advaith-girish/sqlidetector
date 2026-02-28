"""Quantized DistilBERT model for Stage 2 semantic analysis"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig
)
import onnxruntime as ort


class QuantizedDistilBERT:
    """
    Quantized DistilBERT model for SQL injection detection
    
    Uses INT8 quantization for faster inference
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 model_path: Optional[str] = None,
                 quantized: bool = True,
                 max_length: int = 512):
        """
        Initialize DistilBERT model
        
        Args:
            model_name: HuggingFace model name
            model_path: Optional path to load fine-tuned model
            quantized: Whether to use quantized model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.quantized = quantized
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load model
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            # Initialize with base model
            config = DistilBertConfig.from_pretrained(model_name, num_labels=2)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name, config=config
            )
            self.model.to(self.device)
            self.model.eval()
        
        # ONNX runtime session for quantized inference (if available)
        self.ort_session = None
        if quantized and model_path:
            onnx_path = Path(model_path) / "model_quantized.onnx"
            if onnx_path.exists():
                self._load_onnx_model(str(onnx_path))
    
    def _load_model(self, model_path: str):
        """Load fine-tuned model from path"""
        model_path_obj = Path(model_path)
        
        if (model_path_obj / "pytorch_model.bin").exists() or \
           (model_path_obj / "model.safetensors").exists():
            # Load from HuggingFace format
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            # Load from single file
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self, onnx_path: str):
        """Load quantized ONNX model for faster inference"""
        try:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.ort_session = ort.InferenceSession(
                onnx_path,
                providers=providers
            )
            print(f"Loaded quantized ONNX model from {onnx_path}")
        except Exception as e:
            print(f"Warning: Could not load ONNX model: {e}")
            self.ort_session = None
    
    def predict(self, query: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Predict if query is SQL injection
        
        Args:
            query: SQL query string
            threshold: Probability threshold for classification
        
        Returns:
            Tuple of (decision, confidence)
            - decision: 'ALLOW' or 'BLOCK'
            - confidence: Probability of injection
        """
        # Tokenize
        encoded = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        if self.ort_session is not None:
            # Use ONNX runtime for quantized inference
            outputs = self.ort_session.run(
                None,
                {
                    'input_ids': input_ids.cpu().numpy().astype(np.int64),
                    'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
                }
            )
            logits = torch.tensor(outputs[0])
        else:
            # Use PyTorch model
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        prob_injection = float(probs[0][1].item())
        
        # Make decision
        decision = 'BLOCK' if prob_injection >= threshold else 'ALLOW'
        
        return decision, prob_injection
    
    def predict_batch(self, queries: List[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Predict on a batch of queries (single forward pass for GPU efficiency).

        Args:
            queries: List of SQL query strings
            threshold: Probability threshold

        Returns:
            List of (decision, confidence) tuples
        """
        if not queries:
            return []
        # Tokenize batch
        encoded = self.tokenizer(
            queries,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        # Single forward pass
        if self.ort_session is not None:
            outputs = self.ort_session.run(
                None,
                {
                    'input_ids': input_ids.cpu().numpy().astype(np.int64),
                    'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
                }
            )
            logits = torch.tensor(outputs[0])
        else:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prob_injection = probs[:, 1].cpu().numpy()
        return [
            ('BLOCK' if p >= threshold else 'ALLOW', float(p))
            for p in prob_injection
        ]
    
    def save(self, output_path: str, export_onnx: bool = True):
        """
        Save model to disk
        
        Args:
            output_path: Directory to save model
            export_onnx: Whether to export quantized ONNX model
        """
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save in HuggingFace format
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Export ONNX if requested
        if export_onnx:
            self._export_onnx(output_path_obj)
    
    def _export_onnx(self, output_dir: Path):
        """Export model to ONNX format for quantization"""
        try:
            onnx_path = output_dir / "model.onnx"
            
            # Create dummy input
            dummy_input = self.tokenizer(
                "SELECT * FROM users",
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_input['input_ids'].to(self.device),
                 dummy_input['attention_mask'].to(self.device)),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            # Quantize ONNX model (INT8)
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                quantized_path = output_dir / "model_quantized.onnx"
                quantize_dynamic(
                    str(onnx_path),
                    str(quantized_path),
                    weight_type=QuantType.QUInt8
                )
                print(f"Exported quantized ONNX model to {quantized_path}")
            except Exception as e:
                print(f"Warning: Could not quantize ONNX model: {e}")
            
        except Exception as e:
            print(f"Warning: Could not export ONNX model: {e}")