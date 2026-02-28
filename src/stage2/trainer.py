"""Training script for Stage 2 DistilBERT"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

from .distilbert_model import QuantizedDistilBERT
from ..utils.config import get_config


class SQLInjectionDataset(Dataset):
    """Dataset for SQL injection classification"""
    
    def __init__(self, queries: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            queries: List of SQL query strings
            labels: List of labels (0 for safe, 1 for injection)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = str(self.queries[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class Stage2Trainer:
    """Trainer for Stage 2 DistilBERT model"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Stage 2 trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.stage2_config = self.config.get_stage2_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_config = self.stage2_config.get('model', {})
        self.model_name = model_config.get('model_name', 'distilbert-base-uncased')
        self.max_length = model_config.get('max_length', 512)
        self.batch_size = model_config.get('batch_size', 16)
        self.quantized = model_config.get('quantized', True)
    
    def train(self, safe_queries: List[str], injection_queries: List[str],
              validation_split: float = 0.2, test_split: float = 0.1,
              num_epochs: int = 3, learning_rate: float = 2e-5,
              output_path: Optional[str] = None) -> QuantizedDistilBERT:
        """
        Train DistilBERT model
        
        Args:
            safe_queries: List of safe SQL queries
            injection_queries: List of SQL injection queries
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            output_path: Optional path to save trained model
        
        Returns:
            Trained QuantizedDistilBERT instance
        """
        # Prepare data
        safe_labels = [0] * len(safe_queries)
        injection_labels = [1] * len(injection_queries)
        
        all_queries = safe_queries + injection_queries
        all_labels = safe_labels + injection_labels
        
        print(f"Training on {len(safe_queries)} safe and {len(injection_queries)} injection queries")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_queries, all_labels, test_size=test_split, random_state=42, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split / (1 - test_split),
            random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
        
        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        config = DistilBertConfig.from_pretrained(self.model_name, num_labels=2)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )
        model.to(self.device)
        
        # Create datasets
        train_dataset = SQLInjectionDataset(X_train, y_train, tokenizer, self.max_length)
        val_dataset = SQLInjectionDataset(X_val, y_val, tokenizer, self.max_length)
        test_dataset = SQLInjectionDataset(X_test, y_test, tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    predictions = (probs[:, 1] >= 0.5).cpu().numpy()
                    
                    val_predictions.extend(predictions)
                    val_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("New best model!")
        
        # Evaluate on test set
        print("\nTest Set Performance:")
        model.eval()
        test_predictions = []
        test_probs = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = (probs[:, 1] >= 0.5).cpu().numpy()
                
                test_predictions.extend(predictions)
                test_probs.extend(probs[:, 1].cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        print(classification_report(test_labels, test_predictions, target_names=['Safe', 'Injection']))
        
        try:
            auc = roc_auc_score(test_labels, test_probs)
            print(f"ROC-AUC Score: {auc:.4f}")
        except:
            pass
        
        # Create QuantizedDistilBERT wrapper
        distilbert_model = QuantizedDistilBERT(
            model_name=self.model_name,
            quantized=self.quantized,
            max_length=self.max_length
        )
        distilbert_model.model = model
        distilbert_model.tokenizer = tokenizer
        
        # Save model
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            distilbert_model.save(output_path, export_onnx=self.quantized)
            print(f"\nModel saved to {output_path}")
        else:
            model_config = self.stage2_config.get('model', {})
            model_path = model_config.get('model_path', 'models/stage2_distilbert')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            distilbert_model.save(model_path, export_onnx=self.quantized)
            print(f"\nModel saved to {model_path}")
        
        return distilbert_model
    
    def load_trained_model(self, model_path: Optional[str] = None) -> QuantizedDistilBERT:
        """
        Load a trained DistilBERT model
        
        Args:
            model_path: Optional path to model directory
        
        Returns:
            Loaded QuantizedDistilBERT instance
        """
        if model_path is None:
            model_config = self.stage2_config.get('model', {})
            model_path = model_config.get('model_path', 'models/stage2_distilbert')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"DistilBERT model not found at {model_path}")
        
        model_config = self.stage2_config.get('model', {})
        return QuantizedDistilBERT(
            model_name=model_config.get('model_name', 'distilbert-base-uncased'),
            model_path=model_path,
            quantized=model_config.get('quantized', True),
            max_length=model_config.get('max_length', 512)
        )