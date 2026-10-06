import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
import os

def calculate_metrics(y_true, y_prob, y_pred):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = float('nan')
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return metrics, cm

class FusionTrainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer
        opt_name = config.get("optimizer", "AdamW")
        lr = config.get("learning_rate", 1e-3)
        if opt_name == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        elif opt_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
            
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            if len(batch) == 3:
                sem_batch, struct_batch, y_batch = batch
                sem_batch = sem_batch.to(self.device)
                struct_batch = struct_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                logits, _, _, _ = self.model(sem_batch, struct_batch)
                loss = self.criterion(logits, y_batch)
                
            elif len(batch) == 2: # Baseline A
                sem_batch, y_batch = batch
                sem_batch = sem_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                logits, _, _, _ = self.model(sem_batch)
                loss = self.criterion(logits, y_batch)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    sem_batch, struct_batch, y_batch = batch
                    sem_batch = sem_batch.to(self.device)
                    struct_batch = struct_batch.to(self.device)
                    y_batch = y_batch.to(self.device).float().unsqueeze(1)
                    
                    logits, _, _, _ = self.model(sem_batch, struct_batch)
                    
                elif len(batch) == 2:
                    sem_batch, y_batch = batch
                    sem_batch = sem_batch.to(self.device)
                    y_batch = y_batch.to(self.device).float().unsqueeze(1)
                    
                    logits, _, _, _ = self.model(sem_batch)
                    
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())
                
        all_probs = np.array(all_probs).flatten()
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        metrics, cm = calculate_metrics(all_targets, all_probs, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics, cm
        
    def train(self, train_loader, val_loader=None, save_path=None):
        epochs = self.config.get("epochs", 50)
        patience = self.config.get("early_stopping_patience", 10)
        
        best_val_loss = float('inf')
        best_val_f1 = -1.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            if val_loader:
                val_metrics, _ = self.evaluate(val_loader)
                val_loss = val_metrics['loss']
                val_f1 = val_metrics['f1']
                
                # We can do early stopping on loss or F1. Let's do loss for standard deep learning practice.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    
                    if save_path:
                        torch.save(best_state, save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        # Load best model if we have a validation set
        if val_loader and best_state is not None:
            self.model.load_state_dict(best_state)
            if save_path:
                torch.save(best_state, save_path)
        elif save_path:
            # If no validation set, just save the final model
            torch.save(self.model.state_dict(), save_path)
