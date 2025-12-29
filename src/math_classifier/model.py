import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from sklearn.metrics import f1_score
import numpy as np

class MathClassifier(L.LightningModule):
    def __init__(self, input_dim, num_classes, lr=0.01):
        super().__init__()
        self.save_hyperparameters() # Logs params to MLflow automatically
        
        self.lr = lr
        
        # Architecture (Same as your baseline)
        self.model = nn.Linear(input_dim, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

#    def training_step(self, batch, batch_idx):
#        x, y = batch
#        logits = self(x)
#        loss = self.criterion(logits, y)
#        
#        # Log training loss
#        self.log("train_loss", loss, prog_bar=True)
#        return loss
#
#    def validation_step(self, batch, batch_idx):
#        x, y = batch
#        logits = self(x)
#        loss = self.criterion(logits, y)
#        
#        # Calculate Metrics
#        preds = (torch.sigmoid(logits) > 0.5).float()
#        
#        # We need to move tensors to CPU for sklearn metrics
#        y_true = y.cpu().numpy()
#        y_pred = preds.cpu().numpy()
#        
#        # Calculate F1 Micro (as per your notebook)
#        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
#        
#        self.log("val_loss", loss, prog_bar=True)
#        self.log("val_f1", f1, prog_bar=True)
#        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Convert to binary predictions
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        # Move to CPU for sklearn/numpy metrics
        y_true = y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        # 1. Micro F1
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 2. Jaccard Score (Intersection over Union)
        # We calculate it manually or use sklearn.metrics.jaccard_score(..., average='samples')
        # Manual is often faster for tensors, but let's use sklearn for safety
        from sklearn.metrics import jaccard_score
        jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        
        # Log metrics
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_f1", f1, prog_bar=True)
        self.log(f"{prefix}_jaccard", jaccard, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")
        
    def test_step(self, batch, batch_idx):
        # This is what runs when we call trainer.test()
        return self._shared_eval_step(batch, batch_idx, "test")

