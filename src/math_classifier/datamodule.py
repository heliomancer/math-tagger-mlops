import os
import pandas as pd
import numpy as np
import torch
import joblib
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

class MathDataModule(L.LightningDataModule):
    def __init__(self, data_cfg, model_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        
        # Paths
        self.train_path = os.path.join(data_cfg.data_dir, data_cfg.train_file)
        self.test_path = os.path.join(data_cfg.data_dir, data_cfg.test_file)
        
        # State placeholders
        self.vectorizer = None
        self.label2idx = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # 1. Load Data
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Data not found at {self.train_path}. Use data_local.py!")
            
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # 2. Fit Preprocessors (only on train data!)
        print("Fitting TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(max_features=self.model_cfg.max_features, stop_words='english')
        X_train_raw = self.vectorizer.fit_transform(train_df['problem']).toarray()
        
        # 3. Handle Labels
        print("Encoding Labels...")
        unique_labels = sorted(train_df['type'].unique())
        self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        
        # Helper to convert labels to one-hot vectors
        def encode_labels(df):
            y = torch.zeros((len(df), len(unique_labels)))
            for i, row in df.iterrows():
                if row['type'] in self.label2idx:
                    y[i, self.label2idx[row['type']]] = 1.0
            return y

        y_train = encode_labels(train_df)
        
        # 4. Process Validation/Test Data
        # Note: We transform using the fitted vectorizer (no fitting here)
        X_test_raw = self.vectorizer.transform(test_df['problem']).toarray()
        y_test = encode_labels(test_df)
        
        # 5. Convert to PyTorch Tensors
        self.train_dataset = TensorDataset(
            torch.tensor(X_train_raw, dtype=torch.float32), 
            y_train
        )
        self.val_dataset = TensorDataset(
            torch.tensor(X_test_raw, dtype=torch.float32), 
            y_test
        )
        
        # 6. Save Artifacts for Inference
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.vectorizer, "models/vectorizer.joblib")
        joblib.dump(self.label2idx, "models/label2idx.joblib")
        print("Saved preprocessing artifacts to models/")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=True, 
            num_workers=self.data_cfg.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=False, 
            num_workers=self.data_cfg.num_workers
        )

    def test_dataloader(self):
        # repeats val step
        return DataLoader(
            self.val_dataset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=False, 
            num_workers=self.data_cfg.num_workers
        )
