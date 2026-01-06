import os
import pandas as pd
import numpy as np
import torch
import joblib
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class MathDataModule(L.LightningDataModule):
    def __init__(self, data_cfg, model_cfg, seed=99):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.seed = seed 
        
        # Paths
        self.train_path = os.path.join(data_cfg.data_dir, data_cfg.train_file)
        self.test_path = os.path.join(data_cfg.data_dir, data_cfg.test_file)
        
        # State placeholders
        self.vectorizer = None
        self.label2idx = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # 1. Load Data
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Data not found at {self.train_path}. Use math_classifier/commands.py mode=download!")
            
        full_train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        print(f"Loaded {len(full_train_df)} raw training samples and {len(test_df)} test samples.")
        
        # 2. Split Train into Train/Val
        # We use a fixed seed from config so this split is identical every run
        train_df, val_df = train_test_split(
            full_train_df, 
            test_size=self.data_cfg.val_frac, 
            random_state=self.seed,
            stratify=full_train_df['type'] # Good practice: keep label distribution similar
        )
        
        print(f"Splitting Train: {len(train_df)} Training, {len(val_df)} Validation.")
        
        # 3. Fit Preprocessors (Only on the reduced Train split!)
        # Prevents information leakage from validation set
        print("Fitting TF-IDF Vectorizer on Training split...")
        self.vectorizer = TfidfVectorizer(max_features=self.model_cfg.max_features, stop_words='english')
        
        X_train = self.vectorizer.fit_transform(train_df['problem']).toarray()
        X_val = self.vectorizer.transform(val_df['problem']).toarray()
        X_test = self.vectorizer.transform(test_df['problem']).toarray()
        
        # 4. Handle Labels
        print("Encoding Labels...")
        # We determine unique labels from the FULL training set to be safe
        unique_labels = sorted(full_train_df['type'].unique())
        self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        
        def encode_labels(df):
            y = torch.zeros((len(df), len(unique_labels)))
            # FIX: Use enumerate to generate a new 0-based index 'i'
            # We ignore 'original_index' coming from iterrows
            for i, (original_index, row) in enumerate(df.iterrows()):
                label_name = row['type']
                if label_name in self.label2idx:
                    y[i, self.label2idx[label_name]] = 1.0
            return y

        y_train = encode_labels(train_df)
        y_val = encode_labels(val_df)
        y_test = encode_labels(test_df)
        
        
        # 5. Convert to PyTorch Tensors
        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
        self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val)
        self.test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test)
        
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
            self.test_dataset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=False, 
            num_workers=self.data_cfg.num_workers
        )
