import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import hydra
from omegaconf import DictConfig

from math_classifier.datamodule import MathDataModule
from math_classifier.model import MathClassifier

def train(cfg: DictConfig):
    # 1. Reproducibility
    L.seed_everything(cfg.seed)

    # 2. Data Setup
    # We init the datamodule and call setup() manually to get the vocab size
    print("Initializing DataModule...")
    dm = MathDataModule(cfg.data, cfg.model)
    dm.setup() 
    
    # Calculate dimensions for the model
    # TF-IDF vocab size
    input_dim = len(dm.vectorizer.vocabulary_)
    # Number of unique labels
    num_classes = len(dm.label2idx)
    
    print(f"Data ready. Input Dim: {input_dim}, Classes: {num_classes}")

    # 3. Model Setup
    model = MathClassifier(
        input_dim=input_dim, 
        num_classes=num_classes,
        lr=cfg.model.lr
    )

    # 4. Callbacks & Loggers
    
    # MLflow Logger
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=True
    )
    
    # Checkpointing: Save the best model based on Validation F1
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="best-checkpoint",
        monitor="val_f1",
        mode="max",
        save_top_k=1
    )
    
    # Early Stopping: Stop if loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor=cfg.train.monitor_metric,
        patience=cfg.train.patience,
        mode="min"
    )

    # 5. Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    # 6. Start Training
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # --- NEW: Run Test Set on Best Model ---
    print("\n" + "="*40)
    print("FINAL EVALUATION ON TEST SET")
    print("="*40)
    # This automatically loads the best checkpoint
    trainer.test(model, datamodule=dm, ckpt_path="best")
    print("="*40 + "\n")
    
    # 7. Production: Export to ONNX
    # We need a dummy input sample to trace the graph
    print("Exporting to ONNX...")
    model.eval()
    
    # Create a dummy input matching the TF-IDF vector size
    dummy_input = torch.randn(1, input_dim)
    
    onnx_path = "models/model.onnx"
    model.to_onnx(
        onnx_path, 
        dummy_input, 
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model saved to {onnx_path}")
