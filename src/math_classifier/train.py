import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import hydra
from omegaconf import DictConfig
import mlflow


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
    
    # 7. Export to ONNX
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

    # 8. Logging artifacts to MLflow    
    print("Logging serving artifacts to MLflow...")
    # Load the best model weights
    best_model = MathClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Use the existing run context created by Lightning
    with mlflow.start_run(run_id=mlf_logger.run_id): # Log the PyTorch model
        signature = mlflow.models.infer_signature(dummy_input.numpy(), model(dummy_input).detach().numpy())
        # Create the signature using the dummy input we made for ONNX
        # dummy_input is the torch tensor of size (1, 2000) in numpy
        mlflow.pytorch.log_model(best_model, "model", signature=signature, registered_model_name="MathTagger")

        # Log the ONNX version too (optional, but good practice)
        mlflow.log_artifact("models/model.onnx", "model_onnx")

        # Log the preprocessors (CRITICAL for reproducibility)
        mlflow.log_artifact("models/vectorizer.joblib", "preprocessor")
        mlflow.log_artifact("models/label2idx.joblib", "preprocessor")

        # Auto-Promote to Production
        client = mlflow.tracking.MlflowClient()
        # Get the version we just created
        latest_version = client.get_latest_versions("MathTagger", stages=["None"])[0].version
        
        client.transition_model_version_stage(
            name="MathTagger",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True # demotes the old production model automatically
        )
        print(f"Model MathTagger version {latest_version} promoted to Production.")

    print("MLflow artifacts logged successfully.")
