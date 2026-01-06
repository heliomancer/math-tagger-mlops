import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import hydra
import os
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient
from math_classifier.datamodule import MathDataModule
from math_classifier.model import MathClassifier
from math_classifier.inference_pipeline import MathTaggerPipeline
import onnxruntime as ort
import os

def train_logreg(cfg: DictConfig):
    # 1. Reproducibility
    L.seed_everything(cfg.seed)

    # 2. Data Setup
    print("Initializing DataModule...")
    dm = MathDataModule(cfg.data, cfg.model, seed=cfg.seed)
    dm.setup() 
    
    input_dim = len(dm.vectorizer.vocabulary_)
    num_classes = len(dm.label2idx)
    print(f"Data ready. Input Dim: {input_dim}, Classes: {num_classes}")

    # 3. Model Setup
    model = MathClassifier(input_dim=input_dim, num_classes=num_classes, lr=cfg.model.lr)

    # 4. Logger & Callbacks
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=False 
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="best-checkpoint",
        monitor="val_f1",
        mode="max",
        save_top_k=1
    )
    
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
    
    # 7. Test
    print("FINAL EVALUATION ON TEST SET")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    
    # 8. Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    model.to("cpu")
    
    onnx_dir = "models/onnx_export"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    
    dummy_input = torch.randn(1, input_dim)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14 
    )
    print(f"ONNX model saved to {onnx_dir}")

    # 9. Logging Pipeline to MLflow
    print("Logging Pipeline to MLflow...")
    
    artifacts = {
        "vectorizer": "models/vectorizer.joblib",
        "label_map": "models/label2idx.joblib",
        "onnx_model": onnx_dir 
    }

    input_text = cfg.train.get("input_example", "Calculate the area of a circle.")

    # run temporary session to calculate predicted result
    temp_pipeline = MathTaggerPipeline()
    temp_pipeline.mode = "onnx"
    temp_pipeline.threshold = cfg.model.get("threshold", 0.5)
    temp_pipeline.vectorizer = dm.vectorizer
    temp_pipeline.label2idx = dm.label2idx
    temp_pipeline.idx2label = {v: k for k, v in dm.label2idx.items()}
    temp_pipeline.ort_session = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"))
    
    prediction_result = temp_pipeline.predict(context=None, model_input=[input_text])

    signature = mlflow.models.infer_signature(
        model_input=[input_text], 
        model_output=prediction_result
    )

    with mlflow.start_run(run_id=mlf_logger.run_id):

        final_pipeline = MathTaggerPipeline()
        final_pipeline.threshold = cfg.model.get("threshold", 0.5)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=final_pipeline,
            artifacts=artifacts,
            signature=signature,
            input_example=[input_text],
            registered_model_name="MathTagger"
        )
        
        # Auto-Promote
        client = MlflowClient()
        latest_version = client.get_latest_versions("MathTagger", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="MathTagger",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Pipeline registered as MathTagger version {latest_version} (Production)")

    print("Training and Logging Complete.")

