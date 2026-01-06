import os
import joblib
import pandas as pd
import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score
from math_classifier.inference_pipeline import MathTaggerPipeline


def train_catboost(cfg: DictConfig):
    # 1. Load Data
    print("Loading data...")
    train_path = os.path.join(cfg.data.data_dir, cfg.data.train_file)
    full_train_df = pd.read_csv(train_path)
    
    # 2. Split Data (Same seed as PyTorch)
    train_df, val_df = train_test_split(
        full_train_df, 
        test_size=cfg.data.val_frac, 
        random_state=cfg.seed,
        stratify=full_train_df['type']
    )
    
    # 3. Prepare Features & Labels
    X_train = train_df[['problem']]
    y_train = train_df['type']    
    X_val = val_df[['problem']]
    y_val = val_df['type']
    
    # 4. Initialize CatBoost
    unique_labels = sorted(full_train_df['type'].unique())
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    
    y_train_idx = y_train.map(label2idx)
    y_val_idx = y_val.map(label2idx)
    
    print(f"Training CatBoost with params: {cfg.model}")

    eval_metric = cfg.model.get("eval_metric", "Accuracy")
    
    model = CatBoostClassifier(
        iterations=cfg.model.iterations,
        learning_rate=cfg.model.learning_rate,
        depth=cfg.model.depth,
        loss_function='MultiClass',
        eval_metric=cfg.model.get("eval_metric", "Accuracy"), # Value from config
        text_features=['problem'], # Native text support
        random_seed=cfg.seed,
        verbose=100,
        allow_writing_files=False # Cleaner directory
    )
    
    # 5. Fit
    model.fit(
        X_train, y_train_idx,
        eval_set=(X_val, y_val_idx),
        early_stopping_rounds=50
    )

    # 6. Metrics Calculation
    print("Calculating Final Metrics...")
    val_preds = model.predict(X_val).flatten()
   
    final_f1 = f1_score(y_val_idx, val_preds, average='micro')
    final_jaccard = jaccard_score(y_val_idx, val_preds, average='micro')
    
    print(f"Validation F1: {final_f1:.4f}")
    print(f"Validation Jaccard: {final_jaccard:.4f}")
    
    
    # 7. Generate MLflow Signature
    print("Generating MLflow Signature...")
    input_text = cfg.train.get("input_example", "Calculate the area of a circle.")
    
    temp_pipeline = MathTaggerPipeline()
    temp_pipeline.mode = "catboost"
    temp_pipeline.threshold = cfg.model.get("threshold", 0.5)
    temp_pipeline.model = model
    temp_pipeline.label2idx = label2idx
    temp_pipeline.idx2label = {v: k for k, v in label2idx.items()}
    
    # Run prediction
    prediction_result = temp_pipeline.predict(context=None, model_input=[input_text])
    
    signature = mlflow.models.infer_signature(
        model_input=[input_text], 
        model_output=prediction_result
    )

    # 8. Log to MLflow
    print("Logging CatBoost Pipeline to MLflow...")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Save Local Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/catboost_model.cbm"
    model.save_model(model_path)
    joblib.dump(label2idx, "models/label2idx.joblib")
    
    # Define artifacts
    artifacts = {
        "catboost_model": model_path,
        "label_map": "models/label2idx.joblib"
    }
    
    with mlflow.start_run():

        mlflow.log_param("model_type", "catboost")
        mlflow.log_param("iterations", cfg.model.iterations)
        mlflow.log_param("depth", cfg.model.depth)
        mlflow.log_param("learning_rate", cfg.model.learning_rate)
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_param("threshold", cfg.model.get("threshold", 0.5))

        # Log Learning Curves (History) from catboost: train and validation

        evals_result = model.get_evals_result()

        for metric_name, values in evals_result['learn'].items():
            for step, val in enumerate(values):
                mlflow.log_metric(f"train_{metric_name}", val, step=step)
                
        for metric_name, values in evals_result['validation'].items():
            for step, val in enumerate(values):
                mlflow.log_metric(f"val_{metric_name}", val, step=step)

        # Log Final Snapshot Metrics

        mlflow.log_metric("final_f1", final_f1)
        mlflow.log_metric("final_jaccard", final_jaccard)

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
        
        # Auto-promote logic
        client = MlflowClient()
        latest_version = client.get_latest_versions("MathTagger", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="MathTagger", version=latest_version, stage="Production", archive_existing_versions=True
        )
        print(f"CatBoost Model promoted to Production (Version {latest_version})")
