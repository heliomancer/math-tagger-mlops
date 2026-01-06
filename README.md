# Math Tagger MLOps

Math tagger problem is a part of Thesis research on intelligent tutoring systems. This module, developed for an MLOps course, assigns topical tags (e.g., "Algebra", "Geometry") to mathematics problems based on their textual content.

This project implements a full production-grade pipeline:

- **Dual-Backend Training** (PyTorch & CatBoost)
- **Unified Inference Pipeline** (Abstraction layer for model switching)
- **Automatic Model Registry** (MLflow Versioning & Promotion)
- **Reproducibility** (DVC & Hydra)
- **Model Serving** (MLflow Model Registry)

## 📌 Project Overview

### Problem Statement

The main goal is to classify mathematics problems into one or more categories based on problem text. The system supports **Multilabel Classification**, handling cases where a problem belongs to multiple domains (e.g., _Algebra_ AND _Geometry_). Text can contain LaTeX markdown and should be in English.

### Input and Output

- **Input:** Raw string or JSON containing the problem text.
- **Output:** A structured JSON object containing predicted labels, probabilities, and model metadata.
  - _Example:_ `{"labels": ["algebra"], "probabilities": {"algebra": 0.92, ...}, "model_type": "CATBOOST"}`

### Metrics

- **Primary:** Jaccard Score (Intersection over Union).
- **Secondary:** F1-Score (Micro Average).

---

## 🛠️ Technical Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/heliomancer/math-tagger-mlops.git
    cd math-tagger-mlops
    ```

2.  **Install dependencies using `uv`:**
    This will create a virtual environment and sync all locked dependencies.

    ```bash
    uv sync
    ```

3.  **Setup Pre-commit hooks:**
    Ensures code quality before every commit.
    ```bash
    uv run pre-commit install
    ```

### Data Management

This project uses DVC for data versioning. However, for ease of reproduction without shared cloud storage credentials, a download script is provided to fetch the dataset from HuggingFace.

```bash
uv run python math_classifier/commands.py mode=download
```

---

## 🚀 Training Pipeline

The project uses **Hydra** for configuration management. You can switch between different model backends easily. All training runs automatically register the model to MLflow and promote the best result to **Production**.

### Option A: PyTorch Baseline (Logistic Regression)

Trains a linear classifier on TF-IDF vectors using PyTorch Lightning.

```bash
uv run python math_classifier/commands.py mode=train model=logreg
```

### Option B: CatBoost (Gradient Boosting)

Trains a CatBoost classifier using native text support (no external vectorizer).

```bash
uv run python math_classifier/commands.py mode=train model=catboost
```

### Customizing Hyperparameters

You can override any config value from the command line:

```bash
# Example: Train CatBoost with more iterations and a custom experiment name
uv run python math_classifier/commands.py mode=train model=catboost model.iterations=1000 mlflow.experiment_name="catboost_v2"
```

---

## 🔍 Inference & Usage

The project implements a **Unified Inference Pipeline**. Regardless of whether you trained a PyTorch or CatBoost model, the usage interface remains identical.

### CLI Inference

Run predictions on custom text. This command automatically downloads the current **Production** model from the MLflow Registry.

```bash
uv run python math_classifier/commands.py mode=infer input_text="Calculate the area of a circle with radius 5"
```

- **Runtime Thresholding:** You can override the decision boundary dynamically:
  ```bash
  # Force lower threshold to see more candidate labels
  uv run python math_classifier/commands.py mode=infer model.threshold=0.1
  ```

---

## 📡 Serving (MLflow)

We support serving the model via a REST API using MLflow. The server accepts **Raw Text**; the pipeline handles all preprocessing (vectorization/tokenization) internally.

1.  **Start the Server:**
    Serve the model tagged as `Production` on port 5000.

    ```bash
    export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
    uv run mlflow models serve -m "models:/MathTagger/Production" -p 5000 --no-conda
    ```

2.  **Send Requests:**
    - **Input Format:** `{"inputs": ["Problem text"]}`

    Verify connectivity using the check script:

    ```bash
    uv run python server_check.py
    ```

---

## 📁 Project Structure

The project follows a flat-layout structure with clear separation of concerns.

```text
├── configs/               # Hydra configuration files
│   ├── data/              # Dataset configs
│   ├── model/             # Model configs (logreg.yaml, catboost.yaml)
│   └── train/             # General training settings
├── data/                  # Data storage (gitignored)
├── math_classifier/       # Main Package
│   ├── trainers/          # Model-specific training logic
│   │   ├── pytorch_trainer.py
│   │   └── catboost_trainer.py
│   ├── commands.py            # Main CLI entry point
│   ├── datamodule.py          # PyTorch Lightning DataModule
│   ├── model.py               # PyTorch Lightning Module
│   ├── inference_pipeline.py  # Unified PyFunc Wrapper (The "Production" Model)
│   ├── train.py               # Dispatcher (Selects trainer based on config)
│   └── infer.py               # CLI Inference Client
├── models/                # Local artifact staging
├── pyproject.toml         # Dependencies
└── uv.lock                # Locked dependencies
```
