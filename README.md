# Math Tagger MLOps

Math tagger problem is a part of Thesis research on intelligent tutoring systems. This module, developed for an MLOps course, assigns topical tags (e.g., "Algebra", "Geometry") to mathematics problems based on their textual content.

This project implements a full production-grade pipeline:
*   **Data Versioning** (DVC)
*   **Experiment Tracking** (MLflow)
*   **Reproducible Training** (PyTorch Lightning + Hydra)
*   **Inference Pipeline** (Custom PyFunc wrapper for Text-to-Vector-to-Prediction)
*   **Model Serving** (MLflow Model Registry)

## 📌 Project Overview

### Problem Statement
The main goal is to classify mathematics problems into one or more categories based on problem text. Text can contain LaTeX markdown and should be in English.

### Input and Output
*   **Input:** A JSON object or raw string containing the math problem text.
*   **Output:** A list of predicted categories (Multilabel) and their probabilities.
    *   *Example:* `{"labels": ["algebra", "calculus"], "probabilities": {"algebra": 0.95, ...}}`

### Metrics
*   **Primary:** Jaccard Score (Intersection over Union).
*   **Secondary:** F1-Score (Micro Average).

---

## 🛠️ Technical Setup

### Prerequisites
*   Python 3.11+
*   [uv](https://github.com/astral-sh/uv) (for dependency management)
*   Git

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
uv run python src/math_classifier/commands.py mode=download
```

---

## 🚀 Usage

All project commands are unified under a single entry point managed by Hydra configuration.

### 1. Training
Run the training pipeline. This includes data loading, TF-IDF vectorization, PyTorch Lightning training, ONNX export, and **automatic Model Registry promotion**.

```bash
uv run python src/math_classifier/commands.py mode=train
```
*Training takes roughly 1-2 minutes for init and around 1 minute for training (on CPUs).*

*   **Customizing Hyperparameters:** You can override any config value via CLI.
    ```bash
    # Example: Change learning rate and max epochs
    uv run python src/math_classifier/commands.py mode=train model.lr=0.005 train.max_epochs=50
    ```

### 2. Inference (CLI)
Run predictions on custom text input directly from the command line.
This command automatically downloads the **Production** model pipeline from MLflow.

```bash
uv run python src/math_classifier/commands.py mode=infer input_text="Find the area of a circle with radius 5"
```

### 3. Architecture & Artifacts
The training process generates a **Unified Inference Pipeline**:
*   **ONNX Model:** The PyTorch model is exported to ONNX (bundled as a directory to handle large weights).
*   **Preprocessing:** The fitted TF-IDF Vectorizer and Label Encoders are saved.
*   **Wrapper:** A custom `MathTaggerPipeline` (MLflow PyFunc) wraps these artifacts. This ensures that the model accepts **Raw Text** input, handling vectorization internally.

---

## 📡 Serving (MLflow)

The project automatically registers and versions models in the MLflow Model Registry. The best model from the latest training run is automatically promoted to the `Production` stage.

1.  **Start the MLflow Server:**
    We serve the registered `MathTagger` model tagged as `Production`.
    
    ```bash
    export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
    uv run mlflow models serve -m "models:/MathTagger/Production" -p 5000 --no-conda
    ```
    *Server takes around 1-2 minutes to initialize.*

2.  **Send Requests:**
    The server accepts **Raw Text** wrapped in JSON.
    *   **Input Format:** `{"inputs": ["Problem text 1", "Problem text 2"]}`
    
    You can use the provided check script to verify the server is working:
    ```bash
    uv run python server_check.py
    ```

---

## 📁 Project Structure

```text
├── configs/               # Hydra configuration files
│   ├── data/              # Dataset configs
│   ├── model/             # Model hyperparameters
│   └── train/             # Trainer settings
├── data/                  # Data storage (gitignored)
├── models/                # Local artifact storage (staging)
├── src/
│   └── math_classifier/
│       ├── commands.py            # Main CLI entry point
│       ├── datamodule.py          # Lightning DataModule
│       ├── model.py               # Lightning Module (Neural Network)
│       ├── inference_pipeline.py  # Custom PyFunc Wrapper (The "Production" Model)
│       ├── train.py               # Training & Registration logic
│       └── infer.py               # CLI Inference Client
├── pyproject.toml         # Dependencies and project metadata
└── uv.lock                # Pinned dependencies
```

