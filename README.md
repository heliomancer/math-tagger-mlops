
# Math Tagger MLOps

Math tagger problem is a part of Thesis research on intelligent tutoring systems. This module chosen for MLOps course assigns topical tags (e.g., "Algebra", "Geometry") to mathematics problems based on their textual content. 
This project implements a full pipeline: data versioning, experiment tracking, reproducible training, and model serving.

## 📌 Project Overview

### Problem Statement
The main goal is to classify mathematics problems into one or more categories based on problem text. Text can contain LaTeX markdown and should be in English. 

### Input and Output
*   **Input:** A JSON object or raw string containing the math problem text.
*   **Output:** A list of predicted categories (Multilabel).
    *   *Example:* `["algebra", "calculus"]`

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
uv run math-tagger-mlops mode=download
```

---

## 🚀 Usage

All project commands are unified under a single entry point managed by Hydra configuration.

### 1. Training
Run the training pipeline. This includes data loading, TF-IDF vectorization, and PyTorch Lightning training.

```bash
uv run math-tagger-mlops mode=train
```

*   **Customizing Hyperparameters:** You can override any config value via CLI.
    ```bash
    # Example: Change learning rate and max epochs
    uv run math-tagger-mlops mode=train model.lr=0.005 train.max_epochs=50
    ```

### 2. Inference (CLI)
Run predictions on custom text input directly from the command line.

```bash
uv run math-tagger-mlops mode=infer input_text="Find the area of a circle with radius 5"
```

### 3. Production Preparation (ONNX)
The training pipeline automatically exports the best model to ONNX format at the end of the run.
*   **Location:** `models/model.onnx`
*   **Artifacts:** Preprocessing mappings are saved in `models/*.joblib`.

---

## 📡 Serving (MLflow)

The project includes an inference server using MLflow.

1.  **Start the MLflow Server:**
    You need the Run ID of your best training run (found in `mlruns` or the console output after training).
    ```bash
    # Replace <RUN_ID> with your actual ID
    export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
    uv run mlflow models serve -m runs:/<RUN_ID>/model -p 5000 --no-conda
    ```

2.  **Send Requests:**
    *Note: The current server implementation expects pre-vectorized inputs (TF-IDF features).*
    
    You can use the provided check script to verify the server is running:
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
├── models/                # Saved models and artifacts
├── src/
│   └── math_classifier/
│       ├── commands.py    # Main CLI entry point
│       ├── datamodule.py  # Lightning DataModule
│       ├── model.py       # Lightning Module (Architecture)
│       ├── train.py       # Training logic
│       └── infer.py       # Inference logic
├── pyproject.toml         # Dependencies and project metadata
└── uv.lock                # Pinned dependencies
```

