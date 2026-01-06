import os

import joblib
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
from catboost import CatBoostClassifier


class MathTaggerPipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.mode = "unknown"

        self.label2idx = joblib.load(context.artifacts["label_map"])
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # 2. Determine Mode
        if "onnx_model" in context.artifacts:
            self.mode = "onnx"
            self.vectorizer = joblib.load(context.artifacts["vectorizer"])
            model_dir = context.artifacts["onnx_model"]
            model_path = os.path.join(model_dir, "model.onnx")
            self.ort_session = ort.InferenceSession(model_path)

        elif "catboost_model" in context.artifacts:
            self.mode = "catboost"
            self.model = CatBoostClassifier()
            self.model.load_model(context.artifacts["catboost_model"])

    def predict(self, context, model_input):
        # Common Input Handling
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = model_input

        # Branching Inference
        all_probs = None

        if self.mode == "onnx":
            tfidf_vectors = (
                self.vectorizer.transform(texts).toarray().astype(np.float32)
            )
            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            logits = self.ort_session.run([output_name], {input_name: tfidf_vectors})[0]
            all_probs = 1 / (1 + np.exp(-logits))

        elif self.mode == "catboost":
            # CatBoost expects DataFrame with specific column name used in training
            # We assume the column was named "problem" in training
            df = pd.DataFrame({"problem": texts})

            # Predict Proba returns (N_samples, N_classes)
            # CatBoost MultiClass returns Softmax probabilities automatically
            all_probs = self.model.predict_proba(df)

        # Common Post-Processing
        results = []
        threshold = getattr(self, "threshold", 0.5)

        for i, probs in enumerate(all_probs):
            # 1. Filter by threshold
            pred_indices = np.where(probs > threshold)[0]

            # 2. Fallback: Top-1 if none crossed threshold
            if len(pred_indices) == 0:
                pred_indices = [np.argmax(probs)]

            # 3. Map to strings
            labels = [self.idx2label[idx] for idx in pred_indices]

            # 4. Create Dict
            full_prob_dict = {self.idx2label[k]: float(p) for k, p in enumerate(probs)}

            results.append(
                {
                    "labels": labels,
                    "probabilities": full_prob_dict,
                    "model_type": self.mode.upper(),
                }
            )

        return results
