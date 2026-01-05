import mlflow
import joblib
import onnxruntime as ort
import numpy as np
import pandas as pd
import os

class MathTaggerPipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # 1. Load Vectorizer
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        
        # 2. Load Label Map
        self.label2idx = joblib.load(context.artifacts["label_map"])
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        
        # 3. Load ONNX Session
        # FIX: context.artifacts["onnx_model"] now points to the FOLDER
        model_dir = context.artifacts["onnx_model"]
        model_path = os.path.join(model_dir, "model.onnx")
        
        self.ort_session = ort.InferenceSession(model_path)

    def predict(self, context, model_input):
        
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = model_input

        tfidf_vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)

        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        logits = self.ort_session.run([output_name], {input_name: tfidf_vectors})[0]
        
        all_probs = 1 / (1 + np.exp(-logits))
        
        results = []
        threshold = 0.5
        
        for i, probs in enumerate(all_probs):
            pred_indices = np.where(probs > threshold)[0]
            if len(pred_indices) == 0:
                pred_indices = [np.argmax(probs)]
            
            labels = [self.idx2label[idx] for idx in pred_indices]
            
            full_prob_dict = {self.idx2label[k]: float(p) for k, p in enumerate(probs)}
            
            results.append({
                "labels": labels,
                "probabilities": full_prob_dict 
            })
            
        return results
