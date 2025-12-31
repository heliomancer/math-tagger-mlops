import joblib
import onnxruntime as ort
import numpy as np
import os
import re

class MathPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.vectorizer = None
        self.label_map = None
        self.ort_session = None
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load Preprocessing Artifacts
        vec_path = os.path.join(self.model_dir, "vectorizer.joblib")
        label_path = os.path.join(self.model_dir, "label2idx.joblib")
        onnx_path = os.path.join(self.model_dir, "model.onnx")

        if not os.path.exists(vec_path) or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Artifacts not found in {self.model_dir}. Run training first.")

        print("Loading artifacts...")
        self.vectorizer = joblib.load(vec_path)
        self.label2idx = joblib.load(label_path)
        # Create reverse map: Index -> Label Name
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # 2. Load ONNX Model
        # output_names=['output'] ensures we get the logits
        self.ort_session = ort.InferenceSession(onnx_path)
        print("Model loaded successfully.")

    def predict(self, text):
        # 1. Preprocess (TF-IDF)
        # Note: clean text if you had cleaning logic in training (we didn't, just raw)
        tfidf_vector = self.vectorizer.transform([text]).toarray().astype(np.float32)

        # 2. Inference (ONNX)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        
        logits = self.ort_session.run([output_name], {input_name: tfidf_vector})[0]
        
        # 3. Postprocess (Logits -> Probabilities -> Label)
        # Sigmoid
        probs = 1 / (1 + np.exp(-logits))
        
        # Get highest probability class
        pred_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][pred_idx]
        predicted_label = self.idx2label[pred_idx]
        
        return predicted_label, float(confidence)

def infer(cfg):
    # This function is called by commands.py
    input_text = cfg.get("input_text", "Calculate the area of a circle with radius 5.")
    
    predictor = MathPredictor(model_dir="models")
    label, conf = predictor.predict(input_text)
    
    print("\n" + "="*30)
    print(f"Input: {input_text}")
    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {conf:.4f}")
    print("="*30 + "\n")
