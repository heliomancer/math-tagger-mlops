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
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # 2. Load ONNX Model
        self.ort_session = ort.InferenceSession(onnx_path)
        print("Model loaded successfully.")

#    def predict(self, text):
#        # 1. Preprocess (TF-IDF)
#        # raw text no cleaning
#        tfidf_vector = self.vectorizer.transform([text]).toarray().astype(np.float32)
#
#        # 2. Inference (ONNX)
#        input_name = self.ort_session.get_inputs()[0].name
#        output_name = self.ort_session.get_outputs()[0].name
#        
#        logits = self.ort_session.run([output_name], {input_name: tfidf_vector})[0]
#        
#        # 3. Postprocess (Logits -> Probabilities -> Label)
#        # Sigmoid
#        probs = 1 / (1 + np.exp(-logits))
#        # Taking only more than threshold or maximum
#        threshold = 0.5
#        pred_indices = np.where(probs[0] > threshold)[0]
#        if len(pred_indices) == 0:
#            pred_indices = [np.argmax(probs, axis=1)[0]]
#        predicted_labels = [self.idx2label[idx] for idx in pred_indices]
#        return predicted_labels, probs[0][pred_indices]


    def predict(self, text):
        # 1. Preprocess
        tfidf_vector = self.vectorizer.transform([text]).toarray().astype(np.float32)

        # 2. Inference
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        logits = self.ort_session.run([output_name], {input_name: tfidf_vector})[0]
        
        # 3. Postprocess (first item in batch)
        all_probs = 1 / (1 + np.exp(-logits))[0]         
        threshold = 0.5
        pred_indices = np.where(all_probs > threshold)[0]
        
        # Fallback to top-1 if nothing crosses threshold
        if len(pred_indices) == 0:
            pred_indices = [np.argmax(all_probs)]
        predicted_labels = [self.idx2label[i] for i in pred_indices]
        return predicted_labels, all_probs



#def infer(cfg):
#    input_text = cfg.get("input_text", "Calculate the area of a circle with radius 5.")    
#    predictor = MathPredictor(model_dir="models")
#    labels, probs = predictor.predict(input_text)
#    
#    print("\n" + "="*30)
#    print(f"Input: {input_text}")    
#    formatted_labels = ", ".join([l.upper() for l in labels])
#    print(f"Prediction: [{formatted_labels}]") 
#    print(f"Confidences: {probs}") 
#
#    if True: #change to cfg flag later
#        top3_indices = np.argsort(probs[0])[-3:][::-1]
#        print("Top 3 Candidates:")
#        for idx in top3_indices:
#            print(f"  {predictor.idx2label[idx]}: {probs[0][idx]:.4f}")
#
#    print("="*30 + "\n")


def infer(cfg):
    input_text = cfg.get("input_text", "Calculate the area of a circle with radius 5.")
    
    predictor = MathPredictor(model_dir="models")
    # Now probs is a 1D array of all class probabilities (e.g., length 7)
    labels, probs = predictor.predict(input_text) 
    
    print("\n" + "="*30)
    print(f"Input: {input_text}")    
    formatted_labels = ", ".join([l.upper() for l in labels])
    print(f"Prediction: [{formatted_labels}]")
    
    confidences = [
        f"{probs[predictor.label2idx[l]]:.4f}" for l in labels
    ]
    print(f"Confidences: {confidences}")

    # Flag for debugging
    if True: 
        print("-" * 20)
        print("Top 3 Candidates:")
        # argsort gives indices of sorted elements (ascending), so we take last 3 and reverse
        top3_indices = np.argsort(probs)[-3:][::-1]
        
        for idx in top3_indices:
            label_name = predictor.idx2label[idx]
            confidence = probs[idx]
            print(f"  {label_name:<20}: {confidence:.4f}")

    print("="*30 + "\n")
