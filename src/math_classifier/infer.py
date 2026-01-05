import mlflow
import numpy as np

def infer(cfg):
    input_text = cfg.get("input_text", "Calculate the area of a circle.")
    
    print("\n" + "="*30)
    print(f"Input: {input_text}")
    
    model_uri = "models:/MathTagger/Production"
    print(f"Loading model from {model_uri}...")
    
    try:
        loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error: {e}")
        return

    results = loaded_pipeline.predict([input_text])
    result = results[0] # One input -> One result
    
    labels = result["labels"]
    probs_dict = result["probabilities"] 
    
    formatted_labels = ", ".join([l.upper() for l in labels])
    print(f"Prediction: [{formatted_labels}]")
    
    # Print confidence of winners
    winner_confs = [f"{probs_dict[l]:.4f}" for l in labels]
    print(f"Confidences: {winner_confs}")

    if True:
        print("-" * 20)
        # Sort dict items by value descending
        sorted_cands = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
        
        for name, score in sorted_cands:
            print(f"  {name.title():<25}: {score:.4f}")

    print("="*30 + "\n")

