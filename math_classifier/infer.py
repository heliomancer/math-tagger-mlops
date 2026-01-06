import mlflow


def infer(cfg):
    input_text = cfg.get("input_text", "Calculate the area of a circle.")
    runtime_threshold = cfg.model.get("threshold", 0.5)

    print("\n" + "=" * 30)
    print(f"Input: {input_text}")

    model_uri = "models:/MathTagger/Production"
    print(f"Loading model from {model_uri}...")

    try:
        loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. Get Prediction from Model
    results = loaded_pipeline.predict([input_text])
    result = results[0]  # One input -> One result

    # 2. Extract Data
    probs_dict = result["probabilities"]
    backend = result.get("model_type", "UNKNOWN")

    # 3. Runtime threshold application
    active_labels = []
    for label, prob in probs_dict.items():
        if prob >= runtime_threshold:
            active_labels.append(label)

    if not active_labels:
        # Get max
        best_label = max(probs_dict, key=probs_dict.get)
        active_labels.append(best_label)

    # 4. Display
    formatted_labels = ", ".join([al.upper() for al in active_labels])

    print(f"Model type: [{backend}]")
    print(f"Threshold:  {runtime_threshold}")  # Show what we used
    print(f"Prediction: [{formatted_labels}]")

    # Print confidence of winners
    winner_confs = [f"{probs_dict[al]:.4f}" for al in active_labels]
    print(f"Confidences: {winner_confs}")

    if True:
        print("-" * 20)
        sorted_cands = sorted(
            probs_dict.items(), key=lambda item: item[1], reverse=True
        )

        for name, score in sorted_cands:
            print(f"  {name.title():<25}: {score:.4f}")

    print("=" * 30 + "\n")
