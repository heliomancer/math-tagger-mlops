import requests
import joblib
import numpy as np

# Load the vectorizer used during training
vectorizer = joblib.load("models/vectorizer.joblib")

text = "Find the area of a triangle between points (1,1) (2,0) (3,4)"
vector_input = vectorizer.transform([text]).toarray().tolist() # Convert to list for JSON

url = "http://127.0.0.1:5000/invocations"

try:
    response = requests.post(url, json={"inputs": vector_input})
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Connection failed:", e)


print(response.json())
