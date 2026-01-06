import requests

# 1. Define the input
# The model expects a vector of size 2000 (from your config max_features=2000)
# We create a dummy vector of all zeros just to test the connection.
vocab_size = 2000
input_data = {"inputs": [[0.0] * vocab_size]}

# 2. Send the request
url = "http://127.0.0.1:5000/invocations"
try:
    response = requests.post(url, json=input_data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Connection failed:", e)
