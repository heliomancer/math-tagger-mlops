import json

import requests

# New Input Format: Raw Text wrapped in "inputs"
data = {"inputs": ["Find the derivative of 2x + 5.", "Solve for x: 3x = 9"]}

url = "http://127.0.0.1:5000/invocations"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=data, headers=headers)
    print("Status:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
