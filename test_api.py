# Lightweight smoke test helper.
# Run the API first:
#   uvicorn app.main:app --reload
#
# Then:
#   python test_api.py

import requests

BASE_URL = "http://127.0.0.1:8000"

health = requests.get(f"{BASE_URL}/health", timeout=30)
print("HEALTH:", health.status_code, health.json())

response = requests.post(
    f"{BASE_URL}/query",
    json={"question": "What does the pass statement do?", "top_k": 4},
    timeout=120,
)
print("QUERY:", response.status_code)
print(response.json())
