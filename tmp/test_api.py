import requests
import json

url = "http://127.0.0.1:8000/api/generate"
payload = {
    "query": "세종대왕의 업적은?",
    "choices": ["훈민정음", "과학발전", "영토확장"],
    "ground_truth": "훈민정음 창제 등",
    "use_rag": True
}

try:
    response = requests.post(url, json=payload, timeout=300)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Response JSON Structure:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Check for required fields
        required_fields = ["input", "output", "contexts", "ground_truth", "choices"]
        missing = [f for f in required_fields if f not in data]
        if not missing:
            print("\n✅ All Langfuse evaluation fields are present.")
        else:
            print(f"\n❌ Missing fields: {missing}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
