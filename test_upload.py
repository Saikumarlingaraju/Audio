"""Test the Flask prediction endpoint"""
import requests

url = 'http://127.0.0.1:5000/predict'
test_file = 'data/raw/indian_accents/tamil/Tamil_speaker (1).wav'

print(f"Testing prediction with: {test_file}")
print(f"Uploading to: {url}\n")

with open(test_file, 'rb') as f:
    files = {'audio': f}  # Flask app expects 'audio' not 'file'
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

if response.json().get('success'):
    print("\n✅ SUCCESS!")
    print(f"Predicted: {response.json()['predicted_accent']}")
    print(f"Confidence: {response.json()['confidence']:.2f}%")
else:
    print("\n❌ FAILED!")
    print(f"Error: {response.json().get('error', 'Unknown error')}")
