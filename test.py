import requests

url = "http://127.0.0.1:8000/chatbot"
data = {"question": "What services does Promtior offer?"}

response = requests.post(url, json=data)
print(response.json())