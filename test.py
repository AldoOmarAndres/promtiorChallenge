import requests
import json

# test en local, se necesita ejecutar primero 'deploy.py'
url_local = "http://localhost:8000/chatbot"
data = {"question": "When was the company founded?"}

response = requests.post(url_local, json=data).json()
print(response['answer']['answer'])