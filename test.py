import requests
import json

url_local = "http://localhost:8000/chatbot"
#url_app_server = "https://llama-bpe9g6g9gpd2bmbs.eastus2-01.azurewebsites.net/"
data = {"question": "When was the company founded?"}

response = requests.post(url_local, json=data).json()
print(response['answer']['answer'])