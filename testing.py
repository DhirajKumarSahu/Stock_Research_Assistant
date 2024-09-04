import requests

payload = {'user_query':'Should I invest in Yes Bank now?'}

response = requests.post(url = 'http://127.0.0.1:8080/ask_question', data = payload)

print(response)