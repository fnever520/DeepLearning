import requests

# rest api url
url = 'http://0.0.0.0:5000/rest'

# get text to estimate
file = "aclImdb_v1/aclImdb/train/pos/60_8.txt"
with open(file, 'r', encoding="utf-8") as file:
    input_text = file.read()

# key=review
parameters = {'review': input_text}
# perform request
response = requests.get(url, params=parameters)

# result
print(response.json()['review estimation'])

