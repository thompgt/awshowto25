import requests
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# The URL you want to make a GET request to
url = "http://tech.echios.com/search"

headers = {
    "X-API-Key": os.environ['ECHIOS_KEY'],
    "Content-Type": "application/json"
}

if len(sys.argv) > 1:
    keyword = sys.argv[1]
else:
    keyword = "amazon"

# The parameters you want to include in the request
params = {
    "keyword": keyword
}

response = requests.get(url, params=params, headers=headers)

print(response.json())