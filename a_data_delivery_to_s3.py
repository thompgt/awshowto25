import requests
import sys
import os
from dotenv import load_dotenv

load_dotenv()

symbol="TATACOMM.BSE"

url = f"https://tech.echios.com/fetchdata/{symbol}"

# The parameters you want to include in the request
# params = {
#     "symbol": "tsla"
# }

headers = {
    "X-API-Key": os.environ['ECHIOS_KEY'],
    "Content-Type": "application/json"
}
                   
# response = requests.get(url, params=params)
response = requests.get(url, headers=headers)

print(response.json())