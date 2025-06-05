import boto3
import json
from dotenv import load_dotenv
import os

load_dotenv()

client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-2',
    aws_access_key_id=os.getenv('ID'),
    aws_secret_access_key=os.getenv('PASS'),
)

MODEL_ID = 'us.amazon.nova-micro-v1:0'

ACCEPT_HEADER = 'application/json'
CONTENT_TYPE = 'application/json'

price_is = 250
price_is = input("Enter stock price to classify: ")

one_shot_prompt = r"""
You are an expert stock market analyst.

Following is an example of classifying a stock based on price:

based on the rule if stock price is less than 100, classify as 'Low', 
if stock price is between 100 and 500, classify as 'Medium', 
if stock price is greater than 500, classify as 'High'.
Your reply must be in the format of a key-value pair with 'class' as the key.

Example:
stock price: 150.25
{"class" : "Medium"}

stock price: 45.75
{"class" : "Low"}

stock price: 600.50
{"class": "High"}

Now, classify the following stock price:
stock price: """ + f"{price_is}"

prompt = one_shot_prompt

messages = [
    {
      "role": "user",
      "content": [
          {
              "text": prompt
          }
      ]
    }
  ]

params = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}

response = client.converse(
    modelId=MODEL_ID,
    messages=messages,
    inferenceConfig=params
)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("-" * 40)
print(f"Output: {response['output']}")
print("-" * 40)

ans = response['output']
ans_class = json.loads(ans['message']['content'][0]['text'])
print(
    f"Answer:\n\
Stock with price ${float(price_is):.2f} \
is classified as {ans_class['class']}"
)
