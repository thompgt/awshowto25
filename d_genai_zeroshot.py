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

name_of_company = input("Enter the name of Company to find symbol for, Eg.Google: ")
zero_shot_prompt = f"What is the stock ticker symbol for {name_of_company}"

prompt = zero_shot_prompt

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
ans_text = ans['message']['content'][0]['text']
occurence_1=ans_text.find("**")
occurence_2 = ans_text.find("**", occurence_1 + 2)
ansj = ans_text[occurence_1+2:occurence_2] if occurence_1 != -1 and occurence_2 != -1 else "Symbol not found"
print(f"Symbol for {name_of_company} is {ansj}")