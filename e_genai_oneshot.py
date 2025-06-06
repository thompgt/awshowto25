import boto3
import json
from dotenv import load_dotenv
import os

def gen_insight(volatility_value):
    load_dotenv()

    client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-2',
        aws_access_key_id=os.getenv('ID'),
        aws_secret_access_key=os.getenv('PASS'),
    )

    MODEL_ID = 'us.amazon.nova-micro-v1:0'

    one_shot_prompt = r"""
You are an expert stock market analyst.

Following is an example of classifying a stock based on volatility:

based on the rule if stock volatility is less than 0.20 (20%), classify as 'Low', 
if volatility is between 0.20 and 0.40 (20%-40%), classify as 'Medium', 
if volatility is greater than 0.40 (40%), classify as 'High'.
Your reply must be in the format of a key-value pair with 'class' as the key.

Example:
volatility: 0.15
{"class" : "Low"}

volatility: 0.30
{"class" : "Medium"}

volatility: 0.50
{"class": "High"}

Now, classify the following stock volatility:
volatility: """ + f"{volatility_value}"

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

    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig=params
        )

        ans = response['output']
        ans_class = json.loads(ans['message']['content'][0]['text'])
        
        result = f"Stock with volatility {float(volatility_value):.2f} ({float(volatility_value)*100:.1f}%) is classified as {ans_class['class']} volatility"
        return result
    
    except Exception as e:
        return f"Error generating insight: {str(e)}"