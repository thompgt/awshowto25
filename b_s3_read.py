"""
run app using

"""
from dotenv import load_dotenv
import requests
import os
import boto3
import json

load_dotenv()

# AWS S3 configuration (from .env)
AWS_ACCESS_KEY_ID = os.environ.get("ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("PASS")
S3_BUCKET = "stock-details-ech-eg"
S3_PREFIX = "json_by_symbol"
SYMBOL = "APLE"

# print(AWS_ACCESS_KEY_ID)
# print(AWS_SECRET_ACCESS_KEY)

s3 = boto3.client(
  "s3",
  aws_access_key_id=AWS_ACCESS_KEY_ID,
  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

s3_key = f"{S3_PREFIX}/{SYMBOL}.json"
local_filename = f"{SYMBOL}.json"

try:
  s3.download_file(S3_BUCKET, s3_key, local_filename)
  response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
  fcontent = response['Body'].read().decode('utf-8')
  print(fcontent)
except Exception as e:
  print(f"Error downloading file: {e}")