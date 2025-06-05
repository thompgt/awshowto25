# Support for Project

- How to store AWS secrets
- How to view stock data from tech echios end point
- How to download from S3 bucket
- How to use Bedrock
  - Example prompts

### Download file to localhost
- In replit open shell / terminal
- git clone https://github.com/ckechios/awshowto25.git support
- In terminal execute command
```shell
cd support
zip -r support.zip ./*
```
- Download support.zip
zip -r support.zip ./*

### Install requirements
``` 
pip install -r requirements.txt 
```
### Store AWS Secrets
- For replit.com : DO NOT KEEP SECRETS IN .env add to Secrets
- For local dev : .env file is acceptable

### Get stock data from tech echios end point
- Use the API key provided
- example show in a_data_delivery_to_s3.py

### Download from S3 bucket
- example shown in b_s3_read.py

### How to use Bedrock
- example shown in d_genai_zeroshot.py

