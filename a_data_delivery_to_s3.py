import requests

def download_file(url, local_path):
  response = requests.get(url, stream=True)
  response.raise_for_status()
  with open(local_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
      if chunk:
        f.write(chunk)