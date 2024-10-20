import requests
from bs4 import BeautifulSoup
import re


def get_url_file(url):
    match = re.search(r'/([^ /]+)\?', url)
    if match:
        result = match.group(1)
        return result
    else:
        print("No match found.")
        return None


import os

def download(model):
    url = f"http://hf-mirror.com/{model}/tree/main"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    download_links = soup.find_all('a', attrs={'title': 'Download file'})
    for link in download_links:
        url = 'http://hf-mirror.com' + link['href']
        file_name = get_url_file(url)
        if url and file_name:
            os.system(f'wget {url} -O {file_name}')

model = 'gpt2-medium'
download(model)
