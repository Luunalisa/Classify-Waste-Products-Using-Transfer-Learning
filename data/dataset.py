import requests
import zipfile
import os
from tqdm import tqdm

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
file_name = "o-vs-r-split-reduced-1200.zip"

def download_dataset():
    print("Downloading file")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_file_with_progress(file_name):
    print("Extracting file with progress")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        members = zip_ref.infolist() 
        with tqdm(total=len(members), unit='file') as progress_bar:
            for member in members:
                zip_ref.extract(member)
                progress_bar.update(1)
    print("Finished extracting file")

def prepare_dataset():
    if not os.path.exists("o-vs-r-split"):
        download_dataset()
        extract_file_with_progress(file_name)
        os.remove(file_name)
    else:
        print("Dataset already prepared.")