import requests
import tarfile

def download_model(model_name, path="out/models"):
    print("Downloading model...")
    model_url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"
    r = requests.get(model_url, allow_redirects=True)
    open(f"{path}/{model_name}.tar.gz", 'wb').write(r.content)

def decompress_model(model_archive_path, out_path="out/models"):
    file = tarfile.open(model_archive_path)
    file.extractall(out_path)
    file.close()