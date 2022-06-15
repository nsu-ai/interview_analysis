from urllib.parse import urlencode
from zipfile import BadZipFile, ZipFile
from io import BytesIO
import os
import requests
from tqdm import tqdm


def load_model_from_yandex_disk(load_url, model_path,
                                base_url="https://cloud-api.yandex.net/v1/disk/public/resources/download?",
                                model_name=""):
    """
    Function to load model checkpoint from Yandex Disk in zip format and extract to destination folder
    """

    if model_name == "":
        model_name = load_url

    final_url = base_url + urlencode(dict(public_key=load_url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    print(f"Start downloading model {load_url}")
    response = requests.get(download_url, stream=True)
    total = int(response.headers.get('content-length', 0))

    if response.ok:
        try:
            bytes_input = BytesIO()

            with tqdm(desc=model_name, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    size = bytes_input.write(data)
                    bar.update(size)

            z = ZipFile(bytes_input)
            z.extractall(model_path)
            print(f"Model {model_name} successfully loaded and saved to '{model_path}'")
            return True
        except BadZipFile as ex:
            print('Model downloading error: {}'.format(ex))
            return False


models = {
    "RuBERTSentiment": {
        "link": "https://disk.yandex.ru/d/aIKBfBKLHtlpJg",
        "path": "./models/RuBERTSentiment/checkpoints/",
        "checkpoint_name": "blanchefort_rubert-base-cased-sentiment"
    }
}

# TODO: загружать только модели, указанные в конфигурационном файле

# Load all models
for model_name, model_params in models.items():
    if not os.path.isdir(os.path.join(model_params["path"], model_params["checkpoint_name"])):
        load_model_from_yandex_disk(load_url=model_params["link"], model_path=model_params["path"],
                                    model_name=model_name)
