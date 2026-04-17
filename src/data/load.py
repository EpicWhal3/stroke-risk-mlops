import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(output_dir: str = "data/raw"):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "imaadmahmood/stroke-risk-synthetic-2025", path=output_dir, unzip=True
    )
    return pd.read_csv(os.path.join(output_dir, "stroke-dataset.csv"))
