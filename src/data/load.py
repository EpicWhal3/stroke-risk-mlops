from pathlib import Path

import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_dataset() -> pd.DataFrame:
    config = load_config()
    dataset_name = config["dataset"]["kaggle_name"]
    output_dir = Path(config["dataset"]["raw_dir"])
    target_filename = config["dataset"]["raw_filename"]

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=str(output_dir), unzip=True)

    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("CSV file was not found after Kaggle download.")

    source_file = csv_files[0]
    target_file = output_dir / target_filename

    if source_file.name != target_filename:
        source_file.rename(target_file)
    else:
        target_file = source_file

    df = pd.read_csv(target_file)
    print(f"Dataset downloaded: {target_file}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":
    download_dataset()
