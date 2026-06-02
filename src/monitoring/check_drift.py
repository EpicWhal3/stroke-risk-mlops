import os
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    raw_path = os.path.join(
        config["dataset"]["raw_dir"], config["dataset"]["raw_filename"]
    )

    df = pd.read_csv(raw_path)

    reference_data = df.iloc[: int(len(df) * 0.7)].copy()
    current_data = df.iloc[int(len(df) * 0.7) :].copy()

    cols_to_drop = ["id", "stroke"]
    reference_data = reference_data.drop(
        columns=[c for c in cols_to_drop if c in reference_data.columns],
        errors="ignore",
    )
    current_data = current_data.drop(
        columns=[c for c in cols_to_drop if c in current_data.columns], errors="ignore"
    )

    print("Checking for data drift...")

    report = Report(metrics=[DataDriftPreset()])

    my_eval = report.run(reference_data=reference_data, current_data=current_data)

    os.makedirs("monitoring", exist_ok=True)

    my_eval.save_html("monitoring/drift_report.html")

    print("Drift report saved to monitoring/drift_report.html")


if __name__ == "__main__":
    main()
