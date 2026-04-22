import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def check_drift(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("monitoring/drift_report.html")
    return report.as_dict()
