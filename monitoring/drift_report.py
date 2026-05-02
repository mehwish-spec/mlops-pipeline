import pandas as pd
import json
import os
from datetime import datetime

def check_drift(reference_path="data/reference_data.csv", current_path="data/current_data.csv"):
    print("Loading data...")
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    print("Checking drift manually...")
    drift_results = {}
    drift_count = 0

    for col in reference_data.columns:
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        ref_std = reference_data[col].std()
        diff = abs(ref_mean - cur_mean)
        normalized_diff = diff / (ref_std + 1e-10)
        drift = normalized_diff > 0.5
        if drift:
            drift_count += 1
        drift_results[col] = {
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "drift": bool(drift),
            "score": round(normalized_diff, 4)
        }

    drift_share = drift_count / len(reference_data.columns)
    drift_detected = drift_share > 0.15

    os.makedirs("monitoring/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": timestamp,
        "drift_detected": bool(drift_detected),
        "drift_share": round(drift_share, 4),
        "drifted_features": drift_count,
        "total_features": len(reference_data.columns),
        "status": "ALERT" if drift_detected else "OK",
        "feature_drift": drift_results
    }

    with open("monitoring/reports/summary_" + timestamp + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Drift detected: " + str(drift_detected))
    print("Drift share: " + str(round(drift_share, 4)))
    print("Status: " + summary["status"])
    print("Report saved to monitoring/reports/summary_" + timestamp + ".json")
    return summary

if __name__ == "__main__":
    result = check_drift()
    print(result)
