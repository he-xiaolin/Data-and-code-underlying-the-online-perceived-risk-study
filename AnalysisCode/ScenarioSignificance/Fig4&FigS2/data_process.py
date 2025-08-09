import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Script to convert raw Excel sheets into per-clip CSV files for multiple scenarios
# ------------------------------------------------------------

# 1. Define all scenarios with their layout parameters
#    Each entry: scenario_id, feature_nb, clip_nb, condition_nb_list, sheet_names
SCENARIOS = [
    {
        "scenario_id": "HB",
        "feature_nb": 3,
        "clip_nb": 5,
        "condition_nb_list": [3, 3, 3],
        "sheet_names": ["BI", "distance", "speed"],
    },
    {
        "scenario_id": "LC",
        "feature_nb": 3,
        "clip_nb": 6,
        "condition_nb_list": [2, 3, 4],
        "sheet_names": ["Distance", "Driving style", "Lateral behaviour"],
    },
    {
        "scenario_id": "MB",
        "feature_nb": 3,
        "clip_nb": 5,
        "condition_nb_list": [3, 3, 3],
        "sheet_names": ["BI", "distance", "speed"],
    },
    {
        "scenario_id": "SVM",
        "feature_nb": 3,
        "clip_nb": 5,
        "condition_nb_list": [3, 3, 3],
        "sheet_names": ["BI", "distance", "speed"],
    },
]

# Base input and output directories
# RAW_DIR = Path("./raw_data")
RAW_DIR    = Path("../../../Data/SubjectiveRatings_clip")
OUT_BASE = Path("./process_data")

# Iterate over each scenario configuration
for cfg in SCENARIOS:
    sid = cfg["scenario_id"]
    fnb = cfg["feature_nb"]
    cnb = cfg["clip_nb"]
    cond_list = cfg["condition_nb_list"]
    sheets = cfg["sheet_names"]

    # 2. Read Excel file for this scenario
    excel_path = RAW_DIR / f"{sid}.xlsx"
    # Read only the first fnb sheets as a list of DataFrames
    df_list = pd.read_excel(excel_path, sheet_name=list(range(fnb)))
    # Map each sheet to its logical name
    dfs = {name: df_list[i] for i, name in enumerate(sheets)}

    # 3. Loop over each feature, clip, and condition
    for idx, feature in enumerate(sheets):
        df_feature = dfs[feature]
        num_conditions = cond_list[idx]
        for clip in range(1, cnb + 1):
            start_col = (clip - 1) * num_conditions
            end_col = start_col + num_conditions
            df_clip = df_feature.iloc[:, start_col:end_col]
            for cond_idx in range(num_conditions):
                # Extract series, drop NaNs
                series = df_clip.iloc[:, cond_idx].dropna()
                # Prepare output directory
                out_dir = OUT_BASE / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                # Construct filename and save
                fname = f"{feature}_clip_{clip}_condition_{cond_idx+1}.csv"
                series.to_csv(out_dir / fname, index=False)
                print(f"Saved: {sid}_{feature}_clip_{clip}_condition_{cond_idx+1}.csv")