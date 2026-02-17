import json
import pandas as pd
from pathlib import Path

# 1. Define paths
base_path = Path(r"D:\Projects\RF_DETR_Wetland\results")
json_path = base_path / "results_full.json"
excel_path = base_path / "results_summary.xlsx"

# 2. Load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# 3. Extract validation and test class maps
# These are the lists of dictionaries containing metrics for each bird class
valid_metrics = data['class_map']['valid']
test_metrics = data['class_map']['test']

# 4. Convert lists to Pandas DataFrames
df_valid = pd.DataFrame(valid_metrics)
df_test = pd.DataFrame(test_metrics)

# 5. Save to Excel with two sheets
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_valid.to_excel(writer, sheet_name='Validation Set', index=False)
    df_test.to_excel(writer, sheet_name='Test Set', index=False)

print(f"Successfully converted JSON to Excel.")
print(f"File saved at: {excel_path}")