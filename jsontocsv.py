
import pandas as pd
import json
import os

def json_to_csv(json_file, csv_file):
    with open(json_file) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Done,saved {len(df)} rows to {csv_file}")



