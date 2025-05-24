import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

print("Reading from:", raw_dir)
print("Saving to:", processed_dir)

all_dataframes = []

for root, _, files in os.walk(raw_dir):
    for fname in files:
        if not fname.endswith('.csv'):
            continue

        fpath = os.path.join(root, fname)

        try:
            df = pd.read_csv(fpath)

            if df.empty:
                print(f"Skipped empty file: {fpath}")
                continue

            if len(df.columns) <= 1:
                print(f"Skipped malformed file: {fpath}")
                continue

            # Optional: add origin and type tag
            df['source_file'] = fname
            df['entry_type'] = 'comment' if 'comment' in fname.lower() else 'post'

            all_dataframes.append(df)
            print(f"Loaded {fpath} ({df.shape[0]} rows)")

        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

os.makedirs(processed_dir, exist_ok=True)

if all_dataframes:
    combined = pd.concat(all_dataframes, ignore_index=True)
    output_path = os.path.join(processed_dir, 'all_reddit_data.csv')
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} total rows to {output_path}")
else:
    print("No valid data found to concatenate.")
