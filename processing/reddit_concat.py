import os
import pandas as pd

reddit_inventory = []
all_dataframes = []

base_dir = 'data/raw'

for query in os.listdir(base_dir):
    query_path = os.path.join(base_dir, query)
    if not os.path.isdir(query_path):
        continue
    for fname in os.listdir(query_path):
        if fname.endswith('.csv'):
            fpath = os.path.join(query_path, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
                continue

            if df.empty or len(df.columns) <= 1:  # Skip empty or header-only files
                continue

            # Collect metadata
            try:
                year = int(fname.split('_')[-1].split('.')[0])
                month = fname.split('_')[-2]
            except Exception as e:
                print(f"Skipping due to invalid filename format: {fname}")
                continue

            meta = {
                "query": query,
                "filename": fname,
                "filepath": fpath,
                "type": "comment" if "comment" in fname.lower() else "post",
                "year": year,
                "month": month
            }
            reddit_inventory.append(meta)
            all_dataframes.append(df)

# Write combined CSV once
if all_dataframes:
    concatenated_df = pd.concat(all_dataframes, ignore_index=True)
    output_path = 'data/reddit_concatenated.csv'
    concatenated_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")
else:
    print("No valid data found to concatenate.")