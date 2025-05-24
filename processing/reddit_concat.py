import os
import pandas as pd

reddit_inventory = []

base_dir = 'data/raw/'

for query in os.listdir(base_dir):
    query_path = os.path.join(base_dir, query)
    if not os.path.isdir(query_path):
        continue
    for fname in os.listdir(query_path):
        if fname.endswith('.csv'):
            fpath = os.path.join(query_path, fname)
            df = pd.read_csv(fpath)
            if df.empty or len(df.columns) <= 1:  # Skip empty or header-only files
                continue
            meta = {
                "query": query,
                "filename": fname,
                "filepath": fpath,
                "type": "comment" if "comment" in fname.lower() else "post",
                "year": int(fname.split('_')[-1].split('.')[0]),
                "month": fname.split('_')[-2]
            }
            reddit_inventory.append(meta)