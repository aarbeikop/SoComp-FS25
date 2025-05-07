# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd

#kagglehub.login()
# Download latest version
#path = kagglehub.dataset_download("deffro/the-climate-change-twitter-dataset")
#print("Path to dataset files:", path)

# path to dataset: /Users/blueberry/.cache/kagglehub/datasets/deffro/the-climate-change-twitter-dataset/versions/1
data_path = "twitter_data/the_climate_change_twitter_dataset.csv"

df = pd.read_csv(data_path)

print(df.head())

