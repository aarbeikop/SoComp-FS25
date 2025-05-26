import pandas as pd
import os
import matplotlib.pyplot as plt

# File paths
reddit_path = 'data/processed/reddit_data_classified.csv'
twitter_path = 'data/processed/twitter_data_classified.csv'
output_reddit = 'data/processed/reddit_data_downsampled.csv'
output_twitter = 'data/processed/twitter_data_downsampled.csv'

# Plot paths
daily_plot = 'data/processed/daily_comparison.png'
monthly_pre_plot = 'data/processed/monthly_pre_comparison.png'
monthly_post_plot = 'data/processed/monthly_post_comparison.png'

# Read datasets
reddit = pd.read_csv(reddit_path)
twitter = pd.read_csv(twitter_path)

# Parse datetime columns
reddit['created_utc'] = pd.to_datetime(
    reddit['created_utc'],
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
)
twitter['tweet_date'] = pd.to_datetime(
    twitter['tweet_date'],
    format='%Y-%m-%d %H:%M:%S.%f',
    errors='coerce'
)

# Drop missing dates
reddit = reddit.dropna(subset=['created_utc'])
twitter = twitter.dropna(subset=['tweet_date'])

# Extract month period
reddit['month'] = reddit['created_utc'].dt.to_period('M')
twitter['month'] = twitter['tweet_date'].dt.to_period('M')

# Restrict to months present in both
common_months = set(reddit['month']).intersection(twitter['month'])
reddit = reddit[reddit['month'].isin(common_months)]
twitter = twitter[twitter['month'].isin(common_months)]

# Pre-downsample monthly counts
pre_reddit = reddit.groupby('month').size().rename('n_reddit')
pre_twitter = twitter.groupby('month').size().rename('n_tweets')
combined_pre = pd.concat([pre_twitter, pre_reddit], axis=1).fillna(0)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_reddit), exist_ok=True)

# Plot pre-downsample monthly distribution
plt.figure(figsize=(10,5))
combined_pre.plot(kind='bar', title='Pre-Downsample Monthly: Twitter vs Reddit')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(monthly_pre_plot)
print(f'Saved pre-downsample monthly plot to {monthly_pre_plot}')

# Determine target per month: minimum of the two platforms
target_monthly = combined_pre.min(axis=1)

# Downsample both datasets to equal monthly counts
down_reddit = []
down_twitter = []
for month, target in target_monthly.items():
    red_month = reddit[reddit['month'] == month]
    tw_month = twitter[twitter['month'] == month]
    down_reddit.append(red_month.sample(n=target, random_state=42))
    down_twitter.append(tw_month.sample(n=target, random_state=42))

# Concatenate downsampled data
reddit_down = pd.concat(down_reddit).reset_index(drop=True)
twitter_down = pd.concat(down_twitter).reset_index(drop=True)

# Save downsampled CSVs
reddit_down.to_csv(output_reddit, index=False)
twitter_down.to_csv(output_twitter, index=False)
print(f"Saved downsampled Reddit to {output_reddit}\nSaved downsampled Twitter to {output_twitter}")

# Post-downsample monthly counts
post_reddit = reddit_down.groupby(reddit_down['created_utc'].dt.to_period('M')).size().rename('n_reddit')
post_twitter = twitter_down.groupby(twitter_down['tweet_date'].dt.to_period('M')).size().rename('n_tweets')
combined_post = pd.concat([post_twitter, post_reddit], axis=1).fillna(0)

# Plot post-downsample monthly distribution
plt.figure(figsize=(10,5))
combined_post.plot(kind='bar', title='Post-Downsample Monthly: Twitter vs Reddit')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(monthly_post_plot)
print(f'Saved post-downsample monthly plot to {monthly_post_plot}')
print('Downsampling complete.')