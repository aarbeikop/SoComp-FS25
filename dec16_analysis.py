import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS

# File paths for downsampled data
reddit_down_path = 'data/processed/reddit_data_downsampled.csv'
twitter_down_path = 'data/processed/twitter_data_downsampled.csv'
output_dir = 'data/processed/dec2016_analysis'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read downsampled datasets
reddit = pd.read_csv(reddit_down_path, parse_dates=['created_utc'])
twitter = pd.read_csv(twitter_down_path, parse_dates=['tweet_date'])

# Filter to December 2016
target_month = '2016-12'
reddit_dec = reddit[reddit['created_utc'].dt.to_period('M') == target_month]
twitter_dec = twitter[twitter['tweet_date'].dt.to_period('M') == target_month]

# Basic counts
print(f"Reddit posts in {target_month}: {len(reddit_dec)}")
print(f"Twitter posts in {target_month}: {len(twitter_dec)}")

# Sentiment distribution
sent_red = reddit_dec['sentiment_category'].value_counts(normalize=True).sort_index()
sent_tw = twitter_dec['sentiment_category'].value_counts(normalize=True).sort_index()
df_sent = pd.DataFrame({'Reddit': sent_red, 'Twitter': sent_tw}).fillna(0)

# Plot sentiment comparison
plt.figure(figsize=(8,4))
df_sent.plot(kind='bar', title='Sentiment Distribution December 2016', rot=0)
plt.xlabel('Sentiment Category')
plt.ylabel('Proportion')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sentiment_dec2016.png'))
print('Saved sentiment distribution plot.')

# Word clouds without stopwords
custom_stopwords = set(STOPWORDS)

def generate_wordcloud(text, filepath):
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=custom_stopwords
    ).generate(text)
    wc.to_file(filepath)

# Combine and clean text
text_red = ' '.join(reddit_dec['cleaned_text'].dropna())
text_tw = ' '.join(twitter_dec['cleaned_message'].dropna())

# Generate and save wordclouds
generate_wordcloud(text_red, os.path.join(output_dir, 'wordcloud_reddit.png'))
generate_wordcloud(text_tw, os.path.join(output_dir, 'wordcloud_twitter.png'))
print('Saved wordclouds for Reddit and Twitter (without stopwords).')

# Daily activity time series
daily_red = reddit_dec.groupby(reddit_dec['created_utc'].dt.date).size()
daily_tw = twitter_dec.groupby(twitter_dec['tweet_date'].dt.date).size()

plt.figure(figsize=(10,4))
plt.plot(daily_red.index, daily_red.values, label='Reddit')
plt.plot(daily_tw.index, daily_tw.values, label='Twitter')
plt.title('Daily Activity - December 2016')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'daily_activity_dec2016.png'))
print('Saved daily activity plot.')
