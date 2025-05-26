import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import logging
import os

# --------------------------
# Configuration
# --------------------------
reddit_path   = 'data/processed/reddit_data_downsampled.csv'
twitter_path  = 'data/processed/twitter_data_downsampled.csv'
output_dir    = 'data/processed/analysis_results'
window_days   = 30  # rolling window in days

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Main analysis
# --------------------------

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Starting full-period analysis')

    # Load datasets (no date filtering)
    reddit = pd.read_csv(reddit_path, parse_dates=['created_utc'])
    twitter = pd.read_csv(twitter_path, parse_dates=['tweet_date'])
    logging.info(f'Loaded {len(reddit)} Reddit and {len(twitter)} Twitter posts')

    # Validate required numeric sentiment column
    if 'sentiment_compound' not in reddit.columns or 'sentiment_compound' not in twitter.columns:
        logging.error('sentiment_compound column missing in one of the datasets')
        return

    # Compute rolling sentiment using compound score
    logging.info('Computing rolling sentiment for full time series')
        # Sort by date to ensure monotonic index
    r_series = reddit.set_index('created_utc')['sentiment_compound'].astype(float).sort_index()
    t_series = twitter.set_index('tweet_date')['sentiment_compound'].astype(float).sort_index()
#('tweet_date')['sentiment_compound'].astype(float)
    r_roll = r_series.rolling(f'{window_days}D').mean()
    t_roll = t_series.rolling(f'{window_days}D').mean()

    plt.figure(figsize=(10,4))
    r_roll.plot(label=f'Reddit ({window_days}d MA)')
    t_roll.plot(label=f'Twitter ({window_days}d MA)')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Sentiment')
    plt.title('Rolling Average Compound Sentiment (Full Period)')
    plt.legend()
    plt.tight_layout()
    rolling_path = os.path.join(output_dir, 'rolling_sentiment_full.png')
    plt.savefig(rolling_path)
    plt.close()
    logging.info(f'Saved rolling sentiment plot to {rolling_path}')

    # Chi-square test on overall categorical sentiment
    logging.info('Performing chi-square test on full-period sentiment distributions')
    # Use sentiment_category counts
    r_counts = reddit['sentiment_category'].value_counts().sort_index()
    t_counts = twitter['sentiment_category'].value_counts().sort_index()
    contingency = pd.concat([r_counts, t_counts], axis=1, keys=['reddit','twitter']).fillna(0)
    chi2, p, _, _ = chi2_contingency(contingency.values)
    logging.info(f'Chi-square test p-value: {p:.3g}')

    # Save sentiment counts
    counts_path = os.path.join(output_dir, 'sentiment_counts_full.csv')
    contingency.to_csv(counts_path)
    logging.info(f'Saved full-period sentiment counts to {counts_path}')

    logging.info('Full-period analysis complete')

if __name__ == '__main__':
    main()
