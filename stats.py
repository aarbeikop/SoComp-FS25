import pandas as pd
import numpy as np
import os
import json

# ── Configuration ─────────────────────────────────────────────────────────────
reddit_path  = 'data/processed/reddit_data_downsampled.csv'
twitter_path = 'data/processed/twitter_data_downsampled.csv'
output_dir   = 'data/processed/data_profile'
os.makedirs(output_dir, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def profile(df, text_col, date_col, engagement_cols=None, name='data'):
    stats = {}

    # Basic counts & date range
    stats['total_posts'] = len(df)
    stats['date_min']   = str(df[date_col].min())
    stats['date_max']   = str(df[date_col].max())

    # Text length
    lengths = df[text_col].dropna().astype(str).str.split().apply(len)
    stats['length'] = {
        'mean': float(lengths.mean()),
        'std' : float(lengths.std()),
        'min' : int(lengths.min()),
        '25%' : float(lengths.quantile(0.25)),
        'median': float(lengths.median()),
        '75%' : float(lengths.quantile(0.75)),
        'max' : int(lengths.max())
    }

    # Sentiment category distribution
    sent_counts = df['sentiment_category'].value_counts().to_dict()
    stats['sentiment_counts'] = {k: int(v) for k, v in sent_counts.items()}

    # Compound score distribution
    comp = df['sentiment_compound'].dropna()
    stats['compound'] = {
        'mean': float(comp.mean()),
        'std' : float(comp.std()),
        'min' : float(comp.min()),
        '25%' : float(comp.quantile(0.25)),
        'median': float(comp.median()),
        '75%' : float(comp.quantile(0.75)),
        'max' : float(comp.max())
    }

    # Engagement statistics (if available)
    if engagement_cols:
        # sum up any engagement fields that exist
        df['engagement'] = df[engagement_cols].fillna(0).sum(axis=1)
        eng = df['engagement']
        stats['engagement'] = {
            'mean': float(eng.mean()),
            'std' : float(eng.std()),
            'min' : float(eng.min()),
            '25%' : float(eng.quantile(0.25)),
            'median': float(eng.median()),
            '75%' : float(eng.quantile(0.75)),
            'max' : float(eng.max())
        }

    # Monthly volume
    df['month'] = df[date_col].dt.to_period('M')
    monthly = df.groupby('month').size()
    stats['monthly_posts'] = {
        'min'   : int(monthly.min()),
        'mean'  : float(monthly.mean()),
        'max'   : int(monthly.max()),
        'months': len(monthly)
    }

    # Return a dict
    with open(os.path.join(output_dir, f'{name}_profile.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"{name} profile written to {name}_profile.json")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    # Reddit
    reddit = pd.read_csv(reddit_path, parse_dates=['created_utc'])
    profile(
        reddit, 
        text_col='cleaned_text', 
        date_col='created_utc', 
        engagement_cols=['num_comments'] if 'num_comments' in reddit.columns else None,
        name='reddit'
    )

    # Twitter
    twitter = pd.read_csv(twitter_path, parse_dates=['tweet_date'])
    profile(
        twitter,
        text_col='cleaned_message',
        date_col='tweet_date',
        engagement_cols=['retweet_count','favorite_count'] 
            if set(['retweet_count','favorite_count']).issubset(twitter.columns) 
            else None,
        name='twitter'
    )
