"""
    - load twitter csv
    - clean message column
    - filter out empty / very short tweets
    - extract tweet date from tweetid
    - save processed data
    - find tweet date from tweetid
    NOTE: The tweet date extraction relies on the tweetid format and may not be 100% accurate.
            Converter Code inspired by: https://github.com/oduwsdl/tweetedat/tree/master
"""

import pandas as pd
import os
import re
import argparse
import logging
import json
from datetime import datetime
import emoji

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterDataProcessor:
    """Class to process and clean Twitter data for sentiment analysis"""

    def __init__(self, output_dir='data/processed', timeline_file=None):
        self.output_dir = output_dir
        self.timeline_file = timeline_file
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, twitter_file):
        """Load Twitter data from CSV"""
        logger.info(f"Loading Twitter data from: {twitter_file}")
        df = pd.read_csv(twitter_file)
        logger.info(f"Loaded {len(df)} tweets")
        return df

    def clean_text(self, text):
        """Clean and normalize tweet text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = emoji.demojize(text)
        text = re.sub(r':[a-z_]+:', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def filter_data(self, df):
        """Remove deleted/empty/very short tweets"""
        initial_count = len(df)
        df = df[df['message'].notna() & (df['message'].str.strip() != "")]
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        df = df[df['cleaned_message'].str.len() >= 10]
        logger.info(f"Filtered out {initial_count - len(df)} tweets (empty or too short)")
        return df

    def find_tweet_timestamp_post_snowflake(self, tid):
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        return tstamp

    def find_tweet_timestamp_pre_snowflake(self, tid):
        with open(self.timeline_file, "r") as file:
            prev_line_parts = file.readline().rstrip().split(",")
            if tid < int(prev_line_parts[0]):
                return -1
            elif tid == int(prev_line_parts[0]):
                return int(prev_line_parts[1]) * 1000
            for line in file:
                line_parts = line.rstrip().split(",")
                if tid == int(line_parts[0]):
                    return int(line_parts[1]) * 1000
                if int(prev_line_parts[0]) < tid < int(line_parts[0]):
                    est = round(int(prev_line_parts[1]) + (((tid - int(prev_line_parts[0])) / (int(line_parts[0]) - int(prev_line_parts[0]))) * (int(line_parts[1]) - int(prev_line_parts[1]))))
                    return est * 1000
                prev_line_parts = line_parts
        return -1

    def get_tweet_date(self, tid):
        try:
            tid = int(tid)
            pre_snowflake_last_id = 29700859247
            if tid < pre_snowflake_last_id and self.timeline_file:
                ts = self.find_tweet_timestamp_pre_snowflake(tid)
            else:
                ts = self.find_tweet_timestamp_post_snowflake(tid)
            return datetime.fromtimestamp(ts / 1000) if ts > 0 else None
        except:
            return None

    def add_tweet_dates(self, df):
        """Add tweet_date column based on tweetid"""
        if 'tweetid' not in df.columns:
            logger.warning("No tweetid column found. Skipping date extraction.")
            return df
        logger.info("Extracting tweet dates from tweet IDs...")
        df['tweet_date'] = df['tweetid'].apply(self.get_tweet_date)
        return df

    def save_processed_data(self, df):
        """Save processed Twitter data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(self.output_dir, f"processed_twitter_{timestamp}.csv")
        df.to_csv(out_file, index=False)
        logger.info(f"Saved processed Twitter data to: {out_file}")
        return out_file

    def process_data(self, twitter_file):
        df = self.load_data(twitter_file)
        df = self.filter_data(df)
        df = self.add_tweet_dates(df)
        out_file = self.save_processed_data(df)
        return out_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Twitter data for sentiment analysis')
    parser.add_argument('--twitter', type=str, required=True, help='Path to the Twitter CSV file')
    parser.add_argument('--timeline', type=str, default=None, help='Path to TweetTimeline.txt for pre-Snowflake tweets')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = TwitterDataProcessor(output_dir=args.output, timeline_file=args.timeline)
    processor.process_data(args.twitter)

if __name__ == "__main__":
    main()