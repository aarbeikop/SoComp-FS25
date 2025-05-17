"""
    - load twitter csv
    - clean message column
    - filter out empty / very short tweets
    - save processed data
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

    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
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
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = emoji.demojize(text)          # Convert emojis to text
        text = re.sub(r':[a-z_]+:', ' ', text)  # Remove emoji text representations
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()    # Normalize whitespace
        return text

    def filter_data(self, df):
        """Remove deleted/empty/very short tweets"""
        initial_count = len(df)
        df = df[df['message'].notna() & (df['message'].str.strip() != "")]
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        df = df[df['cleaned_message'].str.len() >= 10]
        logger.info(f"Filtered out {initial_count - len(df)} tweets (empty or too short)")
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
        out_file = self.save_processed_data(df)
        return out_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Twitter data for sentiment analysis')
    parser.add_argument('--twitter', type=str, required=True, help='Path to the Twitter CSV file')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = TwitterDataProcessor(output_dir=args.output)
    processor.process_data(args.twitter)

if __name__ == "__main__":
    main()