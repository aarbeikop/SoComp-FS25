"""Simplified sentiment analysis: process a combined CSV, analyze only the 'text' column, and append a numeric 'sentiment' column (-1, 0, 1)."""

import pandas as pd
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import argparse
import os

logger = logging.getLogger('sentiment_analysis')
logging.basicConfig(level=logging.INFO)

class SentimentProcessor:
    """Class for processing sentiment in social media data."""
    
    def __init__(self):
        """Initialize the sentiment processor with VADER."""
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """Return numeric sentiment: -1 negative, 0 neutral, 1 positive."""
        # Handle missing or empty text
        if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
            return 0
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 1
        elif compound <= -0.05:
            return -1
        else:
            return 0

    def process(self, df, text_column='text'):
        """Analyze sentiment for each text and append a 'sentiment' column."""
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame.")
            raise ValueError(f"Column '{text_column}' not found.")
        logger.info(f"Analyzing sentiment for {len(df)} rows using '{text_column}' column")
        df['sentiment'] = df[text_column].apply(self.analyze_text)
        return df


def parse_arguments():
    parser = argparse.ArgumentParser(description='Append numeric sentiment to a CSV with a text column')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output CSV or directory')
    parser.add_argument('--text-col', '-t', default='text',
                        help="Name of the text column (default: 'text')")
    return parser.parse_args()


def main():
    args = parse_arguments()
    inp = args.input
    out = args.output

    # Resolve output path
    if out.lower().endswith('.csv'):
        output_file = out
        output_dir = os.path.dirname(output_file) or '.'
    else:
        output_dir = out
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(inp))[0]
        output_file = os.path.join(output_dir, f"{base}_sentiment.csv")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Input: {inp}")
    logger.info(f"Output file: {output_file}")

    # Read, process, and write
    df = pd.read_csv(inp)
    processor = SentimentProcessor()
    result = processor.process(df, text_column=args.text_col)
    result.to_csv(output_file, index=False)
    logger.info(f"Saved output with sentiment to {output_file}")
    print(f"Saved sentiment CSV to: {output_file}")

if __name__ == '__main__':
    main()
