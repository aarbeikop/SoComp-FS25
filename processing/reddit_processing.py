"""
Reddit Data Processor for Climate Change Sentiment Analysis

This script cleans and processes Reddit data collected from the scraper,
preparing it for sentiment analysis with VADER.

Usage:
    python reddit_data_processor.py --input [INPUT_CSV] --output [OUTPUT_DIR]
"""

import pandas as pd
import numpy as np
import os
import re
import argparse
import logging
import json
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import emoji
import glob
from tqdm import tqdm

os.environ['NLTK_DATA'] = '/Users/blueberry/nltk_data'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RedditDataProcessor:
    """Class to process and clean Reddit data for sentiment analysis"""
    
    def __init__(self, output_dir='data/processed'):
        """Initialize the processor"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download necessary NLTK resources if not already downloaded
        self._setup_nltk()
        
        # Setup stopwords and patterns for text cleaning
        self.stopwords = set(stopwords.words('english'))
        self.climate_stopwords = {
            'climate', 'change', 'global', 'warming', 'environment', 'environmental',
            'weather', 'temperature', 'carbon', 'emissions', 'greenhouse', 'co2'
        }
        
        # Common bot signatures on Reddit (more lenient patterns)
        self.bot_patterns = [
            r'^\s*i am a bot\s*$',
            r'^\s*this action was performed automatically\s*$',
            r'^\s*beep boop\s*$',
            r'^\s*bot\s+disclaimer\s*$'
        ]
        
        self.bot_regex = re.compile('|'.join(self.bot_patterns), re.IGNORECASE)
    
    def _setup_nltk(self):
        """Download necessary NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            logger.info("Downloaded NLTK resources")
    
    def load_data(self, combined_file=None):
        """
        Load Reddit data from a single combined CSV file
        If no file is specified, use the most recent matching file in data/raw/
        """
        if not combined_file:
            logger.info("No input file specified, looking for most recent combined file in data/raw/")
            all_files = sorted(glob.glob('data/raw/*.csv'), key=os.path.getmtime, reverse=True)
            if not all_files:
                raise FileNotFoundError("No data files found in data/raw/")
            combined_file = all_files[0]
        
        logger.info(f"Loading combined data from: {combined_file}")
        df = pd.read_csv(combined_file)
        
        # Debug: Print column names to understand the data structure
        logger.info(f"Available columns: {list(df.columns)}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Entry types: {df['entry_type'].value_counts() if 'entry_type' in df.columns else 'No entry_type column'}")
        
        return df
    
    def clean_text(self, text):
        """Clean and normalize text content (less aggressive)"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove Reddit formatting
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Markdown links
        text = re.sub(r'&amp;', '&', text)  # Convert HTML entities
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        # Handle emojis (convert to text but keep them)
        text = emoji.demojize(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text, english_threshold=0.5):
        """
        Detect if text is in English based on word ratios (more lenient)
        """
        if not text or len(text) < 5:  # Reduced minimum length
            return True  # Assume short text is English rather than filtering it out
        
        # Get words
        words = word_tokenize(text.lower())
        
        if not words:
            return True
        
        # Count English-like words (letters only, not too restrictive)
        english_word_count = sum(1 for word in words 
                               if re.match(r'^[a-zA-Z]+$', word))
        
        if len(words) == 0:
            return True
        
        # Calculate ratio of English-like words
        english_ratio = english_word_count / len(words)
        
        return english_ratio >= english_threshold
    
    def is_bot_content(self, text):
        """Detect if text is likely from a bot (more restrictive patterns)"""
        if not isinstance(text, str) or pd.isna(text):
            return False
        
        # Only flag very obvious bot content
        return bool(self.bot_regex.search(text.strip()))
    
    def parse_created_utc(self, created_utc):
        """Parse UTC timestamp to datetime"""
        try:
            if pd.isna(created_utc):
                return pd.NaT
            return pd.to_datetime(created_utc, unit='s')
        except Exception:
            return pd.NaT

    def add_metadata(self, df):
        """Add metadata and parse timestamps"""
        logger.info("Adding metadata...")
        
        # Parse created_utc timestamps if they exist
        if 'created_utc' in df.columns:
            df['created_date'] = df['created_utc'].apply(self.parse_created_utc)
        else:
            logger.warning("No 'created_utc' column found in data")
            df['created_date'] = pd.NaT
        
        # Parse scraped_utc timestamps if they exist
        if 'scraped_utc' in df.columns:
            df['scraped_date'] = df['scraped_utc'].apply(self.parse_created_utc)
        
        return df
    
    def filter_data(self, df):
        """Filter data based on various criteria (much more lenient)"""
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Starting count
        initial_count = len(data)
        logger.info(f"Starting with {initial_count} entries")
        
        # 1. Remove only explicitly deleted/removed content
        deleted_mask = data['text'].isin(['[deleted]', '[removed]', '', 'nan']) | data['text'].isna()
        data = data[~deleted_mask]
        logger.info(f"Removed {initial_count - len(data)} deleted/removed entries")
        
        # 2. Filter out only very obvious bot content
        data['is_bot'] = data['text'].apply(self.is_bot_content)
        bot_count = data['is_bot'].sum()
        data = data[~data['is_bot']]
        logger.info(f"Removed {bot_count} obvious bot entries")
        
        # 3. Clean text content
        data['cleaned_text'] = data['text'].apply(self.clean_text)
        
        # 4. Filter for English content (more lenient)
        data['is_english'] = data['cleaned_text'].apply(self.detect_language)
        non_english_count = (~data['is_english']).sum()
        data = data[data['is_english']]
        logger.info(f"Removed {non_english_count} non-English entries")
        
        # 5. Remove only very short content (reduced threshold)
        short_content_mask = data['cleaned_text'].str.len() < 3
        short_count = short_content_mask.sum()
        data = data[~short_content_mask]
        logger.info(f"Removed {short_count} entries with very short content")
        
        # Drop temporary columns used for filtering
        data = data.drop(columns=['is_bot', 'is_english'])
        
        logger.info(f"Final count: {len(data)} entries (kept {len(data)/initial_count*100:.1f}%)")
        
        return data
    
    def tokenize_and_analyze_text(self, df):
        """Analyze text characteristics for additional insights"""
        data = df.copy()
        
        # Function to extract words (excluding stopwords)
        def extract_content_words(text):
            if not isinstance(text, str) or pd.isna(text):
                return []
            
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords, punctuation, and very short words
            words = [word for word in tokens 
                    if word not in self.stopwords 
                    and word not in string.punctuation 
                    and len(word) > 2]
            
            return words
        
        # Add word lists
        data['content_words'] = data['cleaned_text'].apply(extract_content_words)
        
        # Extract climate-specific terms
        def extract_climate_terms(words):
            return [word for word in words if word in self.climate_stopwords]
        
        data['climate_terms'] = data['content_words'].apply(extract_climate_terms)
        
        # Count climate terms
        data['climate_term_count'] = data['climate_terms'].apply(len)
        
        # Add text length
        data['text_length'] = data['cleaned_text'].str.len()
        data['word_count'] = data['content_words'].apply(len)
        
        return data
    
    def sample_data_for_manual_validation(self, df, sample_size=50):
        """Create a sample for manual validation of data quality"""
        sample_size = min(sample_size, len(df))
        
        # Stratified sample by subreddit if possible
        if 'subreddit' in df.columns and df['subreddit'].nunique() > 1:
            sample = df.groupby('subreddit', group_keys=False).apply(
                lambda x: x.sample(min(sample_size // df['subreddit'].nunique(), len(x)))
            )
        else:
            sample = df.sample(sample_size)
        
        return sample.reset_index(drop=True)
    
    def save_processed_data(self, df, sample_df=None):
        """Save processed data files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main processed file
        main_file = os.path.join(self.output_dir, f"processed_reddit_data_{timestamp}.csv")
        df.to_csv(main_file, index=False)
        logger.info(f"Saved processed data to: {main_file}")
        
        # Save sample file for validation if available
        sample_file = None
        if sample_df is not None:
            sample_file = os.path.join(self.output_dir, f"sample_for_validation_{timestamp}.csv")
            sample_df.to_csv(sample_file, index=False)
            logger.info(f"Saved sample for validation to: {sample_file}")
        
        # Save metadata about the processing
        metadata = {
            'timestamp': timestamp,
            'total_entries': len(df),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'entry_type_counts': df['entry_type'].value_counts().to_dict() if 'entry_type' in df.columns else {},
            'subreddit_counts': df['subreddit'].value_counts().head(10).to_dict() if 'subreddit' in df.columns else {},
            'files': {
                'processed_data': main_file,
                'sample_data': sample_file
            }
        }
        
        metadata_file = os.path.join(self.output_dir, f"processing_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Saved processing metadata to: {metadata_file}")
        
        return {
            'main_file': main_file,
            'sample_file': sample_file,
            'metadata_file': metadata_file
        }
    
    def process_data(self, combined_file=None, sample_size=50):
        """Main method to process all data"""
        # 1. Load data
        df = self.load_data(combined_file)
        
        # 2. Add metadata
        df = self.add_metadata(df)
        
        # 3. Clean and filter data (much more lenient)
        logger.info("Cleaning and filtering data...")
        df_filtered = self.filter_data(df)
        
        # 4. Tokenize and analyze text
        logger.info("Analyzing text characteristics...")
        df_analyzed = self.tokenize_and_analyze_text(df_filtered)
        
        # 5. Create sample for validation
        logger.info("Creating validation sample...")
        sample_df = self.sample_data_for_manual_validation(df_analyzed, sample_size)
        
        # 6. Save processed data
        logger.info("Saving processed data...")
        file_paths = self.save_processed_data(df_analyzed, sample_df)
        
        # 7. Print summary statistics
        logger.info("\n=== PROCESSING SUMMARY ===")
        logger.info(f"Total entries processed: {len(df_analyzed)}")
        if 'entry_type' in df_analyzed.columns:
            logger.info(f"Entry type distribution:")
            for entry_type, count in df_analyzed['entry_type'].value_counts().items():
                logger.info(f"  {entry_type}: {count}")
        if 'subreddit' in df_analyzed.columns:
            logger.info(f"Top subreddits:")
            for subreddit, count in df_analyzed['subreddit'].value_counts().head(5).items():
                logger.info(f"  {subreddit}: {count}")
        
        logger.info("Data processing complete!")
        
        return file_paths


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process Reddit data for sentiment analysis')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to the combined CSV file (default: most recent in data/raw/)')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed data (default: data/processed)')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples to create for validation (default: 50)')
    
    return parser.parse_args()


def main():
    """Main function to run the data processor"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize processor
        processor = RedditDataProcessor(output_dir=args.output)
        
        # Process data
        file_paths = processor.process_data(
            combined_file=args.input,
            sample_size=args.sample_size
        )
        
        logger.info("Processing complete. Output files:")
        for key, path in file_paths.items():
            if path:
                logger.info(f"{key}: {path}")
                
    except Exception as e:
        logger.exception(f"Error during processing: {e}")


if __name__ == "__main__":
    main()