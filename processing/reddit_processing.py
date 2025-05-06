"""
Reddit Data Processor for Climate Change Sentiment Analysis

This script cleans and processes Reddit data collected from the scraper,
preparing it for sentiment analysis with VADER.

Usage:
    python reddit_data_processor.py --posts [POSTS_CSV] --comments [COMMENTS_CSV] --output [OUTPUT_DIR]
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
        
        # Common bot signatures on Reddit
        self.bot_patterns = [
            r'i am a bot',
            r'bot action performed',
            r'automated message',
            r'this action was performed automatically',
            r'beep boop',
            r'good bot',
            r'bad bot',
            r'^I\'m a bot',
            r'bot\s+disclaimer',
            r'automated\s+response'
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
    
    def load_data(self, posts_file=None, comments_file=None):
        """
        Load Reddit data from CSV files
        
        If no files specified, use the most recent files in data/raw/
        """
        if not posts_file or not comments_file:
            logger.info("No input files specified, looking for most recent files in data/raw/")
            
            # Find most recent posts and comments files
            posts_files = sorted(glob.glob('data/raw/reddit_posts_*.csv'), key=os.path.getmtime, reverse=True)
            comments_files = sorted(glob.glob('data/raw/reddit_comments_*.csv'), key=os.path.getmtime, reverse=True)
            
            if not posts_files or not comments_files:
                raise FileNotFoundError("No Reddit data files found in data/raw/")
            
            posts_file = posts_files[0]
            comments_file = comments_files[0]
        
        logger.info(f"Loading posts from: {posts_file}")
        logger.info(f"Loading comments from: {comments_file}")
        
        # Load data
        posts_df = pd.read_csv(posts_file)
        comments_df = pd.read_csv(comments_file)
        
        logger.info(f"Loaded {len(posts_df)} posts and {len(comments_df)} comments")
        
        return posts_df, comments_df
    
    def clean_text(self, text):
        """Clean and normalize text content"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove Reddit formatting
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Markdown links
        text = re.sub(r'&amp;', '&', text)  # Convert HTML entities
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        # Handle special characters and emojis
        text = emoji.demojize(text)  # Convert emojis to text
        text = re.sub(r':[a-z_]+:', ' ', text)  # Remove emoji text representations
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_language(self, text, english_threshold=0.7):
        """
        Detect if text is in English based on word ratios
        This is a simple approach - for a production system consider using langdetect
        """
        if not text or len(text) < 10:
            return False
            
        # Get a sample of words
        words = word_tokenize(text.lower())[:50]  # Limit to first 50 words for efficiency
        
        if not words:
            return False
            
        # Check against English stopwords
        english_word_count = sum(1 for word in words if word in self.stopwords or word in string.ascii_lowercase)
        
        # Calculate ratio of English words
        english_ratio = english_word_count / len(words)
        
        return english_ratio >= english_threshold
    
    def is_bot_content(self, text):
        """Detect if text is likely from a bot"""
        if not isinstance(text, str) or pd.isna(text):
            return False
            
        return bool(self.bot_regex.search(text))
    
    def filter_data(self, posts_df, comments_df):
        """Filter data based on various criteria"""
        # Make copies to avoid modifying original data
        posts = posts_df.copy()
        comments = comments_df.copy()
        
        # Starting counts
        initial_post_count = len(posts)
        initial_comment_count = len(comments)
        
        # 1. Remove deleted/removed content
        posts = posts[~posts['text'].isin(['[deleted]', '[removed]'])]
        comments = comments[~comments['text'].isin(['[deleted]', '[removed]'])]
        
        logger.info(f"Removed {initial_post_count - len(posts)} deleted/removed posts")
        logger.info(f"Removed {initial_comment_count - len(comments)} deleted/removed comments")
        
        # 2. Filter out bot content
        posts['is_bot'] = posts['text'].apply(self.is_bot_content)
        comments['is_bot'] = comments['text'].apply(self.is_bot_content)
        
        posts_without_bots = posts[~posts['is_bot']]
        comments_without_bots = comments[~comments['is_bot']]
        
        logger.info(f"Removed {len(posts) - len(posts_without_bots)} bot posts")
        logger.info(f"Removed {len(comments) - len(comments_without_bots)} bot comments")
        
        posts = posts_without_bots
        comments = comments_without_bots
        
        # 3. Clean text content
        posts['cleaned_text'] = posts['text'].apply(self.clean_text)
        comments['cleaned_text'] = comments['text'].apply(self.clean_text)
        
        # 4. Filter for English content
        posts['is_english'] = posts['cleaned_text'].apply(self.detect_language)
        comments['is_english'] = comments['cleaned_text'].apply(self.detect_language)
        
        posts_english = posts[posts['is_english']]
        comments_english = comments[comments['is_english']]
        
        logger.info(f"Removed {len(posts) - len(posts_english)} non-English posts")
        logger.info(f"Removed {len(comments) - len(comments_english)} non-English comments")
        
        posts = posts_english
        comments = comments_english
        
        # 5. Remove very short content
        posts = posts[posts['cleaned_text'].str.len() >= 10]
        comments = comments[comments['cleaned_text'].str.len() >= 10]
        
        logger.info(f"Removed {len(posts_english) - len(posts)} posts with very short content")
        logger.info(f"Removed {len(comments_english) - len(comments)} comments with very short content")
        
        # Drop temporary columns used for filtering
        posts = posts.drop(columns=['is_bot', 'is_english'])
        comments = comments.drop(columns=['is_bot', 'is_english'])
        
        return posts, comments
    
    def add_metadata(self, posts_df, comments_df):
        """Add useful metadata for analysis"""
        # Posts metadata
        posts = posts_df.copy()
        
        # Convert timestamps to datetime
        if 'created_utc' in posts.columns:
            posts['created_date'] = pd.to_datetime(posts['created_utc'])
            posts['year_month'] = posts['created_date'].dt.strftime('%Y-%m')
            posts['weekday'] = posts['created_date'].dt.day_name()
            posts['hour'] = posts['created_date'].dt.hour
            
        # Add length metrics
        posts['title_length'] = posts['title'].fillna('').apply(len)
        posts['text_length'] = posts['text'].fillna('').apply(len)
        posts['word_count'] = posts['cleaned_text'].fillna('').apply(lambda x: len(x.split()))
        
        # Add post type
        posts['is_self_post'] = posts['is_self'] if 'is_self' in posts.columns else True
        
        # Comments metadata
        comments = comments_df.copy()
        
        # Convert timestamps to datetime
        if 'created_utc' in comments.columns:
            comments['created_date'] = pd.to_datetime(comments['created_utc'])
            comments['year_month'] = comments['created_date'].dt.strftime('%Y-%m')
            comments['weekday'] = comments['created_date'].dt.day_name()
            comments['hour'] = comments['created_date'].dt.hour
            
        # Add length metrics
        comments['text_length'] = comments['text'].fillna('').apply(len)
        comments['word_count'] = comments['cleaned_text'].fillna('').apply(lambda x: len(x.split()))
        
        return posts, comments
    
    def tokenize_and_analyze_text(self, posts_df, comments_df):
        """Analyze text characteristics for additional insights"""
        posts = posts_df.copy()
        comments = comments_df.copy()
        
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
        
        # Add word lists (excluding stopwords but including climate terms)
        posts['content_words'] = posts['cleaned_text'].apply(extract_content_words)
        comments['content_words'] = comments['cleaned_text'].apply(extract_content_words)
        
        # Extract climate-specific terms
        def extract_climate_terms(words):
            return [word for word in words if word in self.climate_stopwords]
        
        posts['climate_terms'] = posts['content_words'].apply(extract_climate_terms)
        comments['climate_terms'] = comments['content_words'].apply(extract_climate_terms)
        
        # Count climate terms
        posts['climate_term_count'] = posts['climate_terms'].apply(len)
        comments['climate_term_count'] = comments['climate_terms'].apply(len)
        
        return posts, comments
    
    def create_twitter_compatible_subset(self, posts_df, comments_df, max_length=280):
        """
        Create a subset of Reddit content that matches Twitter's length constraints
        This is important for your comparative analysis
        """
        posts = posts_df.copy()
        comments = comments_df.copy()
        
        # Filter posts and comments to Twitter-length compatible
        twitter_compatible_posts = posts[posts['cleaned_text'].str.len() <= max_length].copy()
        twitter_compatible_comments = comments[comments['cleaned_text'].str.len() <= max_length].copy()
        
        # Add flag to indicate this is a Twitter-length compatible subset
        twitter_compatible_posts['twitter_compatible'] = True
        twitter_compatible_comments['twitter_compatible'] = True
        
        logger.info(f"Created Twitter-compatible subset: {len(twitter_compatible_posts)} posts and {len(twitter_compatible_comments)} comments")
        
        return twitter_compatible_posts, twitter_compatible_comments
    
    def sample_data_for_manual_validation(self, posts_df, comments_df, sample_size=50):
        """Create a sample for manual validation of data quality"""
        # Sample posts stratified by subreddit
        posts_sample = posts_df.groupby('subreddit', group_keys=False).apply(
            lambda x: x.sample(min(sample_size // posts_df['subreddit'].nunique(), len(x)))
        )


        # Sample comments
        sample_size = min(sample_size, len(comments_df))
        comments_sample = comments_df.sample(sample_size)
        
        # Reset index
        posts_sample = posts_sample.reset_index(drop=True)
        comments_sample = comments_sample.reset_index(drop=True)
        
        return posts_sample, comments_sample
    
    def save_processed_data(self, posts_df, comments_df, posts_twitter_df=None, comments_twitter_df=None, sample_posts=None, sample_comments=None):
        """Save all processed data files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main processed files
        posts_file = os.path.join(self.output_dir, f"processed_posts_{timestamp}.csv")
        comments_file = os.path.join(self.output_dir, f"processed_comments_{timestamp}.csv")
        
        posts_df.to_csv(posts_file, index=False)
        comments_df.to_csv(comments_file, index=False)
        
        logger.info(f"Saved processed posts to: {posts_file}")
        logger.info(f"Saved processed comments to: {comments_file}")
        
        # Save Twitter-compatible subset if available
        if posts_twitter_df is not None and comments_twitter_df is not None:
            twitter_posts_file = os.path.join(self.output_dir, f"twitter_compatible_posts_{timestamp}.csv")
            twitter_comments_file = os.path.join(self.output_dir, f"twitter_compatible_comments_{timestamp}.csv")
            
            posts_twitter_df.to_csv(twitter_posts_file, index=False)
            comments_twitter_df.to_csv(twitter_comments_file, index=False)
            
            logger.info(f"Saved Twitter-compatible posts to: {twitter_posts_file}")
            logger.info(f"Saved Twitter-compatible comments to: {twitter_comments_file}")
        
        # Save sample files for validation if available
        if sample_posts is not None and sample_comments is not None:
            sample_posts_file = os.path.join(self.output_dir, f"sample_posts_for_validation_{timestamp}.csv")
            sample_comments_file = os.path.join(self.output_dir, f"sample_comments_for_validation_{timestamp}.csv")
            
            sample_posts.to_csv(sample_posts_file, index=False)
            sample_comments.to_csv(sample_comments_file, index=False)
            
            logger.info(f"Saved sample posts for validation to: {sample_posts_file}")
            logger.info(f"Saved sample comments for validation to: {sample_comments_file}")
        
        # Save metadata about the processing
        metadata = {
            'timestamp': timestamp,
            'original_posts': len(posts_df),
            'original_comments': len(comments_df),
            'processed_posts': len(posts_df),
            'processed_comments': len(comments_df),
            'twitter_compatible_posts': len(posts_twitter_df) if posts_twitter_df is not None else 0,
            'twitter_compatible_comments': len(comments_twitter_df) if comments_twitter_df is not None else 0,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'files': {
                'processed_posts': posts_file,
                'processed_comments': comments_file,
                'twitter_compatible_posts': twitter_posts_file if posts_twitter_df is not None else None,
                'twitter_compatible_comments': twitter_comments_file if comments_twitter_df is not None else None,
                'sample_posts': sample_posts_file if sample_posts is not None else None,
                'sample_comments': sample_comments_file if sample_comments is not None else None
            }
        }
        
        metadata_file = os.path.join(self.output_dir, f"processing_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Saved processing metadata to: {metadata_file}")
        
        return {
            'posts_file': posts_file,
            'comments_file': comments_file,
            'twitter_posts_file': twitter_posts_file if posts_twitter_df is not None else None,
            'twitter_comments_file': twitter_comments_file if comments_twitter_df is not None else None,
            'sample_posts_file': sample_posts_file if sample_posts is not None else None,
            'sample_comments_file': sample_comments_file if sample_comments is not None else None,
            'metadata_file': metadata_file
        }
    
    def process_data(self, posts_file=None, comments_file=None, create_twitter_subset=True, sample_size=50):
        """Main method to process all data"""
        # 1. Load data
        posts_df, comments_df = self.load_data(posts_file, comments_file)
        
        # 2. Clean and filter data
        logger.info("Cleaning and filtering data...")
        posts_filtered, comments_filtered = self.filter_data(posts_df, comments_df)
        
        # 3. Add metadata
        logger.info("Adding metadata...")
        posts_with_metadata, comments_with_metadata = self.add_metadata(posts_filtered, comments_filtered)
        
        # 4. Tokenize and analyze text
        logger.info("Analyzing text characteristics...")
        posts_analyzed, comments_analyzed = self.tokenize_and_analyze_text(posts_with_metadata, comments_with_metadata)
        
        # 5. Create Twitter-compatible subset if requested
        twitter_posts, twitter_comments = None, None
        if create_twitter_subset:
            logger.info("Creating Twitter-compatible subset...")
            twitter_posts, twitter_comments = self.create_twitter_compatible_subset(posts_analyzed, comments_analyzed)
        
        # 6. Create sample for validation
        logger.info("Creating validation sample...")
        sample_posts, sample_comments = self.sample_data_for_manual_validation(posts_analyzed, comments_analyzed, sample_size)
        
        # 7. Save all processed data
        logger.info("Saving processed data...")
        file_paths = self.save_processed_data(
            posts_analyzed, 
            comments_analyzed,
            twitter_posts,
            twitter_comments,
            sample_posts,
            sample_comments
        )
        
        logger.info("Data processing complete!")
        return file_paths

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process Reddit data for sentiment analysis')
    
    parser.add_argument('--posts', type=str, default=None,
                        help='Path to the posts CSV file (default: most recent in data/raw/)')
    
    parser.add_argument('--comments', type=str, default=None,
                        help='Path to the comments CSV file (default: most recent in data/raw/)')
    
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data (default: data/processed)')
    
    parser.add_argument('--twitter-subset', type=bool, default=True,
                        help='Create Twitter-compatible subset (default: True)')
    
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
            posts_file=args.posts,
            comments_file=args.comments,
            create_twitter_subset=args.twitter_subset,
            sample_size=args.sample_size
        )
        
        logger.info("Processing complete!")
        logger.info(f"Output files: {json.dumps(file_paths, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()