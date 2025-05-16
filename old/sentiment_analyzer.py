"""Core sentiment analysis logic."""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

logger = logging.getLogger('sentiment_analysis')

class SentimentAnalyzer:
    """Class for analyzing sentiment in social media data."""
    
    def __init__(self, output_dir=None):
        """Initialize the sentiment analyzer.
        
        Args:
            output_dir (str, optional): Directory for output files.
        """
        from config import DEFAULT_OUTPUT_DIR
        
        # Set output directory
        self.output_dir = output_dir if output_dir else DEFAULT_OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        from utils.file_utils import ensure_dir
        ensure_dir(self.output_dir)
        ensure_dir(os.path.join(self.output_dir, 'visualizations'))
        
        # Set up NLTK resources
        from utils.nlp_utils import setup_nltk
        setup_nltk()
        
        # Initialize sentiment processor
        from data.data_processor import SentimentProcessor
        self.processor = SentimentProcessor()
        
        # Track analysis metadata
        self.analysis_stats = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reddit_posts_count': 0,
            'reddit_comments_count': 0,
            'twitter_posts_count': 0,
            'platforms_compared': [],
            'sentiment_distributions': {},
            'end_time': None
        }
    
    def compute_sentiment_stats(self, data):
        """Compute sentiment statistics for each dataset.
        
        Args:
            data (dict): Dictionary of DataFrames.
            
        Returns:
            dict: Sentiment statistics.
        """
        stats = {}
        
        for key, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            logger.info(f"Computing sentiment statistics for {key}")
            
            # Overall sentiment distribution
            sentiment_counts = df['sentiment_category'].value_counts(normalize=True).to_dict()
            
            # Average sentiment scores
            avg_scores = {
                'avg_compound': df['sentiment_compound'].mean(),
                'avg_positive': df['sentiment_pos'].mean(),
                'avg_negative': df['sentiment_neg'].mean(),
                'avg_neutral': df['sentiment_neu'].mean()
            }
            
            # Confidence intervals for sentiment scores
            confidence_intervals = {
                'compound_ci': (
                    df['sentiment_compound'].mean() - 1.96 * df['sentiment_compound'].std() / np.sqrt(len(df)),
                    df['sentiment_compound'].mean() + 1.96 * df['sentiment_compound'].std() / np.sqrt(len(df))
                )
            }
            
            # Store stats
            stats[key] = {
                'count': len(df),
                'sentiment_distribution': sentiment_counts,
                'average_scores': avg_scores,
                'confidence_intervals': confidence_intervals
            }
            
            # Add to analysis metadata
            self.analysis_stats['sentiment_distributions'][key] = sentiment_counts
        
        return stats
    
    def run_analysis(self, reddit_posts=None, reddit_comments=None, twitter=None):
        """Run sentiment analysis on the provided data.
        
        Args:
            reddit_posts (pandas.DataFrame, optional): Reddit posts data.
            reddit_comments (pandas.DataFrame, optional): Reddit comments data.
            twitter (pandas.DataFrame, optional): Twitter data.
            
        Returns:
            dict: Analysis results.
        """
        # Store the dataset counts
        if reddit_posts is not None:
            self.analysis_stats['reddit_posts_count'] = len(reddit_posts)
            if 'reddit' not in self.analysis_stats['platforms_compared']:
                self.analysis_stats['platforms_compared'].append('reddit')
        
        if reddit_comments is not None:
            self.analysis_stats['reddit_comments_count'] = len(reddit_comments)
            if 'reddit' not in self.analysis_stats['platforms_compared']:
                self.analysis_stats['platforms_compared'].append('reddit')
        
        if twitter is not None:
            self.analysis_stats['twitter_posts_count'] = len(twitter)
            if 'twitter' not in self.analysis_stats['platforms_compared']:
                self.analysis_stats['platforms_compared'].append('twitter')
        
        # Prepare dictionary for processed data
        processed_data = {}
        
        # Process Reddit posts
        if reddit_posts is not None and len(reddit_posts) > 0:
            logger.info("Processing Reddit posts")
            processed_data['reddit_posts'] = self.processor.process_dataframe(
                reddit_posts, text_column='text'
            )
        
        # Process Reddit comments
        if reddit_comments is not None and len(reddit_comments) > 0:
            logger.info("Processing Reddit comments")
            processed_data['reddit_comments'] = self.processor.process_dataframe(
                reddit_comments, text_column='text'
            )
        
        # Process Twitter data
        if twitter is not None and len(twitter) > 0:
            logger.info("Processing Twitter data")
            processed_data['twitter'] = self.processor.process_dataframe(
                twitter, text_column='message'
            )
        
        # Combine Reddit posts and comments if both are present
        if 'reddit_posts' in processed_data and 'reddit_comments' in processed_data:
            logger.info("Combining Reddit posts and comments")
            processed_data['reddit_combined'] = self.processor.combine_reddit_data(
                processed_data['reddit_posts'],
                processed_data['reddit_comments']
            )
        
        # Compute sentiment statistics
        logger.info("Computing sentiment statistics")
        sentiment_stats = self.compute_sentiment_stats(processed_data)
        
        # Complete analysis metadata
        self.analysis_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Return results
        results = {
            'processed_data': processed_data,
            'sentiment_stats': sentiment_stats,
            'metadata': self.analysis_stats
        }
        
        return results