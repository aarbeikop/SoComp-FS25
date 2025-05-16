"""Data loading functionalities for sentiment analysis."""

import pandas as pd
import logging

logger = logging.getLogger('sentiment_analysis')

class DataLoader:
    """Class for loading social media data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.datasets = {}
        self.stats = {
            'reddit_posts_count': 0,
            'reddit_comments_count': 0,
            'twitter_posts_count': 0,
            'platforms': []
        }
    
    def load_reddit_posts(self, filepath):
        """Load Reddit posts from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        logger.info(f"Loading Reddit posts from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            self.stats['reddit_posts_count'] = len(df)
            logger.info(f"Loaded {len(df)} Reddit posts")
            
            # Ensure platform column exists
            if 'platform' not in df.columns:
                df['platform'] = 'reddit'
            
            # Add to platforms list if not already included
            if 'reddit' not in self.stats['platforms']:
                self.stats['platforms'].append('reddit')
            
            # Store in datasets dictionary
            self.datasets['reddit_posts'] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading Reddit posts: {str(e)}")
            raise
    
    def load_reddit_comments(self, filepath):
        """Load Reddit comments from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        logger.info(f"Loading Reddit comments from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            self.stats['reddit_comments_count'] = len(df)
            logger.info(f"Loaded {len(df)} Reddit comments")
            
            # Ensure platform column exists
            if 'platform' not in df.columns:
                df['platform'] = 'reddit'
            
            # Add to platforms list if not already included
            if 'reddit' not in self.stats['platforms']:
                self.stats['platforms'].append('reddit')
            
            # Store in datasets dictionary
            self.datasets['reddit_comments'] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading Reddit comments: {str(e)}")
            raise
    
    def load_twitter_data(self, filepath):
        """Load Twitter data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        logger.info(f"Loading Twitter data from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            self.stats['twitter_posts_count'] = len(df)
            logger.info(f"Loaded {len(df)} Twitter posts")
            
            # Ensure platform column exists
            if 'platform' not in df.columns:
                df['platform'] = 'twitter'
            
            # Add to platforms list if not already included
            if 'twitter' not in self.stats['platforms']:
                self.stats['platforms'].append('twitter')
            
            # Store in datasets dictionary
            self.datasets['twitter'] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading Twitter data: {str(e)}")
            raise
    
    def get_datasets(self):
        """Get all loaded datasets.
        
        Returns:
            dict: Dictionary of datasets.
        """
        return self.datasets
    
    def get_stats(self):
        """Get loading statistics.
        
        Returns:
            dict: Loading statistics.
        """
        return self.stats