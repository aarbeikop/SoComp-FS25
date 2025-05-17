"""Data processing utilities for sentiment analysis."""

import pandas as pd
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from data.data_processor import SentimentProcessor

# Load your Twitter data
df = pd.read_csv("twitter_data/twitter_sentiment_data.csv")

# Initialize the processor
processor = SentimentProcessor()

# Process sentiment (it will use the 'message' column)
df_with_sentiment = processor.process_dataframe(df)

# Now df_with_sentiment contains sentiment columns for each tweet
print(df_with_sentiment.head())

logger = logging.getLogger('sentiment_analysis')

class SentimentProcessor:
    """Class for processing sentiment in social media data."""
    
    def __init__(self, analyzer=None):
        """Initialize the sentiment processor.
        
        Args:
            analyzer (nltk.sentiment.vader.SentimentIntensityAnalyzer, optional): 
                VADER sentiment analyzer. If None, a new one will be created.
        """
        # Initialize VADER sentiment analyzer
        if analyzer is None:
            from utils.nlp_utils import get_vader_analyzer
            self.analyzer = get_vader_analyzer()
        else:
            self.analyzer = analyzer
    
    def analyze_text(self, text):
        """Apply sentiment analysis to a text.
        
        Args:
            text (str): Input text.
            
        Returns:
            dict: Sentiment scores.
        """
        # Handle invalid input
        if not isinstance(text, str) or pd.isna(text) or text == "":
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'sentiment_category': 'neutral'
            }
        
        # Get sentiment scores
        sentiment_scores = self.analyzer.polarity_scores(text)
        
        # Add sentiment category
        from config import SENTIMENT_THRESHOLDS
        if sentiment_scores['compound'] >= SENTIMENT_THRESHOLDS['positive']:
            sentiment_scores['sentiment_category'] = 'positive'
        elif sentiment_scores['compound'] <= SENTIMENT_THRESHOLDS['negative']:
            sentiment_scores['sentiment_category'] = 'negative'
        else:
            sentiment_scores['sentiment_category'] = 'neutral'
        
        return sentiment_scores
    
    def process_dataframe(self, df, text_column=None, batch_size=1000):
        """Process sentiment for a DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame.
            text_column (str, optional): Column containing text to analyze.
                If None, will try to automatically detect it.
            batch_size (int, optional): Size of batches for processing. Defaults to 1000.
            
        Returns:
            pandas.DataFrame: Processed DataFrame with sentiment scores.
        """
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame provided for sentiment processing")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Try to find the text column if not specified
        if text_column is None:
            if 'message' in df.columns:
                text_column = 'message'  # Twitter
            elif 'text' in df.columns:
                text_column = 'text'  # Reddit
            else:
                logger.error("Could not determine text column for sentiment analysis")
                raise ValueError("No suitable text column found in the dataset")
        
        logger.info(f"Processing sentiment for {len(df_copy)} items using '{text_column}' column")
        
        # Define sentiment columns
        sentiment_cols = [
            'sentiment_compound', 
            'sentiment_pos', 
            'sentiment_neu', 
            'sentiment_neg', 
            'sentiment_category'
        ]
        
        # Initialize sentiment columns if they don't exist
        for col in sentiment_cols:
            if col not in df_copy.columns:
                df_copy[col] = None
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(df_copy), batch_size), desc="Analyzing sentiment"):
            batch = df_copy.iloc[i:i+batch_size].copy()
            
            # Apply sentiment analysis to each row
            sentiments = batch[text_column].apply(self.analyze_text)
            
            # Extract sentiment scores and categories
            batch['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            batch['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
            batch['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
            batch['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
            batch['sentiment_category'] = sentiments.apply(lambda x: x['sentiment_category'])
            
            # Update the original dataframe
            df_copy.iloc[i:i+batch_size, df_copy.columns.get_indexer(batch.columns)] = batch
        
        logger.info("Sentiment analysis completed")
        return df_copy
    
