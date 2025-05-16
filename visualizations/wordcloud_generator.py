"""Word cloud generation for sentiment analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from wordcloud import WordCloud

from plotter import SentimentPlotter
from config import SENTIMENT_CATEGORIES, WORDCLOUD_COLORMAPS

logger = logging.getLogger('sentiment_analysis')

class WordcloudGenerator(SentimentPlotter):
    """Class for generating word clouds from sentiment data."""
    
    def generate_top_words_by_sentiment(self, df, text_column='cleaned_text', n=20):
        """Generate top words for each sentiment category.
        
        Args:
            df (pandas.DataFrame): Input DataFrame.
            text_column (str, optional): Column containing text. Defaults to 'cleaned_text'.
            n (int, optional): Number of top words to extract. Defaults to 20.
            
        Returns:
            dict: Dictionary of top words by sentiment category.
        """
        from utils.nlp_utils import get_top_words
        
        # Dictionary to store results
        top_words = {}
        
        # Process each sentiment category
        for sentiment in SENTIMENT_CATEGORIES:
            # Filter by sentiment
            category_df = df[df['sentiment_category'] == sentiment]
            
            if len(category_df) == 0:
                top_words[sentiment] = []
                continue
            
            # Combine all text
            all_text = ' '.join(category_df[text_column].fillna(''))
            
            # Get top words
            top_words[sentiment] = get_top_words(all_text, n=n)
        
        return top_words
    
    def plot_top_words(self, data, save_dir=None):
        """Plot top words for each sentiment category across platforms.
        
        Args:
            data (dict): Dictionary of DataFrames.
            save_dir (str, optional): Directory to save figures. Defaults to None.
            
        Returns:
            dict: Dictionary of generated figures.
        """
        logger.info("Creating top words visualizations")
        
        if save_dir is None:
            save_dir = self.viz_dir
        
        figures = {}
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            # Get top words
            top_words = self.generate_top_words_by_sentiment(df)
            
            # Create plots for each sentiment
            for sentiment, words in top_words.items():
                if not words:
                    continue
                
                # Convert to DataFrame for plotting
                words_df = pd.DataFrame(words, columns=['word', 'count'])
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot horizontal bar chart
                bars = sns.barplot(y='word', x='count', data=words_df, 
                                 color=SENTIMENT_COLORS[sentiment], ax=ax)
                
                # Customize plot
                ax.set_title(f'Top Words in {sentiment.title()} Content - {dataset_name}', fontsize=16)
                ax.set_xlabel('Count', fontsize=14)
                ax.set_ylabel('Word', fontsize=14)
                
                # Add value labels
                for i, v in enumerate(words_df['count']):
                    ax.text(v + 0.1, i, str(v), color='black', va='center')
                
                plt.tight_layout()
                
                # Save figure
                filename = f"{dataset_name}_{sentiment}_top_words.png"
                filepath = self.save_figure(fig, filename)
                
                # Store figure reference
                figures[f"{dataset_name}_{sentiment}"] = {
                    'figure': fig,
                    'filepath': filepath
                }
        
        return figures
    
    def generate_wordclouds(self, data):
        """Generate wordclouds for each sentiment category across platforms.
        
        Args:
            data (dict): Dictionary of DataFrames.
            
        Returns:
            dict: Dictionary of generated wordclouds.
        """
        logger.info("Creating wordcloud visualizations")
        
        wordclouds = {}
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            # Process each sentiment category
            for sentiment in SENTIMENT_CATEGORIES:
                # Filter by sentiment
                category_df = df[df['sentiment_category'] == sentiment]
                
                if len(category_df) == 0:
                    continue
                
                # Determine text column
                text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'message')
                
                # Combine all text
                all_text = ' '.join(category_df[text_column].fillna(''))
                
                if not all_text or len(all_text) < 10:
                    continue
                
                # Generate wordcloud
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     max_words=100,
                                     colormap=WORDCLOUD_COLORMAPS[sentiment],
                                     contour_width=1,
                                     contour_color='steelblue')
                
                wordcloud.generate(all_text)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{sentiment.title()} Sentiment Wordcloud - {dataset_name}', fontsize=16)
                ax.axis('off')
                
                # Save figure
                filename = f"{dataset_name}_{sentiment}_wordcloud.png"
                filepath = self.save_figure(fig, filename)
                
                # Store wordcloud
                wordclouds[f"{dataset_name}_{sentiment}"] = {
                    'wordcloud': wordcloud,
                    'figure': fig,
                    'filepath': filepath
                }
        
        return wordclouds