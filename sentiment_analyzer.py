"""
Reddit and Twitter Sentiment Analyzer for Climate Change Discourse

This script performs sentiment analysis on processed Reddit and Twitter data using VADER,
and generates comparative visualizations and insights.

Usage:
    python sentiment_analyzer.py --reddit-posts [POSTS_CSV] --reddit-comments [COMMENTS_CSV] --twitter [TWITTER_CSV] --output [OUTPUT_DIR]
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import nltk
import ssl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Now download NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    """Class to analyze sentiment in Reddit and Twitter data"""
    
    def __init__(self, output_dir='data/sentiment'):
        """Initialize the analyzer"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        # Download necessary NLTK resources if not already downloaded
        self._setup_nltk()
        
        # Initialize VADER sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
        
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
    
    def _setup_nltk(self):
        """Download necessary NLTK resources"""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            logger.info("Downloaded NLTK resources")
    
    def load_data(self, reddit_posts_file=None, reddit_comments_file=None, twitter_file=None):
        """Load processed data files for sentiment analysis"""
        data = {}

        # Load Reddit posts if specified
        if reddit_posts_file:
            logger.info(f"Loading Reddit posts from: {reddit_posts_file}")
            data['reddit_posts'] = pd.read_csv(reddit_posts_file)
            self.analysis_stats['reddit_posts_count'] = len(data['reddit_posts'])
            logger.info(f"Loaded {len(data['reddit_posts'])} Reddit posts")

            if 'platform' not in data['reddit_posts'].columns:
                data['reddit_posts']['platform'] = 'reddit'

        # Load Reddit comments if specified
        if reddit_comments_file:
            logger.info(f"Loading Reddit comments from: {reddit_comments_file}")
            data['reddit_comments'] = pd.read_csv(reddit_comments_file)
            self.analysis_stats['reddit_comments_count'] = len(data['reddit_comments'])
            logger.info(f"Loaded {len(data['reddit_comments'])} Reddit comments")

            if 'platform' not in data['reddit_comments'].columns:
                data['reddit_comments']['platform'] = 'reddit'

        # Load Twitter data if specified
        if twitter_file:
            logger.info(f"Loading Twitter data from: {twitter_file}")
            data['twitter'] = pd.read_csv(twitter_file)
            self.analysis_stats['twitter_posts_count'] = len(data['twitter'])
            logger.info(f"Loaded {len(data['twitter'])} Twitter posts")

            # Ensure the dataset has the required columns
            if 'platform' not in data['twitter'].columns:
                data['twitter']['platform'] = 'twitter' 
            if 'sentiment' not in data['twitter'].columns:
                logger.error("Twitter dataset must contain a 'sentiment' column")
                raise ValueError("Twitter dataset must contain a 'sentiment' column")

        # Check what platforms we have for comparison
        if 'reddit_posts' in data or 'reddit_comments' in data:
            self.analysis_stats['platforms_compared'].append('reddit')
        if 'twitter' in data:
            self.analysis_stats['platforms_compared'].append('twitter')

        return data
    
    def analyze_sentiment(self, text):
        """Apply VADER sentiment analysis to text"""
        if not isinstance(text, str) or pd.isna(text) or text == "":
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'sentiment_category': 'neutral'
            }
        
        # Get VADER sentiment scores
        sentiment_scores = self.sid.polarity_scores(text)
        
        # Add sentiment category based on compound score
        if sentiment_scores['compound'] >= 0.05:
            sentiment_scores['sentiment_category'] = 'positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment_scores['sentiment_category'] = 'negative'
        else:
            sentiment_scores['sentiment_category'] = 'neutral'
        
        logger.debug(f"Sentiment analysis output: {sentiment_scores}")
        return sentiment_scores
    
    def apply_sentiment_analysis(self, df, text_column='cleaned_text'):
        """Apply sentiment analysis to a dataframe"""
        logger.debug(f"Available columns: {df.columns.tolist()}")
        logger.info(f"Applying sentiment analysis to {len(df)} items...")

        # Dynamically determine the text column
        if text_column not in df.columns:
            if 'message' in df.columns:
                text_column = 'message'
            elif 'text' in df.columns:
                text_column = 'text'
            else:
                raise ValueError("No suitable text column found in the dataset.")

        # Define sentiment category column
        sentiment_category_col = 'sentiment_category'

        # Initialize sentiment columns if missing
        required_columns = ['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg', sentiment_category_col]
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Process in batches with progress bar
        batch_size = 1000
        for i in tqdm(range(0, len(df), batch_size), desc="Analyzing sentiment"):
            batch = df.iloc[i:i+batch_size].copy()

            # Apply sentiment analysis to each item in the batch
            sentiments = batch[text_column].apply(self.analyze_sentiment)

            # Extract sentiment scores and categories
            batch['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            batch['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
            batch['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
            batch['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
            batch[sentiment_category_col] = sentiments.apply(lambda x: x.get('sentiment_category', 'neutral'))

            # Update the original dataframe
            df.iloc[i:i+batch_size, df.columns.get_indexer(batch.columns)] = batch

        # Debugging: Check if sentiment category column exists
        if sentiment_category_col not in df.columns:
            logger.error(f"Column '{sentiment_category_col}' is missing after sentiment analysis.")
            raise ValueError(f"Sentiment analysis failed to populate '{sentiment_category_col}'.")

        logger.debug(f"Sample data after sentiment analysis: {df.head()}")
        logger.info(f"Sentiment analysis complete")
        return df
    
    def combine_reddit_data(self, posts_df, comments_df):
        """Combine Reddit posts and comments for overall Reddit sentiment"""
        # Create copies to avoid modifying original data
        posts = posts_df.copy() if posts_df is not None else None
        comments = comments_df.copy() if comments_df is not None else None
        
        if posts is None and comments is None:
            logger.warning("No Reddit data provided to combine")
            return None
        
        # Prepare common columns
        common_columns = ['platform', 'cleaned_text', 'created_date', 'word_count', 
                          'sentiment_compound', 'sentiment_pos', 'sentiment_neu',
                          'sentiment_neg', 'sentiment_category']
        
        # Additional columns specific to each type
        post_columns = common_columns + ['title', 'subreddit', 'score', 'num_comments']
        comment_columns = common_columns + ['subreddit', 'score', 'parent_id']
        
        # Select columns and add content_type
        if posts is not None:
            selected_posts = posts[post_columns].copy()
            selected_posts['content_type'] = 'post'
        else:
            selected_posts = pd.DataFrame(columns=post_columns + ['content_type'])
        
        if comments is not None:
            selected_comments = comments[comment_columns].copy()
            selected_comments['content_type'] = 'comment'
        else:
            selected_comments = pd.DataFrame(columns=comment_columns + ['content_type'])
        
        # Combine data
        combined_data = pd.concat([selected_posts, selected_comments], ignore_index=True)
        logger.info(f"Combined {len(selected_posts)} posts and {len(selected_comments)} comments")
        
        return combined_data
    
    def compute_sentiment_stats(self, data):
        """Compute sentiment statistics for each dataset"""
        stats = {}
        
        for key, df in data.items():
            if df is None or len(df) == 0:
                continue
                
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
    
    def plot_sentiment_distribution(self, data, save_path=None):
        """Plot sentiment distribution comparison between platforms"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        plot_data = []
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
                
            sentiment_counts = df['sentiment_category'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            sentiment_counts['percentage'] = sentiment_counts['count'] / sentiment_counts['count'].sum() * 100
            sentiment_counts['dataset'] = dataset_name
            
            plot_data.append(sentiment_counts)
        
        if not plot_data:
            logger.warning("No data available for sentiment distribution plot")
            return None
            
        plot_df = pd.concat(plot_data, ignore_index=True)
        
        # Create the plot
        sns.barplot(x='sentiment', y='percentage', hue='dataset', data=plot_df, ax=ax)
        
        # Customize the plot
        ax.set_title('Sentiment Distribution Comparison', fontsize=16)
        ax.set_xlabel('Sentiment Category', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(title='Platform', fontsize=12, title_fontsize=14)
        
        # Add percentage labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment distribution plot to: {save_path}")
        
        plt.close()
        return fig
    
    def plot_sentiment_by_content_length(self, data, save_path=None):
        """Plot relationship between content length and sentiment"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0 or 'word_count' not in df.columns:
                continue
                
            # Group by word count ranges for readability
            df['word_count_range'] = pd.cut(df['word_count'], 
                                            bins=[0, 10, 25, 50, 100, 250, 500, 1000, df['word_count'].max()],
                                            labels=['1-10', '11-25', '26-50', '51-100', '101-250', '251-500', '501-1000', '1000+'])
            
            # Calculate average sentiment for each word count range
            sentiment_by_length = df.groupby('word_count_range')['sentiment_compound'].mean().reset_index()
            
            # Plot
            sns.lineplot(x='word_count_range', y='sentiment_compound', data=sentiment_by_length, 
                         marker='o', label=dataset_name, ax=ax)
        
        # Customize the plot
        ax.set_title('Sentiment by Content Length', fontsize=16)
        ax.set_xlabel('Word Count Range', fontsize=14)
        ax.set_ylabel('Average Sentiment Score (Compound)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
        ax.legend(title='Platform', fontsize=12, title_fontsize=14)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment by content length plot to: {save_path}")
        
        plt.close()
        return fig
    
    def plot_sentiment_by_engagement(self, data, save_path=None):
        """Plot relationship between engagement metrics and sentiment"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
                
            # Check if we have engagement metrics
            engagement_col = None
            if 'score' in df.columns:
                engagement_col = 'score'  # Reddit upvotes/score
            elif 'retweet_count' in df.columns:
                engagement_col = 'retweet_count'  # Twitter retweets
            elif 'favorite_count' in df.columns or 'like_count' in df.columns:
                engagement_col = 'favorite_count' if 'favorite_count' in df.columns else 'like_count'  # Twitter likes
                
            if not engagement_col:
                logger.warning(f"No engagement metrics found for {dataset_name}")
                continue
                
            # Create engagement buckets (removing outliers first)
            engagement_threshold = df[engagement_col].quantile(0.99)  # Remove top 1% outliers
            filtered_df = df[df[engagement_col] <= engagement_threshold].copy()
            
            # Create engagement categories
            filtered_df['engagement_category'] = pd.qcut(filtered_df[engagement_col], 
                                                       q=5, 
                                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Calculate average sentiment for each engagement category
            sentiment_by_engagement = filtered_df.groupby('engagement_category')['sentiment_compound'].mean().reset_index()
            
            # Plot
            sns.barplot(x='engagement_category', y='sentiment_compound', data=sentiment_by_engagement, 
                       label=dataset_name, ax=ax, alpha=0.7)
        
        # Customize the plot
        ax.set_title('Sentiment by Engagement Level', fontsize=16)
        ax.set_xlabel('Engagement Level', fontsize=14)
        ax.set_ylabel('Average Sentiment Score (Compound)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
        ax.legend(title='Platform', fontsize=12, title_fontsize=14)
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment by engagement plot to: {save_path}")
        
        plt.close()
        return fig
    
    def generate_top_words_by_sentiment(self, df, text_column='cleaned_text', n=20):
        """Generate top words for each sentiment category"""
        # Dictionary to store results
        top_words = {}
        
        # Process each sentiment category
        for sentiment in ['positive', 'neutral', 'negative']:
            # Filter by sentiment
            category_df = df[df['sentiment_category'] == sentiment]
            
            if len(category_df) == 0:
                top_words[sentiment] = []
                continue
                
            # Combine all text
            all_text = ' '.join(category_df[text_column].fillna(''))
            
            # Tokenize
            tokens = word_tokenize(all_text.lower())
            
            # Remove stopwords and punctuation
            stopwords = set(nltk.corpus.stopwords.words('english'))
            words = [word for word in tokens 
                     if word not in stopwords 
                     and word not in string.punctuation
                     and len(word) > 2]
            
            # Count frequencies
            word_counts = Counter(words)
            
            # Get top words
            top_words[sentiment] = word_counts.most_common(n)
        
        return top_words
    
    def plot_top_words_by_sentiment(self, data, save_dir=None):
        """Plot top words for each sentiment category across platforms"""
        figures = {}
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
                
            # Get top words
            top_words = self.generate_top_words_by_sentiment(df)
            
            # Define colors for each sentiment
            colors = {
                'positive': 'green',
                'neutral': 'blue',
                'negative': 'red'
            }
            
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
                                 color=colors[sentiment], ax=ax)
                
                # Customize plot
                ax.set_title(f'Top Words in {sentiment.title()} Content - {dataset_name}', fontsize=16)
                ax.set_xlabel('Count', fontsize=14)
                ax.set_ylabel('Word', fontsize=14)
                
                # Add value labels
                for i, v in enumerate(words_df['count']):
                    ax.text(v + 0.1, i, str(v), color='black', va='center')
                
                plt.tight_layout()
                
                # Save if directory provided
                if save_dir:
                    filename = f"{dataset_name}_{sentiment}_top_words.png"
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved {sentiment} top words plot for {dataset_name} to: {save_path}")
                
                plt.close()
                
                # Store figure reference
                figures[f"{dataset_name}_{sentiment}"] = fig
        
        return figures
    
    def generate_wordclouds(self, data, save_dir=None):
        """Generate wordclouds for each sentiment category across platforms"""
        wordclouds = {}
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0:
                continue
                
            # Process each sentiment category
            for sentiment in ['positive', 'neutral', 'negative']:
                # Filter by sentiment
                category_df = df[df['sentiment_category'] == sentiment]
                
                if len(category_df) == 0:
                    continue
                    
                # Combine all text
                all_text = ' '.join(category_df['cleaned_text'].fillna(''))
                
                if not all_text or len(all_text) < 10:
                    continue
                
                # Define colors based on sentiment
                if sentiment == 'positive':
                    colormap = 'Greens'
                elif sentiment == 'neutral':
                    colormap = 'Blues'
                else:
                    colormap = 'Reds'
                
                # Generate wordcloud
                wordcloud = WordCloud(width=800, height=400, 
                                       background_color='white',
                                       max_words=100,
                                       colormap=colormap,
                                       contour_width=1,
                                       contour_color='steelblue')
                
                wordcloud.generate(all_text)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{sentiment.title()} Sentiment Wordcloud - {dataset_name}', fontsize=16)
                ax.axis('off')
                
                # Save if directory provided
                if save_dir:
                    filename = f"{dataset_name}_{sentiment}_wordcloud.png"
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved {sentiment} wordcloud for {dataset_name} to: {save_path}")
                
                plt.close()
                
                # Store wordcloud
                wordclouds[f"{dataset_name}_{sentiment}"] = wordcloud
        
        return wordclouds
    
    def generate_sentiment_over_time(self, data, save_path=None):
        """Generate sentiment trends over time"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0 or 'created_date' not in df.columns:
                continue
                
            # Ensure created_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['created_date']):
                try:
                    df['created_date'] = pd.to_datetime(df['created_date'])
                except:
                    logger.warning(f"Cannot convert created_date to datetime for {dataset_name}")
                    continue
            
            # Group by date and calculate average sentiment
            df['date'] = df['created_date'].dt.date
            sentiment_over_time = df.groupby('date')['sentiment_compound'].mean().reset_index()
            
            # Plot
            sns.lineplot(x='date', y='sentiment_compound', data=sentiment_over_time, 
                        label=dataset_name, marker='o', ax=ax)
        
        # Customize plot
        ax.set_title('Sentiment Trends Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Average Sentiment Score (Compound)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
        ax.legend(title='Platform', fontsize=12, title_fontsize=14)
        
        # Format date axis
        plt.xticks(rotation=45)
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment over time plot to: {save_path}")
        
        plt.close()
        return fig
    
    def compare_subreddit_sentiment(self, posts_df, comments_df=None, save_path=None):
        """Compare sentiment across different subreddits"""
        # Combine data if both are provided
        if posts_df is not None and comments_df is not None:
            combined_data = []
            
            if 'subreddit' in posts_df.columns:
                posts_df_copy = posts_df.copy()
                posts_df_copy['content_type'] = 'post'
                combined_data.append(posts_df_copy)
            
            if 'subreddit' in comments_df.columns:
                comments_df_copy = comments_df.copy()
                comments_df_copy['content_type'] = 'comment'
                combined_data.append(comments_df_copy)
            
            if combined_data:
                df = pd.concat(combined_data, ignore_index=True)
            else:
                logger.warning("No subreddit data found in either posts or comments")
                return None
        elif posts_df is not None and 'subreddit' in posts_df.columns:
            df = posts_df.copy()
            df['content_type'] = 'post'
        elif comments_df is not None and 'subreddit' in comments_df.columns:
            df = comments_df.copy()
            df['content_type'] = 'comment'
        else:
            logger.warning("No subreddit data found in either posts or comments")
            return None
        
        # Filter for top subreddits with sufficient data
        subreddit_counts = df['subreddit'].value_counts()
        top_subreddits = subreddit_counts[subreddit_counts >= 10].index.tolist()
        
        if not top_subreddits:
            logger.warning("No subreddits with sufficient data found")
            return None
            
        # Filter for top subreddits
        filtered_df = df[df['subreddit'].isin(top_subreddits)]
        
        # Calculate average sentiment per subreddit
        subreddit_sentiment = filtered_df.groupby('subreddit')['sentiment_compound'].agg(['mean', 'std', 'count']).reset_index()
        subreddit_sentiment = subreddit_sentiment.sort_values('mean', ascending=False)
        
        # Calculate sentiment distribution per subreddit
        sentiment_distribution = filtered_df.groupby(['subreddit', 'sentiment_category']).size().unstack(fill_value=0)
        sentiment_distribution = sentiment_distribution.div(sentiment_distribution.sum(axis=1), axis=0) * 100
        
        # Plot average sentiment by subreddit
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set color based on sentiment value
        colors = ['green' if x >= 0.05 else ('blue' if x > -0.05 else 'red') for x in subreddit_sentiment['mean']]
        
        # Create bar plot
        bars = ax.barh(subreddit_sentiment['subreddit'], subreddit_sentiment['mean'], color=colors)
        
        # Add error bars
        error = 1.96 * subreddit_sentiment['std'] / np.sqrt(subreddit_sentiment['count'])
        ax.errorbar(subreddit_sentiment['mean'], subreddit_sentiment['subreddit'], 
                   xerr=error, fmt='none', ecolor='black', capsize=5)
        
        # Add count annotations
        for i, (mean, count) in enumerate(zip(subreddit_sentiment['mean'], subreddit_sentiment['count'])):
            ax.text(mean, i, f" n={count}", va='center')
        
        # Customize plot
        ax.set_title('Average Sentiment by Subreddit', fontsize=16)
        ax.set_xlabel('Average Sentiment Score (Compound)', fontsize=14)
        ax.set_ylabel('Subreddit', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved subreddit sentiment comparison to: {save_path}")
        
        plt.close()
        
        # Create a second visualization for sentiment distribution by subreddit
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        
        # Plot stacked bar chart of sentiment categories
        sentiment_distribution.plot(kind='barh', stacked=True, ax=ax2, 
                                   color={'positive': 'green', 'neutral': 'blue', 'negative': 'red'})
        
        # Customize plot
        ax2.set_title('Sentiment Distribution by Subreddit', fontsize=16)
        ax2.set_xlabel('Percentage (%)', fontsize=14)
        ax2.set_ylabel('Subreddit', fontsize=14)
        ax2.tick_params(labelsize=12)
        ax2.legend(title='Sentiment', fontsize=12, title_fontsize=14)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            distribution_path = save_path.replace('.png', '_distribution.png')
            plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved subreddit sentiment distribution to: {distribution_path}")
        
        plt.close()
        
        return {"avg_sentiment": fig, "distribution": fig2, "data": subreddit_sentiment}
    
    def generate_sentiment_html_report(self, stats, visualizations, output_file='sentiment_report.html'):
        """Generate an HTML report with all sentiment analysis results"""
        # Create HTML report
        report_path = os.path.join(self.output_dir, output_file)
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cross-Platform Sentiment Analysis: Reddit vs Twitter</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 40px;
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .flex-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .stat-box {{
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    width: 30%;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .image-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .positive {{
                    color: green;
                }}
                .neutral {{
                    color: blue;
                }}
                .negative {{
                    color: red;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    background: #f2f2f2;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cross-Platform Sentiment Analysis</h1>
                <p>Comparing Reddit and Twitter discourse on Climate Change</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Analysis Overview</h2>
                    <p>Analysis performed on {datetime.now().strftime('%Y-%m-%d')} using VADER sentiment analysis.</p>
                    
                    <div class="flex-container">
                        <div class="stat-box">
                            <h3>Reddit Posts</h3>
                            <div class="stat-value">{stats.get('reddit_posts', {}).get('count', 0):,}</div>
                        </div>
                        <div class="stat-box">
                            <h3>Reddit Comments</h3>
                            <div class="stat-value">{stats.get('reddit_comments', {}).get('count', 0):,}</div>
                        </div>
                        <div class="stat-box">
                            <h3>Twitter Posts</h3>
                            <div class="stat-value">{stats.get('twitter', {}).get('count', 0):,}</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Sentiment Distribution</h2>
                    <p>Comparing the distribution of positive, neutral, and negative sentiment between platforms.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_distribution.png" alt="Sentiment Distribution">
                    </div>
                    
                    <table>
                        <tr>
                            <th>Platform</th>
                            <th class="positive">Positive</th>
                            <th class="neutral">Neutral</th>
                            <th class="negative">Negative</th>
                        </tr>
                        <tr>
                            <td>Reddit Posts</td>
                            <td class="positive">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                        <tr>
                            <td>Reddit Comments</td>
                            <td class="positive">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                        <tr>
                            <td>Twitter</td>
                            <td class="positive">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Sentiment by Content Length</h2>
                    <p>How content length affects sentiment expression across platforms.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_by_length.png" alt="Sentiment by Content Length">
                    </div>
                    <p>This visualization shows how average sentiment varies with content length, highlighting platform-specific patterns in sentiment expression.</p>
                </div>
                
                <div class="section">
                    <h2>Sentiment by Engagement</h2>
                    <p>Relationship between engagement metrics and sentiment.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_by_engagement.png" alt="Sentiment by Engagement">
                    </div>
                    <p>This chart reveals how user engagement (upvotes, likes, retweets) correlates with sentiment, providing insights into audience preferences on each platform.</p>
                </div>
                
                <div class="section">
                    <h2>Top Words by Sentiment</h2>
                    <p>Most frequent words used in positive, neutral, and negative content.</p>
                    <h3>Positive Sentiment</h3>
                    <div class="image-container">
                        <img src="visualizations/reddit_posts_positive_top_words.png" alt="Reddit Positive Words">
                        <img src="visualizations/twitter_positive_top_words.png" alt="Twitter Positive Words">
                    </div>
                    
                    <h3>Negative Sentiment</h3>
                    <div class="image-container">
                        <img src="visualizations/reddit_posts_negative_top_words.png" alt="Reddit Negative Words">
                        <img src="visualizations/twitter_negative_top_words.png" alt="Twitter Negative Words">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Wordclouds</h2>
                    <p>Visual representation of term frequency by sentiment category.</p>
                    <h3>Reddit Wordclouds</h3>
                    <div class="flex-container">
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_positive_wordcloud.png" alt="Reddit Positive Wordcloud">
                            <p>Positive Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_neutral_wordcloud.png" alt="Reddit Neutral Wordcloud">
                            <p>Neutral Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_negative_wordcloud.png" alt="Reddit Negative Wordcloud">
                            <p>Negative Sentiment</p>
                        </div>
                    </div>
                    
                    <h3>Twitter Wordclouds</h3>
                    <div class="flex-container">
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/twitter_positive_wordcloud.png" alt="Twitter Positive Wordcloud">
                            <p>Positive Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/twitter_neutral_wordcloud.png" alt="Twitter Neutral Wordcloud">
                            <p>Neutral Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/twitter_negative_wordcloud.png" alt="Twitter Negative Wordcloud">
                            <p>Negative Sentiment</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Subreddit Analysis</h2>
                    <p>Comparison of sentiment across different subreddits.</p>
                    <div class="image-container">
                        <img src="visualizations/subreddit_sentiment.png" alt="Subreddit Sentiment">
                    </div>
                    <div class="image-container">
                        <img src="visualizations/subreddit_sentiment_distribution.png" alt="Subreddit Sentiment Distribution">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Sentiment Over Time</h2>
                    <p>Temporal trends in sentiment expression.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_over_time.png" alt="Sentiment Over Time">
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Cross-Platform Sentiment Analysis Project</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at: {report_path}")
        return report_path

    def run_complete_analysis(self, reddit_posts_file=None, reddit_comments_file=None, twitter_file=None):
            """
            Run complete sentiment analysis workflow and generate all outputs
            
            Parameters:
                reddit_posts_file: Path to Reddit posts CSV
                reddit_comments_file: Path to Reddit comments CSV
                twitter_file: Path to Twitter posts CSV
                
            Returns:
                Dictionary with file paths to all generated outputs
            """
            outputs = {}
        
                # 1. Load data
            logger.info("Loading data files...")
            data = self.load_data(reddit_posts_file, reddit_comments_file, twitter_file)

            # 2. Apply sentiment analysis
            analyzed_data = {}
            for key, df in data.items():
                if df is not None and len(df) > 0:
                    logger.info(f"Analyzing sentiment for {key}...")

                    # Specify the correct text column for each dataset
                    if key == 'twitter':
                        text_column = 'message'
                    elif key in ['reddit_posts', 'reddit_comments']:
                        text_column = 'text'
                    else:
                        logger.error(f"Unknown dataset key: {key}")
                        continue

                    # Apply sentiment analysis
                    analyzed_data[key] = self.apply_sentiment_analysis(df, text_column)

            # 3. Combine Reddit posts and comments if both are present
            if 'reddit_posts' in analyzed_data and 'reddit_comments' in analyzed_data:
                logger.info("Combining Reddit posts and comments for analysis...")
                analyzed_data['reddit_combined'] = self.combine_reddit_data(
                    analyzed_data['reddit_posts'], 
                    analyzed_data['reddit_comments']
                )

            # 4. Compute sentiment statistics
            logger.info("Computing sentiment statistics...")
            sentiment_stats = self.compute_sentiment_stats(analyzed_data)
            
            # 5. Create visualization directory
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 6. Generate visualizations
            logger.info("Generating visualizations...")
            
            # Sentiment distribution
            distribution_path = os.path.join(viz_dir, 'sentiment_distribution.png')
            self.plot_sentiment_distribution(analyzed_data, distribution_path)
            outputs['sentiment_distribution'] = distribution_path
            
            # Sentiment by content length
            length_path = os.path.join(viz_dir, 'sentiment_by_length.png')
            self.plot_sentiment_by_content_length(analyzed_data, length_path)
            outputs['sentiment_by_length'] = length_path
            
            # Sentiment by engagement
            engagement_path = os.path.join(viz_dir, 'sentiment_by_engagement.png')
            self.plot_sentiment_by_engagement(analyzed_data, engagement_path)
            outputs['sentiment_by_engagement'] = engagement_path
            
            # Top words by sentiment
            logger.info("Generating top words visualizations...")
            top_words_visualizations = self.plot_top_words_by_sentiment(analyzed_data, viz_dir)
            outputs['top_words'] = top_words_visualizations
            
            # Wordclouds
            logger.info("Generating wordclouds...")
            wordclouds = self.generate_wordclouds(analyzed_data, viz_dir)
            outputs['wordclouds'] = wordclouds
            
            # Subreddit sentiment comparison (if Reddit data available)
            if 'reddit_posts' in analyzed_data:
                logger.info("Comparing sentiment across subreddits...")
                subreddit_path = os.path.join(viz_dir, 'subreddit_sentiment.png')
                subreddit_results = self.compare_subreddit_sentiment(
                    analyzed_data.get('reddit_posts'),
                    analyzed_data.get('reddit_comments'),
                    subreddit_path
                )
                if subreddit_results:
                    outputs['subreddit_sentiment'] = subreddit_path
            
            # Sentiment over time
            time_path = os.path.join(viz_dir, 'sentiment_over_time.png')
            self.generate_sentiment_over_time(analyzed_data, time_path)
            outputs['sentiment_over_time'] = time_path
            
            # 7. Generate HTML report
            logger.info("Generating HTML report...")
            report_path = self.generate_sentiment_html_report(sentiment_stats, outputs)
            outputs['html_report'] = report_path
            
            # 8. Save the analyzed data
            logger.info("Saving analyzed data...")
            for key, df in analyzed_data.items():
                output_file = os.path.join(self.output_dir, f"{key}_with_sentiment.csv")
                df.to_csv(output_file, index=False)
                outputs[f"{key}_data"] = output_file
            
            # 9. Save sentiment stats as JSON
            stats_file = os.path.join(self.output_dir, "sentiment_stats.json")
            # Convert numpy values to Python primitives for JSON serialization
            json_safe_stats = {}
            for key, stats_dict in sentiment_stats.items():
                json_safe_stats[key] = {}
                for stat_key, stat_value in stats_dict.items():
                    if stat_key == 'sentiment_distribution':
                        json_safe_stats[key][stat_key] = {k: float(v) for k, v in stat_value.items()}
                    elif stat_key == 'average_scores':
                        json_safe_stats[key][stat_key] = {k: float(v) for k, v in stat_value.items()}
                    elif stat_key == 'confidence_intervals':
                        json_safe_stats[key][stat_key] = {k: (float(v[0]), float(v[1])) for k, v in stat_value.items()}
                    else:
                        json_safe_stats[key][stat_key] = stat_value
            
            with open(stats_file, 'w') as f:
                json.dump(json_safe_stats, f, indent=4)
            outputs['stats_json'] = stats_file
            
            # 10. Complete analysis metadata
            self.analysis_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.analysis_stats['output_files'] = outputs
            
            metadata_file = os.path.join(self.output_dir, "sentiment_analysis_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.analysis_stats, f, indent=4)
            outputs['metadata'] = metadata_file
            
            logger.info("Sentiment analysis complete!")
            logger.info(f"All outputs saved to: {self.output_dir}")
            
            return outputs

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis for Reddit and Twitter Climate Change Data')
    
    parser.add_argument('--reddit-posts', type=str, default=None,
                        help='Path to the processed Reddit posts CSV file')
    
    parser.add_argument('--reddit-comments', type=str, default=None,
                        help='Path to the processed Reddit comments CSV file')
    
    parser.add_argument('--twitter', type=str, default=None,
                        help='Path to the processed Twitter posts CSV file')
    
    parser.add_argument('--output', type=str, default='data/sentiment',
                        help='Output directory for sentiment analysis results (default: data/sentiment)')
    
    return parser.parse_args()

def main():
    """Main function to run the analyzer"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer(output_dir=args.output)
        
        # Run complete analysis
        outputs = analyzer.run_complete_analysis(
            reddit_posts_file=args.reddit_posts,
            reddit_comments_file=args.reddit_comments,
            twitter_file=args.twitter
        )
        
        logger.info("Analysis complete!")
        logger.info(f"HTML report available at: {outputs.get('html_report', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()