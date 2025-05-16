"""Temporal trend visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

from plotter import SentimentPlotter
from config import PLOT_FIGSIZE

logger = logging.getLogger('sentiment_analysis')

class TemporalPlotter(SentimentPlotter):
    """Class for creating temporal trend visualizations."""
    
    def plot_sentiment_over_time(self, data, filename='sentiment_over_time.png'):
        """Plot sentiment trends over time.
        
        Args:
            data (dict): Dictionary of DataFrames.
            filename (str, optional): Filename for the saved figure.
            
        Returns:
            tuple: (matplotlib.figure.Figure, str) Figure and path to the saved figure.
        """
        logger.info("Creating sentiment over time plot")
        
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE['wide'])
        
        # Flag to track if any data was plotted
        has_data = False
        
        for dataset_name, df in data.items():
            if df is None or len(df) == 0 or 'created_date' not in df.columns:
                continue
            
            # Ensure created_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['created_date']):
                try:
                    df['created_date'] = pd.to_datetime(df['created_date'])
                except Exception as e:
                    logger.warning(f"Cannot convert created_date to datetime for {dataset_name}: {str(e)}")
                    continue
            
            # Group by date and calculate average sentiment
            df['date'] = df['created_date'].dt.date
            sentiment_over_time = df.groupby('date')['sentiment_compound'].mean().reset_index()
            
            # Plot
            sns.lineplot(x='date', y='sentiment_compound', data=sentiment_over_time, 
                        label=dataset_name, marker='o', ax=ax)
            
            has_data = True
        
        if not has_data:
            logger.warning("No data available for sentiment over time plot")
            plt.close(fig)
            return None, None
        
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
        
        # Save the figure
        filepath = self.save_figure(fig, filename)
        
        return fig, filepath