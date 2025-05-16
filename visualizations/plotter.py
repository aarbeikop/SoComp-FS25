"""Base plotting functionality for sentiment analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger('sentiment_analysis')

class SentimentPlotter:
    """Base class for sentiment visualization."""
    
    def __init__(self, output_dir=None):
        """Initialize the plotter.
        
        Args:
            output_dir (str, optional): Directory for output files.
        """
        from config import DEFAULT_OUTPUT_DIR, DEFAULT_VISUALIZATIONS_DIR
        
        # Set output directory
        self.output_dir = output_dir if output_dir else DEFAULT_OUTPUT_DIR
        self.viz_dir = os.path.join(self.output_dir, DEFAULT_VISUALIZATIONS_DIR)
        
        # Create visualization directory if it doesn't exist
        from utils.file_utils import ensure_dir
        ensure_dir(self.viz_dir)
        
        # Set up Seaborn
        sns.set_style("whitegrid")
        
    def save_figure(self, fig, filename, dpi=300):
        """Save a figure to a file.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            filename (str): Filename for the saved figure.
            dpi (int, optional): DPI for the saved figure. Defaults to 300.
            
        Returns:
            str: Path to the saved figure.
        """
        filepath = os.path.join(self.viz_dir, filename)
        
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved figure to: {filepath}")
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")
            plt.close(fig)
            return None
