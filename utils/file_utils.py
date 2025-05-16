"""File utilities for the sentiment analysis project."""

import os
import json
import pandas as pd
import logging

logger = logging.getLogger('sentiment_analysis')

def ensure_dir(directory):
    """Ensure that a directory exists.
    
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def save_dataframe(df, filepath, index=False):
    """Save a DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        filepath (str): Path to save the CSV file.
        index (bool, optional): Whether to save the DataFrame index. Defaults to False.
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(filepath))
    
    # Save DataFrame
    df.to_csv(filepath, index=index)
    logger.debug(f"Saved DataFrame to: {filepath}")

def save_json(data, filepath, indent=4):
    """Save data to a JSON file.
    
    Args:
        data (dict): Data to save.
        filepath (str): Path to save the JSON file.
        indent (int, optional): JSON indentation. Defaults to 4.
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(filepath))
    
    # Save JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.debug(f"Saved JSON data to: {filepath}")

def save_text(text, filepath):
    """Save text to a file.
    
    Args:
        text (str): Text to save.
        filepath (str): Path to save the text file.
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(filepath))
    
    # Save text
    with open(filepath, 'w') as f:
        f.write(text)
    logger.debug(f"Saved text to: {filepath}")