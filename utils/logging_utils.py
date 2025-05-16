"""Utilities for setting up logging."""

import logging
import os

def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs only to console.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create handlers list
    handlers = [logging.StreamHandler()]
    
    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Get logger
    logger = logging.getLogger('sentiment_analysis')
    
    return logger

