"""NLP utilities for the sentiment analysis project."""

import nltk
import ssl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import logging

logger = logging.getLogger('sentiment_analysis')

def setup_nltk(resources=None):
    """Download and setup necessary NLTK resources.
    
    Args:
        resources (list, optional): List of NLTK resources to download.
    """
    # Handle SSL certificate issues for NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download resources
    if resources is None:
        from sentiment_analysis.config import NLTK_RESOURCES
        resources = NLTK_RESOURCES
        
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                          f'sentiment/{resource}' if resource == 'vader_lexicon' else
                          f'corpora/{resource}')
            logger.debug(f"NLTK resource '{resource}' already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

def get_vader_analyzer():
    """Get a configured VADER sentiment analyzer.
    
    Returns:
        nltk.sentiment.vader.SentimentIntensityAnalyzer: Configured analyzer.
    """
    return SentimentIntensityAnalyzer()

def get_top_words(text, n=20, stopwords=None):
    """Extract the top words from a text.
    
    Args:
        text (str): Input text.
        n (int, optional): Number of top words to return. Defaults to 20.
        stopwords (set, optional): Set of stopwords to exclude. If None, uses NLTK stopwords.
        
    Returns:
        list: List of (word, count) tuples.
    """
    # Get stopwords if not provided
    if stopwords is None:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    words = [word for word in tokens 
             if word not in stopwords 
             and word not in string.punctuation
             and len(word) > 2]
    
    # Count frequencies
    word_counts = Counter(words)
    
    # Get top words
    return word_counts.most_common(n)