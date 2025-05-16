"""Configuration settings for the sentiment analysis project."""

import os

# Default paths
DEFAULT_OUTPUT_DIR = 'data/sentiment'
DEFAULT_VISUALIZATIONS_DIR = 'visualizations'
DEFAULT_REPORT_FILENAME = 'sentiment_report.html'

# NLTK resources
NLTK_RESOURCES = [
    'vader_lexicon',
    'punkt',
    'stopwords'
]

# Sentiment categories
SENTIMENT_CATEGORIES = ['positive', 'neutral', 'negative']

# Sentiment thresholds (VADER)
SENTIMENT_THRESHOLDS = {
    'positive': 0.05,
    'negative': -0.05
}

# Visualization settings
PLOT_FIGSIZE = {
    'default': (12, 8),
    'large': (14, 10),
    'wide': (14, 8)
}

SENTIMENT_COLORS = {
    'positive': 'green',
    'neutral': 'blue',
    'negative': 'red'
}

WORDCLOUD_COLORMAPS = {
    'positive': 'Greens',
    'neutral': 'Blues',
    'negative': 'Reds'
}

