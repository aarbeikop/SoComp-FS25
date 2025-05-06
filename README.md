# Climate Change Sentiment Analysis: Reddit vs Twitter

This repository contains the code for a cross-platform sentiment analysis project comparing Reddit and Twitter discourse on climate change, as part of the Social Computing FS25 course.

## Project Overview

This project analyzes how sentiment varies between Reddit and Twitter when users discuss climate change topics. We focus on understanding how platform-specific features influence user sentiment expression using VADER sentiment analysis.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Reddit API credentials

### Installation

1. Clone this repository
   ```
   git clone https://github.com/aarbeikop/SoComp-FS25.git
   cd climate-sentiment-analysis
   ```

2. Install required packages
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=Climate_Change_Research_Script/1.0 by YourUsername
   
   # Optional settings
   REDDIT_SEARCH_QUERY=climate change
   REDDIT_TIME_FILTER=week
   REDDIT_LIMIT_PER_SUBREDDIT=50
   REDDIT_COMMENT_LIMIT=25
   REDDIT_COMMENT_SORT=top
   
   # Optional subreddits list (comma-separated)
   REDDIT_SUBREDDITS=climatechange,climate,ClimateActionPlan,environment,science,worldnews,news,collapse,sustainability,renewable
   ```

### Getting Reddit API Credentials

1. Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps)
2. Scroll down and click "create another app..."
3. Fill in the details:
   - Name: Climate Change Research Script (or any name)
   - Select "script"
   - Description: Research project for sentiment analysis
   - About URL: (leave blank)
   - Redirect URI: http://localhost:8080
4. Click "create app"
5. The `client_id` is the string under the app name
6. The `client_secret` is labeled "secret"

For more information on setting up PRAW, check [this beginner's guide](https://medium.com/@archanakkokate/scraping-reddit-data-using-python-and-praw-a-beginners-guide-7047962f5d29).

## Running the Scraper

Basic usage:
```
python reddit_scraper.py
```

Customize parameters:
```
python reddit_scraper.py --query "global warming" --time-filter month --limit 100 --comment-limit 50 --subreddits "climatechange,science,environment,worldnews" --output-dir "data/custom_dataset"
```

Available parameters:
- `--query`: Search query (default: "climate change" or from .env)
- `--time-filter`: Time range (options: hour, day, week, month, year, all)
- `--limit`: Maximum posts per subreddit
- `--comment-limit`: Maximum comments per post
- `--subreddits`: Comma-separated list of subreddits to scrape
- `--output-dir`: Output directory for scraped data

## Recommended Subreddits (we should consider exploring)

### Core Climate Science Subreddits
- r/climatescience - Technical discussions about climate research
- r/climate_science - Scientific climate discussion
- r/climateoffensive - Action-oriented climate discussion
- r/ClimateChange - General climate change discussions
- r/GlobalWarming - Focus on warming aspects

### Environmental Policy & Action
- r/environmental_policy - Policy discussions
- r/climate_anxiety - Emotional responses to climate issues
- r/ExtinctionRebellion - Climate activism community
- r/ClimateActionPlan - Solutions-focused discussion
- r/FridaysForFuture - Youth climate movement

### Related Environmental Subreddits
- r/solarpunk - Optimistic environmental futures
- r/Renewable - Renewable energy topics
- r/ZeroWaste - Environmental lifestyle discussions
- r/Permaculture - Sustainable agriculture practices
- r/environment - General environmental issues

### Contrasting Viewpoints
- r/climateskeptics - Alternative perspectives on climate issues
- r/climatedisalarmism - Skeptical climate discussions
- r/energy - Various energy transition perspectives

### General Discussion Forums
- r/futurology - Future implications of climate change
- r/science - Scientific publications including climate science
- r/worldnews - Climate articles in mainstream context
- r/EverythingScience - Broader science discussions
- r/AskScience - Questions about climate science

## Data Processing

After scraping, the data will be saved in `data/raw/` with the following files:
- `reddit_posts_[timestamp].csv`: Contains all collected posts
- `reddit_comments_[timestamp].csv`: Contains all collected comments
- `metadata_[timestamp].json`: Contains metadata about the scraping session

## Project Timeline

- **05.05.2025 - 10.05.2025**: Script development
- **11.05.2025 - 15.05.2025**: Data collection
- **16.05.2025**: Data preprocessing
- **17.05.2025**: Sentiment analysis with VADER
- **18.05.2025 - 21.05.2025**: Analysis and visualization
- **27.05.2025**: Submission deadline
- **28.05.2025**: Final presentations

## Additional Resources

- [PRAW Documentation](https://praw.readthedocs.io/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Twitter Dataset on Kaggle](https://www.kaggle.com/datasets) (search for climate change datasets)
