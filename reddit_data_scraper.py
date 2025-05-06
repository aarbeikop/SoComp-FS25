"""
Reddit Climate Change Data Scraper
This script collects posts and comments related to climate change from specified subreddits.
"""

import praw
import pandas as pd
import datetime
import time
from tqdm import tqdm
import os
import configparser

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)

# Config setup
config = configparser.ConfigParser()
config['Reddit'] = {
    'client_id': 'YOUR_CLIENT_ID',
    'client_secret': 'YOUR_CLIENT_SECRET',
    'user_agent': 'Climate_Change_Research_Script/1.0 by YourUsername',
    'username': '',  # Optional
    'password': ''   # Optional
}

# Save config to file
with open('config.ini', 'w') as configfile:
    config.write(configfile)
print("Created config.ini file. Please fill in your Reddit API credentials.")

def setup_reddit_api():
    """Set up the Reddit API client using PRAW"""
    config.read('config.ini')
    
    reddit = praw.Reddit(
        client_id=config['Reddit']['client_id'],
        client_secret=config['Reddit']['client_secret'],
        user_agent=config['Reddit']['user_agent'],
        username=config['Reddit']['username'] if config['Reddit']['username'] else None,
        password=config['Reddit']['password'] if config['Reddit']['password'] else None
    )
    return reddit

def scrape_subreddit(reddit, subreddit_name, query="climate change", limit=100, time_filter="week"):
    """Scrape posts from a specific subreddit containing the query"""
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    comments_data = []
    
    print(f"Scraping r/{subreddit_name} for posts about '{query}'...")
    
    # Search for posts
    for post in tqdm(subreddit.search(query, limit=limit, time_filter=time_filter), 
                     desc=f"Posts from r/{subreddit_name}", total=limit):
        
        # Get post details
        post_data = {
            'id': post.id,
            'subreddit': subreddit_name,
            'title': post.title,
            'text': post.selftext,
            'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
            'score': post.score,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio,
            'url': post.url,
            'is_original_content': post.is_original_content,
            'post_type': 'submission'
        }
        posts_data.append(post_data)
        
        # Get comments (limited to avoid rate limits)
        try:
            post.comments.replace_more(limit=0)  # Only get top-level comments
            for comment in post.comments.list():
                if hasattr(comment, 'body'):  # Make sure it's a regular comment
                    comment_data = {
                        'id': comment.id,
                        'parent_id': post.id,
                        'subreddit': subreddit_name,
                        'text': comment.body,
                        'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'post_type': 'comment'
                    }
                    comments_data.append(comment_data)
        except Exception as e:
            print(f"Error fetching comments for post {post.id}: {e}")
        
        # Sleep to respect rate limits
        time.sleep(0.5)
    
    return pd.DataFrame(posts_data), pd.DataFrame(comments_data)

def main():
    # Setup Reddit API
    try:
        reddit = setup_reddit_api()
        print("Successfully connected to Reddit API")
    except Exception as e:
        print(f"Error connecting to Reddit API: {e}")
        print("Please check your credentials in config.ini")
        return

    # List of climate-related subreddits to scrape
    subreddits = [
        'climatechange',
        'climate',
        'ClimateActionPlan',
        'environment',
        'science',
        'worldnews',
        'news',
        'collapse',
        'sustainability',
        'renewable'
    ]
    
    # Initialize empty DataFrames for all data
    all_posts = pd.DataFrame()
    all_comments = pd.DataFrame()
    
    # Scrape each subreddit
    for subreddit in subreddits:
        try:
            posts, comments = scrape_subreddit(
                reddit, 
                subreddit, 
                query="climate change", 
                limit=50,  # Adjust as needed
                time_filter="week"  # Options: hour, day, week, month, year, all
            )
            
            print(f"Scraped {len(posts)} posts and {len(comments)} comments from r/{subreddit}")
            
            # Append to the combined DataFrames
            all_posts = pd.concat([all_posts, posts], ignore_index=True)
            all_comments = pd.concat([all_comments, comments], ignore_index=True)
            
            # Sleep to respect rate limits
            time.sleep(2)
            
        except Exception as e:
            print(f"Error scraping r/{subreddit}: {e}")
    
    # Save the data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_posts.to_csv(f'data/raw/reddit_posts_{timestamp}.csv', index=False)
    all_comments.to_csv(f'data/raw/reddit_comments_{timestamp}.csv', index=False)
    
    print(f"Scraping complete. Saved {len(all_posts)} posts and {len(all_comments)} comments.")
    print(f"Files saved as data/raw/reddit_posts_{timestamp}.csv and data/raw/reddit_comments_{timestamp}.csv")

if __name__ == "__main__":
    main()