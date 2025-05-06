"""
Enhanced Reddit Climate Change Data Scraper
This script collects posts and comments related to climate change from specified subreddits
with improved error handling, data quality controls, and better rate limit management.
"""

import praw
import pandas as pd
import datetime
import time
import random
from tqdm import tqdm
import os
import logging
import argparse
import json
from dotenv import load_dotenv
from prawcore.exceptions import PrawcoreException, ResponseException, RequestException
import re
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reddit_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('logs', exist_ok=True)

class RedditScraper:
    """Class to handle Reddit data scraping with better organization"""
    
    def __init__(self, query="climate change", time_filter="week", limit=50, comment_limit=25):
        """Initialize the scraper with parameters"""
        self.query = query
        self.time_filter = time_filter
        self.limit = limit
        self.comment_limit = comment_limit
        self.reddit = self._setup_reddit_api()
        
        # Track statistics
        self.stats = {
            "total_posts": 0,
            "total_comments": 0,
            "failed_subreddits": [],
            "successful_subreddits": [],
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "rate_limit_hits": 0
        }
    
    def _setup_reddit_api(self):
        """Set up the Reddit API client using PRAW with credentials from .env file"""
        
        # Get credentials from environment variables
        client_id = os.environ.get('REDDIT_CLIENT_ID')
        client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        user_agent = os.environ.get('REDDIT_USER_AGENT', 'Climate_Change_Research_Script/1.0')
        
        # Check if credentials exist
        if not client_id or not client_secret:
            raise ValueError("Missing required Reddit API credentials in .env file. "
                             "Please ensure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set.")
        
        # Initialize Reddit API with better error handling
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            # Test the connection
            reddit.user.me()  # Will be None for read-only
            logger.info("Successfully connected to Reddit API")
            return reddit
        except PrawcoreException as e:
            logger.error(f"Error initializing Reddit API: {e}")
            raise
    
    def get_subreddits_list(self, custom_subreddits=None):
        """Get list of subreddits to scrape"""
        if custom_subreddits:
            return custom_subreddits
            
        # Get subreddits from environment variable if available
        subreddits_env = os.environ.get('REDDIT_SUBREDDITS', '')
        if subreddits_env:
            return [s.strip() for s in subreddits_env.split(',')]
        
        # Default list of climate-related subreddits
        return [
            'climatechange',
            'climate',
            'ClimateActionPlan',
            'environment',
            'science',
            'worldnews',
            'news',
            'collapse',
            'sustainability',
            'renewable',
            'energy',
            'green',
            'conservation'
        ]
    
    def scrape_subreddit(self, subreddit_name):
        """Scrape posts and comments from a specific subreddit"""
        logger.info(f"Scraping r/{subreddit_name} for posts about '{self.query}'...")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []
        comments_data = []
        
        try:
            # Search for posts with progress bar
            for post in tqdm(subreddit.search(self.query, limit=self.limit, time_filter=self.time_filter), 
                             desc=f"Posts from r/{subreddit_name}", total=self.limit):
                
                # Check if post has been deleted or removed
                if post.selftext == "[deleted]" or post.selftext == "[removed]":
                    continue
                
                # Get post details with more metadata
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
                    'permalink': post.permalink,
                    'is_original_content': post.is_original_content if hasattr(post, 'is_original_content') else False,
                    'is_self': post.is_self,
                    'is_video': post.is_video if hasattr(post, 'is_video') else False,
                    'post_type': 'submission',
                    'author': str(post.author) if post.author else '[deleted]',
                    'distinguished': post.distinguished if hasattr(post, 'distinguished') else None,
                    'stickied': post.stickied if hasattr(post, 'stickied') else False,
                    'scraped_utc': datetime.datetime.now().timestamp()
                }
                posts_data.append(post_data)
                
                # Get comments with improved handling
                try:
                    # Use a more efficient approach to get comments
                    post.comments.replace_more(limit=0)  # Skip "more comments" for efficiency
                    
                    # Sort comments if possible
                    comment_sort = os.environ.get('REDDIT_COMMENT_SORT', 'top')
                    if hasattr(post, 'comment_sort'):
                        post.comment_sort = comment_sort
                        post.comments.replace_more(limit=0)  # Refresh with new sort
                    
                    # Get top-level comments
                    for i, comment in enumerate(post.comments.list()):
                        if i >= self.comment_limit:
                            break
                            
                        if hasattr(comment, 'body'):  # Make sure it's a regular comment
                            # Skip deleted/removed comments
                            if comment.body == "[deleted]" or comment.body == "[removed]":
                                continue
                                
                            comment_data = {
                                'id': comment.id,
                                'parent_id': post.id,
                                'post_id': post.id,
                                'post_title': post.title,
                                'subreddit': subreddit_name,
                                'text': comment.body,
                                'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                                'score': comment.score,
                                'is_submitter': comment.is_submitter if hasattr(comment, 'is_submitter') else False,
                                'post_type': 'comment',
                                'author': str(comment.author) if comment.author else '[deleted]',
                                'permalink': f"https://www.reddit.com{post.permalink}{comment.id}/",
                                'distinguished': comment.distinguished if hasattr(comment, 'distinguished') else None,
                                'stickied': comment.stickied if hasattr(comment, 'stickied') else False,
                                'scraped_utc': datetime.datetime.now().timestamp()
                            }
                            comments_data.append(comment_data)
                except PrawcoreException as e:
                    logger.warning(f"API error when fetching comments for post {post.id}: {e}")
                    # Implement exponential backoff for rate limits
                    if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                        self.stats["rate_limit_hits"] += 1
                        backoff_time = min(60, 5 * 2**self.stats["rate_limit_hits"])  # Exponential backoff
                        logger.warning(f"Rate limit hit, backing off for {backoff_time} seconds")
                        time.sleep(backoff_time)
                except Exception as e:
                    logger.warning(f"Error fetching comments for post {post.id}: {e}")
                
                # Dynamic sleep to respect rate limits - vary between 0.5 and 1.5 seconds
                time.sleep(0.5 + random.random())
                
            # Record stats
            self.stats["total_posts"] += len(posts_data)
            self.stats["total_comments"] += len(comments_data)
            self.stats["successful_subreddits"].append(subreddit_name)
            
            logger.info(f"Successfully scraped {len(posts_data)} posts and {len(comments_data)} comments from r/{subreddit_name}")
            
        except PrawcoreException as e:
            logger.error(f"Reddit API error while scraping r/{subreddit_name}: {e}")
            self.stats["failed_subreddits"].append({"subreddit": subreddit_name, "reason": str(e)})
            # Handle rate limiting with exponential backoff
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                self.stats["rate_limit_hits"] += 1
                backoff_time = min(300, 5 * 2**self.stats["rate_limit_hits"])  # Maximum 5 minutes
                logger.warning(f"Rate limit hit, backing off for {backoff_time} seconds")
                time.sleep(backoff_time)
        except Exception as e:
            logger.error(f"Unexpected error while scraping r/{subreddit_name}: {e}")
            self.stats["failed_subreddits"].append({"subreddit": subreddit_name, "reason": str(e)})
        
        return pd.DataFrame(posts_data) if posts_data else pd.DataFrame(), pd.DataFrame(comments_data) if comments_data else pd.DataFrame()
    
    def scrape_multiple_subreddits(self, subreddits=None):
        """Scrape multiple subreddits and combine the results"""
        subreddits_to_scrape = self.get_subreddits_list(subreddits)
        logger.info(f"Starting to scrape {len(subreddits_to_scrape)} subreddits: {', '.join(subreddits_to_scrape)}")
        
        all_posts = pd.DataFrame()
        all_comments = pd.DataFrame()
        
        for subreddit in subreddits_to_scrape:
            try:
                posts_df, comments_df = self.scrape_subreddit(subreddit)
                
                # Append to the combined DataFrames if not empty
                if not posts_df.empty:
                    all_posts = pd.concat([all_posts, posts_df], ignore_index=True)
                if not comments_df.empty:
                    all_comments = pd.concat([all_comments, comments_df], ignore_index=True)
                
                # Sleep between subreddits to respect rate limits
                sleep_time = 2 + random.random() * 3  # 2-5 seconds
                logger.info(f"Sleeping for {sleep_time:.2f} seconds before next subreddit")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error processing subreddit {subreddit}: {e}")
                continue
        
        self.stats["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return all_posts, all_comments
    
    def save_data(self, posts_df, comments_df, output_dir='data/raw'):
        """Save the scraped data to CSV and JSON files with metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        posts_file = os.path.join(output_dir, f"reddit_posts_{timestamp}.csv")
        comments_file = os.path.join(output_dir, f"reddit_comments_{timestamp}.csv")
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        
        # Save to CSV
        posts_df.to_csv(posts_file, index=False)
        comments_df.to_csv(comments_file, index=False)
        
        # Add file information to stats
        self.stats.update({
            "posts_file": posts_file,
            "comments_file": comments_file,
            "timestamp": timestamp,
            "query": self.query,
            "time_filter": self.time_filter,
            "limit_per_subreddit": self.limit,
            "comment_limit": self.comment_limit,
            "post_count": len(posts_df),
            "comment_count": len(comments_df),
            "subreddits_attempted": len(self.stats["successful_subreddits"]) + len(self.stats["failed_subreddits"]),
            "subreddits_successful": len(self.stats["successful_subreddits"]),
            "subreddits_failed": len(self.stats["failed_subreddits"])
        })
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        logger.info(f"Saved {len(posts_df)} posts to {posts_file}")
        logger.info(f"Saved {len(comments_df)} comments to {comments_file}")
        logger.info(f"Saved metadata to {metadata_file}")
        
        return {
            "posts_file": posts_file,
            "comments_file": comments_file,
            "metadata_file": metadata_file
        }
    
    def clean_data(self, posts_df, comments_df):
        """Clean and preprocess the data before saving"""
        if posts_df.empty and comments_df.empty:
            logger.warning("No data to clean, both dataframes are empty")
            return posts_df, comments_df
            
        # Clean posts dataframe
        if not posts_df.empty:
            # Handle NaN values
            posts_df['text'] = posts_df['text'].fillna('')
            posts_df['title'] = posts_df['title'].fillna('')
            
            # Remove duplicates
            original_post_count = len(posts_df)
            posts_df = posts_df.drop_duplicates(subset=['id'])
            if len(posts_df) < original_post_count:
                logger.info(f"Removed {original_post_count - len(posts_df)} duplicate posts")
            
            # Convert datetime columns
            if 'created_utc' in posts_df.columns:
                posts_df['created_date'] = pd.to_datetime(posts_df['created_utc'], unit='s')
        
        # Clean comments dataframe
        if not comments_df.empty:
            # Handle NaN values
            comments_df['text'] = comments_df['text'].fillna('')
            
            # Remove duplicates
            original_comment_count = len(comments_df)
            comments_df = comments_df.drop_duplicates(subset=['id'])
            if len(comments_df) < original_comment_count:
                logger.info(f"Removed {original_comment_count - len(comments_df)} duplicate comments")
            
            # Convert datetime columns
            if 'created_utc' in comments_df.columns:
                comments_df['created_date'] = pd.to_datetime(comments_df['created_utc'], unit='s')
        
        logger.info("Data cleaning completed")
        return posts_df, comments_df

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Reddit Climate Change Data Scraper')
    
    parser.add_argument('--query', type=str, default=os.environ.get('REDDIT_SEARCH_QUERY', 'climate change'),
                        help='Search query (default: from .env or "climate change")')
    
    parser.add_argument('--time-filter', type=str, default=os.environ.get('REDDIT_TIME_FILTER', 'week'),
                        choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                        help='Time filter for search results (default: from .env or "week")')
    
    parser.add_argument('--limit', type=int, default=int(os.environ.get('REDDIT_LIMIT_PER_SUBREDDIT', '50')),
                        help='Maximum posts per subreddit (default: from .env or 50)')
    
    parser.add_argument('--comment-limit', type=int, default=int(os.environ.get('REDDIT_COMMENT_LIMIT', '25')),
                        help='Maximum comments per post (default: from .env or 25)')
    
    parser.add_argument('--subreddits', type=str, default=None,
                        help='Comma-separated list of subreddits to scrape (default: from .env or predefined list)')
    
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Output directory for scraped data (default: data/raw)')
    
    return parser.parse_args()

def main():
    """Main function to run the scraper"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Process subreddits argument if provided
    custom_subreddits = None
    if args.subreddits:
        custom_subreddits = [s.strip() for s in args.subreddits.split(',')]
    
    try:
        # Initialize scraper
        scraper = RedditScraper(
            query=args.query,
            time_filter=args.time_filter,
            limit=args.limit,
            comment_limit=args.comment_limit
        )
        
        # Display scraping parameters
        logger.info(f"Using search parameters: query='{args.query}', time_filter='{args.time_filter}', "
                   f"limit={args.limit}, comment_limit={args.comment_limit}")
        
        # Scrape data
        posts_df, comments_df = scraper.scrape_multiple_subreddits(custom_subreddits)
        
        # Clean data
        posts_df, comments_df = scraper.clean_data(posts_df, comments_df)
        
        # Save data
        file_paths = scraper.save_data(posts_df, comments_df, output_dir=args.output_dir)
        
        logger.info(f"Scraping complete. Collected {len(posts_df)} posts and {len(comments_df)} comments.")
        logger.info(f"Files saved to:\n"
                   f"  Posts: {file_paths['posts_file']}\n"
                   f"  Comments: {file_paths['comments_file']}\n"
                   f"  Metadata: {file_paths['metadata_file']}")
        
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main scraping process: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()