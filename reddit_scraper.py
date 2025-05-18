import praw
import yaml
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
from prawcore.exceptions import PrawcoreException
import sys
from tenacity import retry, wait_exponential, stop_after_attempt
from langdetect import detect
import calendar
import csv


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(
        "reddit_scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
os.makedirs('data/raw', exist_ok=True)
os.makedirs('logs', exist_ok=True)


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


class RedditScraper:
    def __init__(self, query="climate change", time_filter="week", limit=50, comment_limit=25):
        self.query = query
        self.time_filter = time_filter
        self.limit = limit
        self.comment_limit = comment_limit
        self.reddit = self._setup_reddit_api()
        self.tone_map = {
            'climatechange': 'action-oriented',
            'climate': 'scientific',
            'ClimateActionPlan': 'activist',
            'environment': 'emotional',
            'science': 'scientific',
            'worldnews': 'mixed',
            'news': 'neutral',
            'collapse': 'skeptical',
            'sustainability': 'action-oriented',
            'renewable': 'action-oriented',
            'energy': 'technical',
            'green': 'lifestyle',
            'conservation': 'scientific',
        }
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
        client_id = os.environ.get('REDDIT_CLIENT_ID')
        client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        user_agent = os.environ.get(
            'REDDIT_USER_AGENT', 'Climate_Change_Research_Script/1.0')
        if not client_id or not client_secret:
            raise ValueError("Missing Reddit API credentials in .env file.")
        try:
            reddit = praw.Reddit(
                client_id=client_id, client_secret=client_secret, user_agent=user_agent)
            reddit.user.me()  # test connection
            logger.info("Connected to Reddit API")
            return reddit
        except PrawcoreException as e:
            logger.error(f"Error initializing Reddit API: {e}")
            raise

    def get_subreddits_list(self, custom_subreddits=None):
        if custom_subreddits:
            return custom_subreddits
        env_subs = os.environ.get('REDDIT_SUBREDDITS', '')
        if env_subs:
            return [s.strip() for s in env_subs.split(',')]
        return list(self.tone_map.keys())

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def scrape_subreddit(self, subreddit_name):
        logger.info(f"Scraping r/{subreddit_name} for query: {self.query}")
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data, comments_data = [], []
        tone = self.tone_map.get(subreddit_name.lower(), 'unknown')

        for post in tqdm(subreddit.search(self.query, limit=self.limit, time_filter=self.time_filter),
                         desc=f"Posts from r/{subreddit_name}", total=self.limit):
            if post.selftext in ["[deleted]", "[removed]"] or not is_english(post.selftext):
                continue
            post_data = {
                'id': post.id,
                'subreddit': subreddit_name,
                'tone': tone,
                'title': post.title,
                'text': post.selftext,
                'created_utc': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments,
                'upvote_ratio': post.upvote_ratio,
                'url': post.url,
                'is_self': post.is_self,
                'author': str(post.author) if post.author else '[deleted]',
                'scraped_utc': time.time(),
                'query_used': self.query
            }
            posts_data.append(post_data)

            try:
                post.comment_sort = os.environ.get(
                    'REDDIT_COMMENT_SORT', 'top')
                post.comments.replace_more(limit=0)
                for i, comment in enumerate(post.comments):
                    if i >= self.comment_limit:
                        break
                    if comment.body in ["[deleted]", "[removed]"] or not is_english(comment.body):
                        continue
                    comment_data = {
                        'id': comment.id,
                        'parent_id': comment.parent_id,
                        'post_id': post.id,
                        'subreddit': subreddit_name,
                        'tone': tone,
                        'text': comment.body,
                        'created_utc': comment.created_utc,
                        'score': comment.score,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'scraped_utc': time.time(),
                        'query_used': self.query
                    }
                    comments_data.append(comment_data)
            except Exception as e:
                logger.warning(
                    f"Error fetching comments for post {post.id}: {e}")
                self.stats['rate_limit_hits'] += 1
                time.sleep(min(60, 5 * 2**self.stats['rate_limit_hits']))

            time.sleep(0.5 + random.random())

        self.stats['total_posts'] += len(posts_data)
        self.stats['total_comments'] += len(comments_data)
        self.stats['successful_subreddits'].append(subreddit_name)
        return pd.DataFrame(posts_data), pd.DataFrame(comments_data)

    def scrape_multiple_subreddits(self, subreddits=None):
        all_posts, all_comments = pd.DataFrame(), pd.DataFrame()
        for sub in self.get_subreddits_list(subreddits):
            try:
                p, c = self.scrape_subreddit(sub)
                if not p.empty:
                    all_posts = pd.concat([all_posts, p.drop_duplicates(
                        subset=["title", "text"])], ignore_index=True)
                if not c.empty:
                    all_comments = pd.concat([all_comments, c.drop_duplicates(
                        subset=["text", "post_id"])], ignore_index=True)
                time.sleep(2 + random.random() * 3)
            except Exception as e:
                logger.error(f"Error scraping {sub}: {e}")
        self.stats['end_time'] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        return all_posts, all_comments

    def save_data(self, posts_df, comments_df, output_dir='data/raw'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        posts_file = os.path.join(output_dir, f"reddit_posts_{timestamp}.csv")
        comments_file = os.path.join(
            output_dir, f"reddit_comments_{timestamp}.csv")
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")

        posts_df.to_csv(posts_file, index=False)
        comments_df.to_csv(comments_file, index=False)
        self.stats.update({
            "posts_file": posts_file,
            "comments_file": comments_file,
            "metadata_file": metadata_file,
            "tone_mapping": self.tone_map
        })
        with open(metadata_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        logger.info(
            f"Saved {len(posts_df)} posts and {len(comments_df)} comments")
        return self.stats


def load_queries_from_yaml(path="utils/queries.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='climate change')
    parser.add_argument('--time-filter', type=str, default='week',
                        choices=['hour', 'day', 'week', 'month', 'year', 'all'])
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--comment-limit', type=int, default=25)
    parser.add_argument('--subreddits', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='data/raw')
    return parser.parse_args()


def main():

    logger.info("=== Reddit scraping session started ===")

    start_time = datetime.datetime.now()
    args = parse_arguments()
    custom_subreddits = [s.strip() for s in args.subreddits.split(
        ',')] if args.subreddits else None
    queries = load_queries_from_yaml()

    for category, query_list in queries.items():
        for query in query_list:
            for year in range(2015, 2020):
                for month in range(1, 13):
                    start_date = datetime.datetime(year, month, 1)
                    last_day = calendar.monthrange(year, month)[1]
                    end_date = datetime.datetime(
                        year, month, last_day, 23, 59, 59)

                    logger.info(f"Scraping '{query}' for {year}-{month:02}")
                    scraper = RedditScraper(
                        query=query, time_filter='all', limit=args.limit, comment_limit=args.comment_limit)
                    posts_df, comments_df = scraper.scrape_multiple_subreddits(
                        custom_subreddits)

                    # Filter by actual timestamp
                    posts_df['created_utc'] = pd.to_datetime(
                        posts_df['created_utc'], unit='s')
                    comments_df['created_utc'] = pd.to_datetime(
                        comments_df['created_utc'], unit='s')
                    posts_df = posts_df[(posts_df['created_utc'] >= start_date) & (
                        posts_df['created_utc'] <= end_date)]
                    comments_df = comments_df[(comments_df['created_utc'] >= start_date) & (
                        comments_df['created_utc'] <= end_date)]

                    if posts_df.empty and comments_df.empty:
                        logger.info(f"No data found for {year}-{month:02}")
                        continue

                    posts_df["query_category"] = category
                    comments_df["query_category"] = category

                    month_dir = os.path.join(args.output_dir, "monthly")
                    os.makedirs(month_dir, exist_ok=True)
                    month_tag = f"{year}_{month:02}"

                    posts_path = os.path.join(
                        month_dir, f"reddit_posts_{month_tag}.csv")
                    comments_path = os.path.join(
                        month_dir, f"reddit_comments_{month_tag}.csv")

                    posts_df.to_csv(posts_path, index=False,
                                    quoting=csv.QUOTE_ALL)
                    comments_df.to_csv(
                        comments_path, index=False, quoting=csv.QUOTE_ALL)

                    logger.info(
                        f"Saved {len(posts_df)} posts and {len(comments_df)} comments for {month_tag}")
                    logger.info(
                        f"Saved {len(posts_df)} posts and {len(comments_df)} comments for {month_tag}")
                    logger.info(
                        "=== Reddit scraping session completed successfully ===")
                    logger.info(
                        f"Total time taken: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    main()
