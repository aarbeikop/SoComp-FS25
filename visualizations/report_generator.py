"""HTML report generation for sentiment analysis."""

import os
import logging
from datetime import datetime

logger = logging.getLogger('sentiment_analysis')

class HTMLReportGenerator:
    """Class for generating HTML reports of sentiment analysis results."""
    
    def __init__(self, output_dir=None):
        """Initialize the report generator.
        
        Args:
            output_dir (str, optional): Directory for output files.
        """
        from sentiment_analysis.config import DEFAULT_OUTPUT_DIR
        
        # Set output directory
        self.output_dir = output_dir if output_dir else DEFAULT_OUTPUT_DIR
        
        # Create output directory if it doesn't exist
        from sentiment_analysis.utils.file_utils import ensure_dir
        ensure_dir(self.output_dir)
    
    def generate_report(self, stats, visualizations, output_file='sentiment_report.html'):
        """Generate an HTML report with all sentiment analysis results.
        
        Args:
            stats (dict): Dictionary of sentiment statistics.
            visualizations (dict): Dictionary of visualization filepaths.
            output_file (str, optional): Output filename. Defaults to 'sentiment_report.html'.
            
        Returns:
            str: Path to the generated HTML report.
        """
        logger.info(f"Generating HTML report: {output_file}")
        
        # Create HTML report
        report_path = os.path.join(self.output_dir, output_file)
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cross-Platform Sentiment Analysis: Reddit vs Twitter</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 40px;
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .flex-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .stat-box {{
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    width: 30%;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .image-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .positive {{
                    color: green;
                }}
                .neutral {{
                    color: blue;
                }}
                .negative {{
                    color: red;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    background: #f2f2f2;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cross-Platform Sentiment Analysis</h1>
                <p>Comparing Reddit and Twitter discourse on Climate Change</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Analysis Overview</h2>
                    <p>Analysis performed on {datetime.now().strftime('%Y-%m-%d')} using VADER sentiment analysis.</p>
                    
                    <div class="flex-container">
                        <div class="stat-box">
                            <h3>Reddit Posts</h3>
                            <div class="stat-value">{stats.get('reddit_posts', {}).get('count', 0):,}</div>
                        </div>
                        <div class="stat-box">
                            <h3>Reddit Comments</h3>
                            <div class="stat-value">{stats.get('reddit_comments', {}).get('count', 0):,}</div>
                        </div>
                        <div class="stat-box">
                            <h3>Twitter Posts</h3>
                            <div class="stat-value">{stats.get('twitter', {}).get('count', 0):,}</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Sentiment Distribution</h2>
                    <p>Comparing the distribution of positive, neutral, and negative sentiment between platforms.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_distribution.png" alt="Sentiment Distribution">
                    </div>
                    
                    <table>
                        <tr>
                            <th>Platform</th>
                            <th class="positive">Positive</th>
                            <th class="neutral">Neutral</th>
                            <th class="negative">Negative</th>
                        </tr>
                        <tr>
                            <td>Reddit Posts</td>
                            <td class="positive">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('reddit_posts', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                        <tr>
                            <td>Reddit Comments</td>
                            <td class="positive">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('reddit_comments', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                        <tr>
                            <td>Twitter</td>
                            <td class="positive">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('positive', 0):.1%}</td>
                            <td class="neutral">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('neutral', 0):.1%}</td>
                            <td class="negative">{stats.get('twitter', {}).get('sentiment_distribution', {}).get('negative', 0):.1%}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Sentiment by Content Length</h2>
                    <p>How content length affects sentiment expression across platforms.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_by_length.png" alt="Sentiment by Content Length">
                    </div>
                    <p>This visualization shows how average sentiment varies with content length, highlighting platform-specific patterns in sentiment expression.</p>
                </div>
                
                <div class="section">
                    <h2>Sentiment by Engagement</h2>
                    <p>Relationship between engagement metrics and sentiment.</p>
                    <div class="image-container">
                        <img src="visualizations/sentiment_by_engagement.png" alt="Sentiment by Engagement">
                    </div>
                    <p>This chart reveals how user engagement (upvotes, likes, retweets) correlates with sentiment, providing insights into audience preferences on each platform.</p>
                </div>
                
                <div class="section">
                    <h2>Top Words by Sentiment</h2>
                    <p>Most frequent words used in positive, neutral, and negative content.</p>
                    <h3>Positive Sentiment</h3>
                    <div class="image-container">
                        <img src="visualizations/reddit_posts_positive_top_words.png" alt="Reddit Positive Words">
                        <img src="visualizations/twitter_positive_top_words.png" alt="Twitter Positive Words">
                    </div>
                    
                    <h3>Negative Sentiment</h3>
                    <div class="image-container">
                        <img src="visualizations/reddit_posts_negative_top_words.png" alt="Reddit Negative Words">
                        <img src="visualizations/twitter_negative_top_words.png" alt="Twitter Negative Words">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Wordclouds</h2>
                    <p>Visual representation of term frequency by sentiment category.</p>
                    <h3>Reddit Wordclouds</h3>
                    <div class="flex-container">
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_positive_wordcloud.png" alt="Reddit Positive Wordcloud">
                            <p>Positive Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_neutral_wordcloud.png" alt="Reddit Neutral Wordcloud">
                            <p>Neutral Sentiment</p>
                        </div>
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/reddit_posts_negative_wordcloud.png" alt="Reddit Negative Wordcloud">
                            <p>Negative Sentiment</p>
                        </div>
                    </div>
                    
                    <h3>Twitter Wordclouds</h3>
                    <div class="flex-container">
                        <div class="image-container" style="width: 30%;">
                            <img src="visualizations/twitter_positive_wordcloud.png" alt="Twitter Positive Wordcloud">
                            <p>Positive Sentiment</p>"""
        
