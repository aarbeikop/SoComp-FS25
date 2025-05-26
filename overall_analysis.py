import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pearsonr, linregress
import logging
import os
from statsmodels.tsa.seasonal import STL

# --------------------------
# Configuration
# --------------------------
reddit_path   = 'data/processed/reddit_data_downsampled.csv'
twitter_path  = 'data/processed/twitter_data_downsampled.csv'
output_dir    = 'data/processed/analysis_results'
window_days   = 30  # rolling window for December 2016

os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Helper functions
# --------------------------

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Saved plot to {path}")

# --------------------------
# Main Analysis
# --------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Starting analysis')

    # Load datasets
    reddit = pd.read_csv(reddit_path, parse_dates=['created_utc'])
    twitter = pd.read_csv(twitter_path, parse_dates=['tweet_date'])
    logging.info(f'Loaded {len(reddit)} Reddit and {len(twitter)} Twitter posts')

    # 1. Overall Sentiment Distribution
    for name, df in [('Reddit', reddit), ('Twitter', twitter)]:
        props = df['sentiment_category'].value_counts(normalize=True).sort_index()
        fig, ax = plt.subplots()
        props.plot(kind='bar', ax=ax, title=f"Sentiment Distribution ({name})")
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Proportion')
        save_plot(fig, os.path.join(output_dir, f'{name.lower()}_sentiment_dist.png'))

    # Effect size (Cramér's V)
    r_cnt = reddit['sentiment_category'].value_counts().sort_index()
    t_cnt = twitter['sentiment_category'].value_counts().sort_index()
    cont = pd.concat([r_cnt, t_cnt], axis=1, keys=['reddit','twitter']).fillna(0)
    chi2, p, _, _ = chi2_contingency(cont.values)
    n = cont.values.sum(); phi2=chi2/n
    r_dim, k_dim = cont.shape
    phi2corr = max(0, phi2 - ((k_dim-1)*(r_dim-1))/(n-1))
    cramers_v = (phi2corr / min(k_dim-1, r_dim-1))**0.5
    with open(os.path.join(output_dir, 'sentiment_effect_size.txt'), 'w') as f:
        f.write(f"p={p:.3g}, V={cramers_v:.3f}\n")
    logging.info(f'Effect size p={p:.3g}, V={cramers_v:.3f}')

    # 2. Sentiment by Content Length
    for name, df, col in [('Reddit', reddit, 'cleaned_text'), ('Twitter', twitter, 'cleaned_message')]:
        lengths = df[col].dropna().str.split().apply(len)
        bins = pd.qcut(lengths, 4, labels=['Q1','Q2','Q3','Q4'])
        ct = pd.crosstab(bins, df.loc[lengths.index, 'sentiment_category'], normalize='index')
        fig, ax = plt.subplots()
        ct.plot(kind='bar', stacked=True, ax=ax, legend=False)
        ax.set_title(f'Sentiment by Length ({name})')
        ax.set_xlabel('Length Quartile')
        ax.set_ylabel('Proportion')
        save_plot(fig, os.path.join(output_dir, f'{name.lower()}_sent_len.png'))

    # 3. Subreddit Sentiment
    if 'subreddit' in reddit.columns:
        ct = pd.crosstab(reddit['subreddit'], reddit['sentiment_category'], normalize='index')
        fig, ax = plt.subplots(figsize=(8,4))
        ct.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Subreddit Sentiment')
        ax.set_xlabel('Subreddit')
        ax.set_ylabel('Proportion')
        save_plot(fig, os.path.join(output_dir, 'subreddit_sent.png'))

    # 4. Engagement vs Sentiment
    if 'num_comments' in reddit: reddit['engagement']=reddit['num_comments']
    if {'retweet_count','favorite_count'}.issubset(twitter):
        twitter['engagement']=twitter['retweet_count']+twitter['favorite_count']
    for name, df in [('Reddit', reddit), ('Twitter', twitter)]:
        if 'engagement' in df and 'sentiment_compound' in df:
            corr, pval = pearsonr(df['sentiment_compound'], df['engagement'])
            with open(os.path.join(output_dir, f'{name.lower()}_eng.csv'),'w') as f:
                f.write(f"r={corr:.3f}, p={pval:.3g}\n")
            logging.info(f'{name} engagement r={corr:.3f}')

    # 5. December 2016 Rolling CI
    for name, df, date_col in [('Reddit', reddit,'created_utc'), ('Twitter', twitter,'tweet_date')]:
        dec = df[df[date_col].dt.to_period('M')=='2016-12']
        if dec.empty: continue
        s=dec.set_index(date_col)['sentiment_compound'].sort_index()
        roll=s.rolling(f'{window_days}D')
        m=roll.mean(); se=roll.std()/roll.count()**0.5
        fig, ax=plt.subplots(figsize=(8,3))
        ax.plot(m.index,m); ax.fill_between(m.index,m-1.96*se,m+1.96*se,alpha=0.3)
        ax.set_title(f'Dec 2016 Sentiment ({name})')
        save_plot(fig, os.path.join(output_dir, f'{name.lower()}_decci.png'))

    # 6. Readability & Regression
    try:
        import textstat
        reddit['read'] = reddit['cleaned_text'].dropna().apply(textstat.flesch_reading_ease)
        twitter['read'] = twitter['cleaned_message'].dropna().apply(textstat.flesch_reading_ease)
        # Combined violin
        fig, ax = plt.subplots(figsize=(6,4))
        data=[reddit[reddit.sentiment_category==c]['read'].dropna() for c in ['negative','neutral','positive']]
        ax.violinplot(data,positions=[0,1,2],showmeans=True)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(['neg','neu','pos'])
        ax.set_title('Readability by Sentiment (Reddit)')
        save_plot(fig, os.path.join(output_dir,'reddit_violin_read.png'))
        fig, ax = plt.subplots(figsize=(6,4))
        data=[twitter[twitter.sentiment_category==c]['read'].dropna() for c in ['negative','neutral','positive']]
        ax.violinplot(data,positions=[0,1,2],showmeans=True)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(['neg','neu','pos'])
        ax.set_title('Readability by Sentiment (Twitter)')
        save_plot(fig, os.path.join(output_dir,'twitter_violin_read.png'))
        # Scatter + regression
        fig, ax = plt.subplots(figsize=(6,5))
        for name, df, col in [('Reddit',reddit,'read'),('Twitter',twitter,'read')]:
            df2=df.dropna(subset=[col,'sentiment_compound'])
            ax.scatter(df2[col],df2['sentiment_compound'],alpha=0.3,label=name)
            slope,intercept,r,_,_=linregress(df2[col],df2['sentiment_compound'])
            x_vals=pd.Series([df2[col].min(),df2[col].max()])
            ax.plot(x_vals,intercept+slope*x_vals,'--',label=f"{name} r={r:.2f}")
        ax.set_xlabel('Flesch Reading Ease'); ax.set_ylabel('Compound Sentiment')
        ax.legend(); fig.tight_layout()
        save_plot(fig, os.path.join(output_dir,'readability_sent_scatter.png'))
    except ImportError:
        logging.warning('textstat missing; skip readability/regression')

    logging.info('Analysis complete')

if __name__=='__main__':
    main()
