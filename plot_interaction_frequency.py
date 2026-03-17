import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

print('Loading Datasets for Temporal Analysis...')
# 1. Moltbook Loading
df_comments = pd.DataFrame(load_dataset('SimulaMet/moltbook-observatory-archive', 'comments')['archive'])
df_comments = df_comments.sort_values('fetched_at').drop_duplicates('id', keep='last')
df_comments['timestamp'] = pd.to_datetime(df_comments['created_at'], format='mixed', utc=True)

# # 2. Reddit Loading
# ds = load_dataset('anhchanghoangsg/reddit_pushshift_dataset_cleaned', split='train', streaming=True)
# records = []
# count = 0
# iterator = iter(ds)

# while count < 1000000:
#     try:
#         row = next(iterator)
#         if all(k in row for k in ['author', 'parent_id', 'subreddit', 'name', 'created_utc']):
#             records.append({k: row[k] for k in ['author', 'parent_id', 'subreddit', 'name', 'created_utc']})
#     except Exception as chunk_e:
#         continue
#     count += 1

# df_reddit = pd.DataFrame(records).dropna()
# df_reddit['timestamp'] = pd.to_datetime(df_reddit['created_utc'], format='mixed', utc=True).dt.floor('D')

# # 3. Calculate Reddit Delays
# df_reddit['comment_id'] = df_reddit['name'].str.split('_').str[-1]
# df_reddit['parent_comment_id'] = df_reddit['parent_id'].where(df_reddit['parent_id'].str.startswith('t1_')).str.split('_').str[-1]
# id2time = df_reddit.set_index('comment_id')['timestamp'].to_dict()

# reddit_edges = (df_reddit.dropna(subset=['parent_comment_id'])
#     .assign(parent_time=lambda d: d['parent_comment_id'].map(id2time))
#     .dropna(subset=['parent_time']))

# reddit_delays = (reddit_edges['timestamp'] - reddit_edges['parent_time']).dt.total_seconds() / (24 * 3600)
# reddit_delays = reddit_delays[reddit_delays >= 0] # Filter out negative anomalies
# # We use +1 before log10 to handle 0 day delays: log10(days + 1)
# reddit_log_delays = np.log10(reddit_delays + 1)

# 4. Calculate Moltbook Delays
time_map = df_comments[['id','timestamp']].rename(columns={'id':'parent_id','timestamp':'parent_time'})
reply_edges = (df_comments.merge(time_map, on='parent_id', how='inner')
    .dropna(subset=['parent_time']))

moltbook_delays = (reply_edges['timestamp'] - reply_edges['parent_time']).dt.total_seconds()
moltbook_delays = moltbook_delays[moltbook_delays >= 0]
moltbook_log_delays = np.log10(moltbook_delays + 1)

# print(f'\nExtracted {len(reddit_delays)} valid Reddit reply delays (days)')
print(f'Extracted {len(moltbook_delays)} valid Moltbook reply delays (seconds)')

# 5. Plotting
plt.figure(figsize=(12, 7))

# We use density=True to normalize the Y-axis so they can overlay perfectly regardless of sample size difference
plt.hist(moltbook_log_delays, bins=100, alpha=0.6, color='royalblue', label='Moltbook (AI)', density=True, edgecolor='black', linewidth=0.5)
# plt.hist(reddit_log_delays, bins=50, alpha=0.6, color='darkorange', label='Reddit (Human)', density=True, edgecolor='black', linewidth=0.5)

plt.title('Reply Delay Distributions (Moltbook AIs)', fontsize=14, pad=15)
plt.xlabel('Log10(Seconds + 1) Delay Between Parent & Reply', fontsize=12)
plt.ylabel('Normalized Frequency Density', fontsize=12)

# X-axis formatting helpers to make the log scale readable
# xticks_log = [np.log10(0+1), np.log10(1+1), np.log10(3+1), np.log10(10+1), np.log10(30+1), np.log10(100+1), np.log10(365+1)]
# xticks_labels = ['Same Day', '1 Day', '3 Days', '10 Days', '1 Month', '100 Days', '1 Year']
# plt.xticks(xticks_log, xticks_labels)

plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('outputs/interaction_frequency_comparison.png', dpi=150)
print('\nSaved plot to interaction_frequency_comparison.png')
