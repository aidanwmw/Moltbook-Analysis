import os
import pickle
import pandas as pd
import networkx as nx
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_platform_colors():
    """Returns the standardized color palette for the project."""
    return {"Moltbook": "#2C5F8A", "Reddit": "#FF4500"}

def _get_cache_path(name):
    # Ensure the Data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"cache_{name}.pkl")

def load_moltbook_graph(cache=True):
    """Loads or builds the Moltbook reply graph."""
    cache_path = _get_cache_path("moltbook_graph")
    
    if cache and os.path.exists(cache_path):
        logging.info(f"Loading Moltbook graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logging.info("Building Moltbook graph from Hugging Face dataset...")
    df_comments = pd.DataFrame(load_dataset("SimulaMet/moltbook-observatory-archive", "comments")["archive"])
    df_comments = df_comments.sort_values("fetched_at").drop_duplicates("id", keep="last")

    parent_map = df_comments[["id", "agent_id"]].rename(columns={"id": "parent_id", "agent_id": "parent_agent"})
    reply_edges = (df_comments.merge(parent_map, on="parent_id", how="left")
                   .dropna(subset=["agent_id", "parent_agent"]))
    reply_edges = reply_edges[reply_edges["agent_id"] != reply_edges["parent_agent"]]

    G = nx.DiGraph(name="Moltbook")
    for (src, tgt), grp in reply_edges.groupby(["agent_id", "parent_agent"]):
        G.add_edge(src, tgt, weight=len(grp))
        
    logging.info(f"Moltbook loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(G, f)
            
    return G

def load_reddit_graph(cache=True, limit=50000):
    """Loads or builds the Reddit reply graph."""
    cache_path = _get_cache_path(f"reddit_graph_{limit}")
    
    if cache and os.path.exists(cache_path):
        logging.info(f"Loading Reddit graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logging.info(f"Building Reddit graph from Hugging Face dataset (limit={limit})...")
    ds = load_dataset("anhchanghoangsg/reddit_pushshift_dataset_cleaned", split="train", streaming=True) 
    
    records = []
    for row in ds.take(limit):
        records.append({k: row[k] for k in ["author", "parent_id", "subreddit", "name"]})
    df_reddit = pd.DataFrame(records).dropna()

    df_reddit["comment_id"] = df_reddit["name"].str.split("_").str[-1]
    df_reddit["parent_comment_id"] = df_reddit["parent_id"].where(
        df_reddit["parent_id"].str.startswith("t1_")
    ).str.split("_").str[-1]

    id2author = df_reddit.set_index("comment_id")["author"].to_dict()
    reddit_edges = (df_reddit.dropna(subset=["parent_comment_id"])
                    .assign(parent_author=lambda d: d["parent_comment_id"].map(id2author))
                    .dropna(subset=["parent_author"]))
    reddit_edges = reddit_edges[reddit_edges["author"] != reddit_edges["parent_author"]]

    G = nx.DiGraph(name="Reddit")
    for (src, tgt), grp in reddit_edges.groupby(["author", "parent_author"]):
        G.add_edge(src, tgt, weight=len(grp))
        
    logging.info(f"Reddit loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(G, f)
            
    return G

if __name__ == "__main__":
    # Test loading
    G_molt = load_moltbook_graph()
    G_red = load_reddit_graph()
    print("Graphs loaded successfully.")
