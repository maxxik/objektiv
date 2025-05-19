import json

from src.article_clustering import run_clustering
from src.rss_fetcher import RSSFetcher


def load_outlets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        outlets = json.load(f)
    return {outlet["name"]: {"rss_url": outlet["rss_url"], "bias": outlet["bias"]} for outlet in outlets}


if __name__ == "__main__":
    # fetch articles from the outlets
    outlets_file = "../config/outlets.json"
    feeds = load_outlets(outlets_file)
    fetcher = RSSFetcher(feeds)
    fetcher.fetch()

    # cluster the articles
    run_clustering("../artifacts/articles", "../artifacts/clustered_articles.json", "../artifacts/processed_pairs.json")


