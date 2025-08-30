import json
import os
from datetime import datetime

from src.article_clustering import run_clustering
from src.rss_fetcher import RSSFetcher


def load_outlets(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        outlets = json.load(f)
    return {outlet["name"]: {"rss_url": outlet["rss_url"], "bias": outlet["bias"]} for outlet in outlets}


if __name__ == "__main__":
    # fetch articles from the outlets
    outlets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/outlets.json")
    feeds = load_outlets(outlets_file)
    fetcher = RSSFetcher(feeds)
    # fetcher.fetch()

    # cluster the articles
    run_clustering("../artifacts/articles", "../artifacts/clustered_articles.json", "../artifacts/processed_pairs.json")

    # copy the clustered articles json to output directory
    clustered_articles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/clustered_articles.json")
    output_fn = f"{datetime.now().strftime('%Y-%m-%d')}.json"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../public/{output_fn}")
    with open(clustered_articles_path, 'r', encoding='utf-8') as f:
        clustered_articles = json.load(f)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clustered_articles, f, ensure_ascii=False, indent=2)

    print(f"Clustered articles saved to {output_path}")

    # overwrite ../public/latest.json with the latest clustered articles filename
    latest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../public/latest_run.json")
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump({"latest": output_fn}, f, ensure_ascii=False, indent=2)

