import os
import json


def __load_clusters(cluster_file_path):
    if os.path.exists(cluster_file_path):
        with open(cluster_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def __load_articles(articles_dir):
    articles = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".json") and not file_name.endswith("_raw.json"):
            with open(os.path.join(articles_dir, file_name), 'r', encoding='utf-8') as f:
                articles.append(json.load(f))
    return articles


def run_post_processing(cluster_file_path = "../artifacts/clustered_articles.json", articles_dir = "../artifacts/articles"):
    clusters = __load_clusters(cluster_file_path)
    articles = __load_articles(articles_dir)

