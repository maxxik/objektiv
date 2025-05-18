import os
import json
import hashlib
import feedparser
from datetime import datetime
import ssl
import certifi
import requests
from bs4 import BeautifulSoup  # Added for HTML parsing

ssl_context = ssl.create_default_context(cafile=certifi.where())

class RSSFetcher:
    def __init__(self, feeds_dict, artifact_dir_path="../artifacts", article_dir_name="articles", state_file="fetched_articles.json"):
        self.feeds_dict = feeds_dict
        self.artifact_dir = artifact_dir_path
        self.article_dir = os.path.join(artifact_dir_path, article_dir_name)
        self.state_file = os.path.join(artifact_dir_path, state_file)
        os.makedirs(self.article_dir, exist_ok=True)
        self.fetched_articles = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def _save_state(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.fetched_articles), f, ensure_ascii=False, indent=2)

    def _article_id(self, entry, feed_name):
        base = entry.get('title', '') + entry.get('published', '')
        article_id = hashlib.sha256(base.encode('utf-8')).hexdigest()
        title_snippet = entry.get('title', '').split()[:5]
        readable_part = f"{feed_name}_{'_'.join(title_snippet)}"
        # Ensure the readable part contains only alphanumeric characters and underscores
        readable_part = ''.join(c if c.isalnum() or c == '_' else '_' for c in readable_part)
        return f"{readable_part}_{article_id}"

    def fetch(self):
        new_articles = 0
        for outlet_name, outlet_data in self.feeds_dict.items():
            feed_url = outlet_data["rss_url"]
            bias = outlet_data["bias"]
            print(f"Fetching RSS feed from {feed_url}")

            d = feedparser.parse(feed_url)
            for entry in d.entries:
                print(f"Processing rss feed entry: {entry.get('title', 'No Title')}")
                article_id = self._article_id(entry, outlet_name)
                if article_id in self.fetched_articles:
                    continue
                self.fetched_articles.add(article_id)
                article = self._entry_to_dict(entry, feed_url, outlet_name, bias)
                
                # Fetch HTML content and extract visible text
                # article_html = self._fetch_article_html(entry.get("link"))
                # if article_html:
                #     article["html_content"] = article_html
                #     article["visible_text"] = self._extract_visible_text(article_html)
                
                self._save_processed_and_raw_article_json(entry, article, article_id)
                new_articles += 1
        self._save_state()
        print(f"Fetched {new_articles} new articles.")

    def _fetch_article_html(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch HTML content from {url}: {e}")
            return None

    def _extract_visible_text(self, html):
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Get visible text
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            print(f"Failed to extract visible text: {e}")
            return ""

    def _entry_to_dict(self, entry, feed_url, feed_name, bias):
        return {
            "id": self._article_id(entry, feed_name),
            "title": entry.get("title"),
            "link": entry.get("link"),
            # use datetime.strptime(str, "%a, %d %b %Y %H:%M:%S %z") to parse the date
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
            "feed_url": feed_url,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "outlet": feed_name,
            "bias": bias
        }

    def _save_processed_and_raw_article_json(self, entry, article, article_id):
        fname = os.path.join(self.article_dir, f"{article_id}.json")
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)

        # raw_fname = os.path.join(self.article_dir, f"{article_id}_raw.json")
        # with open(raw_fname, 'w', encoding='utf-8') as f:
        #     json.dump(entry, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    feeds = {
        "Telex": "https://telex.hu/rss",
        # "Mandiner": "https://mandiner.hu/rss",
        # "Origo": "https://www.origo.hu/publicapi/hu/rss/origo/articles",
        # "444": "https://444.hu/feed"
    }
    fetcher = RSSFetcher(feeds)
    fetcher.fetch()
