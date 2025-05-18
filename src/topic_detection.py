import os
import json
from datetime import datetime

from openai import OpenAI

client = OpenAI()


class TopicDetector:
    def __init__(self, topics_file, processed_articles_file):
        self.topics_file = topics_file
        self.processed_articles_file = processed_articles_file
        self.topics = self._load_topics()
        self.processed_articles = self._load_processed_articles()

        self.embedded_articles = self._load_article_embeddings()

    def _load_topics(self):
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_topics(self):
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.topics, f, ensure_ascii=False, indent=2)

    def _load_processed_articles(self):
        if os.path.exists(self.processed_articles_file):
            with open(self.processed_articles_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_processed_articles(self):
        with open(self.processed_articles_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_articles, f, ensure_ascii=False, indent=2)

    def _prepare_existing_topics(self):
        # Prepare existing topics and their articles for the prompt
        existing_topics = []
        for topic in self.topics:
            topic_name, articles = topic["name"], topic["articles"]
            if articles:
                if any(article["id"] == article_id for article in articles):
                    print(f"Article {article_id} already exists in topic {topic_name}.")
                    return topic_name

                # skip if the topic is older than 1 day
                topic_created_at = datetime.fromisoformat(topic["createdAt"])
                if (datetime.now() - topic_created_at).days > 1:
                    continue

                existing_topics.append(
                    f"Topic name: \"{topic_name}\"\nRelated articles: {', '.join([article['title'] for article in articles])}\n")
        existing_topics_str = "\n".join(existing_topics) if existing_topics else "None"
        return existing_topics_str


    def detect_topic(self, article_id, title, content):
        if article_id in self.processed_articles:
            print(f"Article has already been processed. Skipping.")
            return None

        existing_topics_str = self._prepare_existing_topics()

        prompt = (
            "Your task is to group news articles that report on the exact same real-world event or incident—not just similar topics or themes. "
            "Carefully compare the main content and details of each article to determine if they are reporting on the same specific event. "
            "ONLY assign an article to an existing group if it clearly covers the same specific event, development, or announcement as the articles already in that group. "
            "If the article is about a different event, create a new group for it.\n"
            "When creating a new group, the 'news_group_title' must be a clear, descriptive, and concise summary of the specific event or incident in Hungarian. "
            "Use a neutral, factual tone, similar in style to article headlines. Do NOT use general categories or thematic titles—always refer to the precise event (e.g., 'Szabó István életműdíjat kapott a 2025-ös Cannes-i Filmfesztiválon').\n"
            "Additionally, classify whether the article is factually reporting on an event (as opposed to being opinionated or subjective content).\n"
            "\n------\n"
            f"Existing news groups and their articles:\n{existing_topics_str}\n\n"
            "\n------\n"
            "Article to classify:\n"
            f"Title: {title}\n\nContent: {content}\n\n"
            "Response format: {\"news_group_title\": \"Konkrét esemény magyarul, semleges stílusban\", \"is_factual\": true/false}"
        )
        print(f"Prompt: {prompt}")
        try:
            response = client.chat.completions.create(model="gpt-4.1-nano",
                                                      messages=[{"role": "user", "content": prompt}],
                                                      max_tokens=100,
                                                      temperature=0.7)
            result = json.loads(response.choices[0].message.content)
            topic = result.get("news_group_title")
            to_be_skipped = not result.get("is_factual", False)

            if topic and not to_be_skipped:
                # Check if topic already exists
                existing_topic = next((t for t in self.topics if t["name"] == topic), None)
                if existing_topic:
                    # Append article to existing topic
                    existing_topic["articles"].append({"id": article_id, "title": title})
                else:
                    # Create new topic
                    self.topics.append({"name": topic, "articles": [{"id": article_id, "title": title}],
                                        "createdAt": datetime.now().isoformat()})
                self._save_topics()
                self.processed_articles.append(article_id)
                self._save_processed_articles()
                print(f"Detected topic: {topic}")
                return topic
            else:
                print(f"Topic detection skipped for article {article_id}.")
        except Exception as e:
            print(f"Failed to detect topic: {e}")
            return None, False

    def detect_topics(self, articles_dir):
        """Detect topics for all articles in the specified directory."""
        for filename in os.listdir(articles_dir):
            if filename.endswith(".json") and not filename.endswith("_raw.json"):
                file_path = os.path.join(articles_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"Processing file: {file_path}")
                    article = json.load(f)
                    article_id = article.get("id")
                    title = article.get("title")
                    content = f'{article.get("summary", "")}\n'
                    # f'Scraped text from website: {article.get("visible_text", "")}'
                    topic = self.detect_topic(article_id, title, content)
                    print(f"Article ID: {article_id}, Topic: {topic}")

    def print_largest_topics(self):
        """Print the 5 largest topics based on the number of articles."""
        sorted_topics = sorted(self.topics, key=lambda x: len(x["articles"]), reverse=True)
        for topic in sorted_topics[:5]:
            print(f"Topic: {topic['name']}, Articles: {len(topic['articles'])}, Created At: {topic['createdAt']}")

if __name__ == "__main__":
    topics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/topics.json")
    processed_articles_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts/grouped_articles.json")
    topic_detector = TopicDetector(topics_file, processed_articles_file)

    # Example usage
    article_id = "12345"
    title = "Interesting change in politics"
    content = "it is a great change in the political landscape of the country."
    topic = topic_detector.detect_topic(article_id, title, content)
    print(f"Detected Topic: {topic}")

