import json
import os
from datetime import datetime

clustered_articles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"../public/2025-05-19.json")
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"../public/{datetime.now().strftime('%Y-%m-%d')}.json")
with open(clustered_articles_path, 'r', encoding='utf-8') as f:
    clustered_articles = json.load(f)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(clustered_articles, f, ensure_ascii=False, indent=2)

print(f"Clustered articles saved to {output_path}")
