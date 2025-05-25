import os
import json
from datetime import datetime, timezone, timedelta

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import numpy as np
from sklearn.cluster import KMeans


def load_articles(directory, age_limit=1):
    """Load articles from JSON files in the specified directory."""
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.endswith("_raw.json"):
            with open(os.path.join(directory, filename), "r") as file:
                data = json.load(file)
                hungarian_tz = timezone(timedelta(hours=1))  # Hungarian timezone (CET, UTC+1)
                article_published_at = datetime.strptime(data["published"].replace("GMT", "+0000"),
                                                         "%a, %d %b %Y %H:%M:%S %z")
                if (datetime.now(hungarian_tz) - article_published_at).days > age_limit:
                    print(f"Skipping article {filename} due to age limit.")
                    continue
                articles.append(data)  # Assuming each file contains a list of articles
    print("length of articles: ", len(articles))
    return articles


def generate_embeddings(articles):
    """Generate embeddings for article content using OpenAI, skipping already embedded articles."""
    embeddings = []
    for article in articles:
        article_id = article["id"]
        embedding_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"../artifacts/embeddings/{article_id}.json"
        )

        if os.path.exists(embedding_file):
            # Load existing embedding
            with open(embedding_file, "r") as file:
                embedding_data = json.load(file)
                embeddings.append(embedding_data["embedding"])
        else:
            # Generate new embedding
            text = (article['title'] + " " + article['summary'])
            # + (" " + article['visible_text'] if 'visible_text' in article else ""))
            print(f"Generating embedding for: \n{text}\n")
            response = client.embeddings.create(input=text, model="text-embedding-3-small")
            embedding = response.data[0].embedding
            embeddings.append(embedding)

            # Save the embedding to a file
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            with open(embedding_file, "w") as file:
                json.dump({"id": article_id, "embedding": embedding}, file, indent=4, ensure_ascii=False)

    return np.array(embeddings)


def get_semantically_similar_articles(embeddings, threshold=0.8):
    """Get semantically similar articles using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Find similar articles based on the threshold
    similar_articles = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                similar_articles.append((i, j))

    return similar_articles, similarity_matrix


def print_similarity_stats(articles, similar_articles):
    print(f"{len(similar_articles)} similar articles found.")

    # print statistics about the similar articles
    print(f"Number of articles: {len(articles)}")
    print(f"Number of similar article pairs: {len(similar_articles)}")
    # print number of articles with at least 1, 2, 3 similar articles

    similar_article_counts = {}
    for i, j in similar_articles:
        if articles[i]["id"] not in similar_article_counts:
            similar_article_counts[articles[i]["id"]] = 0
        similar_article_counts[articles[i]["id"]] += 1
        if articles[j]["id"] not in similar_article_counts:
            similar_article_counts[articles[j]["id"]] = 0
        similar_article_counts[articles[j]["id"]] += 1
    # print number of articles with at least 1, 2, 3 similar articles
    for i in range(1, 4):
        count = len([k for k, v in similar_article_counts.items() if v >= i])
        print(f"Number of articles with at least {i} similar articles: {count}")


def export_articles_and_similarities_to_neo4j(articles, similarity_matrix, labels):
    """Export articles and their similarities to Neo4j."""
    from py2neo import Graph

    # Connect to Neo4j
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "objektiv"))

    # Clear existing data
    graph.run("MATCH (n) DETACH DELETE n")

    # Create nodes for each article with a dynamic label based on clusterId
    for i, article in enumerate(articles):
        cluster_label = f"Cluster_{labels[i]}"  # Create a dynamic label for the cluster
        graph.run(
            f"CREATE (a:Article:{cluster_label} {{id: $id, title: $title, summary: $summary, outlet: $outlet, bias: $bias, link: $link, clusterId: $clusterId}})",
            id=article["id"],
            title=article["title"],
            summary=article["summary"],
            outlet=article["outlet"],
            bias=article["bias"],
            link=article["link"],
            clusterId=str(labels[i]),
        )

    # Create relationships based on similarity and weights
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.8:
                graph.run(
                    "MATCH (a:Article {id: $id1}), (b:Article {id: $id2}) "
                    "CREATE (a)-[:SIMILAR {weight: $weight}]->(b)",
                    id1=articles[i]["id"],
                    id2=articles[j]["id"],
                    weight=int(similarity_matrix[i][j] * 100),
                )

    print("Exported articles and similarities to Neo4j.")


def leiden_clustering(similarity_matrix, resolution_parameter=1):
    """
    Perform Leiden clustering on the similarity matrix.
    Converts the similarity matrix to a graph and applies the Leiden algorithm.
    """
    import igraph as ig
    import leidenalg

    # Convert similarity matrix to a graph
    edges = np.transpose(np.triu_indices_from(similarity_matrix, k=1))
    weights = similarity_matrix[edges[:, 0], edges[:, 1]]

    # Create graph
    g = ig.Graph(directed=False)
    g.add_vertices(similarity_matrix.shape[0])  # Each observation is a node
    g.add_edges(edges)

    # Assign weights to edges
    g.es['weight'] = weights

    # Perform Leiden clustering
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights=g.es['weight'], resolution_parameter=resolution_parameter
    )
    groups = np.array(partition.membership)

    print("Executed Leiden clustering.")

    return groups


def pairwise_article_comparison(article1, article2, processed_pairs_file):
    processed_pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), processed_pairs_file)
    processed_pairs = []
    if os.path.exists(processed_pairs_path):
        with open(processed_pairs_path, "r") as file:
            processed_pairs = json.load(file)

        # Check if the pair has already been processed
        for pair in processed_pairs:
            if (pair["articleId1"] == article1["id"] and pair["articleId2"] == article2["id"]) or \
                    (pair["articleId1"] == article2["id"] and pair["articleId2"] == article1["id"]):
                print(f"Pair {pair['articleId1']} and {pair['articleId2']} already processed.")
                return pair["result"]

    # if titles and summaries are the same, return True
    if article1["title"] == article2["title"] and article1["summary"] == article2["summary"]:
        print(f"Pair {article1['id']} and {article2['id']} are exactly the same.")
        processed_pairs.append({"articleId1": article1["id"], "articleId2": article2["id"], "result": True})
        with open(processed_pairs_path, "w") as file:
            json.dump(processed_pairs, file, indent=4, ensure_ascii=False)
        return True

    prompt = (
        f"Given the following two articles, determine if they are about the same event or incident.\n"
        f"Article 1: {article1['title']}\n"
        f"{article1['summary']}\n"
        f"Article 2: {article2['title']}\n"
        f"{article2['summary']}\n"
        f"Answer with 'yes' or 'no' ONLY.\n"

    )
    print(f"Prompt: {prompt}")
    try:
        response = client.chat.completions.create(model="gpt-4.1-nano",
                                                  messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=5,
                                                  temperature=0.7)
        result_str = response.choices[0].message.content
        result = result_str.strip().lower() == "yes"
        print(f"Result: {result}")
        # Save the processed pair and result
        processed_pairs.append({"articleId1": article1["id"], "articleId2": article2["id"], "result": result})
        with open(processed_pairs_path, "w") as file:
            json.dump(processed_pairs, file, indent=4, ensure_ascii=False)


    except Exception as e:
        print(f"pairwise article comparison failed: {e}")


def generate_title_for_a_cluster(cluster):
    # merge the titles and summaries of the articles in the cluster
    combined_articles = "\n".join(
        ["Title: " + article["articleTitle"] + "\nSummary: " + article["articleSummary"] for article in
         cluster["articles"]])

    # Generate a title using OpenAI
    prompt = (f"Generate a title for the following articles:\n{combined_articles}"
              "When creating a new group, the 'news_group_title' must be a clear, descriptive, and concise summary of the specific event or incident in Hungarian. "
              "Use a neutral, factual tone, similar in style to article headlines. Do NOT use general categories or thematic titles—always refer to the precise event (e.g., 'Szabó István életműdíjat kapott a 2025-ös Cannes-i Filmfesztiválon').\n")

    response = client.chat.completions.create(model="gpt-4o",
                                              messages=[{"role": "user", "content": prompt}],
                                              max_tokens=50,
                                              temperature=0.7)
    generated_title = response.choices[0].message.content.strip()
    print(prompt)
    print(f"Generated title: {generated_title}")
    return generated_title


def run_clustering(input_directory="../artifacts/articles", output_file="../artifacts/clustered_articles.json",
                   processed_pairs_file="../artifacts/processed_pairs.json"):
    input_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_directory)
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)

    # Load articles
    articles = load_articles(input_directory)

    # Generate embeddings
    embeddings = generate_embeddings(articles)

    # Get semantically similar articles
    similar_articles, similarity_matrix = get_semantically_similar_articles(embeddings, threshold=0.6)

    print_similarity_stats(articles, similar_articles)

    # Perform pairwise article comparison
    matching_matrix = np.zeros((len(articles), len(articles)), dtype=bool)
    counter = 0
    total = len(similar_articles)
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if similarity_matrix[i][j] > 0.6:
                print(f"[{counter}/{total}] Pairwise comparison of articles")
                counter += 1
                result = pairwise_article_comparison(articles[i], articles[j], processed_pairs_file)
                matching_matrix[i][j] = result
                matching_matrix[j][i] = result

    # Perform clustering
    # labels = louvain_clustering(similarity_matrix, resolution_parameter=1.5)
    labels = leiden_clustering(matching_matrix, resolution_parameter=1)

    # Export articles and their similarities to Neo4j
    # export_articles_and_similarities_to_neo4j(articles, matching_matrix.astype(int), labels)

    # print stats about the clusters, print the articles in a cluster
    clusters = []
    for i, label in enumerate(labels):
        label = str(label)
        current_article = {"articleId": articles[i]["id"], "articleTitle": articles[i]["title"],
                           "articleSummary": articles[i]["summary"], "articleOutlet": articles[i]["outlet"],
                           "articleBias": articles[i]["bias"], "articleUrl": articles[i]["link"],}
        # if label is not found among clusterIds
        if not any(cluster["clusterId"] == label for cluster in clusters):
            clusters.append({"clusterId": label,
                             "articles": [current_article]})
        else:
            for cluster in clusters:
                if cluster["clusterId"] == label:
                    cluster["articles"].append(current_article)

    # drop clusters with less than 3 articles
    clusters = [cluster for cluster in clusters if len(cluster["articles"]) >= 3]

    # print number of clusters
    print(f"Number of clusters: {len(clusters)}")

    # generate title for each cluster
    for cluster in clusters:
        cluster["clusterTitle"] = generate_title_for_a_cluster(cluster)

    # save the clusters with article id only
    with open(output_file, "w") as file:
        json.dump(clusters, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    run_clustering()

