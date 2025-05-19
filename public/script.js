document.addEventListener("DOMContentLoaded", () => {
    const clustersContainer = document.getElementById("clusters-container");

    fetch("2025-05-19.json")
        .then(response => response.json())
        .then(data => {
            // Sort clusters by the number of articles in descending order
            data.sort((a, b) => b.articles.length - a.articles.length);

            data.forEach(cluster => {
                const clusterDiv = document.createElement("div");
                clusterDiv.className = "cluster";

                const clusterTitle = document.createElement("h2");
                clusterTitle.textContent = cluster.clusterTitle;
                clusterDiv.appendChild(clusterTitle);

                const toggleButton = document.createElement("button");
                toggleButton.textContent = "Cikkek mutatása";
                toggleButton.className = "toggle-button";
                clusterDiv.appendChild(toggleButton);

                const articlesContainer = document.createElement("div");
                articlesContainer.className = "articles-container";
                articlesContainer.style.display = "none"; // Initially hidden
                cluster.articles.forEach(article => {
                    const articleDiv = document.createElement("div");
                    articleDiv.className = "article";

                    const articleLink = document.createElement("a");
                    articleLink.href = article.articleUrl;
                    articleLink.textContent = article.articleTitle;
                    articleLink.target = "_blank";
                    articleDiv.appendChild(articleLink);

                    const articleSummary = document.createElement("p");
                    articleSummary.className = "article-summary";
                    articleSummary.textContent = article.articleSummary;
                    articleDiv.appendChild(articleSummary);

                    const articleOutlet = document.createElement("p");
                    articleOutlet.className = "article-outlet";
                    articleOutlet.textContent = `Forrás: ${article.articleOutlet}`;
                    articleDiv.appendChild(articleOutlet);

                    const articleBias = document.createElement("p");
                    articleBias.className = "article-bias";
                    articleBias.textContent = `Irányultság: ${article.articleBias}`;
                    articleDiv.appendChild(articleBias);

                    articlesContainer.appendChild(articleDiv);
                });
                clusterDiv.appendChild(articlesContainer);

                // Bias Bar
                const biasCounts = { Left: 0, Neutral: 0, Right: 0 };
                cluster.articles.forEach(article => {
                    if (biasCounts[article.articleBias] !== undefined) {
                        biasCounts[article.articleBias]++;
                    }
                });

                const totalArticles = cluster.articles.length;
                const leftRatio = (biasCounts.Left / totalArticles) * 100;
                const centerRatio = (biasCounts.Neutral / totalArticles) * 100;
                const rightRatio = (biasCounts.Right / totalArticles) * 100;

                const biasBar = document.createElement("div");
                biasBar.className = "bias-bar";

                const left = document.createElement("div");
                left.className = "left";
                left.style.flex = leftRatio;
                biasBar.appendChild(left);

                const center = document.createElement("div");
                center.className = "center";
                center.style.flex = centerRatio;
                biasBar.appendChild(center);

                const right = document.createElement("div");
                right.className = "right";
                right.style.flex = rightRatio;
                biasBar.appendChild(right);

                clusterDiv.appendChild(biasBar);

                // Source Count
                const leftSources = biasCounts.Left;
                const centerSources = biasCounts.Neutral; // Adjust for possible naming inconsistencies
                const rightSources = biasCounts.Right;

                const sourceCount = document.createElement("div");
                sourceCount.className = "source-count";
                sourceCount.innerHTML = `
                    <p>baloldali: ${leftSources}</p>
                    <p>semleges: ${centerSources}</p>
                    <p>jobboldali: ${rightSources}</p>
                `;
                clusterDiv.appendChild(sourceCount);

                toggleButton.addEventListener("click", () => {
                    if (articlesContainer.style.display === "none") {
                        articlesContainer.style.display = "block";
                        toggleButton.textContent = "Cikkek elrejtése";
                    } else {
                        articlesContainer.style.display = "none";
                        toggleButton.textContent = "Cikkek mutatása";
                    }
                });

                clustersContainer.appendChild(clusterDiv);
            });
        })
        .catch(error => {
            console.error("Error loading JSON data:", error);
        });
});

