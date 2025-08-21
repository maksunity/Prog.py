# health_news_clustering_improved.py
import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, LatentDirichletAllocation
from kneed import KneeLocator
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(directory):
    tweets = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            tweets.append(parts[2])
            except UnicodeDecodeError:
                with open(os.path.join(directory, filename), 'r', encoding='latin-1') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            tweets.append(parts[2])
    return tweets


def enhanced_preprocessor(text):
    # Удаление URL, упоминаний, хэштегов
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    # Удаление пунктуации (кроме апострофа)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    # Удаление цифр и специальных символов
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()

    # Токенизация
    tokens = word_tokenize(text)

    # Удаление стоп-слов и коротких токенов
    custom_stopwords = {
        'say', 'said', 'like', 'get', 'make', 'take', 'use', 'go', 'know', 'one',
        'would', 'could', 'also', 'many', 'new', 'year', 'time', 'day', 'want'
    }
    stop_words = set(stopwords.words('english')) | custom_stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 3]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def find_optimal_clusters(X, max_k=10):
    sse = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    try:
        kl = KneeLocator(range(2, max_k + 1), sse, curve='convex', direction='decreasing')
        return kl.elbow if kl.elbow is not None else 5  # Возвращаем значение по умолчанию
    except:
        return 5


def cluster_analysis(vectorizer, model, n_top_words=10):
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    if hasattr(model, 'cluster_centers_'):
        # Для KMeans
        for i in range(model.n_clusters):
            centroid = model.cluster_centers_[i]
            top_indices = centroid.argsort()[-n_top_words:][::-1]
            cluster_keywords[f"Cluster {i}"] = [terms[ind] for ind in top_indices]

    return cluster_keywords


def visualize_clusters(X, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                          cmap='viridis', alpha=0.6, s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True)
    plt.show()


def main():
    # Параметры
    DATA_DIR = 'Health-Tweets'
    MIN_TWEET_LENGTH = 7  # Минимальное количество слов в твите

    # 1. Загрузка данных
    print("Loading data...")
    raw_tweets = load_data(DATA_DIR)
    print(f"Loaded {len(raw_tweets)} raw tweets")

    # 2. Улучшенная предобработка
    print("Preprocessing text...")
    processed_tweets = [enhanced_preprocessor(tweet) for tweet in raw_tweets]

    # Фильтрация коротких твитов
    filtered_tweets = [t for t in processed_tweets if len(t.split()) >= MIN_TWEET_LENGTH]
    print(f"After filtering: {len(filtered_tweets)} tweets")

    # 3. Векторизация с биграммами
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=MIN_TWEET_LENGTH,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(filtered_tweets)

    # 4. Определение оптимального числа кластеров
    print("Finding optimal cluster number...")
    optimal_k = find_optimal_clusters(X)
    # Защита от None
    if optimal_k is None:
        optimal_k = 5  # Значение по умолчанию
        print(f"Cannot determine optimal clusters. Using default: {optimal_k}")
    else:
        print(f"Optimal number of clusters: {optimal_k}")

    # 5. Кластеризация KMeans
    print("\nRunning KMeans clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_

    # 6 Оценка качества
    silhouette_kmeans = silhouette_score(X, kmeans_labels)
    print(f"KMeans Silhouette Score: {silhouette_kmeans:.3f}")


    # 7. Визуализация
    visualize_clusters(X, kmeans_labels, "KMeans Clustering")

    # 8. Анализ кластеров
    print("\nKMeans Cluster Topics:")
    kmeans_keywords = cluster_analysis(vectorizer, kmeans)
    for cluster, words in kmeans_keywords.items():
        print(f"{cluster}: {', '.join(words)}")

    # 9. Примеры твитов для каждого кластера (KMeans)
    print("\nExample tweets per cluster (KMeans):")
    df = pd.DataFrame({
        'original_text': raw_tweets[:len(filtered_tweets)],
        'processed_text': filtered_tweets,
        'cluster': kmeans_labels
    })

    for cluster in sorted(df['cluster'].unique()):
        cluster_tweets = df[df['cluster'] == cluster]['original_text'].sample(2, random_state=42)
        print(f"\nCluster {cluster} examples:")
        for tweet in cluster_tweets:
            print(f"- {tweet[:150]}...")


if __name__ == "__main__":
    main()