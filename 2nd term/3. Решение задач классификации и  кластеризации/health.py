import os
import re
import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


# Функция для корректной кодировки файлов
def safe_read_file(filepath):
    encodings = ['utf-8', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with codecs.open(filepath, 'r', encoding=enc, errors='ignore') as f:
                return f.readlines()
        except Exception as e:
            print(f"Не удалось прочитать {filepath} с кодировкой {enc}: {e}")
    return []


# Загрузка данных
folder = "Health-Tweets"  # путь к папке с .txt файлами
texts = []

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        lines = safe_read_file(os.path.join(folder, filename))
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                texts.append(parts[2]) # заголовок новости

print("Количество заголовков:", len(texts))
cleaned_texts = [re.sub(r"http\S+", "", t) for t in texts]

# Векторизация
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=2)
X_tfidf = vectorizer.fit_transform(cleaned_texts)

# Кластеризация
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_tfidf)

print("\nТоп ключевых слов по кластерам:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(kmeans.n_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"🔹 Кластер {i}: {', '.join(top_terms)}")

print("\nПримеры заголовков по кластерам:")
for cluster_id in range(kmeans.n_clusters):
    print(f"\nКластер {cluster_id}")
    cluster_texts = [text for i, text in enumerate(texts) if labels[i] == cluster_id]
    for example in cluster_texts[:10]:  # 10 заголовков
        print(f"  - {example}")


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="tab10")
plt.title("Кластеризация заголовков новостей о здоровье")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
