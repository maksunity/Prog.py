import os
import re
import codecs
import string
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap.umap_ as umap


# Загрузка необходимых ресурсов NLTK с обработкой ошибок
def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        stopwords.words('english')
    except LookupError:
        print("Загрузка необходимых ресурсов NLTK...")
        nltk.download(['stopwords', 'wordnet', 'omw-1.4'], quiet=True)


download_nltk_resources()

# Инициализация инструментов предобработки
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def preprocess_text(text):
    """Функция для комплексной предобработки текста"""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Удаление ссылок, цифр и спецсимволов
    text = re.sub(r"http\S+|\d+|[@#]", "", text.lower())

    # Удаление пунктуации и лишних пробелов
    text = ''.join([c for c in text if c not in punctuation])
    text = re.sub(r"\s+", " ", text).strip()

    # Лемматизация и удаление стоп-слов
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2])


# Загрузка и предобработка данных с обработкой ошибок
def load_data(folder):
    texts = []
    print("Загрузка и предобработка данных...")

    for filename in tqdm(os.listdir(folder)):
        if filename.endswith(".txt"):
            try:
                with codecs.open(os.path.join(folder, filename), 'r',
                                 encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"\nОшибка при чтении файла {filename}: {e}")
                continue

            for line in lines:
                try:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        processed = preprocess_text(parts[2])
                        if processed:
                            texts.append(processed)
                except Exception as e:
                    print(f"\nОшибка при обработке строки: {line[:50]}...: {e}")
                    continue

    print(f"\nУспешно загружено {len(texts)} документов после предобработки")
    return texts


# Основной код
if __name__ == "__main__":
    folder = "Health-Tweets"
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Папка с данными '{folder}' не найдена!")

    texts = load_data(folder)

    if not texts:
        raise ValueError("Не удалось загрузить ни одного документа!")

    # Векторизация с улучшенными параметрами
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=3,
        max_df=0.7
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # Поиск оптимального числа кластеров
    print("\nПоиск оптимального числа кластеров...")
    inertia = []
    silhouettes = []
    k_range = range(3, 11)

    for k in tqdm(k_range):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_tfidf)
            inertia.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_tfidf, labels))
        except Exception as e:
            print(f"\nОшибка при кластеризации для k={k}: {e}")
            continue

    # Визуализация результатов
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.title('Метод локтя')
    plt.xlabel('Число кластеров')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, 'ro-')
    plt.title('Силуэт-анализ')
    plt.xlabel('Число кластеров')
    plt.tight_layout()
    plt.show()

    # Выбор оптимального k
    optimal_k = k_range[np.argmax(silhouettes)]
    print(f"\nОптимальное число кластеров: {optimal_k}")

    # Финальная кластеризация
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tfidf)

    # Оценка качества
    print(f"\nСилуэт-коэффициент: {silhouette_score(X_tfidf, labels):.2f}")

    # Визуализация с UMAP
    print("\nВизуализация кластеров...")
    try:
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_tfidf.toarray())

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="Spectral", s=5)
        plt.title(f"2D визуализация {optimal_k} кластеров (UMAP)")
        plt.colorbar(scatter)
        plt.show()
    except Exception as e:
        print(f"\nОшибка при визуализации UMAP: {e}")
        print("Попытка использовать TSNE вместо UMAP...")

        try:
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_tfidf.toarray())

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="Spectral", s=5)
            plt.title(f"2D визуализация {optimal_k} кластеров (TSNE)")
            plt.colorbar(scatter)
            plt.show()
        except Exception as e:
            print(f"\nОшибка при визуализации TSNE: {e}")

    # Анализ результатов
    print("\nТоп ключевых слов по кластерам:")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"🔹 Кластер {i}: {', '.join(top_terms)}")

    # Примеры документов из кластеров
    print("\nПримеры документов:")
    for cluster_id in range(kmeans.n_clusters):
        print(f"\nКластер {cluster_id}")
        cluster_texts = [text for idx, text in enumerate(texts) if labels[idx] == cluster_id]
        for example in cluster_texts[:3]:
            print(f"  - {example}")