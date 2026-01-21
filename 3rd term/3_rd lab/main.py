import json
import re
import os
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from gensim import corpora, models
from gensim.models import CoherenceModel

warnings.filterwarnings('ignore')


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_texts_from_news(data):
    texts = []
    for item in data:
        if 'content' in item and item['content']:
            text = item['content']
            if len(text) > 20:
                texts.append(text)
    return texts


def extract_texts_from_products(data):
    texts = []
    for item in data:
        if 'text' in item and item['text']:
            text = item['text']
            if len(text) > 20:
                texts.append(text)
    return texts


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_texts(texts):
    processed = []
    for text in texts:
        processed_text = preprocess_text(text)
        if len(processed_text) > 10:
            processed.append(processed_text)
    return processed


def create_tfidf_vectors(texts, max_features=100):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        max_df=0.85,
        min_df=2,
        stop_words='english'
    )
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors


def create_count_vectors(texts, max_features=100):
    vectorizer = CountVectorizer(
        max_features=max_features,
        max_df=0.85,
        min_df=2,
        stop_words='english'
    )
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors


def apply_nmf(vectors, n_components=10):
    nmf_model = NMF(
        n_components=n_components,
        random_state=42,
        max_iter=200
    )
    nmf_model.fit(vectors)
    return nmf_model


def apply_lda_sklearn(vectors, n_components=10):
    lda_model = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method='online',
        learning_offset=50.0,
        random_state=42
    )
    lda_model.fit(vectors)
    return lda_model


def prepare_gensim_data(texts):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return dictionary, corpus, tokenized_texts


def apply_lda_gensim(corpus, dictionary, n_topics=10):
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        iterations=100,
        alpha='auto',
        per_word_topics=True
    )
    return lda_model


def calculate_coherence_score(model, texts, dictionary, corpus):
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def plot_top_words(model, feature_names, n_top_words, title):
    n_components = model.components_.shape[0]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx >= 10:
            break
        top_features_ind = topic.argsort()[-n_top_words:][::-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        ax.set_title(f'Topic {topic_idx + 1}', fontsize=12)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def get_top_words_list(model, feature_names, n_top_words=10):
    topics_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:][::-1]
        top_features = [(feature_names[i], topic[i]) for i in top_features_ind]
        topics_words.append(top_features)
    return topics_words


def get_document_topics(model, vectors):
    doc_topics = model.transform(vectors)
    return doc_topics


def save_topics_to_json(topics_words, output_file):
    topics_dict = {}
    for idx, words in enumerate(topics_words):
        topics_dict[f'topic_{idx + 1}'] = [
            {'word': word, 'weight': float(weight)} for word, weight in words
        ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(topics_dict, f, ensure_ascii=False, indent=2)


def save_document_topics(doc_topics, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('doc_id,')
        f.write(','.join([f'topic_{i+1}' for i in range(doc_topics.shape[1])]))
        f.write('\n')
        
        for doc_idx, topics in enumerate(doc_topics):
            f.write(f'{doc_idx},')
            f.write(','.join([f'{score:.4f}' for score in topics]))
            f.write('\n')


def process_dataset(dataset_name, texts, n_components=10, n_top_words=10, output_dir='output'):
    print(f'\n{"="*50}')
    print(f'Processing {dataset_name} dataset')
    print(f'Number of documents: {len(texts)}')
    print(f'{"="*50}\n')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_texts = preprocess_texts(texts)
    print(f'After preprocessing: {len(processed_texts)} documents')
    
    print('\n--- NMF with TF-IDF ---')
    tfidf_vectorizer, tfidf_vectors = create_tfidf_vectors(processed_texts, max_features=100)
    nmf_model = apply_nmf(tfidf_vectors, n_components=n_components)
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    fig = plot_top_words(nmf_model, tfidf_feature_names, n_top_words, 
                         f'NMF Topics - {dataset_name}')
    fig.savefig(os.path.join(output_dir, f'{dataset_name}_nmf_topics.png'), dpi=150)
    plt.close()
    print(f'Saved NMF visualization: {dataset_name}_nmf_topics.png')
    
    nmf_topics = get_top_words_list(nmf_model, tfidf_feature_names, n_top_words)
    save_topics_to_json(nmf_topics, os.path.join(output_dir, f'{dataset_name}_nmf_topics.json'))
    
    nmf_doc_topics = get_document_topics(nmf_model, tfidf_vectors)
    save_document_topics(nmf_doc_topics, os.path.join(output_dir, f'{dataset_name}_nmf_doc_topic.csv'))
    print(f'Saved NMF document topics: {dataset_name}_nmf_doc_topic.csv')
    
    print('\n--- LDA with Sklearn ---')
    count_vectorizer, count_vectors = create_count_vectors(processed_texts, max_features=100)
    lda_sklearn_model = apply_lda_sklearn(count_vectors, n_components=n_components)
    
    count_feature_names = count_vectorizer.get_feature_names_out()
    fig = plot_top_words(lda_sklearn_model, count_feature_names, n_top_words, 
                         f'LDA Topics (Sklearn) - {dataset_name}')
    fig.savefig(os.path.join(output_dir, f'{dataset_name}_lda_topics.png'), dpi=150)
    plt.close()
    print(f'Saved LDA visualization: {dataset_name}_lda_topics.png')
    
    lda_topics = get_top_words_list(lda_sklearn_model, count_feature_names, n_top_words)
    save_topics_to_json(lda_topics, os.path.join(output_dir, f'{dataset_name}_lda_topics.json'))
    
    lda_doc_topics = get_document_topics(lda_sklearn_model, count_vectors)
    save_document_topics(lda_doc_topics, os.path.join(output_dir, f'{dataset_name}_lda_doc_topic.csv'))
    print(f'Saved LDA document topics: {dataset_name}_lda_doc_topic.csv')
    
    print('\n--- LDA with Gensim ---')
    dictionary, corpus, tokenized_texts = prepare_gensim_data(processed_texts)
    lda_gensim_model = apply_lda_gensim(corpus, dictionary, n_topics=n_components)
    
    coherence_score = calculate_coherence_score(
        lda_gensim_model, 
        tokenized_texts, 
        dictionary, 
        corpus
    )
    print(f'Coherence Score (Gensim LDA): {coherence_score:.4f}')
    
    print('\nTop words per topic (Gensim LDA):')
    for idx, topic in lda_gensim_model.print_topics(num_words=n_top_words):
        print(f'Topic {idx + 1}: {topic}')
    
    print(f'\nCompleted processing {dataset_name} dataset!\n')


def main():
    n_components = 10
    n_top_words = 10
    
    news_file = '../2_nd lab/news.json'
    pstu_file = '../2_nd lab/pstu.json'
    
    print('Loading news dataset...')
    try:
        news_data = load_json_data(news_file)
        news_texts = extract_texts_from_news(news_data)
        print(f'Loaded {len(news_texts)} news articles')
        
        process_dataset('news', news_texts, n_components, n_top_words, output_dir='out_news')
    except FileNotFoundError:
        print(f'News file not found: {news_file}')
    except Exception as e:
        print(f'Error processing news: {e}')
    
    print('\n' + '='*70 + '\n')
    
    print('Loading PSTU news dataset...')
    try:
        pstu_data = load_json_data(pstu_file)
        pstu_texts = extract_texts_from_news(pstu_data)
        print(f'Loaded {len(pstu_texts)} PSTU news articles')
        
        process_dataset('pstu', pstu_texts, n_components, n_top_words, output_dir='out_pstu')
    except FileNotFoundError:
        print(f'PSTU file not found: {pstu_file}')
    except Exception as e:
        print(f'Error processing PSTU news: {e}')
    
    print('\n' + '='*70)
    print('All processing completed!')
    print('='*70)


if __name__ == '__main__':
    main()
