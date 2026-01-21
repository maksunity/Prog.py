import os
import json
import cv2
import yt_dlp
import whisper
import torch
from ultralytics import YOLO
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def get_device():
    """Определение доступного устройства для вычислений"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ GPU не доступен, используется CPU")
    return device


def download_video(url, output_path='videos'):
    """Скачивание видео с YouTube/RuTube"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"Видео скачано: {filename}")
            return filename, info
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return None, None


def extract_frames(video_path, fps=1):
    """Извлечение кадров из видео"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    timestamps = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            timestamp = frame_count / video_fps
            timestamps.append(timestamp)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Извлечено {extracted_count} кадров из видео")
    return frames, timestamps


def detect_objects_yolo(frames, timestamps, model_name='yolov8n.pt', device='cpu'):
    """Детекция объектов на кадрах с помощью YOLO"""
    model = YOLO(model_name)
    
    detections = []
    all_objects = []
    
    print(f"Анализ кадров с YOLO на {device.upper()}...")
    for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        results = model(frame, verbose=False, device=device)
        
        frame_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                if confidence > 0.5:
                    frame_objects.append({
                        'object': class_name,
                        'confidence': confidence,
                        'timestamp': timestamp
                    })
                    all_objects.append(class_name)
        
        if frame_objects:
            detections.append({
                'timestamp': timestamp,
                'objects': frame_objects
            })
        
        if (idx + 1) % 10 == 0:
            print(f"Обработано кадров: {idx + 1}/{len(frames)}")
    
    return detections, all_objects


def transcribe_audio(video_path, model_name='base', device='cpu'):
    """Транскрипция аудио с помощью Whisper"""
    print(f"Загрузка модели Whisper на {device.upper()}...")
    model = whisper.load_model(model_name, device=device)
    
    print("Транскрибирование аудио...")
    result = model.transcribe(video_path, language='ru', verbose=False)
    
    transcription = {
        'text': result['text'],
        'segments': result['segments']
    }
    
    print(f"Транскрипция завершена. Найдено сегментов: {len(result['segments'])}")
    return transcription


def find_significant_events(detections, transcription):
    """Определение значимых событий в видео"""
    events = []
    
    # События на основе детекций объектов (снижен порог с 3 до 2)
    for detection in detections:
        if len(detection['objects']) >= 2:
            objects_list = [obj['object'] for obj in detection['objects']]
            events.append({
                'timestamp': detection['timestamp'],
                'type': 'visual',
                'description': f"Обнаружено {len(objects_list)} объектов: {', '.join(set(objects_list))}",
                'objects': objects_list
            })
    
    # События на основе речи
    for segment in transcription['segments']:
        if len(segment['text'].split()) > 10:
            events.append({
                'timestamp': segment['start'],
                'type': 'speech',
                'description': segment['text'].strip(),
                'duration': segment['end'] - segment['start']
            })
    
    events.sort(key=lambda x: x['timestamp'])
    return events


def cluster_content(transcription_text, n_clusters=3):
    """Кластеризация контента по темам"""
    if not transcription_text or len(transcription_text.strip()) < 20:
        return None, None
    
    # Разбиваем текст на сегменты (по точкам, восклицательным и вопросительным знакам)
    import re
    sentences = re.split(r'[.!?]', transcription_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) < 2:
        return None, None
    
    # Адаптируем количество кластеров под количество предложений
    actual_clusters = min(n_clusters, max(2, len(sentences) // 3))
    
    try:
        # Убираем stop_words для лучшей работы с русским текстом
        vectorizer = TfidfVectorizer(max_features=50, min_df=1)
        X = vectorizer.fit_transform(sentences)
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Группируем предложения по кластерам
        clustered_sentences = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            clustered_sentences[cluster_id].append(sentences[idx])
        
        # Находим ключевые слова для каждого кластера
        topics = {}
        feature_names = vectorizer.get_feature_names_out()
        
        for cluster_id in range(actual_clusters):
            cluster_center = kmeans.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[f"Тема {cluster_id + 1}"] = {
                'keywords': top_words,
                'sentences': clustered_sentences[cluster_id][:3],
                'count': len(clustered_sentences[cluster_id])
            }
        
        return clusters, topics
    except Exception as e:
        print(f"Ошибка кластеризации: {e}")
        return None, None


def generate_statistics(detections, all_objects, transcription, events):
    """Генерация статистики по видео"""
    object_counts = Counter(all_objects)
    
    stats = {
        'total_frames_analyzed': len(detections),
        'total_objects_detected': len(all_objects),
        'unique_objects': len(set(all_objects)),
        'most_common_objects': object_counts.most_common(10),
        'transcription_length': len(transcription['text']),
        'total_events': len(events),
        'visual_events': len([e for e in events if e['type'] == 'visual']),
        'speech_events': len([e for e in events if e['type'] == 'speech'])
    }
    
    return stats


def save_results(video_name, detections, transcription, events, stats, topics):
    """Сохранение результатов анализа"""
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_basename = os.path.splitext(os.path.basename(video_name))[0]
    
    results = {
        'video_name': video_name,
        'detections': detections,
        'transcription': transcription,
        'events': events,
        'statistics': stats,
        'topics': topics
    }
    
    output_file = os.path.join(output_dir, f"{video_basename}_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в: {output_file}")
    return output_file


def display_timeline(events):
    """Отображение временной шкалы событий"""
    print("\n" + "="*80)
    print("ВРЕМЕННАЯ ШКАЛА ЗНАЧИМЫХ СОБЫТИЙ")
    print("="*80)
    print(f"Всего событий: {len(events)}\n")
    
    page_size = 20
    current_page = 0
    
    while True:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(events))
        
        if start_idx >= len(events):
            print("\nДостигнут конец списка")
            break
        
        for idx in range(start_idx, end_idx):
            event = events[idx]
            timestamp = event['timestamp']
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            
            print(f"\n[{idx+1}] {minutes:02d}:{seconds:02d} - {event['type'].upper()}")
            print(f"    {event['description'][:100]}...")
        
        if end_idx >= len(events):
            print("\n(Конец списка)")
            break
        
        print(f"\n--- Страница {current_page + 1} из {(len(events) + page_size - 1) // page_size} ---")
        choice = input("Enter - следующая страница, q - выход: ").strip().lower()
        
        if choice == 'q':
            break
        
        current_page += 1


def search_events(events, query):
    """Поиск событий по ключевому слову"""
    query_lower = query.lower()
    found_events = []
    
    for event in events:
        if query_lower in event['description'].lower():
            found_events.append(event)
        elif event['type'] == 'visual' and 'objects' in event:
            if any(query_lower in obj.lower() for obj in event['objects']):
                found_events.append(event)
    
    return found_events


def interactive_interface(events, stats, topics):
    """Простой интерактивный интерфейс"""
    while True:
        print("\n" + "="*80)
        print("ИНТЕРАКТИВНЫЙ АНАЛИЗ ВИДЕО")
        print("="*80)
        print("1. Показать временную шкалу")
        print("2. Статистика")
        print("3. Темы контента")
        print("4. Поиск событий")
        print("5. Выход")
        
        choice = input("\nВыберите действие: ").strip()
        
        if choice == '1':
            display_timeline(events)
        
        elif choice == '2':
            print("\n" + "="*80)
            print("СТАТИСТИКА")
            print("="*80)
            for key, value in stats.items():
                if key == 'most_common_objects':
                    print(f"\n{key}:")
                    for obj, count in value:
                        print(f"  - {obj}: {count}")
                else:
                    print(f"{key}: {value}")
        
        elif choice == '3':
            if topics:
                print("\n" + "="*80)
                print("ТЕМАТИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ КОНТЕНТА")
                print("="*80)
                print("\nКластеризация текста на основе TF-IDF и K-Means")
                print("Группировка предложений по смысловой близости\n")
                for topic_name, topic_data in topics.items():
                    print(f"\n{topic_name}: ({topic_data.get('count', 0)} предложений)")
                    print(f"  Ключевые слова: {', '.join(topic_data['keywords'])}")
                    if topic_data['sentences']:
                        print(f"  Примеры предложений:")
                        for sentence in topic_data['sentences']:
                            print(f"    - {sentence[:150]}...")
            else:
                print("\nТемы не определены (недостаточно данных для кластеризации)")
                print("Для анализа тем нужно минимум 2 предложения в транскрипции")
        
        elif choice == '4':
            query = input("\nВведите поисковый запрос: ").strip()
            found = search_events(events, query)
            
            if found:
                print(f"\nНайдено событий: {len(found)}")
                for idx, event in enumerate(found[:10]):
                    timestamp = event['timestamp']
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    print(f"\n[{idx+1}] {minutes:02d}:{seconds:02d} - {event['type'].upper()}")
                    print(f"    {event['description'][:100]}...")
            else:
                print("\nСобытия не найдены")
        
        elif choice == '5':
            print("\nВыход из программы")
            break
        
        else:
            print("\nНеверный выбор, попробуйте снова")


def analyze_video(video_source, is_url=False):
    """Основная функция анализа видео"""
    print("\n" + "="*80)
    print("СИСТЕМА АНАЛИЗА ВИДЕОДАННЫХ")
    print("="*80 + "\n")
    
    # Определение устройства
    device = get_device()
    print()
    
    # Шаг 1: Получение видео
    if is_url:
        print("Шаг 1: Скачивание видео...")
        video_path, info = download_video(video_source)
        if not video_path:
            print("Не удалось скачать видео")
            return
    else:
        video_path = video_source
        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return
    
    # Шаг 2: Извлечение кадров
    print("\nШаг 2: Извлечение кадров...")
    frames, timestamps = extract_frames(video_path, fps=1)
    
    # Шаг 3: Детекция объектов
    print("\nШаг 3: Детекция объектов...")
    detections, all_objects = detect_objects_yolo(frames, timestamps, device=device)
    
    # Шаг 4: Транскрипция аудио
    print("\nШаг 4: Транскрипция аудио...")
    transcription = transcribe_audio(video_path, device=device)
    
    # Шаг 5: Определение значимых событий
    print("\nШаг 5: Определение значимых событий...")
    events = find_significant_events(detections, transcription)
    
    # Шаг 6: Кластеризация контента
    print("\nШаг 6: Кластеризация контента...")
    clusters, topics = cluster_content(transcription['text'])
    
    # Шаг 7: Генерация статистики
    print("\nШаг 7: Генерация статистики...")
    stats = generate_statistics(detections, all_objects, transcription, events)
    
    # Шаг 8: Сохранение результатов
    print("\nШаг 8: Сохранение результатов...")
    output_file = save_results(video_path, detections, transcription, events, stats, topics)
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)
    
    # Запуск интерактивного интерфейса
    interactive_interface(events, stats, topics)


def load_existing_results(results_dir='results'):
    """Загрузка существующих результатов анализа"""
    if not os.path.exists(results_dir):
        print("Папка с результатами не найдена")
        return
    
    files = [f for f in os.listdir(results_dir) if f.endswith('_analysis.json')]
    
    if not files:
        print("Нет сохраненных результатов")
        return
    
    print("\nДоступные результаты анализа:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    
    try:
        choice = int(input("\nВыберите файл (номер): ").strip())
        if 1 <= choice <= len(files):
            file_path = os.path.join(results_dir, files[choice - 1])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"\nЗагружен анализ: {results['video_name']}")
            
            events = results.get('events', [])
            stats = results.get('statistics', {})
            topics = results.get('topics')
            
            interactive_interface(events, stats, topics)
        else:
            print("Неверный номер")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Ошибка при загрузке: {e}")


def main():
    """Главная функция"""
    print("\n" + "="*80)
    print("СИСТЕМА АНАЛИЗА ВИДЕОДАННЫХ")
    print("="*80)
    print("1. Проанализировать видео по URL (YouTube/RuTube)")
    print("2. Проанализировать локальный видео файл")
    print("3. Открыть результаты предыдущего анализа")
    print("4. Выход")
    
    choice = input("\nВаш выбор: ").strip()
    
    if choice == '1':
        url = input("Введите URL видео: ").strip()
        analyze_video(url, is_url=True)
    
    elif choice == '2':
        path = input("Введите путь к видео файлу: ").strip()
        analyze_video(path, is_url=False)
    
    elif choice == '3':
        load_existing_results()
    
    elif choice == '4':
        print("Выход")
    
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()
