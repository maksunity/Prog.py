import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random
def load_data(data_dir='fontan_test_image', label_file='_annotations.csv'):
    labels_path = os.path.join(data_dir, label_file)
    df = pd.read_csv(labels_path)

    grouped = df.groupby('filename')

    images_data = []
    for filename, group in grouped:
        img_path = os.path.join(data_dir, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading: {img_path}")
                continue

            # Для каждого объекта в изображении
            for _, row in group.iterrows():
                # Вырезаем ROI по bounding box
                x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                roi = img[y1:y2, x1:x2]

                images_data.append({
                    'filename': filename,
                    'roi': roi,
                    'label': row['class']  # class содержит цифру (0-9)
                })

        except Exception as e:
            print(f"Error {e} with image: {img_path}")

    return images_data



def preprocess_digit(roi, target_size=(64, 64)):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Изменение размера с сохранением пропорций
    h, w = thresh.shape
    scale = target_size[1] / max(h, w)
    resized = cv2.resize(thresh, (int(w * scale), int(h * scale)))

    pad_h = target_size[1] - resized.shape[0]
    pad_w = target_size[0] - resized.shape[1]
    padded = cv2.copyMakeBorder(resized,
                                pad_h // 2, pad_h - pad_h // 2,
                                pad_w // 2, pad_w - pad_w // 2,
                                cv2.BORDER_CONSTANT,
                                value=0)
    return padded

# Фонтанное преобразование
def fountain_features(roi, max_points=200):
    # Поиск контуров
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    features = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:  # Фильтр мелких шумов
            for point in cnt:
                features.extend(point[0].tolist())

    # Нормализация длины
    if len(features) > max_points:
        features = features[:max_points]
    else:
        features += [0] * (max_points - len(features))

    return np.array(features)


def visualize_predictions(model, data, num_samples=3):
    plt.figure(figsize=(15, 5))
    indices = random.sample(range(len(data)), num_samples)

    for i, idx in enumerate(indices):
        item = data[idx]
        roi = item['roi'].copy()
        processed = preprocess_digit(roi)
        features = fountain_features(processed)

        # Предсказание
        pred = model.predict([features])[0]
        true_label = item['label']

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        title_color = 'green' if pred == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred}", color=title_color)
        plt.axis('off')

    plt.show()

def visualize_by_filename(model, data, filename):
    plt.figure(figsize=(15, 5))

    # Найти все ROI из указанного файла
    matches = [item for item in data if item['filename'] == filename]

    if not matches:
        print(f"Файл {filename} не найден в данных!")
        return

    for i, item in enumerate(matches[:3]):
        # item = data[idx]
        roi = item['roi'].copy()
        processed = preprocess_digit(roi)
        features = fountain_features(processed)

        pred = model.predict([features])[0]
        true_label = item['label']

        plt.subplot(1, min(3, len(matches)), i + 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        title_color = 'green' if pred == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred}", color=title_color)
        plt.axis('off')

    plt.show()
def main():
    data = load_data()

    # Подготовка датасета
    X = []
    y = []
    for item in data:
        try:
            processed = preprocess_digit(item['roi'])
            features = fountain_features(processed)
            X.append(features)
            y.append(item['label'])
        except Exception as e:
            print(f"Skipping item due to error: {e}")

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(),PCA(n_components=0.95),SVC(kernel='rbf', C=10, gamma='scale'))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    visualize_predictions(model, data)
    user_file = "Video_2024-01-25_142650420_jpg.rf.49e408f43d058a4d5e2cdf26a89eb09e.jpg"
    visualize_by_filename(model, data, user_file)

if __name__ == "__main__":
    main()