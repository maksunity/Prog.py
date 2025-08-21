from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics.utils.metrics import ap_per_class
import cv2

# 1. Загрузка модели
# model = YOLO("my_yolo/v1.1/weights/best.pt")
model = YOLO("my_yolo/v1.2/weights/best.pt")

# 2. Пути к данным
test_images_dir = "my_dataset_v2/test/images"
test_labels_dir = "my_dataset_v2/test/labels"
output_dir = "output_new"
os.makedirs(output_dir, exist_ok=True)


# 3. Функция загрузки истинных меток
def load_true_labels(labels_dir):
    true_labels = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f.readlines():
                    class_id = int(line.strip().split()[0])
                    true_labels.append(class_id)
    return true_labels


# 4. Сбор предсказаний и истинных меток
y_true = load_true_labels(test_labels_dir)
y_pred = []
confidences = []

# for image_file in os.listdir(test_images_dir):
#     if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#         # Предсказание модели
#         results = model(os.path.join(test_images_dir, image_file))
#
#         # Обработка результатов
#         for r in results:
#             boxes = r.boxes
#             if boxes:
#                 for box in boxes:
#                     y_pred.append(int(box.cls.item()))
#                     confidences.append(float(box.conf.item()))

for image_file in os.listdir(test_images_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_images_dir, image_file)

        # Предсказание модели
        results = model(img_path)

        # Сохранение результата с детекциями
        if results:
            output_path = os.path.join(output_dir, image_file)
            results[0].save(filename=output_path)

        # Обработка результатов
        for r in results:
            boxes = r.boxes
            if boxes:
                for box in boxes:
                    y_pred.append(int(box.cls.item()))
                    confidences.append(float(box.conf.item()))

# 5. Выравнивание данных
min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

# 6. Генерация отчетов
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# 7. Визуализация
def plot_class_distribution(y_true, y_pred):
    classes = sorted(set(y_true + y_pred))

    plt.figure(figsize=(12, 6))
    plt.hist([y_true, y_pred], bins=len(classes), label=['True', 'Predicted'])
    plt.xticks(classes)
    plt.title('Class Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


plot_class_distribution(y_true, y_pred)

