import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

def train_knn_mnist():
    print("[INFO] Загрузка MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(np.uint8)

    print("[INFO] Обучение KNN...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn


def preprocess_brake_image(img_path):
    print(f"[INFO] Загрузка и предобработка изображения: {img_path}")
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    plt.imshow(thresh, cmap='gray')
    plt.title("Бинаризация")
    plt.axis("off")
    plt.show()

    return image, thresh


def extract_digits(thresh_img):
    print("[INFO] Поиск и извлечение цифр...")
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if 20 < w < 100 and 20 < h < 100:
        # if 10 < w < 150 and 10 < h < 150:
        aspect_ratio = w / float(h)
        if 10 < w < 150 and 10 < h < 150 and 0.3 < aspect_ratio < 1.3:
            roi = thresh_img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.flatten().reshape(1, -1)
            digits.append(roi)
            boxes.append((x, y, w, h))

    boxes_digits = sorted(zip(boxes, digits), key=lambda b: b[0][0])
    boxes = [b[0] for b in boxes_digits]
    digits = [b[1] for b in boxes_digits]

    for idx, roi in enumerate(digits):
        digit_img = roi.reshape(28, 28)
        plt.subplot(1, len(digits), idx + 1)
        plt.imshow(digit_img, cmap='gray')
        plt.axis('off')
    plt.show()

    return digits, boxes



def recognize_digits(knn, digits):
    print("[INFO] Распознавание цифр...")
    results = []
    for digit_img in digits:
        results.append(knn.predict(digit_img)[0])
    return results

def detect_brake_number_knn(img_path):
    knn = train_knn_mnist()
    image, thresh = preprocess_brake_image(img_path)
    digits, boxes = extract_digits(thresh)
    results = recognize_digits(knn, digits)

    for (x, y, w, h), digit in zip(boxes, results):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    number = ''.join(str(d) for d in results)
    print(f"[RESULT] Распознанный номер башмака: {number}")

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Номер башмака: {number}")
    plt.axis("off")
    plt.show()

    return number

# Пример запуска
if __name__ == "__main__":
    image_path = "data/Video_2024-01-25_1426501314.jpg"  # Укажи путь к своему изображению
    detect_brake_number_knn(image_path)
