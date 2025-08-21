from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# Обучение

# if __name__ == '__main__':
#     model = YOLO("yolo11n.pt")
#     results = model.train(data="C:/Users/maksunity/PycharmProjects/Prog_Den/Prog.py/2nd term/4. Детекция объектов/my_dataset_v2/data.yaml",
#                           epochs=120,
#                           imgsz=640,
#                           batch=30,
#                           patience=30,
#                           cache="ram",
#                           project="my_yolo",
#                           name="v1.2",
#                           device="cuda")


# использование
# model = YOLO("my_yolo/v1/weights/best.pt")
# model = YOLO("my_yolo/v1.1/weights/best.pt")
model = YOLO("my_yolo/v1.2/weights/best.pt")
# image_path = "data/Video_2024-01-25_1426501277.jpg"
image_path = "data/Video_2024-01-25_142650513.jpg"


results = model(image_path)
if results:
    # Показать изображение с результатами
    results[0].show()
    # Сохранить результат в файл
    results[0].save(filename="output.jpg")

    for r in results:
        boxes = r.boxes
        if boxes:
            names = r.names
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                print(f"Объект: {names[cls_id]}, Точность: {conf:.2f}, Координаты: {xyxy}")
else:
    print("Нет объектов для распознавания.")

