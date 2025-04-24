from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Загрузка модели
model = YOLO("yolo11n.pt")

# model = YOLO(r"runs\detect\train\weights\best.pt")
# model = YOLO(r"mnist\train2\weights\best.pt")

results = model.train(data="datasets\my_dataset\data.yaml", epochs=30, imgsz=640, project="my_yolo",
    name="v1")

# # Загрузка изображения
# image_path = "data/Video_2024-01-25_1426501330.jpg"
# results = model(image_path)
#
# if results:
#     results[0].show()
#     results[0].save(filename="output.jpg")
#
#     for r in results:
#         boxes = r.boxes
#         if boxes:
#             names = r.names
#             for i in range(len(boxes)):
#                 cls_id = int(boxes.cls[i].item())
#                 conf = float(boxes.conf[i].item())
#                 xyxy = boxes.xyxy[i].tolist()
#                 print(f"Объект: {names[cls_id]}, Точность: {conf:.2f}, Координаты: {xyxy}")
# else:
#     print("Нет объектов для распознавания.")
#

