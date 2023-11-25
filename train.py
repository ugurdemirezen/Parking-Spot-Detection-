
from ultralytics import YOLO
model=YOLO('yolov8s.pt')
results = model.train(data="C:/Users/Administrator/PycharmProjects/YoloProjects/car/data.yaml", epochs=100,device="0")
