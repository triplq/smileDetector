import onnx
from onnx import helper

# Загружаем модель ONNX
model = onnx.load("Resources/best.onnx")  # запускать из корня репозитория

# Получаем информацию о графе модели
graph = model.graph

# Выводим имена входных узлов
print("Input names:")
for input in graph.input:
    print(input.name)

# Выводим имена выходных узлов
print("Output names:")
for output in graph.output:
    print(output.name)

# yolo task=detect mode=predict model=Resources/best.pt source=Resources/face.jpg show=True imgsz=640 name=yolov8n_v8_50e_infer1280 show_labels=False