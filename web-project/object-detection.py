from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
import sys
import os

os.chdir(sys.path[0])

app = Flask(__name__)


@app.route("/")
def root():
    """
    Статичный контент index.html
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Ручка /detect с методом POST забирает
    изображение и передает в функцию detect_objects_on_image
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(json.dumps(boxes), mimetype="application/json")


def detect_objects_on_image(buf):
    with open('../data/classes-names.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = YOLO("yolov10m.pt")
    results = model.predict(buf)
    result = results[0]
    data = {int(k):v for k,v in data.items()}
    for r in results: r.names=data
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
