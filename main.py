
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from typing import List
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load YOLO model
yolo = cv2.dnn.readNet("./yolov3-tiny.weights", "./yolov3-tiny.cfg")
with open("./coco.names", 'r') as f:
    classes = f.read().splitlines()

def detect_objects(image: np.ndarray) -> np.ndarray:
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        yolo.setInput(blob)
        output_layers_names = yolo.getUnconnectedOutLayersNames()
        layer_output = yolo.forward(output_layers_names)

        boxes, confidences, class_ids = [], [], []

        for output in layer_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error: {e}")
        return image

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)
    result_image = detect_objects(image)
    _, buffer = cv2.imencode('.jpg', result_image)
    return JSONResponse(content={"image": buffer.tobytes()})
