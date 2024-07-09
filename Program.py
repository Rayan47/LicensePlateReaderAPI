__MODEL__ = "best.pt"
IMAGEDIR = "images/"

import cv2
import easyocr
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile

import uuid

model = YOLO(__MODEL__)

reader = easyocr.Reader(['en'])


def Run(path: str):
    results = model.predict(path, show=False)

    boxes = []
    for box in results[0].boxes.xyxy.cpu().detach().numpy():
        boxes.append(box)

    images = []

    for boxy in boxes:
        box = list(map(int, boxy))
        img = cv2.imread(path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grey = cv2.threshold(img_grey, 100, 255, cv2.THRESH_BINARY_INV)[1]

        img_rgb = img_grey[box[1]:box[3], box[0]:box[2]]
        images.append(img_rgb)

    final = [reader.readtext(im)[0][1] for im in images]
    print(final)
    outputs: list[str] = []
    for out in final:
        curr = ""
        for ch in out:
            if ch.isalnum():
                curr += ch
        outputs.append(curr)
    return {i: outputs[i] for i in range(len(outputs))}


app = FastAPI()


@app.get("/")
def read_root():
    return "Hello World!"


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
    ret = Run(f"{IMAGEDIR}{file.filename}")
    return ret
