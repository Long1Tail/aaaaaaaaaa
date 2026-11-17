
import io
import os
import json
import zipfile
from pathlib import Path
from threading import Lock

import torch

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO


MODEL_PATH = "/models/best.pt"
IMAGE_SIZE = 1280
CONFIDENCE = 0.25

DEVICE = (
    "0" if torch.cuda.is_available()
    else "cpu"
)

MAX_IMAGE_SIZE = 20 * 1024 * 1024

app = FastAPI()

model = None
inference_lock = Lock()


@app.on_event("startup")
def startup():

    global model

    if not Path(MODEL_PATH).is_file():
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully")
    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)


@app.get("/health")
def health():

    if model is None:
        raise HTTPException(503, "Model not loaded")

    return {
        "status": "ok",
        "device": DEVICE,
        "imgsz": IMAGE_SIZE
    }


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...)
):

    if model is None:
        raise HTTPException(503, "Model unavailable")

    if not files:
        raise HTTPException(400, "No images")

    output = io.BytesIO()

    metadata = {
        "model": "YOLOv8s E2",
        "imgsz": IMAGE_SIZE,
        "images": []
    }

    with inference_lock:

        with zipfile.ZipFile(
            output,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            for index, file in enumerate(files):

                content = await file.read()

                if not content or len(content) > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        413,
                        f"Invalid or oversized image: {file.filename}"
                    )

                try:
                    image = Image.open(
                        io.BytesIO(content)
                    ).convert("RGB")

                except (UnidentifiedImageError, OSError):
                    raise HTTPException(
                        415,
                        f"Cannot decode: {file.filename}"
                    )

                result = model.predict(
                    source=image,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    device=DEVICE,
                    verbose=False
                )[0]

                # Ultralytics returns an annotated BGR image.
                annotated = result.plot()

                image_bytes = io.BytesIO()

                Image.fromarray(
                    annotated[:, :, ::-1]
                ).save(
                    image_bytes,
                    format="JPEG",
                    quality=92
                )

                original_name = Path(
                    file.filename or f"image_{index}.jpg"
                ).name

                output_name = (
                    f"{index:03d}_"
                    f"{Path(original_name).stem[:70]}.jpg"
                )

                archive.writestr(
                    f"images/{output_name}",
                    image_bytes.getvalue()
                )

                detections = []

                if result.boxes is not None:

                    boxes = result.boxes

                    coordinates = (
                        boxes.xyxy.cpu().tolist()
                    )

                    classes = (
                        boxes.cls.cpu().tolist()
                    )

                    confidences = (
                        boxes.conf.cpu().tolist()
                    )

                    for bbox, cls, conf in zip(
                        coordinates,
                        classes,
                        confidences
                    ):

                        class_id = int(cls)

                        detections.append({
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": float(conf),
                            "bbox_xyxy": bbox
                        })

                metadata["images"].append({
                    "input": original_name,
                    "output": output_name,
                    "width": image.width,
                    "height": image.height,
                    "detections": detections
                })

            archive.writestr(
                "detections.json",
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2
                )
            )

    return Response(
        content=output.getvalue(),
        media_type="application/zip"
    )
