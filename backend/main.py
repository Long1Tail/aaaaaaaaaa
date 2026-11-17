
import os
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

app = FastAPI()

INFERENCE_URL = os.getenv(
    "INFERENCE_URL",
    "http://inference:8001"
)

MAX_FILES = 20
MAX_SIZE = 100 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff"
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_images(
    files: list[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(400, "No files provided")

    if len(files) > MAX_FILES:
        raise HTTPException(413, "Too many files")

    payload = []
    total_size = 0

    for file in files:

        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                415,
                f"Unsupported file: {file.filename}"
            )

        content = await file.read()
        total_size += len(content)

        if total_size > MAX_SIZE:
            raise HTTPException(
                413,
                "Total upload size exceeds 100 MB"
            )

        if not content:
            raise HTTPException(
                400,
                f"Empty file: {file.filename}"
            )

        payload.append((
            "files",
            (
                file.filename or "image.jpg",
                content,
                file.content_type
            )
        ))

    try:
        async with httpx.AsyncClient(
            timeout=300
        ) as client:

            response = await client.post(
                f"{INFERENCE_URL}/predict",
                files=payload
            )

    except httpx.TimeoutException:
        raise HTTPException(
            504,
            "Inference timeout"
        )

    except httpx.HTTPError as e:
        raise HTTPException(
            502,
            f"Inference service unavailable: {e}"
        )

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail", "Inference error"
            )
        except Exception:
            detail = "Inference error"

        raise HTTPException(
            response.status_code,
            detail
        )

    return Response(
        content=response.content,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                'attachment; filename="results.zip"'
        }
    )
