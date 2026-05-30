from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from PIL import Image

import io   
import os  
import uuid


app = FastAPI()#create an object of the FastAPI class.
templates = Jinja2Templates(directory="templates")# Create FastAPI app instance
model = YOLO("yolov8n.pt")# Load the YOLOv8 model. You can choose different versions like yolov8s.pt, yolov8m.pt, etc.

app.mount("/static", StaticFiles(directory="static"), name="static")# Set up Jinja2. For rendering html in python and setting the directory for html files. In this case, we will create a folder called "templates" and put our html files in there.


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"image": None}
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")#

    results = model(image)

    result_image = results[0].plot()

    filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join("static", filename)

    Image.fromarray(result_image).save(output_path)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "image": f"/static/{filename}"
        }
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}
