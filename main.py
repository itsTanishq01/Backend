from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uvicorn
from diseases import DISEASE_DATABASE

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "crop_disease_model.keras"
IMG_SIZE = 128
PLANT_DISEASES = list(DISEASE_DATABASE.keys())

model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
async def home():
    return {
        "api_name": "Plant Disease Detection API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Upload an image for disease detection",
            "/": "GET - API information"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image uploaded")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Try again later.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])

    disease_code = PLANT_DISEASES[predicted_class_idx]
    disease_info = DISEASE_DATABASE[disease_code]

    return {
        "prediction": {
            "disease_code": disease_code,
            "disease_name": disease_info["name"],
            "confidence": round(confidence * 100, 2)
        },
        "details": disease_info,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
