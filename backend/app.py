import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# -------------------------------
# 1. Initialize App & CORS
# -------------------------------
app = FastAPI(title="CIFAR-10 Image Classifier API")

# Allow requests from React frontend (Vite default is 5173, CRA default is 3000)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. Load Model
# -------------------------------
MODEL_PATH = "cifar_model.h5"

# Check if model file exists before trying to load
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ WARNING: Model file '{MODEL_PATH}' not found in {os.getcwd()}")
    print("Please place your cifar_model.h5 file in the backend folder.")
    # We create a dummy model so the server doesn't crash immediately for testing
    # model = None 
else:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

# CIFAR-10 class names
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

IMG_SIZE = 48 

# -------------------------------
# 3. Helper: Process Uploaded Image
# -------------------------------
def read_image(file_data: bytes):
    try:
        image = Image.open(io.BytesIO(file_data))
        image = image.convert("RGB") # Ensure 3 channels
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0) # Batch dimension
        return image_array
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

# -------------------------------
# 4. Prediction Endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file content immediately (UploadFile is a generator/spooled file)
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Process the image
    try:
        img = read_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Make prediction
    try:
        # Ensure model is loaded
        if 'model' not in globals() or model is None:
             # Fallback for testing if model is missing
             return {"class": "Model Missing", "confidence": 0.0}

        preds = model.predict(img)
        pred_class_index = np.argmax(preds[0])
        confidence = float(preds[0][pred_class_index])
        
        return {
            "class": CLASS_NAMES[pred_class_index], 
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------
# 5. Root Endpoint
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "CIFAR-10 API is running", "model_status": "loaded" if os.path.exists(MODEL_PATH) else "missing"}